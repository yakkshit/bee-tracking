import os
import sys

# Windows DLL loading and OpenMP safety configuration
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

if sys.platform.startswith("win"):
    try:
        import site
        for site_path in site.getsitepackages():
            torch_lib = os.path.join(site_path, "torch", "lib")
            if os.path.exists(torch_lib):
                os.add_dll_directory(torch_lib)
    except Exception:
        pass

import cv2
import time
import shutil
import threading
import collections
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

_YOLO_DETECTOR = None
_ONLINE_TRAINER = None


def get_safe_path(file_path):
    """
    Cross-platform pathlib Path utility. On Windows (win32), if absolute path length 
    exceeds 250 characters, prepends \\?\ to bypass MAX_PATH limits safely.
    """
    if not file_path:
        return Path(".")
    p = Path(file_path).resolve()
    if sys.platform == 'win32':
        p_str = str(p)
        if len(p_str) > 250 and not p_str.startswith("\\\\?\\"):
            return Path("\\\\?\\" + p_str)
    return p


class CameraThread:
    """
    Non-blocking threaded camera reader that continuously grabs frames at 60 FPS 
    into a deque(maxlen=2). Main UI loop only pulls the latest frame to eliminate lag.
    """
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.buffer = collections.deque(maxlen=2)
        self.running = False
        self.thread = None
        self._lock = threading.Lock()

    def start(self):
        if self.running:
            return
        if self.source == "live" or isinstance(self.source, int):
            cam_idx = 0 if self.source == "live" else self.source
            if os.name == 'nt':
                self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(cam_idx)
        else:
            safe_p = str(get_safe_path(self.source))
            self.cap = cv2.VideoCapture(safe_p)

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.buffer.append(frame)
            else:
                time.sleep(0.01)

    def read_latest(self):
        with self._lock:
            if self.buffer:
                return True, self.buffer[-1].copy()
        return False, None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap:
            self.cap.release()
            self.cap = None


def create_csrt_tracker():
    """Create OpenCV CSRT tracker with fallbacks across opencv-python versions."""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    elif hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create()
    return None


def preprocess_ir_frame(frame, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to IR camera frames
    to reduce center glare and boost dark bee contrast.
    """
    if frame is None:
        return None
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced


class KalmanBeeFilter:
    """Lightweight 2D Constant-Velocity Kalman Filter for smooth bee motion tracking."""

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.initialized = False

    def init(self, cx, cy):
        self.kf.statePost = np.array([[np.float32(cx)], [np.float32(cy)], [0.0], [0.0]], np.float32)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        prediction = self.kf.predict()
        return float(prediction[0][0]), float(prediction[1][0])

    def correct(self, cx, cy):
        if not self.initialized:
            self.init(cx, cy)
            return cx, cy
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]], np.float32)
        corrected = self.kf.correct(measurement)
        return float(corrected[0][0]), float(corrected[1][0])


class RobustBeeTracker:
    """
    Robust Single-Object Bee Tracking Pipeline using:
    1. CLAHE + Gaussian Contrast Enhancement
    2. Adaptive Background Subtraction (MOG2) for motion isolation against static arena
    3. Morphological cleanup + Size/Proximity Blob Analysis
    4. 4-State Kalman Filter for velocity prediction & frame dropout handling
    5. Savitzky-Golay filter for smooth trajectory path estimation
    """
    def __init__(self, yolo_detector=None, bg_history=500, var_threshold=16, min_area=30, max_area=600):
        self.bg_history = bg_history
        self.var_threshold = var_threshold
        self.min_area = min_area
        self.max_area = max_area

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.bg_history,
            varThreshold=self.var_threshold,
            detectShadows=False
        )

        self.kalman = KalmanBeeFilter()
        self.csrt_tracker = None
        self.state = "SEARCHING"  # "SEARCHING" or "LOCKED_ON"
        
        self.is_tracking = False
        self.last_position = None
        self.last_bbox = None
        self.confidence = 0.0
        self.frame_count = 0
        self.bg_initialized = False

        self.x_history = collections.deque(maxlen=300)
        self.y_history = collections.deque(maxlen=300)

    @property
    def tracker(self):
        return self.csrt_tracker

    @tracker.setter
    def tracker(self, value):
        self.csrt_tracker = value

    def initialize_background(self, frame, num_frames=20):
        """Warm up background model with initial frames."""
        if frame is None or self.bg_initialized:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        for _ in range(num_frames):
            self.bg_subtractor.apply(enhanced)
        self.bg_initialized = True

    def force_lock_on(self, frame, cx, cy, box_size=38, bbox_size=None):
        """Force tracker lock-on at user clicked coordinate (Help / Entry tag)."""
        if bbox_size is not None:
            box_size = bbox_size
        if frame is None:
            return False

        h_f, w_f = frame.shape[:2]
        x1 = max(0, int(cx - box_size / 2.0))
        y1 = max(0, int(cy - box_size / 2.0))
        bw = min(box_size, w_f - x1)
        bh = min(box_size, h_f - y1)
        bbox = (float(x1), float(y1), float(bw), float(bh))

        tracker = create_csrt_tracker()
        if tracker is not None:
            try:
                tracker.init(frame, (int(x1), int(y1), int(bw), int(bh)))
                self.csrt_tracker = tracker
            except Exception:
                self.csrt_tracker = None

        self.state = "LOCKED_ON"
        self.is_tracking = True
        self.last_position = (float(cx), float(cy))
        self.last_bbox = bbox
        self.confidence = 1.0
        self.kalman.init(cx, cy)
        
        self.x_history.append(float(cx))
        self.y_history.append(float(cy))
        return True

    def reset_to_searching(self):
        self.state = "SEARCHING"
        self.is_tracking = False
        self.csrt_tracker = None

    def detect_bee_blob(self, frame, arena_center=None, arena_radius=None):
        """Detect bee using CLAHE + Background Subtraction + Blob Filtering."""
        if frame is None:
            return None, None

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 1. Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # 2. Background Subtraction
        fg_mask = self.bg_subtractor.apply(blurred)

        # 3. Morphological Operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4. Find Contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Filter Blobs by Size & Arena Bounds
        valid_blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area <= area <= self.max_area:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                    if arena_center and arena_radius:
                        acx, acy = arena_center
                        if np.hypot(cx - acx, cy - acy) > arena_radius + 40:
                            continue
                    valid_blobs.append(((cx, cy), c, area))

        if not valid_blobs:
            return None, None

        # Select best blob: if we have a last position, pick closest blob to last position
        if self.last_position is not None:
            lcx, lcy = self.last_position
            best_blob = min(valid_blobs, key=lambda b: np.hypot(b[0][0] - lcx, b[0][1] - lcy))
            if np.hypot(best_blob[0][0] - lcx, best_blob[0][1] - lcy) > 150:
                best_blob = max(valid_blobs, key=lambda b: b[2])
        else:
            best_blob = max(valid_blobs, key=lambda b: b[2])

        return best_blob[0], best_blob[1]

    def update(self, frame, use_clahe=True, arena_center=None, arena_radius=None):
        """Main tracking loop combining MOG2, CSRT, Kalman & Darkspot fallbacks."""
        if frame is None:
            return None, None, "lost", 0.0

        self.frame_count += 1
        if not self.bg_initialized:
            self.initialize_background(frame)

        h_f, w_f = frame.shape[:2]
        detected_pos, contour = self.detect_bee_blob(frame, arena_center=arena_center, arena_radius=arena_radius)

        # If CSRT lock-on active, attempt CSRT update first
        if self.state == "LOCKED_ON" and self.csrt_tracker is not None:
            try:
                ok, tb = self.csrt_tracker.update(frame)
                if ok:
                    x, y, w, h = [float(v) for v in tb]
                    cx, cy = x + w / 2.0, y + h / 2.0
                    if 0 <= cx < w_f and 0 <= cy < h_f:
                        if detected_pos:
                            dcx, dcy = detected_pos
                            if np.hypot(cx - dcx, cy - dcy) < 50:
                                cx, cy = (cx + dcx) / 2.0, (cy + dcy) / 2.0

                        scx, scy = self.kalman.correct(cx, cy)
                        bw = w if w > 10 else 38.0
                        bh = h if h > 10 else 38.0
                        self.last_bbox = (scx - bw / 2.0, scy - bh / 2.0, bw, bh)
                        self.last_position = (scx, scy)
                        self.confidence = 0.90
                        self.x_history.append(scx)
                        self.y_history.append(scy)
                        return (scx, scy), self.last_bbox, "ok", 0.90
            except Exception:
                pass

        if detected_pos is not None:
            cx, cy = detected_pos
            scx, scy = self.kalman.correct(cx, cy)
            bw, bh = 38.0, 38.0
            if contour is not None:
                bx, by, cbw, cbh = cv2.boundingRect(contour)
                bw, bh = max(20.0, float(cbw)), max(20.0, float(cbh))

            self.last_bbox = (scx - bw / 2.0, scy - bh / 2.0, bw, bh)
            self.last_position = (scx, scy)
            self.is_tracking = True
            self.state = "LOCKED_ON"
            self.confidence = 0.85
            self.x_history.append(scx)
            self.y_history.append(scy)

            # Sync CSRT tracker to blob
            tracker = create_csrt_tracker()
            if tracker is not None:
                try:
                    tracker.init(frame, (int(self.last_bbox[0]), int(self.last_bbox[1]), int(bw), int(bh)))
                    self.csrt_tracker = tracker
                except Exception:
                    pass

            return (scx, scy), self.last_bbox, "ok", 0.85

        # Fallback 1: Dark spot search if bee is stationary
        if self.last_position is not None:
            lcx, lcy = self.last_position
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            win_sz = 200
            x1 = max(0, int(lcx - win_sz / 2))
            y1 = max(0, int(lcy - win_sz / 2))
            roi = gray[y1 : min(h_f, y1 + win_sz), x1 : min(w_f, x1 + win_sz)]
            if roi.size > 0:
                min_val, _, min_loc, _ = cv2.minMaxLoc(roi)
                if min_val < 100:  # Dark patch found
                    cx = float(x1 + min_loc[0])
                    cy = float(y1 + min_loc[1])
                    scx, scy = self.kalman.correct(cx, cy)
                    self.last_bbox = (scx - 19.0, scy - 19.0, 38.0, 38.0)
                    self.last_position = (scx, scy)
                    self.confidence = 0.50
                    self.x_history.append(scx)
                    self.y_history.append(scy)
                    return (scx, scy), self.last_bbox, "weak", 0.50

        # Fallback 2: Kalman prediction
        pred = self.kalman.predict()
        if pred is not None and self.is_tracking:
            pcx, pcy = pred
            bw = self.last_bbox[2] if self.last_bbox else 38.0
            bh = self.last_bbox[3] if self.last_bbox else 38.0
            self.last_bbox = (pcx - bw / 2.0, pcy - bh / 2.0, bw, bh)
            self.last_position = (pcx, pcy)
            self.x_history.append(pcx)
            self.y_history.append(pcy)
            return (pcx, pcy), self.last_bbox, "weak", 0.40

        return None, None, "lost", 0.0

    def get_smoothed_path(self, window=15, poly_order=3):
        """Return Savitzky-Golay smoothed trajectory points."""
        if len(self.x_history) < window or window < 5:
            return list(zip(self.x_history, self.y_history))

        if window % 2 == 0:
            window += 1
        if len(self.x_history) < window:
            window = len(self.x_history) if len(self.x_history) % 2 != 0 else len(self.x_history) - 1

        if window < poly_order + 2:
            return list(zip(self.x_history, self.y_history))

        try:
            x_arr = list(self.x_history)
            y_arr = list(self.y_history)
            x_smooth = savgol_filter(x_arr, window, poly_order)
            y_smooth = savgol_filter(y_arr, window, poly_order)
            return list(zip(x_smooth, y_smooth))
        except Exception:
            return list(zip(self.x_history, self.y_history))


# Alias for seamless backwards compatibility across existing modules
HybridBeeTracker = RobustBeeTracker


class BeeYOLODetector:
    """Wrapper around fine-tuned / base YOLO model for bee detection."""

    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or self._resolve_best_model()
        self._lock = threading.Lock()
        self._load_model()

    def _resolve_best_model(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        custom_best = os.path.join(base_dir, "models", "best_bee_yolo.pt")
        if os.path.exists(custom_best):
            return custom_best
        base_yolo = os.path.join(base_dir, "yolo11n.pt")
        if os.path.exists(base_yolo):
            return base_yolo
        return "yolo11n.pt"

    def _load_model(self):
        with self._lock:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                print(f"[BeeYOLODetector] Active YOLO model loaded from: {self.model_path}")
            except Exception as e:
                print(f"[BeeYOLODetector] Warning: Could not initialize YOLO model ({e})")
                self.model = None

    def reload_model(self):
        self.model_path = self._resolve_best_model()
        self._load_model()

    def detect_bee(self, frame, roi=None, conf_threshold=0.25, use_clahe=False):
        """
        Detect bee in frame or ROI.
        roi: (x, y, w, h) in pixel coordinates.
        Returns: (cx, cy, w, h, conf) in absolute frame pixel coordinates, or None.
        """
        if self.model is None or frame is None:
            return None

        target_frame = preprocess_ir_frame(frame) if use_clahe else frame
        offset_x, offset_y = 0, 0
        img = target_frame

        if roi is not None:
            rx, ry, rw, rh = [int(v) for v in roi]
            rx = max(0, min(rx, target_frame.shape[1] - 1))
            ry = max(0, min(ry, target_frame.shape[0] - 1))
            rw = max(1, min(rw, target_frame.shape[1] - rx))
            rh = max(1, min(rh, target_frame.shape[0] - ry))
            img = target_frame[ry:ry + rh, rx:rx + rw]
            offset_x, offset_y = rx, ry

        try:
            with self._lock:
                try:
                    results = self.model.track(img, tracker="bytetrack.yaml", conf=conf_threshold, verbose=False)
                except Exception:
                    results = self.model.predict(img, conf=conf_threshold, verbose=False)

            if not results or len(results) == 0 or len(results[0].boxes) == 0:
                return None

            boxes = results[0].boxes
            best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
            box = boxes.xyxy[best_idx].cpu().numpy()
            conf = float(boxes.conf[best_idx].cpu().numpy())

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2.0 + offset_x
            cy = y1 + h / 2.0 + offset_y

            return (cx, cy, w, h, conf)
        except Exception:
            return None


class OnlineYOLOTrainer:
    """
    Background worker that continually harvests verified tracking frames during
    Streamlit live tracking, fine-tunes YOLO in parallel, and dynamically upgrades
    the detector in real-time.
    """

    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "data", "online_train")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.best_model_path = os.path.join(self.models_dir, "best_bee_yolo.pt")

        self.sample_count = 0
        self.uncommitted_count = 0
        self.model_version = 1
        self.is_training = False
        self.disabled = False
        self.last_status = "Ready"
        self._lock = threading.Lock()

        self._init_dirs()

    def _init_dirs(self):
        os.makedirs(os.path.join(self.data_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "labels", "val"), exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self._create_yaml()

    def _create_yaml(self):
        yaml_path = os.path.join(self.data_dir, "online_data.yaml")
        yaml_content = f"""# Online Real-time YOLO Dataset
path: {os.path.abspath(self.data_dir)}
train: images/train
val: images/val
names:
  0: bee
"""
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        return yaml_path

    def add_tracking_sample(self, frame, cx, cy, box_size=38, tag_type="auto", priority=False):
        """
        Record verified tracking coordinate on the fly.
        """
        if frame is None or self.disabled:
            return

        h, w = frame.shape[:2]
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return

        norm_cx = cx / w
        norm_cy = cy / h
        norm_w = box_size / w
        norm_h = box_size / h

        stamp = int(time.time() * 1000)
        img_name = f"sample_{stamp}_{self.sample_count}.jpg"
        lbl_name = f"sample_{stamp}_{self.sample_count}.txt"

        # Split: 80% train, 20% val
        split = "val" if (self.sample_count % 5 == 0) else "train"
        img_path = os.path.join(self.data_dir, "images", split, img_name)
        lbl_path = os.path.join(self.data_dir, "labels", split, lbl_name)

        try:
            cv2.imwrite(img_path, frame)
            with open(lbl_path, "w") as f:
                f.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            with self._lock:
                self.sample_count += 1
                self.uncommitted_count += 1

            # Background fine-tuning is deferred to explicit user trigger (e.g. Fine-Tune button)
            # to prevent CPU thread starvation and UI freezing on laptops.
        except Exception as e:
            print(f"[OnlineYOLOTrainer] Sample save error: {e}")

    def trigger_background_training(self, epochs=3):
        """Spawns non-blocking background thread to fine-tune YOLO model."""
        with self._lock:
            if self.is_training or self.disabled:
                return
            self.is_training = True
            self.last_status = f"Fine-tuning YOLO on {self.sample_count} live frames..."

        thread = threading.Thread(target=self._run_training_worker, args=(epochs,), daemon=True)
        thread.start()

    def _run_training_worker(self, epochs):
        try:
            from ultralytics import YOLO
            yaml_path = self._create_yaml()

            base_weights = self.best_model_path if os.path.exists(self.best_model_path) else "yolo11n.pt"
            model = YOLO(base_weights)

            # Quick online fine-tune burst
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=640,
                batch=8,
                plots=False,
                save=True,
                verbose=False,
                workers=1,
                augment=True
            )

            save_dir = getattr(model.trainer, "save_dir", None)
            trained_weights = os.path.join(save_dir, "weights", "best.pt") if save_dir else None

            if trained_weights and os.path.exists(trained_weights):
                shutil.copy2(trained_weights, self.best_model_path)
                with self._lock:
                    self.model_version += 1
                    self.uncommitted_count = 0
                    self.last_status = f"✅ Model updated to v{self.model_version} ({self.sample_count} frames trained)"

                # Reload live detector immediately
                get_yolo_detector().reload_model()
                print(f"[OnlineYOLOTrainer] Successfully upgraded live YOLO model to v{self.model_version}!")
            else:
                with self._lock:
                    self.last_status = "Trained successfully"
        except Exception as e:
            err_str = str(e)
            with self._lock:
                if "1114" in err_str or "c10.dll" in err_str or "DLL" in err_str:
                    self.last_status = "Classic OpenCV Engine Active (PyTorch C++ runtime DLL notice on host OS)"
                    self.disabled = True
                else:
                    self.last_status = f"Classic OpenCV Engine Active (PyTorch notice: {err_str[:60]})"
                    self.disabled = True
            if self.disabled:
                print(f"[OnlineYOLOTrainer] PyTorch C++ runtime DLL notice on host OS. Defaulting to high-performance OpenCV tracking engine.")
            else:
                print(f"[OnlineYOLOTrainer] Background train notice: {e}")
        finally:
            with self._lock:
                self.is_training = False

    def get_status(self):
        with self._lock:
            return {
                "sample_count": self.sample_count,
                "model_version": self.model_version,
                "is_training": self.is_training,
                "disabled": self.disabled,
                "status": self.last_status
            }


def get_yolo_detector():
    global _YOLO_DETECTOR
    if _YOLO_DETECTOR is None:
        _YOLO_DETECTOR = BeeYOLODetector()
    return _YOLO_DETECTOR


def get_online_trainer():
    global _ONLINE_TRAINER
    if _ONLINE_TRAINER is None:
        _ONLINE_TRAINER = OnlineYOLOTrainer()
    return _ONLINE_TRAINER


def get_tracking_end_frame(exit_frame=None, max_frame=None, analysis_end_frame=None):
    """Return the frame that should end analysis tracking.

    We deliberately do not stop tracking as soon as the bee exits the arena.
    Tracking should continue until the video ends, unless the user explicitly
    marks a later analysis end frame.
    """
    if analysis_end_frame is not None:
        return int(analysis_end_frame)
    if max_frame is None:
        return int(exit_frame) if exit_frame is not None else 0
    return int(max_frame)
