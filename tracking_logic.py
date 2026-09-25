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


class HybridBeeTracker:
    """
    Hybrid YOLO + OpenCV CSRT Single-Object Tracker Pipeline.
    
    States:
    - SEARCHING (YOLO Active): Runs YOLO/Detector to locate the bee bounding box.
      Once found (conf > 0.25), initializes TrackerCSRT and transitions to LOCKED_ON.
    - LOCKED_ON (CSRT Active): Does NOT run YOLO every frame. Runs tracker.update(frame) at 200+ FPS.
      Handles motion blur and fast movement. If CSRT loses lock (success == False),
      transitions back to SEARCHING for instant YOLO re-acquisition.
    """
    def __init__(self, yolo_detector=None):
        self.state = "SEARCHING"  # "SEARCHING" or "LOCKED_ON"
        self.yolo_detector = yolo_detector or get_yolo_detector()
        self.csrt_tracker = None
        self.last_bbox = None
        self.last_center = None
        self.confidence = 0.0
        self.kalman = KalmanBeeFilter()
        self.miss_count = 0

    def reset_to_searching(self):
        self.state = "SEARCHING"
        self.csrt_tracker = None
        self.miss_count = 0

    def force_lock_on(self, frame, cx, cy, box_size=38):
        """Force CSRT tracker lock-on at user clicked coordinate (Help / Entry tag)."""
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
                self.state = "LOCKED_ON"
                self.last_bbox = bbox
                self.last_center = (cx, cy)
                self.confidence = 1.0
                self.kalman.init(cx, cy)
                self.miss_count = 0
                return True
            except Exception as e:
                print(f"[HybridBeeTracker] CSRT init exception: {e}")
        self.state = "SEARCHING"
        return False

    def update(self, frame, use_clahe=True, arena_center=None, arena_radius=None):
        """
        Processes frame using Hybrid State Machine:
        Returns: (center, bbox, status, confidence)
        """
        if frame is None:
            return None, None, "lost", 0.0

        h_f, w_f = frame.shape[:2]
        processed_frame = preprocess_ir_frame(frame) if use_clahe else frame

        # --- STATE B: LOCKED_ON (CSRT Active - NO YOLO) ---
        if self.state == "LOCKED_ON" and self.csrt_tracker is not None:
            try:
                success, tb = self.csrt_tracker.update(processed_frame)
                if success:
                    x, y, w, h = [float(v) for v in tb]
                    cx, cy = x + w / 2.0, y + h / 2.0
                    
                    # Verify boundary sanity
                    in_bounds = (0 <= cx < w_f and 0 <= cy < h_f)
                    if arena_center and arena_radius:
                        acx, acy = arena_center
                        dist_arena = np.hypot(cx - acx, cy - acy)
                        if dist_arena > arena_radius + 50:
                            in_bounds = False

                    if in_bounds:
                        self.last_bbox = (x, y, w, h)
                        self.last_center = (cx, cy)
                        self.confidence = 0.90
                        self.kalman.correct(cx, cy)
                        self.miss_count = 0
                        return (cx, cy), self.last_bbox, "ok", 0.90
            except Exception:
                pass

            # CSRT lost target -> transition back to SEARCHING for YOLO re-acquisition
            self.miss_count += 1
            if self.miss_count >= 2:
                self.state = "SEARCHING"
                self.csrt_tracker = None

        # --- STATE A: SEARCHING (YOLO Active) ---
        self.state = "SEARCHING"
        yolo_res = self.yolo_detector.detect_bee(
            processed_frame,
            roi=self.last_bbox if self.last_bbox else None,
            conf_threshold=0.20,
            use_clahe=False
        )
        if yolo_res is None and self.last_center is not None:
            # Search wider ROI around last known position
            lcx, lcy = self.last_center
            win_sz = 300
            roi_search = (lcx - win_sz / 2.0, lcy - win_sz / 2.0, win_sz, win_sz)
            yolo_res = self.yolo_detector.detect_bee(processed_frame, roi=roi_search, conf_threshold=0.15, use_clahe=False)

        if yolo_res is not None:
            ycx, ycy, yw, yh, yconf = yolo_res
            bbox = (ycx - yw / 2.0, ycy - yh / 2.0, yw, yh)
            
            # Transition to LOCKED_ON with CSRT
            tracker = create_csrt_tracker()
            if tracker is not None:
                try:
                    tracker.init(processed_frame, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                    self.csrt_tracker = tracker
                    self.state = "LOCKED_ON"
                except Exception:
                    self.csrt_tracker = None

            self.last_bbox = bbox
            self.last_center = (ycx, ycy)
            self.confidence = yconf
            self.kalman.correct(ycx, ycy)
            self.miss_count = 0
            return (ycx, ycy), bbox, "ok", yconf

        # Try Kalman Filter prediction fallback if temporary dropout
        pred = self.kalman.predict()
        if pred is not None and self.last_center is not None:
            pcx, pcy = pred
            dist_pred = np.hypot(pcx - self.last_center[0], pcy - self.last_center[1])
            if dist_pred < 150:
                bw = self.last_bbox[2] if self.last_bbox else 38
                bh = self.last_bbox[3] if self.last_bbox else 38
                bbox = (pcx - bw / 2.0, pcy - bh / 2.0, bw, bh)
                return (pcx, pcy), bbox, "weak", 0.40

        return None, None, "lost", 0.0


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
