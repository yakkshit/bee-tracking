import os
import cv2
import time
import shutil
import threading
import numpy as np

_YOLO_DETECTOR = None
_ONLINE_TRAINER = None


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

    def detect_bee(self, frame, roi=None, conf_threshold=0.25):
        """
        Detect bee in frame or ROI.
        roi: (x, y, w, h) in pixel coordinates.
        Returns: (cx, cy, w, h, conf) in absolute frame pixel coordinates, or None.
        """
        if self.model is None or frame is None:
            return None

        offset_x, offset_y = 0, 0
        img = frame

        if roi is not None:
            rx, ry, rw, rh = [int(v) for v in roi]
            rx = max(0, min(rx, frame.shape[1] - 1))
            ry = max(0, min(ry, frame.shape[0] - 1))
            rw = max(1, min(rw, frame.shape[1] - rx))
            rh = max(1, min(rh, frame.shape[0] - ry))
            img = frame[ry:ry + rh, rx:rx + rw]
            offset_x, offset_y = rx, ry

        try:
            with self._lock:
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
        if frame is None:
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

            # Trigger training if threshold reached or if priority tag (help/entry tag)
            batch_threshold = 20 if not priority else 5
            if self.uncommitted_count >= batch_threshold and not self.is_training:
                self.trigger_background_training()
        except Exception as e:
            print(f"[OnlineYOLOTrainer] Sample save error: {e}")

    def trigger_background_training(self, epochs=3):
        """Spawns non-blocking background thread to fine-tune YOLO model."""
        with self._lock:
            if self.is_training:
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
            with self._lock:
                self.last_status = f"Online train notice: {e}"
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
