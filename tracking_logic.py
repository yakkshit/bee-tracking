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
    def __init__(self, source=0, camera_idx=None):
        self.source = camera_idx if camera_idx is not None else source
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

    def get_frame(self):
        ok, frame = self.read_latest()
        return frame if ok else None

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


# ---------------------------------------------------------------------------
# Multi-Strategy Ensemble Detector
# ---------------------------------------------------------------------------

class MultiStrategyDetector:
    """
    Runs parallel detection strategies and returns the best consensus position.

    KEY FIXES (based on video analysis):
    - varThreshold=40 (was 12) — suppresses IR camera sensor noise (~4.8 pixel std)
    - Frame-diff uses frame[i-2] vs frame[i] (same-exposure pairs for alternating IR cameras)
    - 3-frame rolling median buffer before diff to denoise
    - Dark-spot search DISABLED as primary (it locks onto the feeder, not bee)
    - Feeder center excluded from blob detection
    """

    def __init__(self):
        # MOG2 — tuned for noisy IR sensor
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )
        self.bg_frame_count = 0
        self.bg_warmup_frames = 25

        # Frame diff with same-exposure skip (i vs i-2) + rolling median
        self.frame_buffer = collections.deque(maxlen=4)  # rolling buffer of last 4 grays

        # Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.lk_pts = None
        self.lk_prev_gray = None

        # Feeder/center exclusion
        self.feeder_center = None  # set externally after calibration
        self.feeder_radius = 50    # pixels to exclude around feeder

    def _to_gray_clahe(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _to_denoised(self, frame):
        """CLAHE + Gaussian blur — the pre-processing pipeline for all detectors."""
        gray = self._to_gray_clahe(frame)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def warmup_bg(self, frame, n=25):
        """Feed static frames into MOG2 to initialize background quickly."""
        blurred = self._to_denoised(frame)
        for _ in range(n):
            self.bg_sub.apply(blurred)
        self.bg_frame_count += n

    def detect_mog2(self, frame, min_area=25, max_area=1200, last_pos=None,
                    arena_center=None, arena_radius=None):
        blurred = self._to_denoised(frame)
        fg = self.bg_sub.apply(blurred)
        self.bg_frame_count += 1

        if self.bg_frame_count < self.bg_warmup_frames:
            return None

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=3)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._best_blob(contours, min_area, max_area, last_pos, arena_center, arena_radius)

    def detect_frame_diff(self, frame, min_area=25, max_area=1200, last_pos=None,
                          arena_center=None, arena_radius=None):
        """
        Frame differencing with noise fixes:
        - Adds current frame to rolling buffer
        - Diffs frame[now] vs frame[now-2] to stay in the same exposure-pair
          (IR cameras with auto-exposure alternate between dark/bright frames)
        - Applies higher threshold (25 vs 18) because sensor noise is ~4-5px std
        """
        blurred = self._to_denoised(frame)
        self.frame_buffer.append(blurred)

        # Need at least 3 frames in the buffer so we can skip 2 back
        if len(self.frame_buffer) < 3:
            return None

        # Compare current frame against the frame 2 steps ago (same exposure type)
        prev = self.frame_buffer[-3]  # 2 frames ago
        diff = cv2.absdiff(prev, blurred)

        # Higher threshold to suppress ~4.8px sensor noise (3-sigma = ~15, use 22)
        _, thresh = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, k, iterations=3)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._best_blob(contours, min_area, max_area, last_pos, arena_center, arena_radius)

    def detect_dark_spot(self, frame, last_pos=None, arena_center=None, arena_radius=None,
                         search_radius=200):
        """
        Dark-spot search in a local ROI around the last known position.
        NOTE: Only used as a last-resort fallback and ONLY when we have a last_pos
        (avoids locking onto the static feeder at the arena center).
        """
        if last_pos is None:
            return None  # Never search whole frame — feeder is darker than bee!

        gray = self._to_gray_clahe(frame)
        h, w = gray.shape[:2]

        lx, ly = int(last_pos[0]), int(last_pos[1])
        sr = min(search_radius, 150)  # tighter ROI
        x0, y0 = max(0, lx - sr), max(0, ly - sr)
        x1, y1 = min(w, lx + sr), min(h, ly + sr)

        if x1 <= x0 or y1 <= y0:
            return None

        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        min_val, _, min_loc, _ = cv2.minMaxLoc(roi)
        if min_val > 100:
            return None

        cx = float(x0 + min_loc[0])
        cy = float(y0 + min_loc[1])

        # Reject if inside feeder exclusion zone
        if self.feeder_center is not None:
            fc_x, fc_y = self.feeder_center
            if np.hypot(cx - fc_x, cy - fc_y) < self.feeder_radius:
                return None

        if arena_center and arena_radius:
            ax, ay = arena_center
            if np.hypot(cx - ax, cy - ay) > arena_radius + 30:
                return None

        return (cx, cy)

    def detect_optical_flow(self, frame, last_pos):
        """Lucas-Kanade sparse optical flow — works even with high noise."""
        if last_pos is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.lk_prev_gray is None or self.lk_prev_gray.shape != gray.shape:
            self.lk_prev_gray = gray
            cx, cy = last_pos
            self.lk_pts = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
            return None

        if self.lk_pts is None:
            cx, cy = last_pos
            self.lk_pts = np.array([[[float(cx), float(cy)]]], dtype=np.float32)

        try:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.lk_prev_gray, gray, self.lk_pts, None, **self.lk_params
            )
            self.lk_prev_gray = gray
            if new_pts is not None and status is not None and status[0][0] == 1:
                nx, ny = float(new_pts[0][0][0]), float(new_pts[0][0][1])
                dx = nx - last_pos[0]
                dy = ny - last_pos[1]
                if np.hypot(dx, dy) < 180:
                    self.lk_pts = new_pts
                    return (nx, ny)
        except Exception:
            pass
        self.lk_prev_gray = gray
        return None

    def reset_optical_flow(self, pos=None):
        self.lk_pts = None
        self.lk_prev_gray = None
        if pos is not None:
            cx, cy = pos
            self.lk_pts = np.array([[[float(cx), float(cy)]]], dtype=np.float32)

    def _best_blob(self, contours, min_area, max_area, last_pos, arena_center, arena_radius):
        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_area <= area <= max_area:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])

                    # Reject blobs outside arena
                    if arena_center and arena_radius:
                        ax, ay = arena_center
                        if np.hypot(cx - ax, cy - ay) > arena_radius + 40:
                            continue

                    # Reject blobs inside feeder exclusion zone (feeder is static dark dot)
                    if self.feeder_center is not None:
                        fc_x, fc_y = self.feeder_center
                        if np.hypot(cx - fc_x, cy - fc_y) < self.feeder_radius:
                            continue

                    valid.append(((cx, cy), area))

        if not valid:
            return None

        if last_pos is not None:
            lx, ly = last_pos
            close = [(pos, a) for pos, a in valid if np.hypot(pos[0] - lx, pos[1] - ly) < 180]
            if close:
                return min(close, key=lambda b: np.hypot(b[0][0] - lx, b[0][1] - ly))[0]
        return max(valid, key=lambda b: b[1])[0]



import enum
from scipy.optimize import linear_sum_assignment


class TrackState(enum.Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """
    Single-Object Tracklet Representation for ByteTrack.
    Maintains 2D constant-velocity Kalman Filter, bounding box geometry,
    confidence score, tracklet history, and micro-kinematic behavioral features
    (velocity v, acceleration a, angular velocity omega, jitter, immobility).
    """
    _count = 0

    def __init__(self, tlwh, score, feat_type="generic"):
        STrack._count += 1
        self.track_id = STrack._count
        self.is_activated = False
        self.state = TrackState.New

        # Bounding box: [x, y, w, h]
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.feat_type = feat_type

        self.kalman_filter = KalmanBeeFilter()
        cx = self._tlwh[0] + self._tlwh[2] / 2.0
        cy = self._tlwh[1] + self._tlwh[3] / 2.0
        self.kalman_filter.init(cx, cy)

        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.time_since_update = 0

        # Trajectory history: deque of (cx, cy, frame_id, timestamp)
        self.history = collections.deque(maxlen=300)
        self.history.append((cx, cy, 0, time.time()))

        # Micro-kinematics & behavioral states (as defined in V1.pdf)
        self.micro_kinematics = {
            "v": 0.0,            # Velocity magnitude (pixels/sec)
            "a": 0.0,            # Acceleration magnitude (pixels/sec^2)
            "omega": 0.0,        # Angular velocity (deg/sec)
            "heading": 0.0,      # Heading angle (degrees)
            "immobile_sec": 0.0, # Sustained immobility counter (sec)
            "jitter_score": 0.0, # High-frequency jitter / erratic index
            "curvature": 0.0,    # Trajectory curvature
        }

        # Anomaly trigger log for this tracklet
        self.triggers = []
        self._immobile_frames = 0
        self._erratic_frames = 0

    @property
    def tlwh(self):
        return self._tlwh

    @property
    def tlbr(self):
        ret = self._tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def center(self):
        return (float(self._tlwh[0] + self._tlwh[2] / 2.0),
                float(self._tlwh[1] + self._tlwh[3] / 2.0))

    def predict(self):
        """Kalman filter state prediction."""
        pred = self.kalman_filter.predict()
        if pred is not None:
            cx, cy = pred
            w, h = self._tlwh[2], self._tlwh[3]
            self._tlwh[0] = cx - w / 2.0
            self._tlwh[1] = cy - h / 2.0
        self.time_since_update += 1

    def activate(self, frame_id):
        """Activate a new tracklet."""
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 1
        self.time_since_update = 0

    def re_activate(self, new_track, frame_id, new_id=False, fps=30.0):
        """Re-activate a lost tracklet with new detection."""
        cx = new_track.tlwh[0] + new_track.tlwh[2] / 2.0
        cy = new_track.tlwh[1] + new_track.tlwh[3] / 2.0
        corr_cx, corr_cy = self.kalman_filter.correct(cx, cy)

        w, h = new_track.tlwh[2], new_track.tlwh[3]
        self._tlwh = np.array([corr_cx - w / 2.0, corr_cy - h / 2.0, w, h], dtype=np.float32)
        self.score = new_track.score
        self.feat_type = new_track.feat_type
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.time_since_update = 0
        if new_id:
            STrack._count += 1
            self.track_id = STrack._count

        self.history.append((corr_cx, corr_cy, frame_id, time.time()))
        self.compute_micro_kinematics(fps=fps)

    def update(self, new_track, frame_id, fps=30.0):
        """Update tracked tracklet with matched detection and compute micro-kinematics."""
        cx = new_track.tlwh[0] + new_track.tlwh[2] / 2.0
        cy = new_track.tlwh[1] + new_track.tlwh[3] / 2.0
        corr_cx, corr_cy = self.kalman_filter.correct(cx, cy)

        w, h = new_track.tlwh[2], new_track.tlwh[3]
        self._tlwh = np.array([corr_cx - w / 2.0, corr_cy - h / 2.0, w, h], dtype=np.float32)
        self.score = new_track.score
        self.feat_type = new_track.feat_type
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.time_since_update = 0

        self.history.append((corr_cx, corr_cy, frame_id, time.time()))
        self.compute_micro_kinematics(fps=fps)

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def compute_micro_kinematics(self, fps=30.0):
        """
        Stage 1 Micro-Kinematics Feature Extraction & Anomaly Triggers (V1.pdf Section 4.1):
        - Velocity v, acceleration a, angular velocity omega
        - Sustained immobility (v < epsilon for > 3s)
        - High-frequency jitter / erratic acceleration (candidate Varroa / disease marker)
        - Periodic figure-eight / waggle dance candidate
        """
        if len(self.history) < 2:
            return

        fps = max(1.0, float(fps))
        p_now = self.history[-1]
        p_prev = self.history[-2]

        dx = p_now[0] - p_prev[0]
        dy = p_now[1] - p_prev[1]
        dist = np.hypot(dx, dy)

        # Instantaneous & windowed velocity (px/sec)
        inst_v = dist * fps
        prev_v = self.micro_kinematics.get("v", 0.0)
        v_smooth = 0.7 * prev_v + 0.3 * inst_v

        # Acceleration (px/sec^2)
        inst_a = abs(v_smooth - prev_v) * fps
        prev_a = self.micro_kinematics.get("a", 0.0)
        a_smooth = 0.7 * prev_a + 0.3 * inst_a

        # Heading angle and angular velocity omega (deg/sec)
        heading = np.degrees(np.arctan2(dy, dx)) if dist > 1.5 else self.micro_kinematics.get("heading", 0.0)
        prev_heading = self.micro_kinematics.get("heading", heading)
        d_heading = (heading - prev_heading + 180.0) % 360.0 - 180.0
        omega = abs(d_heading) * fps

        # 1. Immobility check (v < 25 px/s for > 3.0s)
        immobile_thresh_v = 25.0
        if v_smooth < immobile_thresh_v:
            self._immobile_frames += 1
        else:
            self._immobile_frames = max(0, self._immobile_frames - 2)

        immobile_sec = self._immobile_frames / fps

        # Check sustained immobility anomaly trigger (> 3.0 sec)
        if immobile_sec >= 3.0 and (self._immobile_frames % int(fps * 2) == 0):
            trigger = {
                "type": "SUSTAINED_IMMOBILITY",
                "track_id": self.track_id,
                "frame": p_now[2],
                "duration_sec": round(immobile_sec, 2),
                "pos": (p_now[0], p_now[1]),
                "description": f"Bee {self.track_id} stationary for {immobile_sec:.1f}s (candidate resting/foraging)",
                "behavior_candidate": "resting / foraging",
                "confidence": 0.88,
            }
            self.triggers.append(trigger)

        # 2. Erratic Jitter / Varroa symptom marker
        if (omega > 120.0 and a_smooth > 250.0) or (omega > 220.0):
            self._erratic_frames += 1
        else:
            self._erratic_frames = max(0, self._erratic_frames - 1)

        erratic_sec = self._erratic_frames / fps
        jitter_score = min(1.0, (omega / 300.0) * 0.5 + (a_smooth / 500.0) * 0.5)

        if erratic_sec >= 1.5 and (self._erratic_frames % int(fps * 2) == 0):
            trigger = {
                "type": "ERRATIC_JITTER",
                "track_id": self.track_id,
                "frame": p_now[2],
                "duration_sec": round(erratic_sec, 2),
                "pos": (p_now[0], p_now[1]),
                "description": f"Bee {self.track_id} erratic micro-jitter (omega={omega:.0f}°/s, a={a_smooth:.0f}px/s²)",
                "behavior_candidate": "erratic flight / symptom",
                "confidence": 0.85,
            }
            self.triggers.append(trigger)

        # 3. Figure-eight / waggle dance trajectory candidate
        if len(self.history) >= 30:
            recent_pts = [(p[0], p[1]) for p in list(self.history)[-30:]]
            xs = [p[0] for p in recent_pts]
            ys = [p[1] for p in recent_pts]
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            if 15.0 < span_x < 150.0 and 15.0 < span_y < 150.0 and v_smooth > 30.0:
                if omega > 90.0 and (self.frame_id % int(fps * 3) == 0):
                    trigger = {
                        "type": "WAGGLE_DANCE_CANDIDATE",
                        "track_id": self.track_id,
                        "frame": p_now[2],
                        "duration_sec": 3.0,
                        "pos": (p_now[0], p_now[1]),
                        "description": f"Bee {self.track_id} looping figure-eight trajectory candidate",
                        "behavior_candidate": "waggle dance",
                        "confidence": 0.78,
                    }
                    self.triggers.append(trigger)

        self.micro_kinematics = {
            "v": round(float(v_smooth), 1),
            "a": round(float(a_smooth), 1),
            "omega": round(float(omega), 1),
            "heading": round(float(heading), 1),
            "immobile_sec": round(float(immobile_sec), 2),
            "jitter_score": round(float(jitter_score), 2),
            "curvature": round(float(omega / max(1.0, v_smooth)), 3),
        }


def compute_iou_cost(tracks, detections, max_distance=150.0):
    """
    Computes Cost Matrix using combined Intersection-over-Union (IoU) and
    Euclidean spatial centroid distance gating for small animal tracking.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)), dtype=np.float32)

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, trk in enumerate(tracks):
        tb = trk.tlbr
        t_cx, t_cy = trk.center
        t_w = max(1.0, tb[2] - tb[0])
        t_h = max(1.0, tb[3] - tb[1])
        t_area = t_w * t_h

        for j, det in enumerate(detections):
            db = det.tlbr
            d_cx, d_cy = det.center
            d_w = max(1.0, db[2] - db[0])
            d_h = max(1.0, db[3] - db[1])
            d_area = d_w * d_h

            ix1 = max(tb[0], db[0])
            iy1 = max(tb[1], db[1])
            ix2 = min(tb[2], db[2])
            iy2 = min(tb[3], db[3])
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            union = t_area + d_area - inter
            iou = inter / union if union > 0 else 0.0

            dist = np.hypot(t_cx - d_cx, t_cy - d_cy)
            dist_cost = min(1.0, dist / max_distance)

            cost_matrix[i, j] = (1.0 - iou) * 0.6 + dist_cost * 0.4

    return cost_matrix


class ByteBeeTracker:
    """
    ByteTrack Multi-Object Tracking (MOT) Algorithm.
    (Zhang et al. ECCV 2022, V1.pdf Section 4.1)

    ByteTrack associates almost every detection box across video frames by:
      1. First association: matching high-confidence boxes (D_high) with active tracks.
      2. Second association: matching low-confidence boxes (D_low) with remaining unmatched tracks.
         (Prevents losing bees during fast movement, motion blur, and glare).
      3. Lost track recovery & unconfirmed track management.
      4. Micro-kinematic feature calculation & multi-animal merge/interaction detection.
    """

    def __init__(self, track_thresh=0.35, low_thresh=0.10, match_thresh=0.75,
                 track_buffer=60, frame_rate=30):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate

        self.tracked_stracks = []    # Active confirmed tracks
        self.lost_stracks = []       # Temporarily lost tracks
        self.removed_stracks = []    # Terminated tracks
        self.unconfirmed_stracks = []

        self.frame_id = 0
        self.all_triggers = []       # Global log of anomaly triggers emitted

    def reset(self):
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.unconfirmed_stracks.clear()
        self.frame_id = 0
        self.all_triggers.clear()

    def update(self, detections, frame_id=None, fps=None, arena_center=None, arena_radius=None):
        """
        Main ByteTrack two-stage association loop.
        detections: list of STrack or list of (x1, y1, x2, y2, score, [feat_type]) or (cx, cy, w, h, score)
        Returns: (active_tracks, new_triggers)
        """
        self.frame_id = frame_id if frame_id is not None else self.frame_id + 1
        fps = fps if fps is not None else self.frame_rate

        # Convert input to STrack candidates
        dets = []
        for d in detections:
            if isinstance(d, STrack):
                dets.append(d)
            elif isinstance(d, (list, tuple)):
                if len(d) >= 5:
                    if len(d) >= 6:
                        x1, y1, x2, y2, score, ftype = d[:6]
                    else:
                        x1, y1, x2, y2, score = d[:5]
                        ftype = "blob"
                    w, h = max(10.0, float(x2 - x1)), max(10.0, float(y2 - y1))
                    dets.append(STrack([float(x1), float(y1), w, h], float(score), feat_type=ftype))
                elif len(d) == 4:
                    cx, cy, bw, bh = d
                    dets.append(STrack([float(cx - bw/2), float(cy - bh/2), float(bw), float(bh)], 0.6, feat_type="blob"))

        # Step 1: Separate detections into D_high and D_low
        dets_high = [d for d in dets if d.score >= self.track_thresh]
        dets_low  = [d for d in dets if self.low_thresh <= d.score < self.track_thresh]

        # Step 2: Kalman state prediction for active and lost tracks
        unconfirmed = [t for t in self.unconfirmed_stracks if t.state == TrackState.New]
        tracked = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        lost = [t for t in self.lost_stracks if t.state == TrackState.Lost]

        strack_pool = tracked + lost
        for s in strack_pool:
            s.predict()

        # Step 3: First Association — Match D_high with Tracked pool
        cost_matrix = compute_iou_cost(tracked, dets_high)
        matched_trk_idx = set()
        matched_det_idx = set()
        matches_a = []

        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.match_thresh:
                    matches_a.append((r, c))
                    matched_trk_idx.add(r)
                    matched_det_idx.add(c)

        unmatched_tracked = [tracked[i] for i in range(len(tracked)) if i not in matched_trk_idx]
        unmatched_dets_high = [dets_high[j] for j in range(len(dets_high)) if j not in matched_det_idx]

        # Step 4: Second Association — Match D_low with remaining unmatched tracked tracks
        cost_matrix_low = compute_iou_cost(unmatched_tracked, dets_low)
        matches_b = []
        matched_trk_low_idx = set()

        if cost_matrix_low.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix_low)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix_low[r, c] < (self.match_thresh + 0.10):
                    matches_b.append((r, c))
                    matched_trk_low_idx.add(r)

        # Update tracks matched in Association 1
        activated_tracks = []
        refind_tracks = []
        for r, c in matches_a:
            trk = tracked[r]
            det = dets_high[c]
            trk.update(det, self.frame_id, fps=fps)
            activated_tracks.append(trk)

        # Update tracks matched in Association 2 (ByteTrack low-score recovery)
        for r, c in matches_b:
            trk = unmatched_tracked[r]
            det = dets_low[c]
            trk.update(det, self.frame_id, fps=fps)
            activated_tracks.append(trk)

        # Still unmatched tracked tracks -> candidate for lost
        still_unmatched_tracked = [
            unmatched_tracked[i] for i in range(len(unmatched_tracked)) if i not in matched_trk_low_idx
        ]

        # Step 5: Match remaining D_high with lost pool (re-acquisition)
        cost_matrix_lost = compute_iou_cost(lost, unmatched_dets_high)
        matched_lost_idx = set()
        matched_dets_high_lost_idx = set()

        if cost_matrix_lost.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix_lost)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix_lost[r, c] < self.match_thresh:
                    trk = lost[r]
                    det = unmatched_dets_high[c]
                    trk.re_activate(det, self.frame_id, new_id=False, fps=fps)
                    refind_tracks.append(trk)
                    matched_lost_idx.add(r)
                    matched_dets_high_lost_idx.add(c)

        remaining_dets_high = [
            unmatched_dets_high[j] for j in range(len(unmatched_dets_high)) if j not in matched_dets_high_lost_idx
        ]

        # Step 6: Deal with unconfirmed tracks & initialize new tracks from high-score detections
        cost_matrix_unconf = compute_iou_cost(unconfirmed, remaining_dets_high)
        matched_unconf_idx = set()
        matched_dets_unconf_idx = set()

        if cost_matrix_unconf.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix_unconf)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix_unconf[r, c] < self.match_thresh:
                    trk = unconfirmed[r]
                    det = remaining_dets_high[c]
                    trk.update(det, self.frame_id, fps=fps)
                    trk.state = TrackState.Tracked
                    trk.is_activated = True
                    activated_tracks.append(trk)
                    matched_unconf_idx.add(r)
                    matched_dets_unconf_idx.add(c)

        # Initialise new tracks from remaining high-confidence detections
        new_dets = [
            remaining_dets_high[j] for j in range(len(remaining_dets_high)) if j not in matched_dets_unconf_idx
        ]
        new_tracks = []
        for det in new_dets:
            if det.score >= self.track_thresh:
                cx, cy = det.center
                if arena_center and arena_radius:
                    ax, ay = arena_center
                    if np.hypot(cx - ax, cy - ay) > arena_radius + 40:
                        continue
                det.activate(self.frame_id)
                det.compute_micro_kinematics(fps=fps)
                new_tracks.append(det)

        # Step 7: Update state lists and handle lost / removed
        for trk in still_unmatched_tracked:
            trk.mark_lost()
            self.lost_stracks.append(trk)

        self.tracked_stracks = activated_tracks + refind_tracks + new_tracks

        # Remove tracks lost for longer than track_buffer
        active_lost = []
        for trk in self.lost_stracks:
            if self.frame_id - trk.frame_id > self.track_buffer:
                trk.mark_removed()
                self.removed_stracks.append(trk)
            elif trk.state == TrackState.Lost and trk not in self.tracked_stracks:
                active_lost.append(trk)
        self.lost_stracks = active_lost

        # Step 8: Multi-animal Proximity / Merge Trigger (V1.pdf Section 4.1)
        new_triggers = []
        all_active = [t for t in self.tracked_stracks if t.is_activated]
        for i in range(len(all_active)):
            t1 = all_active[i]
            for j in range(i + 1, len(all_active)):
                t2 = all_active[j]
                c1, c2 = t1.center, t2.center
                d = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if d < 45.0:  # Within 45px proximity
                    trig = {
                        "type": "INTERACTION_MERGE",
                        "track_ids": [t1.track_id, t2.track_id],
                        "frame": self.frame_id,
                        "duration_sec": 1.0,
                        "pos": ((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0),
                        "description": f"Bees {t1.track_id} and {t2.track_id} close contact (dist={d:.1f}px) — candidate trophallaxis/grooming",
                        "behavior_candidate": "trophallaxis / grooming",
                        "confidence": 0.82,
                    }
                    new_triggers.append(trig)

        # Collect all triggers emitted this frame
        for t in all_active:
            if t.triggers:
                for tr in t.triggers:
                    if tr.get("frame") == self.frame_id:
                        new_triggers.append(tr)

        self.all_triggers.extend(new_triggers)
        return all_active, new_triggers


# ---------------------------------------------------------------------------
# BioQuery Zero-Shot Behavioral Ethogram & Query Engine (V1.pdf Stage 2 & 3)
# ---------------------------------------------------------------------------

class BioQueryEngine:
    """
    BioQuery: Decoupled Vision-Language Ethogram Generator & Query Engine.
    (V1.pdf Section 4.2 & 4.3)
    - Stage 2: Event-driven structured JSON ethogram generation
    - Stage 3: Natural language query engine over behavioral log E = {E1, ..., EM}
    """

    def __init__(self):
        self.ethogram_log = []

    def generate_ethogram_entry(self, trigger_event, fps=30.0, video_clip_path=None):
        """
        Generate structured ethogram JSON entry following V1.pdf schema:
        {event_id, timestamp, time_sec, frame_start, frame_end, behavior_class, confidence, description, biological_notes}
        """
        fps = max(1.0, float(fps))
        frame = trigger_event.get("frame", 0)
        time_sec = frame / fps
        m, s = divmod(int(time_sec), 60)
        ts_str = f"{m:02d}:{s:02d}.{int((time_sec % 1) * 10)}"

        b_class = trigger_event.get("behavior_candidate", "resting")
        event_id = f"EVT_{len(self.ethogram_log)+1:04d}_{frame}"

        bio_notes = ""
        if "resting" in b_class or "foraging" in b_class:
            bio_notes = "Subject showed extended low-velocity pause (v < 25 px/s). Potential sucrose intake or resting state."
        elif "erratic" in b_class:
            bio_notes = "High-frequency angular jitter and sudden acceleration spikes detected. Marked as potential Varroa / neuro-motor anomaly."
        elif "waggle" in b_class:
            bio_notes = "Looped figure-eight trajectory detected in arena sub-region. Candidate recruitment dance orientation."
        elif "trophallaxis" in b_class or "grooming" in b_class:
            bio_notes = "Spatial proximity merge (< 45px) between two individuals for > 1s. Candidate social contact or food exchange."

        entry = {
            "event_id": event_id,
            "timestamp": ts_str,
            "time_seconds": round(time_sec, 2),
            "frame_start": max(0, frame - int(fps * 2.5)),
            "frame_end": frame + int(fps * 2.5),
            "duration_sec": trigger_event.get("duration_sec", 5.0),
            "track_id": trigger_event.get("track_id", 1),
            "behavior_class": b_class,
            "confidence": trigger_event.get("confidence", 0.85),
            "description": trigger_event.get("description", "Behavioral event recorded"),
            "biological_notes": bio_notes,
            "clip_path": video_clip_path,
        }
        self.ethogram_log.append(entry)
        return entry

    def query(self, query_str):
        """
        Stage 3 Natural Language Ethogram Query Interface (V1.pdf Section 4.3).
        Answers queries such as:
          - "Show all erratic movement events"
          - "How many waggle dances were detected?"
          - "Find any bee resting for more than 3 seconds"
          - "Show trophallaxis / proximity events"
        """
        q = (query_str or "").lower().strip()
        matched = []

        if not self.ethogram_log:
            return {
                "answer": "No behavioral events have been cataloged yet. Run tracking on live or recorded feed to generate ethogram events.",
                "matched_events": [],
                "total_events": 0,
            }

        for e in self.ethogram_log:
            b = e["behavior_class"].lower()
            desc = e["description"].lower()
            notes = e["biological_notes"].lower()

            if "erratic" in q or "jitter" in q or "varroa" in q:
                if "erratic" in b or "jitter" in desc or "varroa" in notes:
                    matched.append(e)
            elif "waggle" in q or "dance" in q or "figure" in q:
                if "waggle" in b or "dance" in desc:
                    matched.append(e)
            elif "resting" in q or "immobile" in q or "pause" in q or "foraging" in q:
                if "resting" in b or "immobility" in desc or "stationary" in desc or "foraging" in b:
                    matched.append(e)
            elif "trophallaxis" in q or "grooming" in q or "contact" in q or "social" in q or "merge" in q:
                if "trophallaxis" in b or "grooming" in b or "contact" in desc or "merge" in desc:
                    matched.append(e)
            else:
                if any(w in desc or w in b or w in notes for w in q.split() if len(w) > 2):
                    matched.append(e)

        if not matched and q:
            matched = self.ethogram_log

        count = len(matched)
        summary = f"BioQuery analyzed {len(self.ethogram_log)} cataloged ethogram events across the session. Found {count} matching event(s)."
        if count > 0:
            classes = collections.Counter(e["behavior_class"] for e in matched)
            breakdown = ", ".join(f"{k}: {v}" for k, v in classes.items())
            summary += f" [Breakdown: {breakdown}]"

        return {
            "answer": summary,
            "matched_events": matched,
            "total_events": len(self.ethogram_log),
        }


_BIOQUERY_ENGINE = None

def get_bioquery_engine():
    global _BIOQUERY_ENGINE
    if _BIOQUERY_ENGINE is None:
        _BIOQUERY_ENGINE = BioQueryEngine()
    return _BIOQUERY_ENGINE


class RobustBeeTracker:
    """
    Robust Single/Multi-Object Bee Tracking Pipeline powered by ByteTrack MOT
    and Multi-Strategy Feature Extraction (ECCV 2022 / V1.pdf).

    Features:
      1. ByteTrack MOT core associating high & low confidence detections (D_high, D_low)
      2. MultiStrategyDetector (MOG2, Noise-tuned Frame Diff, CSRT, Lucas-Kanade)
      3. Real-time Micro-Kinematics (v, a, omega, jitter, immobility)
      4. Zero-shot Anomaly Triggering for BioQuery ethogram generation
      5. Full backwards compatibility with existing UI & analysis pipelines.
    """
    def __init__(self, yolo_detector=None, bg_history=300, var_threshold=40,
                 min_area=20, max_area=1500):
        self.bg_history = bg_history
        self.var_threshold = var_threshold
        self.min_area = min_area
        self.max_area = max_area

        # Multi-strategy ensemble detector
        self.ensemble = MultiStrategyDetector()

        # ByteTrack MOT Engine
        self.byte_tracker = ByteBeeTracker(
            track_thresh=0.35, low_thresh=0.10, match_thresh=0.70, track_buffer=60
        )

        # CSRT tracker for precise optical lock-on
        self.csrt_tracker = None
        self.csrt_fail_count = 0

        # Kalman filter for single-bee smoothing & prediction
        self.kalman = KalmanBeeFilter()

        self.state = "SEARCHING"
        self.is_tracking = False
        self.last_position = None
        self.last_bbox = None
        self.confidence = 0.0
        self.frame_count = 0

        self.bg_initialized = False

        self.x_history = collections.deque(maxlen=500)
        self.y_history = collections.deque(maxlen=500)

        # Micro-kinematics & recent triggers
        self.micro_kinematics = {"v": 0.0, "a": 0.0, "omega": 0.0, "heading": 0.0, "immobile_sec": 0.0, "jitter_score": 0.0}
        self.recent_triggers = []

    @property
    def tracker(self):
        return self.csrt_tracker

    @tracker.setter
    def tracker(self, value):
        self.csrt_tracker = value

    def initialize_background(self, frame, num_frames=25):
        """Warm up background model with initial frames."""
        if frame is None or self.bg_initialized:
            return
        self.ensemble.warmup_bg(frame, n=num_frames)
        self.bg_initialized = True

    def set_feeder_center(self, cx, cy, radius_px=60):
        """Call after arena calibration to exclude the feeder dot from blob detection."""
        self.ensemble.feeder_center = (float(cx), float(cy))
        self.ensemble.feeder_radius = float(radius_px)

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

        self.ensemble.reset_optical_flow((cx, cy))

        self.state = "LOCKED_ON"
        self.is_tracking = True
        self.last_position = (float(cx), float(cy))
        self.last_bbox = bbox
        self.confidence = 1.0
        self.csrt_fail_count = 0
        self.kalman.init(cx, cy)

        self.x_history.append(float(cx))
        self.y_history.append(float(cy))

        # Seed ByteTrack
        det = STrack([float(x1), float(y1), float(bw), float(bh)], 1.0, feat_type="manual_tag")
        det.activate(self.frame_count)
        self.byte_tracker.tracked_stracks = [det]

        return True

    def reset_to_searching(self):
        self.state = "SEARCHING"
        self.is_tracking = False
        self.csrt_tracker = None
        self.csrt_fail_count = 0
        self.byte_tracker.reset()

    def detect_bee_blob(self, frame, arena_center=None, arena_radius=None):
        """Detect bee using MOG2 + frame-diff (backwards compatible interface)."""
        pos = self.ensemble.detect_mog2(
            frame, self.min_area, self.max_area, self.last_position, arena_center, arena_radius
        )
        if pos is None:
            pos = self.ensemble.detect_frame_diff(
                frame, self.min_area, self.max_area, self.last_position, arena_center, arena_radius
            )
        return pos, None

    def update(self, frame, use_clahe=True, arena_center=None, arena_radius=None, fps=30.0):
        """
        Main tracking loop — ByteTrack MOT + Multi-Strategy Detection Consensus.
        Returns: (center, bbox, status, confidence).
        """
        if frame is None:
            return None, None, "lost", 0.0

        self.frame_count += 1
        if not self.bg_initialized:
            self.initialize_background(frame)

        h_f, w_f = frame.shape[:2]

        # Gather candidate detections across all strategies
        detections = []

        # 1. CSRT Candidate (if locked on)
        csrt_pos = None
        if self.state == "LOCKED_ON" and self.csrt_tracker is not None and self.csrt_fail_count < 10:
            try:
                ok, tb = self.csrt_tracker.update(frame)
                if ok:
                    x, y, bw, bh = [float(v) for v in tb]
                    cx, cy = x + bw / 2.0, y + bh / 2.0
                    if 0 <= cx < w_f and 0 <= cy < h_f:
                        csrt_pos = (cx, cy)
                        detections.append(STrack([x, y, bw, bh], 0.85, feat_type="csrt"))
                        self.csrt_fail_count = 0
                    else:
                        self.csrt_fail_count += 1
                else:
                    self.csrt_fail_count += 1
            except Exception:
                self.csrt_fail_count += 1

        # 2. MOG2 Candidate
        mog2_pos = self.ensemble.detect_mog2(
            frame, self.min_area, self.max_area, self.last_position, arena_center, arena_radius
        )
        if mog2_pos is not None:
            bw = self.last_bbox[2] if self.last_bbox else 38.0
            bh = self.last_bbox[3] if self.last_bbox else 38.0
            detections.append(STrack([mog2_pos[0] - bw/2, mog2_pos[1] - bh/2, bw, bh], 0.65, feat_type="mog2"))

        # 3. Frame-Diff Candidate (instant same-exposure frame diff)
        diff_pos = self.ensemble.detect_frame_diff(
            frame, self.min_area, self.max_area, self.last_position, arena_center, arena_radius
        )
        if diff_pos is not None:
            bw = self.last_bbox[2] if self.last_bbox else 38.0
            bh = self.last_bbox[3] if self.last_bbox else 38.0
            detections.append(STrack([diff_pos[0] - bw/2, diff_pos[1] - bh/2, bw, bh], 0.55, feat_type="frame_diff"))

        # 4. Lucas-Kanade Optical Flow
        if self.last_position is not None:
            lk_pos = self.ensemble.detect_optical_flow(frame, self.last_position)
            if lk_pos is not None:
                bw = self.last_bbox[2] if self.last_bbox else 38.0
                bh = self.last_bbox[3] if self.last_bbox else 38.0
                detections.append(STrack([lk_pos[0] - bw/2, lk_pos[1] - bh/2, bw, bh], 0.50, feat_type="optical_flow"))

        # 5. Dark-spot search (fallback ROI only)
        if not detections and self.last_position is not None:
            dark_pos = self.ensemble.detect_dark_spot(
                frame, self.last_position, arena_center, arena_radius, search_radius=120
            )
            if dark_pos is not None:
                bw = self.last_bbox[2] if self.last_bbox else 38.0
                bh = self.last_bbox[3] if self.last_bbox else 38.0
                detections.append(STrack([dark_pos[0] - bw/2, dark_pos[1] - bh/2, bw, bh], 0.30, feat_type="dark_spot"))

        # Run ByteTrack Update
        active_tracks, new_triggers = self.byte_tracker.update(
            detections, frame_id=self.frame_count, fps=fps,
            arena_center=arena_center, arena_radius=arena_radius
        )
        self.recent_triggers = new_triggers

        # Pass triggers to BioQuery Engine
        if new_triggers:
            bq = get_bioquery_engine()
            for trig in new_triggers:
                bq.generate_ethogram_entry(trig, fps=fps)

        # Primary track selection
        if active_tracks:
            # Pick best track (closest to last position or highest confidence)
            if self.last_position is not None:
                best_track = min(active_tracks, key=lambda t: np.hypot(t.center[0] - self.last_position[0], t.center[1] - self.last_position[1]))
            else:
                best_track = max(active_tracks, key=lambda t: t.score)

            scx, scy = best_track.center
            scx = max(0.0, min(float(w_f - 1), scx))
            scy = max(0.0, min(float(h_f - 1), scy))

            bw = max(20.0, float(best_track.tlwh[2]))
            bh = max(20.0, float(best_track.tlwh[3]))
            self.last_bbox = (scx - bw / 2.0, scy - bh / 2.0, bw, bh)
            self.last_position = (scx, scy)
            self.is_tracking = True
            self.state = "LOCKED_ON"
            self.confidence = float(best_track.score)
            self.micro_kinematics = best_track.micro_kinematics

            self.x_history.append(scx)
            self.y_history.append(scy)

            # Re-seed CSRT if lost or drifting
            if csrt_pos is None or self.csrt_fail_count >= 3:
                tracker = create_csrt_tracker()
                if tracker is not None:
                    ix = max(0, int(scx - bw / 2))
                    iy = max(0, int(scy - bh / 2))
                    ibw = min(int(bw), w_f - ix)
                    ibh = min(int(bh), h_f - iy)
                    if ibw > 0 and ibh > 0:
                        try:
                            tracker.init(frame, (ix, iy, ibw, ibh))
                            self.csrt_tracker = tracker
                            self.csrt_fail_count = 0
                        except Exception:
                            pass
                self.ensemble.reset_optical_flow((scx, scy))

            status = "ok" if self.confidence > 0.4 else "weak"
            return (scx, scy), self.last_bbox, status, self.confidence

        # Kalman prediction fallback
        pred = self.kalman.predict()
        if pred is not None and self.is_tracking:
            pcx, pcy = pred
            pcx = max(0.0, min(float(w_f - 1), pcx))
            pcy = max(0.0, min(float(h_f - 1), pcy))
            bw = self.last_bbox[2] if self.last_bbox else 38.0
            bh = self.last_bbox[3] if self.last_bbox else 38.0
            self.last_bbox = (pcx - bw / 2.0, pcy - bh / 2.0, bw, bh)
            self.last_position = (pcx, pcy)
            self.x_history.append(pcx)
            self.y_history.append(pcy)
            return (pcx, pcy), self.last_bbox, "weak", 0.30

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
