# Streamlit compatibility patch for streamlit_drawable_canvas
import streamlit.elements.image as st_image

image_to_url_fn = None
try:
    from streamlit.elements.lib.image_utils import image_to_url as image_to_url_fn
except ImportError:
    pass
if not image_to_url_fn:
    try:
        from streamlit.elements.image_utils import image_to_url as image_to_url_fn
    except ImportError:
        pass
if not image_to_url_fn:
    try:
        from streamlit.elements.image import image_to_url as image_to_url_fn
    except ImportError:
        pass

if image_to_url_fn:
    class MockLayoutConfig:
        def __init__(self, width):
            self.width = width

    _orig = image_to_url_fn

    def wrapped_image_to_url(*args, **kwargs):
        new_args = list(args)
        if len(new_args) > 1:
            w = new_args[1]
            if not hasattr(w, "width"):
                new_args[1] = MockLayoutConfig(w)
        elif "width" in kwargs:
            kwargs["layout_config"] = MockLayoutConfig(kwargs.pop("width"))
        return _orig(*new_args, **kwargs)

    st_image.image_to_url = wrapped_image_to_url

import io
import os
import tempfile
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import LineCollection
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from tracking_logic import get_tracking_end_frame

st.set_page_config(page_title="Bee Arena Tracker", page_icon="🐝", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; max-width: 1200px; }
    .player-shell {
        background: #111; border-radius: 12px; padding: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .player-time { color: #eee; font-family: monospace; font-size: 14px; }
    .tag-entry { border-left: 4px solid #2A9D8F !important; }
    .tag-exit  { border-left: 4px solid #E63946 !important; }
    .tag-help  { border-left: 4px solid #FFB703 !important; }
    div[data-testid="stSidebar"] { background: #1a1a2e; }
</style>
""",
    unsafe_allow_html=True,
)

OUTER_RADIUS_MM = 420.0
INNER_RADIUS_MM = 210.0
TAG_BOX_PX = 44
CANVAS_W = 900

TRACK_SETTINGS = {
    "template_threshold": 0.28,
    "search_margin": 0.55,
    "lost_threshold": 0.30,
    "template_update_interval": 25,
    "feeder_radius_mm": 40.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fit_circle(points):
    if len(points) < 3:
        return None
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    A = np.column_stack((x, y, np.ones_like(x)))
    B = -(x ** 2 + y ** 2)
    try:
        w, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        a, b_val, c = w
        xc, yc = -a / 2.0, -b_val / 2.0
        return xc, yc, np.sqrt(xc ** 2 + yc ** 2 - c)
    except Exception:
        return None


def pixel_to_mm(x, y, center, scale):
    xc, yc = center
    x_mm = (x - xc) * scale
    y_mm = -(y - yc) * scale
    d_mm = float(np.sqrt(x_mm ** 2 + y_mm ** 2))
    return x_mm, y_mm, d_mm


def point_to_bbox(x, y, frame_shape, size=TAG_BOX_PX):
    h, w = frame_shape[:2]
    half = size / 2.0
    return clamp_bbox((x - half, y - half, size, size), frame_shape)


def draw_calibration_overlay(img, xc, yc, r_pixels):
    out = img.copy()
    # Outer Circle - Bright Yellow (0, 255, 255)
    cv2.circle(out, (int(xc), int(yc)), int(r_pixels), (0, 255, 255), 2)
    
    # Inner Circle - Bright Cyan (255, 255, 0)
    xc_inner = st.session_state.get("inner_circle_center")
    r_inner = st.session_state.get("inner_circle_radius")
    
    if xc_inner is not None and r_inner is not None:
        cv2.circle(out, (int(xc_inner[0]), int(xc_inner[1])), int(r_inner), (255, 255, 0), 2)
    else:
        # Fallback to standard 21cm inner circle
        r_inner_calc = INNER_RADIUS_MM / (OUTER_RADIUS_MM / r_pixels)
        cv2.circle(out, (int(xc), int(yc)), int(r_inner_calc), (255, 255, 0), 2)
        
    # Feeder Center - Bright Orange
    cv2.circle(out, (int(xc), int(yc)), 6, (0, 140, 255), -1)
    
    # Hive Entry - Bright Green
    hive_entry = st.session_state.get("hive_entry_point")
    if hive_entry is not None:
        hx, hy = hive_entry
        cv2.drawMarker(out, (int(hx), int(hy)), (0, 255, 0), cv2.MARKER_STAR, 15, 2)
        cv2.putText(out, "HIVE", (int(hx) + 8, int(hy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return out


def preprocess_frame(frame, apply_clahe, clahe_clip, clahe_grid, apply_blur, blur_ksize, mask_circle=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        gray = clahe.apply(gray)
    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if apply_blur:
        processed = cv2.GaussianBlur(processed, (blur_ksize, blur_ksize), 0)
    if mask_circle is not None:
        xc, yc, r = mask_circle
        mask = np.zeros(processed.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(xc), int(yc)), int(r), 255, -1)
        processed = cv2.bitwise_and(processed, processed, mask=mask)
    return processed


def find_darkest_spot_in_roi(gray, center_xy, window_size):
    if center_xy is None:
        return None
    cx, cy = center_xy
    half_w = max(40, int(window_size[0] / 2))
    half_h = max(40, int(window_size[1] / 2))
    x0, y0 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
    x1 = min(gray.shape[1], int(cx + half_w))
    y1 = min(gray.shape[0], int(cy + half_h))
    if x1 <= x0 or y1 <= y0:
        return center_xy
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return center_xy
    _, _, min_loc, _ = cv2.minMaxLoc(roi)
    return float(x0 + min_loc[0]), float(y0 + min_loc[1])


def find_darkest_in_arena(gray, xc, yc, r_pixels, exclude_center_px=0):
    mask = np.zeros_like(gray)
    cv2.circle(mask, (int(xc), int(yc)), int(max(10, r_pixels - 10)), 255, -1)
    if exclude_center_px > 0:
        cv2.circle(mask, (int(xc), int(yc)), int(exclude_center_px), 0, -1)
    masked = cv2.bitwise_or(gray, cv2.bitwise_not(mask))
    _, _, min_loc, _ = cv2.minMaxLoc(masked)
    return float(min_loc[0]), float(min_loc[1])


def template_match(frame_gray, last_bbox, template, search_margin, threshold):
    if last_bbox is None or template is None or template.size == 0:
        return None
    x, y, w, h = last_bbox
    x, y = int(max(0, x)), int(max(0, y))
    w, h = int(max(10, w)), int(max(10, h))
    sx0 = max(0, int(x - w * search_margin))
    sy0 = max(0, int(y - h * search_margin))
    sx1 = min(frame_gray.shape[1], int(x + w + w * search_margin))
    sy1 = min(frame_gray.shape[0], int(y + h + h * search_margin))
    region = frame_gray[sy0:sy1, sx0:sx1]
    if region.size == 0 or region.shape[0] < template.shape[0] or region.shape[1] < template.shape[1]:
        return None
    result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        return None
    return float(sx0 + max_loc[0]), float(sy0 + max_loc[1]), float(w), float(h), float(max_val)


def create_tracker():
    for factory in (
        getattr(cv2, "TrackerCSRT_create", None),
        getattr(cv2, "TrackerMIL_create", None),
        getattr(cv2, "legacy", None) and getattr(cv2.legacy, "TrackerCSRT_create", None),
        getattr(cv2, "legacy", None) and getattr(cv2.legacy, "TrackerMIL_create", None),
    ):
        if factory:
            try:
                return factory()
            except Exception:
                continue
    return None


def bbox_center(bbox):
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def clamp_bbox(bbox, frame_shape):
    h, w = frame_shape[:2]
    x, y, bw, bh = bbox
    bw = max(12, min(bw, w - 1))
    bh = max(12, min(bh, h - 1))
    x = max(0, min(x, w - bw))
    y = max(0, min(y, h - bh))
    return float(x), float(y), float(bw), float(bh)


def extract_template(gray, bbox):
    x, y, w, h = [int(v) for v in bbox]
    patch = gray[y : y + h, x : x + w]
    return patch.copy() if patch.size else None


def init_track_state(frame, bbox):
    tracker = create_tracker()
    ib = [int(v) for v in clamp_bbox(bbox, frame.shape)]
    tracker_ok = False
    if tracker is not None:
        tracker.init(frame, tuple(ib))
        tracker_ok = True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cx, cy = bbox_center(ib)
    was_on_feeder = False
    if st.session_state.circle_center and st.session_state.scale_factor:
        _, _, d = pixel_to_mm(cx, cy, st.session_state.circle_center, st.session_state.scale_factor)
        was_on_feeder = d <= st.session_state.feeder_radius_mm
    return {
        "bbox": tuple(float(v) for v in ib),
        "center": (cx, cy),
        "template": extract_template(gray, ib),
        "tracker": tracker,
        "tracker_initialized": tracker_ok,
        "was_on_feeder": was_on_feeder,
        "frame_idx": 0,
    }


def track_single_frame(frame, state, settings):
    xc, yc = st.session_state.circle_center
    r_px = st.session_state.circle_radius
    scale = st.session_state.scale_factor
    feeder_mm = settings["feeder_radius_mm"]
    threshold = settings["template_threshold"]
    base_margin = settings["search_margin"]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_bbox = state["bbox"]
    last_center = state["center"]
    template = state["template"]
    tracker = state["tracker"]
    tracker_ok = state["tracker_initialized"]
    was_on_feeder = state["was_on_feeder"]

    near_feeder = was_on_feeder
    if last_center:
        _, _, last_d = pixel_to_mm(last_center[0], last_center[1], (xc, yc), scale)
        near_feeder = near_feeder or last_d <= feeder_mm * 2.0

    search_margin = base_margin * 4.0 if near_feeder else base_margin
    detected, confidence, method = None, 0.0, "hold"

    if tracker is not None and tracker_ok:
        ok, tb = tracker.update(frame)
        if ok:
            x, y, w, h = [float(v) for v in tb]
            cx, cy = x + w / 2.0, y + h / 2.0
            _, _, d = pixel_to_mm(cx, cy, (xc, yc), scale)
            if d <= OUTER_RADIUS_MM + 40:
                detected = (cx, cy)
                last_bbox = clamp_bbox((x, y, w, h), frame.shape)
                confidence, method = 0.85, "tracker"

    if detected is None and last_bbox is not None:
        match = template_match(gray, last_bbox, template, search_margin, threshold * (0.65 if near_feeder else 1.0))
        if match:
            x, y, w, h, conf = match
            detected = (x + w / 2.0, y + h / 2.0)
            last_bbox = clamp_bbox((x, y, w, h), frame.shape)
            confidence, method = conf, "template"

    if detected is None and last_center is not None:
        win = 200 if near_feeder else 90
        spot = find_darkest_spot_in_roi(gray, last_center, (win, win))
        if spot:
            dist_px = np.hypot(spot[0] - last_center[0], spot[1] - last_center[1])
            if dist_px <= (220 if near_feeder else 90):
                detected = spot
                bw = last_bbox[2] if last_bbox else TAG_BOX_PX
                bh = last_bbox[3] if last_bbox else TAG_BOX_PX
                last_bbox = clamp_bbox((spot[0] - bw / 2, spot[1] - bh / 2, bw, bh), frame.shape)
                confidence, method = 0.45, "darkspot"

    if detected is None and near_feeder:
        exclude_px = feeder_mm / scale
        spot = find_darkest_in_arena(gray, xc, yc, r_px, exclude_center_px=exclude_px)
        if spot and last_center:
            dist_px = np.hypot(spot[0] - last_center[0], spot[1] - last_center[1])
            _, _, d_spot = pixel_to_mm(spot[0], spot[1], (xc, yc), scale)
            if dist_px <= 300 and d_spot > feeder_mm * 0.8:
                detected = spot
                bw = last_bbox[2] if last_bbox else TAG_BOX_PX
                last_bbox = clamp_bbox((spot[0] - bw / 2, spot[1] - bh / 2, bw, bh), frame.shape)
                confidence, method = 0.4, "arena_scan"

    if detected is None:
        return None, None, "lost", state

    cx, cy = detected
    _, _, d_mm = pixel_to_mm(cx, cy, (xc, yc), scale)
    now_on_feeder = d_mm <= feeder_mm
    leaving_feeder = was_on_feeder and not now_on_feeder

    if tracker is not None and (leaving_feeder or method in ("template", "darkspot", "arena_scan")):
        tracker = create_tracker()
        if tracker is not None:
            tracker.init(frame, tuple(int(v) for v in last_bbox))
            tracker_ok = True

    if state["frame_idx"] % settings["template_update_interval"] == 0:
        new_tpl = extract_template(gray, last_bbox)
        if new_tpl is not None:
            template = new_tpl

    new_state = {
        "bbox": last_bbox,
        "center": (cx, cy),
        "template": template,
        "tracker": tracker,
        "tracker_initialized": tracker_ok,
        "was_on_feeder": now_on_feeder,
        "frame_idx": state["frame_idx"],
    }
    status = "ok" if confidence >= settings["lost_threshold"] else "weak"
    return (cx, cy), last_bbox, status, new_state


def make_coord(frame_idx, cx, cy, fps, tag_type="auto", status="ok"):
    xc, yc = st.session_state.circle_center
    scale = st.session_state.scale_factor
    x_mm, y_mm, d_mm = pixel_to_mm(cx, cy, (xc, yc), scale)
    t0 = st.session_state.entry_frame if st.session_state.entry_frame is not None else 0
    return {
        "frame": frame_idx,
        "time_sec": (frame_idx - t0) / fps,
        "x_pixel": float(cx),
        "y_pixel": float(cy),
        "x_mm": float(x_mm),
        "y_mm": float(y_mm),
        "distance_mm": float(d_mm),
        "tag_type": tag_type,
        "status": status,
    }


def upsert_coord(coord):
    st.session_state.track_coords = [c for c in st.session_state.track_coords if c["frame"] != coord["frame"]]
    st.session_state.track_coords.append(coord)
    st.session_state.track_coords.sort(key=lambda c: c["frame"])


def render_player_frame(frame, coords_up_to_frame, markers, cur_center=None, status="idle"):
    vis = draw_calibration_overlay(
        frame,
        st.session_state.circle_center[0],
        st.session_state.circle_center[1],
        st.session_state.circle_radius,
    )
    if len(coords_up_to_frame) > 1:
        pts = np.array([[int(c["x_pixel"]), int(c["y_pixel"])] for c in coords_up_to_frame], np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], False, (0, 200, 255), 2)

    for m in markers:
        mx, my, color, label = m
        cv2.drawMarker(vis, (int(mx), int(my)), color, cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(vis, label, (int(mx) + 10, int(my) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if cur_center:
        cx, cy = int(cur_center[0]), int(cur_center[1])
        col = (0, 255, 0) if status in ("ok", "idle") else (0, 180, 255) if status == "manual" else (0, 80, 255)
        cv2.circle(vis, (cx, cy), 8, col, -1)
        cv2.circle(vis, (cx, cy), 14, col, 2)
    return vis


def build_processed_df(coords, fps, feeder_mm):
    records, prev_zone = [], None
    for c in coords:
        d = float(np.sqrt(c["x_mm"] ** 2 + c["y_mm"] ** 2))
        zone = "Inner" if d <= INNER_RADIUS_MM else "Outer" if d <= OUTER_RADIUS_MM else "Exited"
        transition = f"{prev_zone} -> {zone}" if prev_zone and zone != prev_zone else None
        prev_zone = zone
        records.append(
            {
                "frame": int(c["frame"]),
                "time_sec": round(float(c["time_sec"]), 2),
                "x_pixel": round(float(c["x_pixel"]), 2),
                "y_pixel": round(float(c["y_pixel"]), 2),
                "x_mm": round(float(c["x_mm"]), 2),
                "y_mm": round(float(c["y_mm"]), 2),
                "distance_from_center_mm": round(d, 2),
                "current_zone": zone,
                "in_arena": d <= OUTER_RADIUS_MM,
                "on_feeder": d <= feeder_mm,
                "transition_event": transition,
                "tag_type": c.get("tag_type", "auto"),
            }
        )
    return pd.DataFrame(records)


def plot_trajectory(df, entry_frame=None, exit_frame=None, title="Bee Trajectory", bee_id="unknown", outcome="Unknown"):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    ax.add_patch(plt.Circle((0, 0), OUTER_RADIUS_MM, fill=False, color="#ccc", lw=2, label="Outer (42 cm)"))
    ax.add_patch(plt.Circle((0, 0), INNER_RADIUS_MM, fill=False, color="#aaa", lw=1.5, ls="--", label="Inner (21 cm)"))
    x_mm, y_mm, t_sec = df["x_mm"].values, df["y_mm"].values, df["time_sec"].values
    if len(x_mm) >= 2:
        pts = np.array([x_mm, y_mm]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = plt.Normalize(t_sec.min(), t_sec.max())
        lc = LineCollection(segs, cmap="viridis", norm=norm, linewidth=2.5)
        lc.set_array(t_sec)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, shrink=0.8, label="Time (s)")
    ax.plot(x_mm[0], y_mm[0], "^", color="#2A9D8F", ms=10, label="Entry")
    ax.plot(x_mm[-1], y_mm[-1], "X", color="#E63946", ms=10, label="Exit")
    ax.plot(0, 0, "o", color="#FB8500", ms=10, label="Feeder")
    ax.set_xlim(-460, 460)
    ax.set_ylim(-460, 460)
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)
    
    # Overlay metadata text box inside the plot
    textstr = f"Bee ID: {bee_id}\nOutcome: {outcome}"
    props = dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#cccccc', alpha=0.85)
    ax.text(0.03, 0.03, textstr, transform=ax.transAxes, fontsize=10,
            fontweight='bold', verticalalignment='bottom', bbox=props)
            
    ax.legend(loc="upper right", fontsize=9)
    return fig


def generate_tracked_video(video_path, coords, entry_frame, exit_frame, output_path, circle_center, circle_radius):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    coords_sorted = sorted(coords, key=lambda c: c["frame"])
    entry_f = entry_frame if entry_frame is not None else 0
    exit_f = exit_frame if exit_frame is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, entry_f)
    
    for f_idx in range(entry_f, exit_f + 1):
        ret, frame = cap.read()
        if not ret:
            break
            
        vis = draw_calibration_overlay(frame, circle_center[0], circle_center[1], circle_radius)
        
        coords_up_to_frame = [c for c in coords_sorted if c["frame"] <= f_idx]
        if len(coords_up_to_frame) > 1:
            pts = np.array([[int(c["x_pixel"]), int(c["y_pixel"])] for c in coords_up_to_frame], np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], False, (0, 200, 255), 2)
            
        for c in coords_up_to_frame:
            if c.get("tag_type") == "entry":
                cv2.drawMarker(vis, (int(c["x_pixel"]), int(c["y_pixel"])), (42, 157, 143), cv2.MARKER_TILTED_CROSS, 18, 2)
                cv2.putText(vis, "ENTRY", (int(c["x_pixel"]) + 10, int(c["y_pixel"]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (42, 157, 143), 2)
            elif c.get("tag_type") == "exit":
                cv2.drawMarker(vis, (int(c["x_pixel"]), int(c["y_pixel"])), (230, 57, 70), cv2.MARKER_TILTED_CROSS, 18, 2)
                cv2.putText(vis, "EXIT", (int(c["x_pixel"]) + 10, int(c["y_pixel"]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 57, 70), 2)
                
        coord_at_cur = next((c for c in coords_up_to_frame if c["frame"] == f_idx), None)
        if coord_at_cur:
            cx, cy = int(coord_at_cur["x_pixel"]), int(coord_at_cur["y_pixel"])
            status = coord_at_cur.get("status", "ok")
            col = (0, 255, 0) if status in ("ok", "idle") else (0, 180, 255) if status == "manual" else (0, 80, 255)
            cv2.circle(vis, (cx, cy), 8, col, -1)
            cv2.circle(vis, (cx, cy), 14, col, 2)
            
        out.write(vis)
        
    cap.release()
    out.release()
    return True


def fmt_time(frame, fps):
    sec = frame / fps
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"


def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return ok, frame


def video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    meta = {
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def apply_point_tag(px, py, tag_mode, frame, fps):
    """Apply entry / exit / help / analysis-end tag at pixel coords on current frame."""
    bbox = point_to_bbox(px, py, frame.shape)
    cx, cy = bbox_center(bbox)
    cur = st.session_state.player_frame

    if tag_mode == "entry":
        st.session_state.entry_frame = cur
        st.session_state.entry_point = (cx, cy)
        st.session_state.track_coords = []
        st.session_state.track_state = init_track_state(frame, bbox)
        st.session_state.track_state["frame_idx"] = cur
        upsert_coord(make_coord(cur, cx, cy, fps, tag_type="entry", status="manual"))
        st.session_state.track_phase = "ready"
        st.session_state.tracking_lost = False

    elif tag_mode == "help":
        st.session_state.track_coords = [c for c in st.session_state.track_coords if c["frame"] < cur]
        st.session_state.track_state = init_track_state(frame, bbox)
        st.session_state.track_state["frame_idx"] = cur
        upsert_coord(make_coord(cur, cx, cy, fps, tag_type="help", status="manual"))
        st.session_state.track_phase = "tracking"
        st.session_state.tracking_lost = False
        st.session_state.is_playing = True

    elif tag_mode == "exit":
        st.session_state.track_coords = [c for c in st.session_state.track_coords if c["frame"] < cur]
        st.session_state.exit_frame = cur
        st.session_state.exit_point = (cx, cy)
        upsert_coord(make_coord(cur, cx, cy, fps, tag_type="exit", status="manual"))
        st.session_state.track_phase = "tracking"
        st.session_state.is_playing = True
        st.session_state.tracking_lost = False

    elif tag_mode == "analysis_end":
        st.session_state.track_coords = [c for c in st.session_state.track_coords if c["frame"] < cur]
        st.session_state.analysis_end_frame = cur
        st.session_state.analysis_end_point = (cx, cy)
        upsert_coord(make_coord(cur, cx, cy, fps, tag_type="analysis_end", status="manual"))
        st.session_state.track_phase = "complete"
        st.session_state.is_playing = False
        st.session_state.tracking_lost = False


def process_tracking_frame(frame, frame_idx, fps, settings):
    """Track one frame; returns True if ok, False if lost."""
    if st.session_state.track_state is None:
        return False
    state = st.session_state.track_state.copy()
    state["frame_idx"] = frame_idx
    center, _, status, new_state = track_single_frame(frame, state, settings)
    if center is None or status == "lost":
        st.session_state.tracking_lost = True
        st.session_state.is_playing = False
        st.session_state.track_phase = "paused_lost"
        return False
    cx, cy = center
    upsert_coord(make_coord(frame_idx, cx, cy, fps, tag_type="auto", status=status))
    st.session_state.track_state = new_state
    return True


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
DEFAULTS = {
    "tab": "load",
    "video_path": None,
    "video_name": "",
    "circle_center": None,
    "circle_radius": None,
    "inner_circle_center": None,
    "inner_circle_radius": None,
    "hive_entry_point": None,
    "scale_factor": None,
    "tracking_fps": 30.0,
    "apply_clahe": True,
    "clahe_clip": 3.0,
    "clahe_grid": 8,
    "apply_blur": True,
    "blur_ksize": 5,
    "mask_background": False,
    "feeder_radius_mm": 40.0,
    "player_frame": 0,
    "last_player_frame": 0,
    "timeline_slider": 0,
    "frame_number_input": 0,
    "is_playing": False,
    "tag_mode": None,
    "entry_frame": None,
    "entry_point": None,
    "exit_frame": None,
    "exit_point": None,
    "analysis_end_frame": None,
    "analysis_end_point": None,
    "track_coords": [],
    "track_state": None,
    "track_phase": "idle",
    "tracking_lost": False,
    "track_stride": 1,
    "bee_went_back": "unknown",
    "processed_df": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

settings = {**TRACK_SETTINGS, "feeder_radius_mm": st.session_state.feeder_radius_mm}


# ---------------------------------------------------------------------------
# Top navigation
# ---------------------------------------------------------------------------
st.title("🐝 Bee Arena Tracker")
tabs = {"load": "1 · Video", "calibrate": "2 · Calibrate", "track": "3 · Track", "analysis": "4 · Analysis"}
tab_keys = list(tabs.keys())
tab_labels = [tabs[k] for k in tab_keys]
disabled = []
if not st.session_state.video_path:
    disabled = ["calibrate", "track", "analysis"]
elif not st.session_state.circle_center:
    disabled = ["track", "analysis"]
elif st.session_state.track_phase != "complete":
    disabled = ["analysis"]

choice = st.radio(
    "Workflow",
    tab_keys,
    format_func=lambda k: tabs[k],
    horizontal=True,
    index=tab_keys.index(st.session_state.tab),
)
if choice in disabled:
    st.warning(f"Complete previous steps before opening **{tabs[choice]}**.")
    choice = st.session_state.tab
st.session_state.tab = choice


# ===================================================================
# TAB 1 — Load video
# ===================================================================
if st.session_state.tab == "load":
    st.subheader("Load your arena video")
    workspace_video = "2024-11-17 17-04-16.R13.LR.P0U8.mp4"
    use_ws = os.path.exists(workspace_video) and st.checkbox(f"Use `{workspace_video}`", value=True)
    if use_ws:
        st.session_state.video_path = workspace_video
        st.session_state.video_name = workspace_video
    else:
        up = st.file_uploader("Upload .mp4 / .mov", type=["mp4", "avi", "mov"])
        if up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(up.read())
            tmp.close()
            st.session_state.video_path = tmp.name
            st.session_state.video_name = up.name

    if st.session_state.video_path:
        meta = video_meta(st.session_state.video_path)
        st.session_state.tracking_fps = meta["fps"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Resolution", f"{meta['w']}×{meta['h']}")
        c2.metric("FPS", f"{meta['fps']:.0f}")
        c3.metric("Duration", fmt_time(meta["frames"], meta["fps"]))
        ok, f0 = read_frame(st.session_state.video_path, 0)
        if ok:
            st.image(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Preview")
        if st.button("Next: Calibrate arena →", type="primary"):
            st.session_state.tab = "calibrate"
            st.rerun()


# ===================================================================
# TAB 2 — Calibrate
# ===================================================================
elif st.session_state.tab == "calibrate":
    st.subheader("Calibrate the circular arena")
    st.caption("Click exactly 9 points: 4 on the outer rim, 4 on the inner rim, and 1 at the hive entry.")

    if st.button("Load demo calibration"):
        st.session_state.circle_center = (942.0, 433.0)
        st.session_state.circle_radius = 379.0
        st.session_state.scale_factor = OUTER_RADIUS_MM / 379.0
        st.session_state.inner_circle_center = (942.0, 433.0)
        st.session_state.inner_circle_radius = 189.5
        st.session_state.hive_entry_point = (1300.0, 433.0)
        st.rerun()

    ok, frame0 = read_frame(st.session_state.video_path, 0)
    if not ok:
        st.error("Cannot read video.")
        st.stop()

    oh, ow = frame0.shape[:2]
    ch = int(oh * CANVAS_W / ow)
    ratio = ow / CANVAS_W
    pil = Image.fromarray(cv2.cvtColor(cv2.resize(frame0, (CANVAS_W, ch)), cv2.COLOR_BGR2RGB))

    result = st_canvas(
        fill_color="rgba(0, 255, 255, 0.4)",
        stroke_width=2,
        stroke_color="#00FFFF",
        background_image=pil,
        update_streamlit=True,
        height=ch,
        width=CANVAS_W,
        drawing_mode="point",
        key="calib_canvas",
    )

    num_clicked = 0
    if result.json_data:
        for obj in result.json_data.get("objects", []):
            if obj.get("type") == "circle":
                num_clicked += 1

    st.markdown(f"**Points clicked:** `{num_clicked} / 9`")

    if result.json_data:
        pts = []
        for obj in result.json_data.get("objects", []):
            if obj.get("type") == "circle":
                r = obj.get("radius", 0)
                pts.append((obj["left"] + r, obj["top"] + r))
        orig = [(p[0] * ratio, p[1] * ratio) for p in pts]

        if len(orig) >= 9:
            fit_outer = fit_circle(orig[:4])
            fit_inner = fit_circle(orig[4:8])
            hive_entry = orig[8]

            if fit_outer and fit_inner:
                xc_o, yc_o, r_o = fit_outer
                xc_i, yc_i, r_i = fit_inner

                st.session_state.circle_center = (xc_o, yc_o)
                st.session_state.circle_radius = r_o
                st.session_state.scale_factor = OUTER_RADIUS_MM / r_o

                st.session_state.inner_circle_center = (xc_i, yc_i)
                st.session_state.inner_circle_radius = r_i
                st.session_state.hive_entry_point = hive_entry

                overlay = draw_calibration_overlay(frame0, xc_o, yc_o, r_o)
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Fitted Calibration Overlay")

                if st.button("Next: Start tracking →", type="primary"):
                    st.session_state.tab = "track"
                    st.session_state.player_frame = 0
                    st.session_state.last_player_frame = 0
                    st.session_state.timeline_slider = 0
                    st.session_state.frame_number_input = 0
                    st.session_state.track_coords = []
                    st.session_state.track_phase = "idle"
                    st.rerun()
            else:
                st.error("Could not fit circles — please try again.")
        elif len(orig) >= 4:
            st.info("Keep clicking: click 4 points for the inner circle, and then 1 point for the hive entry.")


# ===================================================================
# TAB 3 — Track (video player)
# ===================================================================
elif st.session_state.tab == "track":
    meta = video_meta(st.session_state.video_path)
    fps = st.session_state.tracking_fps
    max_frame = max(0, meta["frames"] - 1)
    cur = int(st.session_state.player_frame)
    cur = max(0, min(cur, max_frame))
    last_val = st.session_state.get("last_player_frame", 0)

    # Detect user interaction on slider or number input vs programmatic changes
    if "timeline_slider" in st.session_state and st.session_state.timeline_slider != last_val:
        cur = int(st.session_state.timeline_slider)
        cur = max(0, min(cur, max_frame))
        st.session_state.is_playing = False
    elif "frame_number_input" in st.session_state and st.session_state.frame_number_input != last_val:
        cur = int(st.session_state.frame_number_input)
        cur = max(0, min(cur, max_frame))
        st.session_state.is_playing = False

    st.session_state.player_frame = cur
    st.session_state.timeline_slider = cur
    st.session_state.frame_number_input = cur
    st.session_state.last_player_frame = cur

    ok, frame = read_frame(st.session_state.video_path, cur)
    if not ok:
        st.error("Cannot read frame.")
        st.stop()

    stride = int(st.session_state.get("track_stride", 1))

    # Inject keyboard shortcut Shift+D
    st.components.v1.html(
        """
        <script>
        const doc = window.parent.document;
        if (!window.parent.hasShiftDShortcut) {
            window.parent.hasShiftDShortcut = true;
            doc.addEventListener("keydown", function(e) {
                if (e.shiftKey && (e.key === "D" || e.key === "d")) {
                    e.preventDefault();
                    const buttons = Array.from(doc.querySelectorAll("button"));
                    const stepBtn = buttons.find(btn => btn.innerText && btn.innerText.includes("▶▶"));
                    if (stepBtn) {
                        stepBtn.click();
                    }
                }
            });
        }
        </script>
        """,
        height=0,
        width=0,
    )

    # --- Tracker state synchronization ---
    coord_at_cur = next((c for c in st.session_state.track_coords if c["frame"] == cur), None)
    if coord_at_cur:
        st.session_state.tracking_lost = False
        if st.session_state.track_phase == "paused_lost":
            st.session_state.track_phase = "ready"
        if st.session_state.track_state is None or st.session_state.track_state.get("frame_idx") != cur:
            bbox = point_to_bbox(coord_at_cur["x_pixel"], coord_at_cur["y_pixel"], frame.shape)
            st.session_state.track_state = init_track_state(frame, bbox)
            st.session_state.track_state["frame_idx"] = cur
    else:
        if st.session_state.entry_frame is not None:
            if cur < st.session_state.entry_frame:
                st.session_state.track_phase = "idle"
                st.session_state.tracking_lost = False
            elif cur > st.session_state.entry_frame:
                st.session_state.tracking_lost = True
                st.session_state.track_phase = "paused_lost"

    # --- Status banner ---
    phase = st.session_state.track_phase
    if phase == "idle":
        st.info("**Step 1:** Scrub to where the bee enters. Select **Entry tag** and click on the bee.")
    elif phase == "ready":
        st.info("**Step 2:** Press **Play** to run live tracking. Use **Help tag** if tracking is lost. Use **Exit tag** when the bee leaves.")
    elif phase == "tracking":
        st.success("Live tracking… Press **Pause** anytime. Place **Exit tag** when done.")
    elif phase == "paused_lost":
        st.warning("Tracking lost — select **Help tag**, click the bee, tracking resumes automatically.")
    elif phase == "complete":
        st.success("Tracking complete! Go to **4 · Analysis**.")

    # --- Tag tool buttons ---
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        if st.button("🟢 Entry tag", use_container_width=True, type="primary" if st.session_state.tag_mode == "entry" else "secondary"):
            st.session_state.tag_mode = "entry" if st.session_state.tag_mode != "entry" else None
            st.session_state.is_playing = False
            st.rerun()
    with t2:
        if st.button("🟡 Help tag", use_container_width=True, type="primary" if st.session_state.tag_mode == "help" else "secondary"):
            st.session_state.tag_mode = "help" if st.session_state.tag_mode != "help" else None
            st.session_state.is_playing = False
            st.rerun()
    with t3:
        if st.button("🔴 Exit tag", use_container_width=True, type="primary" if st.session_state.tag_mode == "exit" else "secondary"):
            st.session_state.tag_mode = "exit" if st.session_state.tag_mode != "exit" else None
            st.session_state.is_playing = False
            st.rerun()
    with t4:
        if st.button("⏹ End tag", use_container_width=True, type="primary" if st.session_state.tag_mode == "analysis_end" else "secondary"):
            st.session_state.tag_mode = "analysis_end" if st.session_state.tag_mode != "analysis_end" else None
            st.session_state.is_playing = False
            st.rerun()
    with t5:
        if st.button("↺ Reset session", use_container_width=True):
            for k in ("entry_frame", "entry_point", "exit_frame", "exit_point", "analysis_end_frame", "analysis_end_point", "track_state"):
                st.session_state[k] = None
            st.session_state.track_coords = []
            st.session_state.track_phase = "idle"
            st.session_state.is_playing = False
            st.session_state.player_frame = 0
            st.session_state.timeline_slider = 0
            st.session_state.tag_mode = None
            st.session_state.tracking_lost = False
            st.rerun()

    with st.expander("⚙️ Tracking Options", expanded=False):
        st.session_state.track_stride = st.slider(
            "Tracking Speed / Frame Stride",
            min_value=1,
            max_value=60,
            value=st.session_state.track_stride,
            step=1,
            help="1x tracks every frame, 2x tracks alternate frames, up to 60x."
        )

    if st.session_state.tag_mode:
        mode_labels = {
            "entry": "Entry — click the bee where tracking should start",
            "help": "Help — click the bee to re-acquire tracking",
            "exit": "Exit — click the bee (or its last position) to mark an exit event",
            "analysis_end": "End — click the frame where analysis should finish",
        }
        st.caption(f"**Active tool:** {mode_labels[st.session_state.tag_mode]}")

    # --- Build display frame ---
    coords_sorted = sorted(st.session_state.track_coords, key=lambda c: c["frame"])
    coords_visible = [c for c in coords_sorted if c["frame"] <= cur]
    coord_at_cur = next((c for c in coords_sorted if c["frame"] == cur), None)

    markers = []
    if st.session_state.entry_point:
        markers.append((*st.session_state.entry_point, (42, 157, 143), "ENTRY"))
    if st.session_state.exit_point:
        markers.append((*st.session_state.exit_point, (230, 57, 70), "EXIT"))
    if st.session_state.analysis_end_point:
        markers.append((*st.session_state.analysis_end_point, (255, 183, 77), "END"))

    cur_center = (coord_at_cur["x_pixel"], coord_at_cur["y_pixel"]) if coord_at_cur else None
    cur_status = coord_at_cur.get("status", "ok") if coord_at_cur else "idle"
    vis = render_player_frame(frame, coords_visible, markers, cur_center, cur_status)

    oh, ow = vis.shape[:2]
    ch = int(oh * CANVAS_W / ow)
    ratio = ow / CANVAS_W
    vis_rgb = cv2.cvtColor(cv2.resize(vis, (CANVAS_W, ch)), cv2.COLOR_BGR2RGB)

    st.markdown('<div class="player-shell">', unsafe_allow_html=True)

    if st.session_state.tag_mode:
        canvas = st_canvas(
            fill_color="rgba(255,255,255,0)",
            stroke_width=0,
            background_image=Image.fromarray(vis_rgb),
            update_streamlit=True,
            height=ch,
            width=CANVAS_W,
            drawing_mode="point",
            point_display_radius=8,
            key=f"tag_{st.session_state.tag_mode}_{cur}",
        )
        if canvas.json_data:
            for obj in canvas.json_data.get("objects", []):
                if obj.get("type") == "circle":
                    r = obj.get("radius", 0)
                    cx_c = (obj["left"] + r) * ratio
                    cy_c = (obj["top"] + r) * ratio
                    apply_point_tag(cx_c, cy_c, st.session_state.tag_mode, frame, fps)
                    st.session_state.tag_mode = None
                    if st.session_state.track_phase == "complete":
                        st.session_state.tab = "analysis"
                    st.rerun()
    else:
        st.image(vis_rgb, use_container_width=True)

    # --- Player controls ---
    p1, p2, p3, p4, p5, p6 = st.columns([1, 1, 1, 1, 4, 2])
    with p1:
        if st.button("⏮", help="Back 5s"):
            st.session_state.player_frame = max(0, cur - int(fps * 5))
            st.session_state.is_playing = False
            st.rerun()
    with p2:
        if st.button("◀"):
            st.session_state.player_frame = max(0, cur - stride)
            st.session_state.is_playing = False
            st.rerun()
    with p3:
        play_label = "⏸" if st.session_state.is_playing else "▶"
        if st.button(play_label, help="Play / Pause"):
            if st.session_state.entry_frame is None:
                st.warning("Set an Entry tag first.")
            elif st.session_state.track_phase in ("ready", "tracking", "paused_lost"):
                if st.session_state.track_phase == "paused_lost" and st.session_state.tracking_lost:
                    st.warning("Use Help tag to re-acquire the bee first.")
                else:
                    st.session_state.is_playing = not st.session_state.is_playing
                    if st.session_state.is_playing:
                        st.session_state.track_phase = "tracking"
                        # Discard coordinates beyond the current frame
                        st.session_state.track_coords = [c for c in st.session_state.track_coords if c["frame"] <= cur]
                    st.rerun()
            elif st.session_state.track_phase == "complete":
                st.info("Tracking finished — open Analysis tab.")
    with p4:
        if st.button("▶▶"):
            st.session_state.player_frame = min(max_frame, cur + stride)
            st.session_state.is_playing = False
            st.rerun()
    with p5:
        st.slider(
            "Timeline",
            0,
            max_frame,
            value=st.session_state.timeline_slider,
            label_visibility="collapsed",
            key="timeline_slider",
        )
    with p6:
        st.number_input(
            "Frame",
            min_value=0,
            max_value=max_frame,
            value=st.session_state.frame_number_input,
            step=1,
            label_visibility="collapsed",
            key="frame_number_input",
        )

    end_f = get_tracking_end_frame(
        exit_frame=st.session_state.exit_frame,
        max_frame=max_frame,
        analysis_end_frame=st.session_state.analysis_end_frame,
    )
    st.markdown(
        f'<p class="player-time">{fmt_time(cur, fps)} / {fmt_time(end_f, fps)} &nbsp;·&nbsp; Frame {cur} &nbsp;·&nbsp; {len(coords_sorted)} tracked points</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Live tracking engine (one frame per rerun while playing) ---
    if st.session_state.is_playing and st.session_state.track_phase == "tracking":
        entry_f = st.session_state.entry_frame or 0
        end_f = get_tracking_end_frame(
            exit_frame=st.session_state.exit_frame,
            max_frame=max_frame,
            analysis_end_frame=st.session_state.analysis_end_frame,
        )

        if cur >= end_f:
            st.session_state.is_playing = False
            st.session_state.track_phase = "complete"
            st.rerun()

        next_f = cur + stride
        if next_f > end_f:
            st.session_state.is_playing = False
            st.session_state.track_phase = "complete"
            st.rerun()

        ok_n, frame_n = read_frame(st.session_state.video_path, next_f)
        if ok_n and st.session_state.track_state is not None:
            if not process_tracking_frame(frame_n, next_f, fps, settings):
                st.session_state.player_frame = next_f
                st.rerun()
            st.session_state.player_frame = next_f
            time.sleep(0.015)
            st.rerun()
        else:
            st.session_state.is_playing = False
            st.rerun()


# ===================================================================
# TAB 4 — Analysis
# ===================================================================
elif st.session_state.tab == "analysis":
    st.subheader("Trajectory analysis")

    if st.session_state.track_phase != "complete" or not st.session_state.track_coords:
        st.warning("Finish tracking first (place an End tag on the Track tab, or let the video reach its end).")
        if st.button("← Back to tracking"):
            st.session_state.tab = "track"
            st.rerun()
        st.stop()

    import re
    import shutil

    # 1. Parse Bee ID, Orientation, and Stimulus from filename
    video_name = st.session_state.video_name
    bee_id = "unknown"
    orientation = "unknown"
    p_val = None
    u_val = None

    # Parse ID (e.g. R13)
    match_id = re.search(r'\b(R\d+)\b', video_name, re.IGNORECASE)
    if match_id:
        bee_id = match_id.group(1).upper()

    # Parse Orientation (LR or TB)
    match_ori = re.search(r'\b(LR|TB)\b', video_name, re.IGNORECASE)
    if match_ori:
        orientation = match_ori.group(1).upper()

    # Parse Stimulus P...U...
    match_pu = re.search(r'\bP(\d+(?:\.\d+)?)U(\d+(?:\.\d+)?)\b', video_name, re.IGNORECASE)
    if match_pu:
        p_val = match_pu.group(1)
        u_val = match_pu.group(2)

    # 2. Lookup calibration data in results folder
    intensity = "unknown"
    hdr = "unknown"
    
    csv_path = None
    results_dir = "results"
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.startswith("extended_bbcalibrations") and f.endswith(".csv"):
                csv_path = os.path.join(results_dir, f)
                break

    if csv_path and p_val and u_val and orientation != "unknown":
        try:
            calib_df = pd.read_csv(csv_path)
            target_stim = f"p{float(p_val):.1f}u{float(u_val):.1f}"
            for idx, row in calib_df.iterrows():
                row_stim = str(row.get("stimulus", "")).strip().lower().replace(" ", "")
                row_stim_m = re.match(r'p(\d+(?:\.\d+)?)u(\d+(?:\.\d+)?)', row_stim)
                if row_stim_m:
                    row_stim_norm = f"p{float(row_stim_m.group(1)):.1f}u{float(row_stim_m.group(2)):.1f}"
                else:
                    row_stim_norm = row_stim
                
                row_ori = str(row.get("orientation", "")).strip().upper()
                if row_stim_norm == target_stim and row_ori == orientation:
                    intensity = str(row.get("intensity", "unknown")).strip()
                    hdr_raw = row.get("hdr", None)
                    if pd.notna(hdr_raw):
                        try:
                            hdr = f"{float(hdr_raw):.4f}"
                        except ValueError:
                            hdr = str(hdr_raw)
                    break
        except Exception:
            pass

    # 3. Create heading
    if p_val and u_val:
        heading = f"Bee Trajectory - ID: {bee_id} | Stimulus: p{p_val} u{u_val} | Ori: {orientation}"
        if intensity != "unknown" or hdr != "unknown":
            heading += f" (Intensity: {intensity}, HDR: {hdr})"
    else:
        heading = f"Bee Trajectory - ID: {bee_id} | Video: {os.path.splitext(video_name)[0]}"

    # Translate outcome
    outcome_str = "Unknown"
    if st.session_state.bee_went_back is True:
        outcome_str = "Went back"
    elif st.session_state.bee_went_back is False:
        outcome_str = "Still in arena"
    elif st.session_state.bee_went_back == "unknown":
        outcome_str = "Unknown"

    fps = st.session_state.tracking_fps
    df = build_processed_df(st.session_state.track_coords, fps, st.session_state.feeder_radius_mm)
    # Add Bee ID and Trial Outcome columns to the CSV
    df["bee_id"] = bee_id
    df["trial_outcome"] = outcome_str
    st.session_state.processed_df = df

    step = df["frame"].diff().dropna().median() if len(df) > 1 else 1
    pts = df[["x_mm", "y_mm"]].values
    path_len = float(np.sum(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)))) if len(pts) > 1 else 0
    t_total = len(df) * step / fps

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked duration", f"{t_total:.1f} s")
    c2.metric("Path length", f"{path_len / 10:.1f} cm")
    c3.metric("Time on feeder", f"{df['on_feeder'].sum() * step / fps:.1f} s")
    c4.metric("Frames", len(df))

    fig = plot_trajectory(df, st.session_state.entry_frame, st.session_state.exit_frame, title=heading, bee_id=bee_id, outcome=outcome_str)
    st.pyplot(fig)

    # 4. Trial Outcome Question
    st.markdown("---")
    st.markdown("### Trial Outcome")
    st.markdown("**Did the bee return to the feeder / go back or is it still in the arena?**")
    col_out1, col_out2, col_out3 = st.columns(3)
    with col_out1:
        if st.button("🟢 Yes, the bee went back", use_container_width=True):
            st.session_state.bee_went_back = True
            st.rerun()
    with col_out2:
        if st.button("🔴 No, still in arena", use_container_width=True):
            st.session_state.bee_went_back = False
            st.rerun()
    with col_out3:
        if st.button("⚪ Unknown", use_container_width=True):
            st.session_state.bee_went_back = "unknown"
            st.rerun()

    if st.session_state.bee_went_back is True:
        st.success("🟢 Success: Bee successfully returned to the feeder.")
    elif st.session_state.bee_went_back is False:
        st.error("🔴 Warning: Bee remained in the arena.")
    else:
        st.info("⚪ Outcome: Unknown (unspecified outcome).")

    # 5. Export Files & Create ZIP
    video_dir_name = os.path.splitext(video_name)[0].replace(" ", "_")
    export_dir = os.path.join("results", video_dir_name)
    os.makedirs(export_dir, exist_ok=True)

    # Generate tracked video if it doesn't exist yet
    video_export_path = os.path.join(export_dir, f"tracked_preview_{os.path.splitext(video_name)[0]}.mp4")
    if not os.path.exists(video_export_path):
        with st.spinner("Generating tracked preview video (this may take a moment)..."):
            generate_tracked_video(
                st.session_state.video_path,
                st.session_state.track_coords,
                st.session_state.entry_frame,
                st.session_state.exit_frame,
                video_export_path,
                st.session_state.circle_center,
                st.session_state.circle_radius
            )

    csv_export_path = os.path.join(export_dir, f"bee_track_{os.path.splitext(video_name)[0]}.csv")
    png_export_path = os.path.join(export_dir, f"trajectory_{os.path.splitext(video_name)[0]}.png")
    
    df.to_csv(csv_export_path, index=False)
    fig.savefig(png_export_path, dpi=200, bbox_inches="tight")

    if st.session_state.bee_went_back is not None:
        with open(os.path.join(export_dir, "trial_outcome.txt"), "w") as f:
            f.write(f"Bee ID: {bee_id}\n")
            f.write(f"Outcome: {outcome_str}\n")

    # Zip the archive
    tmp_zip_base = os.path.join(tempfile.gettempdir(), video_dir_name)
    zip_archive_path = shutil.make_archive(tmp_zip_base, 'zip', export_dir)

    with open(zip_archive_path, "rb") as f:
        zip_bytes = f.read()

    st.markdown("### Export results")
    st.download_button(
        label="📥 Export & Download entire results folder (ZIP)",
        data=zip_bytes,
        file_name=f"{video_dir_name}.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary",
    )

    trans = df[df["transition_event"].notna()][["frame", "time_sec", "transition_event"]]
    if len(trans):
        st.markdown("**Zone transitions**")
        st.dataframe(trans, use_container_width=True, hide_index=True)

    help_tags = df[df["tag_type"] == "help"][["frame", "time_sec", "x_mm", "y_mm"]]
    if len(help_tags):
        st.markdown("**Manual help points**")
        st.dataframe(help_tags, use_container_width=True, hide_index=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"bee_track_{os.path.splitext(st.session_state.video_name)[0]}.csv",
        mime="text/csv",
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        "Download trajectory plot (PNG)",
        buf,
        file_name=f"trajectory_{os.path.splitext(st.session_state.video_name)[0]}.png",
        mime="image/png",
    )
