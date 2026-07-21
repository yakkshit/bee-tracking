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
import re
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import LineCollection
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from tracking_logic import get_tracking_end_frame

# ---------------------------------------------------------------------------
# Page configuration & UI style rules
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Bee Arena Tracker", page_icon="🐝", layout="wide", initial_sidebar_state="collapsed")

# Premium Linear/Vercel styling
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', system-ui, sans-serif;
        background-color: #0B0F17;
        color: #F3F4F6;
    }
    
    .app-header {
        font-size: 32px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 8px;
        letter-spacing: -0.025em;
    }
    .app-subheader {
        font-size: 15px;
        color: #9CA3AF;
        margin-bottom: 24px;
    }
    
    .premium-card {
        background-color: #161B26;
        border: 1px solid #232D3F;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .player-shell {
        background-color: #0F131E;
        border: 1px solid #1F293D;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.2);
    }
    .player-time {
        color: #9CA3AF;
        font-family: 'Inter', monospace;
        font-size: 13px;
        margin-top: 8px;
    }
    
    div[data-baseweb="tab-list"] {
        border-bottom: 1px solid #1F293D;
        gap: 8px;
    }
    button[data-baseweb="tab"] {
        color: #9CA3AF;
        font-weight: 500;
        padding: 12px 16px;
        border-radius: 6px 6px 0 0;
        transition: all 0.2s ease;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4F46E5 !important;
        border-bottom: 2px solid #4F46E5 !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 4px;
    }
    .status-idle { background-color: #374151; color: #D1D5DB; }
    .status-ready { background-color: #065F46; color: #A7F3D0; }
    .status-tracking { background-color: #1E3A8A; color: #BFDBFE; }
    .status-lost { background-color: #7F1D1D; color: #FCA5A5; }
    .status-complete { background-color: #14532D; color: #86EFAC; }
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


def draw_calibration_overlay(img, xc, yc, r_pixels, slot_idx=0):
    slot = st.session_state.slots[slot_idx]
    out = img.copy()
    # Outer Circle - Bright Yellow (0, 255, 255)
    cv2.circle(out, (int(xc), int(yc)), int(r_pixels), (0, 255, 255), 2)
    
    # Inner Circle - Bright Cyan (255, 255, 0)
    xc_inner = slot.get("inner_circle_center")
    r_inner = slot.get("inner_circle_radius")
    
    if xc_inner is not None and r_inner is not None:
        cv2.circle(out, (int(xc_inner[0]), int(xc_inner[1])), int(r_inner), (255, 255, 0), 2)
    else:
        # Fallback to standard 21cm inner circle
        r_inner_calc = INNER_RADIUS_MM / (OUTER_RADIUS_MM / r_pixels)
        cv2.circle(out, (int(xc), int(yc)), int(r_inner_calc), (255, 255, 0), 2)
        
    # Feeder Center - Bright Orange
    cv2.circle(out, (int(xc), int(yc)), 6, (0, 140, 255), -1)
    
    # Hive Entry - Bright Green
    hive_entry = slot.get("hive_entry_point")
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


def init_track_state(frame, bbox, slot_idx=0):
    slot = st.session_state.slots[slot_idx]
    tracker = create_tracker()
    ib = [int(v) for v in clamp_bbox(bbox, frame.shape)]
    tracker_ok = False
    if tracker is not None:
        tracker.init(frame, tuple(ib))
        tracker_ok = True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cx, cy = bbox_center(ib)
    was_on_feeder = False
    if slot["circle_center"] and slot["scale_factor"]:
        _, _, d = pixel_to_mm(cx, cy, slot["circle_center"], slot["scale_factor"])
        was_on_feeder = d <= slot["feeder_radius_mm"]
    return {
        "bbox": tuple(float(v) for v in ib),
        "center": (cx, cy),
        "template": extract_template(gray, ib),
        "tracker": tracker,
        "tracker_initialized": tracker_ok,
        "was_on_feeder": was_on_feeder,
        "frame_idx": 0,
    }


def track_single_frame(frame, state, settings, slot_idx=0):
    slot = st.session_state.slots[slot_idx]
    xc, yc = slot["circle_center"]
    r_px = slot["circle_radius"]
    scale = slot["scale_factor"]
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
                bh = last_bbox[3] if last_bbox else TAG_BOX_PX
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


def make_coord(frame_idx, cx, cy, fps, slot_idx=0, tag_type="auto", status="ok"):
    slot = st.session_state.slots[slot_idx]
    xc, yc = slot["circle_center"]
    scale = slot["scale_factor"]
    x_mm, y_mm, d_mm = pixel_to_mm(cx, cy, (xc, yc), scale)
    t0 = slot["entry_frame"] if slot["entry_frame"] is not None else 0
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


def upsert_coord(coord, slot_idx=0):
    slot = st.session_state.slots[slot_idx]
    slot["track_coords"] = [c for c in slot["track_coords"] if c["frame"] != coord["frame"]]
    slot["track_coords"].append(coord)
    slot["track_coords"].sort(key=lambda c: c["frame"])


def render_player_frame(frame, coords_up_to_frame, markers, slot_idx=0, cur_center=None, status="idle"):
    slot = st.session_state.slots[slot_idx]
    vis = draw_calibration_overlay(
        frame,
        slot["circle_center"][0],
        slot["circle_center"][1],
        slot["circle_radius"],
        slot_idx
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


def plot_trajectory(df, entry_frame=None, exit_frame=None, title="Bee Trajectory", bee_id="unknown", outcome="Unknown", hive_entry_mm=None, orientation="unknown", p_val="unknown", u_val="unknown", dop="unknown"):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    
    # Outer circle boundary (dimgrey, solid)
    ax.add_patch(plt.Circle((0, 0), OUTER_RADIUS_MM, fill=False, color="#333333", lw=2, label="Outer Boundary (r = 42 cm)"))
    # Inner circle boundary (darkgrey, dashed)
    ax.add_patch(plt.Circle((0, 0), INNER_RADIUS_MM, fill=False, color="#666666", lw=1.5, ls="--", label="Inner Boundary (r = 21 cm)"))
    
    # Text labels at the top of circles
    ax.text(0, OUTER_RADIUS_MM + 10, "Outer Boundary (r = 42 cm)", ha="center", va="bottom", color="#333333", fontsize=9, fontweight="semibold")
    ax.text(0, INNER_RADIUS_MM + 10, "Inner Boundary (r = 21 cm)", ha="center", va="bottom", color="#666666", fontsize=9, fontweight="semibold")

    x_mm, y_mm, t_sec = df["x_mm"].values, df["y_mm"].values, df["time_sec"].values

    # Plot Hive Entry location if available (in mm coordinates)
    if hive_entry_mm is not None:
        hx_mm, hy_mm = hive_entry_mm
        ax.plot(hx_mm, hy_mm, "*", color="#2ecc71", ms=14, mew=1.5, mec="#1a7a42", label="Hive Entry")
        ax.annotate("HIVE", (hx_mm, hy_mm), textcoords="offset points",
                     xytext=(12, 8), fontsize=9, fontweight="bold", color="#1a7a42",
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#2ecc71', alpha=0.7))

    if len(x_mm) >= 2:
        pts = np.array([x_mm, y_mm]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = plt.Normalize(t_sec.min(), t_sec.max())
        lc = LineCollection(segs, cmap="turbo", norm=norm, linewidth=3.0)
        lc.set_array(t_sec)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, shrink=0.8, label="Time (s)")
        cbar.ax.yaxis.label.set_fontweight("bold")

        # Add time progress markers at 25%, 50%, 75% along the trajectory
        n = len(x_mm)
        for pct, marker_style, marker_color in [
            (0.25, "o", "#FFD700"),
            (0.50, "s", "#FF8C00"),
            (0.75, "D", "#FF4500"),
        ]:
            idx = int(n * pct)
            if idx < n:
                t_val = t_sec[idx]
                ax.plot(x_mm[idx], y_mm[idx], marker_style, color=marker_color, ms=7,
                        mec="black", mew=0.5, zorder=5)
                ax.annotate(f"{t_val:.1f}s", (x_mm[idx], y_mm[idx]),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=7.5, fontweight="bold", color=marker_color,
                            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor=marker_color, alpha=0.7))

        # Add a direction arrow at the end of the path
        if n >= 4:
            dx = x_mm[-1] - x_mm[-3]
            dy = y_mm[-1] - y_mm[-3]
            ax.arrow(x_mm[-3], y_mm[-3], dx * 0.3, dy * 0.3,
                     head_width=12, head_length=8, fc="#E63946", ec="#E63946",
                     alpha=0.7, zorder=6, label="Direction")

    # Plot entry/exit points and write circled "①" labels next to them
    ax.plot(x_mm[0], y_mm[0], "^", color="#2ecc71", ms=10, label="Entry Point", zorder=10)
    ax.plot(x_mm[-1], y_mm[-1], "o", color="#1a53ff", ms=10, label="Exit Point", zorder=10)
    
    # Circled number labels
    ax.annotate("①", (x_mm[0], y_mm[0]), textcoords="offset points", xytext=(0, -18),
                fontsize=9, color="#2ecc71", fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="#2ecc71", lw=1))
    ax.annotate("①", (x_mm[-1], y_mm[-1]), textcoords="offset points", xytext=(0, 18),
                fontsize=9, color="#1a53ff", fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="#1a53ff", lw=1))

    ax.plot(0, 0, "o", color="#FB8500", ms=10, label="Feeder (0,0)")
    ax.set_xlim(-460, 460)
    ax.set_ylim(-460, 460)
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    
    fig.suptitle(title, fontsize=12, fontweight="bold")
    
    # Overlay metadata text box inside the plot
    textstr = f"Bee ID: {bee_id}\nOutcome: {outcome}"
    props = dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#cccccc', alpha=0.85)
    ax.text(0.03, 0.03, textstr, transform=ax.transAxes, fontsize=10,
            fontweight='bold', verticalalignment='bottom', bbox=props)
            
    ax.legend(loc="upper right", fontsize=9, framealpha=1, facecolor="white", edgecolor="#cccccc")
    return fig


def generate_tracked_video(video_path, coords, entry_frame, exit_frame, output_path, circle_center, circle_radius):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    coords_sorted = sorted(coords, key=lambda c: c["frame"]) if coords else []

    if coords_sorted:
        coords_min = int(coords_sorted[0]["frame"])
        coords_max = int(coords_sorted[-1]["frame"])
    else:
        coords_min, coords_max = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    entry_f = int(entry_frame) if entry_frame is not None else coords_min
    exit_f = int(exit_frame) if exit_frame is not None else coords_max

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    entry_f = max(0, min(entry_f, total_frames - 1))
    exit_f = max(0, min(exit_f, total_frames - 1))

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


def apply_point_tag(px, py, tag_mode, frame, fps, slot_idx=0):
    slot = st.session_state.slots[slot_idx]
    bbox = point_to_bbox(px, py, frame.shape)
    cx, cy = bbox_center(bbox)
    cur = slot["player_frame"]

    if tag_mode == "entry":
        slot["entry_frame"] = cur
        slot["entry_point"] = (cx, cy)
        slot["track_coords"] = [c for c in slot["track_coords"] if c["frame"] >= cur]
        slot["track_state"] = init_track_state(frame, bbox, slot_idx)
        slot["track_state"]["frame_idx"] = cur
        upsert_coord(make_coord(cur, cx, cy, fps, slot_idx, tag_type="entry", status="manual"), slot_idx)
        slot["track_phase"] = "ready"
        slot["tracking_lost"] = False

    elif tag_mode == "help":
        slot["track_coords"] = [c for c in slot["track_coords"] if c["frame"] < cur]
        slot["track_state"] = init_track_state(frame, bbox, slot_idx)
        slot["track_state"]["frame_idx"] = cur
        upsert_coord(make_coord(cur, cx, cy, fps, slot_idx, tag_type="help", status="manual"), slot_idx)
        slot["track_phase"] = "tracking"
        slot["tracking_lost"] = False
        st.session_state.is_playing = True

    elif tag_mode == "exit":
        slot["track_coords"] = [c for c in slot["track_coords"] if c["frame"] < cur]
        slot["exit_frame"] = cur
        slot["exit_point"] = (cx, cy)
        upsert_coord(make_coord(cur, cx, cy, fps, slot_idx, tag_type="exit", status="manual"), slot_idx)
        slot["track_phase"] = "tracking"
        st.session_state.is_playing = True
        slot["tracking_lost"] = False

    elif tag_mode == "analysis_end":
        slot["track_coords"] = [c for c in slot["track_coords"] if c["frame"] < cur]
        slot["analysis_end_frame"] = cur
        slot["analysis_end_point"] = (cx, cy)
        upsert_coord(make_coord(cur, cx, cy, fps, slot_idx, tag_type="analysis_end", status="manual"), slot_idx)
        slot["track_phase"] = "complete"
        slot["tracking_lost"] = False


def process_tracking_frame(frame, frame_idx, fps, settings, slot_idx=0):
    slot = st.session_state.slots[slot_idx]
    if slot["track_state"] is None:
        return False
    state = slot["track_state"].copy()
    state["frame_idx"] = frame_idx
    center, _, status, new_state = track_single_frame(frame, state, settings, slot_idx)
    if center is None or status == "lost":
        slot["tracking_lost"] = True
        slot["track_phase"] = "paused_lost"
        return False
    cx, cy = center
    upsert_coord(make_coord(frame_idx, cx, cy, fps, slot_idx, tag_type="auto", status=status), slot_idx)
    slot["track_state"] = new_state
    return True


# ---------------------------------------------------------------------------
# Session state & Slot layout configuration
# ---------------------------------------------------------------------------
DEFAULTS = {
    "tab": "load",
    "video_path": None,
    "video_name": "",
    "video_folder": "",
    "local_videos": [],
    "selected_video_index": None,
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

SLOT_KEYS = [
    "video_path",
    "video_name",
    "selected_video_index",
    "circle_center",
    "circle_radius",
    "inner_circle_center",
    "inner_circle_radius",
    "hive_entry_point",
    "scale_factor",
    "entry_frame",
    "entry_point",
    "exit_frame",
    "exit_point",
    "analysis_end_frame",
    "analysis_end_point",
    "track_coords",
    "track_state",
    "track_phase",
    "tracking_lost",
    "bee_went_back",
    "processed_df",
    "player_frame",
    "last_player_frame",
    "timeline_slider",
    "frame_number_input",
]

# Initialize multi-slot state list
if "slots" not in st.session_state:
    st.session_state.slots = [
        {
            "video_path": None,
            "video_name": "",
            "selected_video_index": None,
            "circle_center": None,
            "circle_radius": None,
            "inner_circle_center": None,
            "inner_circle_radius": None,
            "hive_entry_point": None,
            "scale_factor": None,
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
            "bee_went_back": "unknown",
            "processed_df": None,
            "player_frame": 0,
            "last_player_frame": 0,
            "timeline_slider": 0,
            "frame_number_input": 0,
            "feeder_radius_mm": 40.0,
            "tracking_fps": 30.0,
        }
        for _ in range(4)
    ]

if "num_slots" not in st.session_state:
    st.session_state.num_slots = 1

if "active_slot" not in st.session_state:
    st.session_state.active_slot = 0

def sync_active_slot_to_flat():
    active = st.session_state.active_slot
    slot = st.session_state.slots[active]
    for k in SLOT_KEYS:
        st.session_state[k] = slot.get(k, None)
    st.session_state.feeder_radius_mm = slot.get("feeder_radius_mm", 40.0)
    st.session_state.tracking_fps = slot.get("tracking_fps", 30.0)

def sync_flat_to_active_slot():
    active = st.session_state.active_slot
    slot = st.session_state.slots[active]
    for k in SLOT_KEYS:
        slot[k] = st.session_state.get(k, None)
    slot["feeder_radius_mm"] = st.session_state.feeder_radius_mm
    slot["tracking_fps"] = st.session_state.tracking_fps

# Bidirectional sync on script load
sync_active_slot_to_flat()

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

settings = {**TRACK_SETTINGS, "feeder_radius_mm": st.session_state.feeder_radius_mm}

# ---------------------------------------------------------------------------
# Navigation Tabs Layout
# ---------------------------------------------------------------------------
st.markdown('<div class="app-header">🐝 Bee Arena Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subheader">Linear-inspired concurrent animal tracking system</div>', unsafe_allow_html=True)

tabs = {
    "load": "1 · Video Setup",
    "calibrate": "2 · Arena Calibration",
    "track": "3 · Tracking Room",
    "analysis": "4 · Statistics & Analysis"
}
tab_keys = list(tabs.keys())

disabled_tabs = []
if not st.session_state.slots[0]["video_path"]:
    disabled_tabs = ["calibrate", "track", "analysis"]
elif not st.session_state.slots[0]["circle_center"]:
    disabled_tabs = ["track", "analysis"]

choice = st.radio(
    "Workflow Navigation",
    tab_keys,
    format_func=lambda k: tabs[k],
    horizontal=True,
    index=tab_keys.index(st.session_state.tab),
    label_visibility="collapsed"
)

if choice in disabled_tabs:
    st.warning(f"Please complete previous workflow steps before unlocking **{tabs[choice]}**.")
    choice = st.session_state.tab
st.session_state.tab = choice

st.markdown("<br>", unsafe_allow_html=True)

# ===================================================================
# TAB 1 — Load video Setup
# ===================================================================
if st.session_state.tab == "load":
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 📽 Setup Tracking Videos")
    st.write("Specify how many videos you wish to track concurrently and select the paths or files.")

    num_slots = st.selectbox(
        "Concurrently tracked videos (maximum 4):",
        [1, 2, 3, 4],
        index=st.session_state.num_slots - 1,
        help="Choose the number of video streams to process side-by-side."
    )
    if num_slots != st.session_state.num_slots:
        sync_flat_to_active_slot()
        st.session_state.num_slots = num_slots
        if st.session_state.active_slot >= num_slots:
            st.session_state.active_slot = 0
        sync_active_slot_to_flat()
        st.rerun()

    if num_slots > 1:
        slot_names = [f"Slot {i+1}: {st.session_state.slots[i]['video_name'] or 'Empty'}" for i in range(num_slots)]
        active_sel = st.radio(
            "Select Video Slot to Configure:",
            range(num_slots),
            format_func=lambda idx: slot_names[idx],
            index=st.session_state.active_slot,
            horizontal=True,
            help="Select which slot you are loading a video file for."
        )
        if active_sel != st.session_state.active_slot:
            sync_flat_to_active_slot()
            st.session_state.active_slot = active_sel
            sync_active_slot_to_flat()
            st.rerun()

    st.markdown(f"#### Loading Video for **Slot {st.session_state.active_slot + 1}**")
    
    workspace_video = "2024-11-17 17-04-16.R13.LR.P0U8.mp4"
    if os.path.exists(workspace_video):
        if st.checkbox(f"Use default workspace video `{workspace_video}` (Slot {st.session_state.active_slot + 1})", value=False, key=f"use_workspace_{st.session_state.active_slot}"):
            st.session_state.video_path = workspace_video
            st.session_state.video_name = workspace_video
            st.session_state.selected_video_index = None

    upload_file = st.file_uploader(
        f"Upload a video file (.mp4, .avi, .mov) for Slot {st.session_state.active_slot + 1}",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=False,
        key=f"upload_{st.session_state.active_slot}",
        help="Upload a video file to run tracking on."
    )
    if upload_file is not None:
        suffix = os.path.splitext(upload_file.name)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(upload_file.read())
        tmp.close()
        st.session_state.video_path = tmp.name
        st.session_state.video_name = upload_file.name
        st.session_state.video_folder = ""
        st.session_state.local_videos = []
        st.session_state.selected_video_index = None

    path_input = st.text_input(
        f"Or local path to folder/video file for Slot {st.session_state.active_slot + 1}:",
        value=st.session_state.video_folder,
        placeholder="Enter path to directory containing video files",
        key=f"path_input_{st.session_state.active_slot}",
        help="Enter the folder path or exact file location on your computer."
    )
    if path_input != st.session_state.video_folder:
        st.session_state.video_folder = path_input
        st.session_state.local_videos = []
        st.session_state.selected_video_index = None
        st.session_state.video_path = None
        st.session_state.video_name = ""

    if st.button("Refresh file list", key=f"ref_{st.session_state.active_slot}"):
        st.session_state.local_videos = []
        st.session_state.selected_video_index = None

    if st.session_state.video_folder:
        folder = st.session_state.video_folder
        if os.path.isdir(folder):
            found = sorted([f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in {".mp4", ".mov", ".avi"}])
            st.session_state.local_videos = [{"name": f, "path": os.path.join(folder, f)} for f in found]
        elif os.path.isfile(folder):
            st.session_state.local_videos = [{"name": os.path.basename(folder), "path": folder}]
        else:
            st.warning("Specified path is not a valid folder or file location.")

    if st.session_state.local_videos:
        file_options = []
        for video in st.session_state.local_videos:
            video_dir_name = os.path.splitext(video["name"])[0].replace(" ", "_")
            export_dir = os.path.join("results", video_dir_name)
            tracked = os.path.exists(export_dir) and any(f.startswith("bee_track_") and f.endswith(".csv") for f in os.listdir(export_dir))
            label = f"{video['name']} {'✅ tracked' if tracked else '• new'}"
            file_options.append(label)

        choice_index = st.selectbox(
            "Select Video File:",
            list(range(len(file_options))),
            format_func=lambda i: file_options[i],
            index=st.session_state.selected_video_index if st.session_state.selected_video_index is not None else 0,
            key=f"choice_{st.session_state.active_slot}",
            help="Choose a file from the folder to load."
        )
        st.session_state.selected_video_index = choice_index
        selected_video = st.session_state.local_videos[choice_index]
        st.session_state.video_path = selected_video["path"]
        st.session_state.video_name = selected_video["name"]

    if st.session_state.video_path:
        meta = video_meta(st.session_state.video_path)
        st.session_state.tracking_fps = meta["fps"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Resolution", f"{meta['w']}×{meta['h']}")
        c2.metric("FPS", f"{meta['fps']:.0f}")
        c3.metric("Duration", fmt_time(meta["frames"], meta["fps"]))
        ok, f0 = read_frame(st.session_state.video_path, 0)
        if ok:
            st.image(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Initial Frame Preview")
            
        video_dir_name = os.path.splitext(st.session_state.video_name)[0].replace(" ", "_")
        export_dir = os.path.join("results", video_dir_name)
        existing_csv = None
        if os.path.exists(export_dir):
            for f in os.listdir(export_dir):
                if f.startswith("bee_track_") and f.endswith(".csv"):
                    existing_csv = os.path.join(export_dir, f)
                    break

        if existing_csv:
            st.info(f"Existing tracking results found in {export_dir}.")
            if st.button("Load existing results into slot", key=f"load_res_{st.session_state.active_slot}"):
                try:
                    df_existing = pd.read_csv(existing_csv)
                    st.session_state.processed_df = df_existing
                    st.session_state.track_coords = df_existing.to_dict("records")
                    if "tag_type" in df_existing.columns:
                        entries = df_existing[df_existing["tag_type"] == "entry"]["frame"]
                        exits = df_existing[df_existing["tag_type"] == "exit"]["frame"]
                        if not entries.empty:
                            st.session_state.entry_frame = int(entries.iloc[0])
                        if not exits.empty:
                            st.session_state.exit_frame = int(exits.iloc[-1])
                    st.session_state.track_phase = "complete"
                    st.success("Loaded existing results successfully.")
                    sync_flat_to_active_slot()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load existing results: {e}")

        if st.button("Next: Calibrate arena →", type="primary", key="load_done_btn"):
            sync_flat_to_active_slot()
            st.session_state.tab = "calibrate"
            st.rerun()
    else:
        st.info("Empty State: Please select or upload a video file to begin.")
    st.markdown("</div>", unsafe_allow_html=True)


# ===================================================================
# TAB 2 — Calibrate
# ===================================================================
elif st.session_state.tab == "calibrate":
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 📐 Arena Circle Calibration")
    st.caption("Click exactly 9 points on the canvas: 4 on the outer rim, 4 on the inner rim, and 1 at the hive entry.")

    num_slots = st.session_state.num_slots
    if num_slots > 1:
        slot_names = []
        for i in range(num_slots):
            sl = st.session_state.slots[i]
            v_name = sl["video_name"] or "Empty"
            cal_status = "✅ Calibrated" if sl["circle_center"] is not None else "❌ Not Calibrated"
            slot_names.append(f"Slot {i+1} ({cal_status}): {v_name}")
            
        active_sel = st.radio(
            "Select Video Slot to Calibrate:",
            range(num_slots),
            format_func=lambda idx: slot_names[idx],
            index=st.session_state.active_slot,
            horizontal=True,
            help="Switch to select which video stream to calibrate."
        )
        if active_sel != st.session_state.active_slot:
            sync_flat_to_active_slot()
            st.session_state.active_slot = active_sel
            sync_active_slot_to_flat()
            st.rerun()

    st.markdown(f"#### Calibrating **Slot {st.session_state.active_slot + 1}**: `{st.session_state.video_name}`")
    
    if not st.session_state.video_path:
        st.warning("Please setup a video in Video Setup first.")
        st.stop()

    if st.button("Load demo calibration parameters", key=f"demo_calib_{st.session_state.active_slot}"):
        st.session_state.circle_center = (942.0, 433.0)
        st.session_state.circle_radius = 379.0
        st.session_state.scale_factor = OUTER_RADIUS_MM / 379.0
        st.session_state.inner_circle_center = (942.0, 433.0)
        st.session_state.inner_circle_radius = 189.5
        st.session_state.hive_entry_point = (1300.0, 433.0)
        sync_flat_to_active_slot()
        st.rerun()

    ok, frame0 = read_frame(st.session_state.video_path, 0)
    if not ok:
        st.error("Cannot read video stream.")
        st.stop()

    oh, ow = frame0.shape[:2]
    ch = int(oh * CANVAS_W / ow)
    ratio = ow / CANVAS_W
    
    # If already calibrated, render the overlay directly on the canvas background
    bg_frame = frame0.copy()
    if st.session_state.circle_center is not None and st.session_state.circle_radius is not None:
        bg_frame = draw_calibration_overlay(
            bg_frame,
            st.session_state.circle_center[0],
            st.session_state.circle_center[1],
            st.session_state.circle_radius,
            st.session_state.active_slot
        )
        
    pil = Image.fromarray(cv2.cvtColor(cv2.resize(bg_frame, (CANVAS_W, ch)), cv2.COLOR_BGR2RGB))

    # Single-slot active calibration canvas to avoid mixing coordinates
    result = st_canvas(
        fill_color="rgba(0, 255, 255, 0.4)",
        stroke_width=2,
        stroke_color="#00FFFF",
        background_image=pil,
        update_streamlit=True,
        height=ch,
        width=CANVAS_W,
        drawing_mode="point",
        key=f"calib_canvas_slot_{st.session_state.active_slot}",
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

                overlay = draw_calibration_overlay(frame0, xc_o, yc_o, r_o, st.session_state.active_slot)
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Fitted Calibration Overlay Preview")

                if st.button("Next: Start tracking room →", type="primary", key="done_calib_btn"):
                    sync_flat_to_active_slot()
                    st.session_state.tab = "track"
                    for i in range(num_slots):
                        slot = st.session_state.slots[i]
                        slot["player_frame"] = 0
                        slot["last_player_frame"] = 0
                        slot["timeline_slider"] = 0
                        slot["frame_number_input"] = 0
                        slot["track_coords"] = []
                        slot["track_phase"] = "idle"
                    st.session_state.is_playing = False
                    sync_active_slot_to_flat()
                    st.rerun()
            else:
                st.error("Could not fit circles. Verify you clicked exactly on the rims.")
        elif len(orig) >= 4:
            st.info("Keep clicking: click 4 points for the inner circle, and then 1 point for the hive entry.")
    st.markdown("</div>", unsafe_allow_html=True)


# ===================================================================
# TAB 3 — Track (video player)
# ===================================================================
elif st.session_state.tab == "track":
    num_slots = st.session_state.num_slots

    # --- Callbacks for Slider / Input changes (User Interacted) ---
    def on_timeline_change(slot_idx):
        slide_key = f"timeline_slider_widget_{slot_idx}"
        new_val = st.session_state[slide_key]
        st.session_state.slots[slot_idx]["player_frame"] = new_val
        st.session_state.slots[slot_idx]["last_player_frame"] = new_val
        st.session_state.is_playing = False

    def on_num_change(slot_idx):
        num_key = f"frame_num_widget_{slot_idx}"
        new_val = st.session_state[num_key]
        st.session_state.slots[slot_idx]["player_frame"] = new_val
        st.session_state.slots[slot_idx]["last_player_frame"] = new_val
        st.session_state.is_playing = False

    if num_slots > 1:
        slot_names = []
        for idx in range(num_slots):
            sl = st.session_state.slots[idx]
            v_name = sl["video_name"] or f"Slot {idx+1}"
            phase_str = sl["track_phase"].upper()
            if sl["tracking_lost"]:
                phase_str = "LOST"
            slot_names.append(f"Slot {idx+1} ({phase_str}): {v_name}")
            
        active_sel = st.radio(
            "Select active video to apply manual tags to:",
            range(num_slots),
            format_func=lambda idx: slot_names[idx],
            index=st.session_state.active_slot,
            horizontal=True,
            help="Choose which video you want to interact with on the manual tag buttons."
        )
        if active_sel != st.session_state.active_slot:
            sync_flat_to_active_slot()
            st.session_state.active_slot = active_sel
            sync_active_slot_to_flat()
            st.rerun()

    # --- Status banners at the top of columns ---
    cols_status = st.columns(num_slots)
    for i in range(num_slots):
        slot = st.session_state.slots[i]
        with cols_status[i]:
            phase = slot["track_phase"]
            vname = slot["video_name"] or f"Slot {i+1}"
            if phase == "idle":
                st.markdown(f'<div class="status-badge status-idle">Slot {i+1}: Tag Entry Point</div>', unsafe_allow_html=True)
            elif phase == "ready":
                st.markdown(f'<div class="status-badge status-ready">Slot {i+1}: Ready</div>', unsafe_allow_html=True)
            elif phase == "tracking":
                st.markdown(f'<div class="status-badge status-tracking">Slot {i+1}: Tracking</div>', unsafe_allow_html=True)
            elif phase == "paused_lost":
                st.markdown(f'<div class="status-badge status-lost">Slot {i+1}: Lost</div>', unsafe_allow_html=True)
            elif phase == "complete":
                st.markdown(f'<div class="status-badge status-complete">Slot {i+1}: Complete</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Tag tool buttons ---
    t1, t2, t3, t4, t5, t6 = st.columns(6)
    with t1:
        if st.button("🟢 Entry tag", use_container_width=True, type="primary" if st.session_state.tag_mode == "entry" else "secondary", help="Click on the bee where it enters the arena."):
            st.session_state.tag_mode = "entry" if st.session_state.tag_mode != "entry" else None
            st.session_state.is_playing = False
            st.rerun()
    with t2:
        if st.button("🟡 Help tag", use_container_width=True, type="primary" if st.session_state.tag_mode == "help" else "secondary", help="Use if the tracker loses tracking; click the bee to re-acquire."):
            st.session_state.tag_mode = "help" if st.session_state.tag_mode != "help" else None
            st.session_state.is_playing = False
            st.rerun()
    with t3:
        if st.button("🔴 Exit tag", use_container_width=True, type="primary" if st.session_state.tag_mode == "exit" else "secondary", help="Mark where the bee exits the arena."):
            st.session_state.tag_mode = "exit" if st.session_state.tag_mode != "exit" else None
            st.session_state.is_playing = False
            st.rerun()
    with t4:
        if st.button("⏹ End tag", use_container_width=True, type="primary" if st.session_state.tag_mode == "analysis_end" else "secondary", help="Manually terminate tracking boundary."):
            st.session_state.tag_mode = "analysis_end" if st.session_state.tag_mode != "analysis_end" else None
            st.session_state.is_playing = False
            st.rerun()
    with t5:
        if st.button("↺ Reset Active Slot", use_container_width=True, help="Reset the tracking state of the currently active video slot."):
            for k in ("entry_frame", "entry_point", "exit_frame", "exit_point", "analysis_end_frame", "analysis_end_point", "track_state"):
                st.session_state[k] = None
            st.session_state.track_coords = []
            st.session_state.track_phase = "idle"
            st.session_state.is_playing = False
            st.session_state.player_frame = 0
            st.session_state.timeline_slider = 0
            st.session_state.frame_number_input = 0
            st.session_state.tag_mode = None
            st.session_state.tracking_lost = False
            sync_flat_to_active_slot()
            st.rerun()
    with t6:
        if st.button("↺ Reset All Slots", use_container_width=True, help="Reset tracking states of all loaded slots."):
            for idx in range(num_slots):
                sl = st.session_state.slots[idx]
                for k in ("entry_frame", "entry_point", "exit_frame", "exit_point", "analysis_end_frame", "analysis_end_point", "track_state"):
                    sl[k] = None
                sl["track_coords"] = []
                sl["track_phase"] = "idle"
                sl["player_frame"] = 0
                sl["timeline_slider"] = 0
                sl["frame_number_input"] = 0
                sl["tracking_lost"] = False
            st.session_state.is_playing = False
            st.session_state.tag_mode = None
            sync_active_slot_to_flat()
            st.rerun()

    with st.expander("⚙️ Tracking Config", expanded=False):
        st.session_state.track_stride = st.slider(
            "Frame Stride",
            min_value=1,
            max_value=60,
            value=st.session_state.track_stride,
            step=1,
            help="1x tracks every frame, 2x tracks every other frame, increasing playback speed."
        )

    if st.session_state.tag_mode:
        mode_labels = {
            "entry": "Entry — click the bee on the active canvas",
            "help": "Help — click the bee to re-acquire",
            "exit": "Exit — click the exit location",
            "analysis_end": "End — click the end frame location",
        }
        st.info(f"**Tool Active:** {mode_labels[st.session_state.tag_mode]} on **Slot {st.session_state.active_slot+1}**")

    # --- Inject keyboard shortcut Shift+D ---
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

    stride = int(st.session_state.get("track_stride", 1))

    # --- Render Grid Layout of Players ---
    import math
    num_rows = math.ceil(num_slots / 2)
    grid_cols = []
    if num_slots == 1:
        grid_cols = [st.container()]
    else:
        for r in range(num_rows):
            grid_cols.extend(st.columns(2))

    for i in range(num_slots):
        slot = st.session_state.slots[i]
        if not slot["video_path"]:
            with grid_cols[i]:
                st.warning(f"Slot {i+1}: No video loaded.")
            continue

        slot_meta = video_meta(slot["video_path"])
        slot_fps = slot.get("tracking_fps") or slot_meta["fps"] or 30.0
        slot_max = max(0, slot_meta["frames"] - 1)

        slot_cur = int(slot["player_frame"])
        slot_cur = max(0, min(slot_cur, slot_max))
        slot["last_player_frame"] = slot_cur

        # Force state update of the widgets before they are drawn
        st.session_state[f"timeline_slider_widget_{i}"] = slot_cur
        st.session_state[f"frame_num_widget_{i}"] = slot_cur

        ok_f, frame = read_frame(slot["video_path"], slot_cur)
        if not ok_f:
            with grid_cols[i]:
                st.error(f"Cannot read frame {slot_cur} for Slot {i+1}")
            continue

        slot_coords = slot["track_coords"]
        coord_at_cur = next((c for c in slot_coords if c["frame"] == slot_cur), None)
        if coord_at_cur:
            slot["tracking_lost"] = False
            if slot["track_phase"] == "paused_lost":
                slot["track_phase"] = "ready"
            if slot["track_state"] is None or slot["track_state"].get("frame_idx") != slot_cur:
                bbox = point_to_bbox(coord_at_cur["x_pixel"], coord_at_cur["y_pixel"], frame.shape)
                slot["track_state"] = init_track_state(frame, bbox, i)
                slot["track_state"]["frame_idx"] = slot_cur
        else:
            if slot["entry_frame"] is not None:
                if slot_cur < slot["entry_frame"]:
                    slot["track_phase"] = "idle"
                    slot["tracking_lost"] = False
                elif slot_cur > slot["entry_frame"]:
                    slot["tracking_lost"] = True
                    slot["track_phase"] = "paused_lost"

        coords_sorted = sorted(slot["track_coords"], key=lambda c: c["frame"])
        coords_visible = [c for c in coords_sorted if c["frame"] <= slot_cur]

        markers = []
        if slot["entry_point"]:
            markers.append((*slot["entry_point"], (42, 157, 143), "ENTRY"))
        if slot["exit_point"]:
            markers.append((*slot["exit_point"], (230, 57, 70), "EXIT"))
        if slot["analysis_end_point"]:
            markers.append((*slot["analysis_end_point"], (255, 183, 77), "END"))

        cur_center = (coord_at_cur["x_pixel"], coord_at_cur["y_pixel"]) if coord_at_cur else None
        cur_status = coord_at_cur.get("status", "ok") if coord_at_cur else "idle"
        vis = render_player_frame(frame, coords_visible, markers, i, cur_center, cur_status)

        oh, ow = vis.shape[:2]
        canvas_width = 440 if num_slots > 1 else CANVAS_W
        ch = int(oh * canvas_width / ow)
        ratio = ow / canvas_width
        vis_rgb = cv2.cvtColor(cv2.resize(vis, (canvas_width, ch)), cv2.COLOR_BGR2RGB)

        with grid_cols[i]:
            title_suffix = " ⭐️ (ACTIVE)" if i == st.session_state.active_slot else ""
            st.markdown(f"##### Video Slot {i+1}: `{slot['video_name']}`{title_suffix}")
            st.markdown('<div class="player-shell">', unsafe_allow_html=True)

            if st.session_state.tag_mode and i == st.session_state.active_slot:
                canvas = st_canvas(
                    fill_color="rgba(255,255,255,0)",
                    stroke_width=0,
                    background_image=Image.fromarray(vis_rgb),
                    update_streamlit=True,
                    height=ch,
                    width=canvas_width,
                    drawing_mode="point",
                    point_display_radius=8,
                    key=f"tag_{st.session_state.tag_mode}_slot_{i}_{slot_cur}",
                )
                if canvas.json_data:
                    for obj in canvas.json_data.get("objects", []):
                        if obj.get("type") == "circle":
                            r = obj.get("radius", 0)
                            cx_c = (obj["left"] + r) * ratio
                            cy_c = (obj["top"] + r) * ratio
                            apply_point_tag(cx_c, cy_c, st.session_state.tag_mode, frame, slot_fps, i)
                            st.session_state.tag_mode = None
                            if slot["track_phase"] == "complete" and num_slots == 1:
                                st.session_state.tab = "analysis"
                            sync_active_slot_to_flat()
                            st.rerun()
            else:
                st.image(vis_rgb, use_container_width=True)

            pc1, pc2, pc3 = st.columns([5, 2, 2])
            with pc1:
                st.slider(
                    f"Timeline Slot {i+1}",
                    0,
                    slot_max,
                    value=slot["player_frame"],
                    label_visibility="collapsed",
                    key=f"timeline_slider_widget_{i}",
                    on_change=on_timeline_change,
                    args=(i,)
                )
            with pc2:
                st.number_input(
                    f"Frame Slot {i+1}",
                    min_value=0,
                    max_value=slot_max,
                    value=slot["player_frame"],
                    step=1,
                    label_visibility="collapsed",
                    key=f"frame_num_widget_{i}",
                    on_change=on_num_change,
                    args=(i,)
                )
            with pc3:
                if i != st.session_state.active_slot:
                    if st.button("Activate", key=f"activate_slot_btn_{i}", use_container_width=True, help="Activate this slot for tagging"):
                        sync_flat_to_active_slot()
                        st.session_state.active_slot = i
                        sync_active_slot_to_flat()
                        st.rerun()
                else:
                    st.markdown("<div style='text-align: center; color: #2A9D8F; font-weight: bold; margin-top: 4px;'>Active</div>", unsafe_allow_html=True)

            slot_end_f = get_tracking_end_frame(
                exit_frame=slot["exit_frame"],
                max_frame=slot_max,
                analysis_end_frame=slot["analysis_end_frame"],
            )
            st.markdown(
                f'<p class="player-time">{fmt_time(slot_cur, slot_fps)} / {fmt_time(slot_end_f, slot_fps)} &nbsp;·&nbsp; Frame {slot_cur} &nbsp;·&nbsp; {len(slot_coords)} pts</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # --- Global Controls Below Players ---
    st.markdown("---")
    p1, p2, p3, p4, p5 = st.columns([1, 1, 1, 1, 4])
    
    active_slot = st.session_state.active_slot
    active_s = st.session_state.slots[active_slot]
    active_cur = active_s["player_frame"]
    active_meta = video_meta(active_s["video_path"]) if active_s["video_path"] else {"fps": 30.0, "frames": 1}
    active_fps = active_s.get("tracking_fps") or active_meta["fps"] or 30.0
    active_max = max(0, active_meta["frames"] - 1)
    
    with p1:
        if st.button("⏮", help="Back 5s (Active Slot)"):
            active_s["player_frame"] = max(0, active_cur - int(active_fps * 5))
            st.session_state.is_playing = False
            sync_active_slot_to_flat()
            st.rerun()
    with p2:
        if st.button("◀", help="Back 1 Stride (Active Slot)"):
            active_s["player_frame"] = max(0, active_cur - stride)
            st.session_state.is_playing = False
            sync_active_slot_to_flat()
            st.rerun()
    with p3:
        play_label = "⏸" if st.session_state.is_playing else "▶"
        if st.button(play_label, help="Play / Pause All"):
            any_ready = False
            for idx in range(num_slots):
                sl = st.session_state.slots[idx]
                if sl["video_path"] and sl["entry_frame"] is not None:
                    any_ready = True
            if not any_ready:
                st.warning("Set an Entry tag on at least one video first.")
            else:
                st.session_state.is_playing = not st.session_state.is_playing
                if st.session_state.is_playing:
                    for idx in range(num_slots):
                        sl = st.session_state.slots[idx]
                        if sl["track_phase"] in ("ready", "tracking", "paused_lost") and not sl["tracking_lost"]:
                            sl["track_phase"] = "tracking"
                            sl["track_coords"] = [c for c in sl["track_coords"] if c["frame"] <= sl["player_frame"]]
                sync_active_slot_to_flat()
                st.rerun()
    with p4:
        if st.button("▶▶", help="Forward 1 Stride (Active Slot)"):
            active_s["player_frame"] = min(active_max, active_cur + stride)
            st.session_state.is_playing = False
            sync_active_slot_to_flat()
            st.rerun()
    with p5:
        st.markdown(f"**Playback Controls (Stepping targets Slot {active_slot+1})** | Speed: {stride} frames")

    # --- Live tracking engine (synchronized concurrent frame loops) ---
    if st.session_state.is_playing:
        any_advanced = False
        for idx in range(num_slots):
            slot = st.session_state.slots[idx]
            if not slot["video_path"] or slot["track_state"] is None:
                continue
            if slot["track_phase"] != "tracking" or slot["tracking_lost"]:
                continue

            slot_meta = video_meta(slot["video_path"])
            slot_max = max(0, slot_meta["frames"] - 1)
            slot_cur = int(slot["player_frame"])

            end_f = get_tracking_end_frame(
                exit_frame=slot["exit_frame"],
                max_frame=slot_max,
                analysis_end_frame=slot["analysis_end_frame"],
            )

            if slot_cur >= end_f:
                slot["track_phase"] = "complete"
                continue

            next_f = slot_cur + stride
            if next_f > end_f:
                slot["track_phase"] = "complete"
                continue

            ok_n, frame_n = read_frame(slot["video_path"], next_f)
            if ok_n:
                slot_fps = slot.get("tracking_fps") or slot_meta["fps"] or 30.0
                if process_tracking_frame(frame_n, next_f, slot_fps, settings, idx):
                    slot["player_frame"] = next_f
                    slot["last_player_frame"] = next_f
                    any_advanced = True
                else:
                    slot["player_frame"] = next_f
                    slot["last_player_frame"] = next_f
                    any_advanced = True
            else:
                slot["track_phase"] = "complete"

        if any_advanced:
            time.sleep(0.015)
            sync_active_slot_to_flat()
            st.rerun()
        else:
            st.session_state.is_playing = False
            sync_active_slot_to_flat()
            st.rerun()


# ===================================================================
# TAB 4 — Analysis
# ===================================================================
elif st.session_state.tab == "analysis":
    st.subheader("Trajectory analysis")

    num_slots = st.session_state.num_slots
    
    # 1. Verification that tracking is complete
    uncomplete = [i for i in range(num_slots) if st.session_state.slots[i]["track_phase"] != "complete" or not st.session_state.slots[i]["track_coords"]]
    if uncomplete:
        st.warning(f"Complete tracking for all active slots first. Unfinished slots: {', '.join([f'Slot {i+1}' for i in uncomplete])}")
        if st.button("← Back to tracking"):
            st.session_state.tab = "track"
            st.rerun()
        st.stop()

    # 2. If multiple slots, show comparative metrics at the top
    if num_slots > 1:
        st.markdown("### 📊 Metrics Comparison")
        comp_records = []
        for i in range(num_slots):
            slot = st.session_state.slots[i]
            video_name = slot["video_name"]
            import re
            bee_id = "unknown"
            match_id = re.search(r'\b(R[_\s]*\d+)\b', video_name, re.IGNORECASE)
            if match_id:
                bee_id = match_id.group(1).upper().replace("_", "").replace(" ", "")
            
            df_temp = build_processed_df(slot["track_coords"], slot.get("tracking_fps") or 30.0, slot.get("feeder_radius_mm") or 40.0)
            step = df_temp["frame"].diff().dropna().median() if len(df_temp) > 1 else 1
            pts = df_temp[["x_mm", "y_mm"]].values
            path_len = float(np.sum(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)))) if len(pts) > 1 else 0
            
            outcome_str = "Unknown"
            if slot["bee_went_back"] is True:
                outcome_str = "Went back"
            elif slot["bee_went_back"] is False:
                outcome_str = "Still in arena"
                
            comp_records.append({
                "Slot": f"Slot {i+1}",
                "Video Name": video_name,
                "Bee ID": bee_id,
                "Outcome": outcome_str,
                "Duration (s)": round(len(df_temp) * step / (slot.get("tracking_fps") or 30.0), 1),
                "Path Length (cm)": round(path_len / 10.0, 1),
                "Time on Feeder (s)": round(df_temp['on_feeder'].sum() * step / (slot.get("tracking_fps") or 30.0), 1),
            })
        st.dataframe(pd.DataFrame(comp_records), use_container_width=True, hide_index=True)
        st.markdown("---")
        
        # Select active slot to display detailed breakdown
        active_sel = st.selectbox(
            "Select Video Slot for Detailed Analysis & Export:",
            range(num_slots),
            format_func=lambda idx: f"Video Slot {idx+1} : {st.session_state.slots[idx]['video_name']}",
            index=st.session_state.active_slot
        )
        if active_sel != st.session_state.active_slot:
            sync_flat_to_active_slot()
            st.session_state.active_slot = active_sel
            sync_active_slot_to_flat()
            st.rerun()

    import re
    import shutil

    # 1. Parse Bee ID, Orientation, and Stimulus from filename
    video_name = st.session_state.video_name
    bee_id = "unknown"
    orientation = "unknown"
    p_val = "unknown"
    u_val = "unknown"

    # Robust parser for R13 / R_13 / R 13
    match_id = re.search(r'\b(R[_\s]*\d+)\b', video_name, re.IGNORECASE)
    if match_id:
        bee_id = match_id.group(1).upper().replace("_", "").replace(" ", "")

    # Parse Orientation (LR or TB)
    match_ori = re.search(r'\b(LR|TB)\b', video_name, re.IGNORECASE)
    if match_ori:
        orientation = match_ori.group(1).upper()

    # Robust parser for P_X.X_U_Y.Y or P0.1U8 or P4.3U8.0
    match_pu = re.search(r'\bP[_\s]*(\d+(?:\.\d+)?)[_\s]*U[_\s]*(\d+(?:\.\d+)?)\b', video_name, re.IGNORECASE)
    if match_pu:
        p_val = match_pu.group(1)
        u_val = match_pu.group(2)

    # 2. Lookup calibration data in results folder
    intensity = "unknown"
    hdr = "unknown"
    xhdr = "unknown"
    xdop = "unknown"
    x_val = "unknown"
    edge_intensity = "unknown"
    signal_rank = "unknown"
    calibration_date = "unknown"
    
    csv_path = None
    results_dir = "calibrations"
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.startswith("extended_bbcalibrations") and f.endswith(".csv"):
                csv_path = os.path.join(results_dir, f)
                break

    if csv_path and p_val != "unknown" and u_val != "unknown" and orientation != "unknown":
        try:
            calib_df = pd.read_csv(csv_path)
            target_p = float(p_val)
            target_u = float(u_val)
            
            for idx, row in calib_df.iterrows():
                row_stim = str(row.get("stimulus", "")).strip().lower()
                row_stim_m = re.match(r'p\s*(\d+(?:\.\d+)?)\s*u\s*(\d+(?:\.\d+)?)', row_stim)
                if row_stim_m:
                    rp = float(row_stim_m.group(1))
                    ru = float(row_stim_m.group(2))
                    row_ori = str(row.get("orientation", "")).strip().upper()
                    
                    if abs(rp - target_p) < 0.01 and abs(ru - target_u) < 0.01 and row_ori == orientation:
                        # Extract and format calibration fields
                        intensity_raw = row.get("intensity", "unknown")
                        intensity = str(intensity_raw).strip() if pd.notna(intensity_raw) else "unknown"

                        hdr_raw = row.get("hdr", None)
                        if pd.notna(hdr_raw):
                            try:
                                hdr = f"{float(hdr_raw):.4f}"
                            except Exception:
                                hdr = str(hdr_raw)

                        try:
                            xhdr_raw = row.get("xhdr", None)
                            xhdr = f"{float(xhdr_raw):.4f}" if pd.notna(xhdr_raw) else "unknown"
                        except Exception:
                            xhdr = str(row.get("xhdr", "unknown"))

                        try:
                            xdop_raw = row.get("xdop", None)
                            xdop = f"{float(xdop_raw):.4f}" if pd.notna(xdop_raw) else "unknown"
                        except Exception:
                            xdop = str(row.get("xdop", "unknown"))

                        try:
                            x_raw = row.get("x", None)
                            x_val = f"{float(x_raw):.4f}" if pd.notna(x_raw) else "unknown"
                        except Exception:
                            x_val = str(row.get("x", "unknown"))

                        try:
                            edge_raw = row.get("edge intensity (L/R)", None)
                            edge_intensity = f"{float(edge_raw):.4f}" if pd.notna(edge_raw) else str(edge_raw).strip()
                        except Exception:
                            edge_intensity = str(row.get("edge intensity (L/R)", "unknown"))

                        try:
                            sig_raw = row.get("signal rank", None)
                            signal_rank = str(sig_raw).strip() if pd.notna(sig_raw) else "unknown"
                        except Exception:
                            signal_rank = str(row.get("signal rank", "unknown"))

                        try:
                            date_raw = row.get("calibration_date", None)
                            calibration_date = str(date_raw).strip() if pd.notna(date_raw) else "unknown"
                        except Exception:
                            calibration_date = str(row.get("calibration_date", "unknown"))

                        break
        except Exception:
            pass

    # 3. Create heading title exactly matching design format
    heading = f"Cue: {orientation} | p{p_val} u{u_val} | DoP = {xdop} (n = 1 trials)"

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
    
    # Add Bee ID, Trial Outcome and all calibration metadata columns to the CSV
    df["bee_id"] = bee_id
    df["trial_outcome"] = outcome_str
    df["stimulus"] = f"p{p_val} u{u_val}"
    df["orientation"] = orientation
    df["x"] = x_val
    df["xdop"] = xdop
    df["dop"] = xdop
    df["xhdr"] = xhdr
    df["hdr"] = hdr
    df["intensity"] = intensity
    df["edge_intensity"] = edge_intensity
    df["signal_rank"] = signal_rank
    df["calibration_date"] = calibration_date
    
    # Add high-precision timestamp (MM:SS.mmm)
    timestamps = []
    for idx, row_df in df.iterrows():
        sec = row_df["time_sec"]
        timestamps.append(f"{int(sec//60):02d}:{int(sec%60):02d}.{int((sec%1)*1000):03d}")
    df["timestamp"] = timestamps
    
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

    # Convert hive entry point from pixel to mm coordinates for the plot
    hive_entry_mm = None
    if st.session_state.hive_entry_point is not None and st.session_state.circle_center and st.session_state.scale_factor:
        hx_mm, hy_mm, _ = pixel_to_mm(
            st.session_state.hive_entry_point[0],
            st.session_state.hive_entry_point[1],
            st.session_state.circle_center,
            st.session_state.scale_factor,
        )
        hive_entry_mm = (hx_mm, hy_mm)

    fig = plot_trajectory(df, st.session_state.entry_frame, st.session_state.exit_frame, title=heading, bee_id=bee_id, outcome=outcome_str, hive_entry_mm=hive_entry_mm, orientation=orientation, p_val=p_val, u_val=u_val, dop=xdop)
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

    video_export_path = os.path.join(export_dir, f"tracked_preview_{os.path.splitext(video_name)[0]}.mp4")
    if not os.path.exists(video_export_path):
        with st.spinner("Generating tracked preview video (this may take a moment)..."):
            max_f = meta["frames"] - 1
            end_f = get_tracking_end_frame(
                exit_frame=st.session_state.exit_frame,
                max_frame=max_f,
                analysis_end_frame=st.session_state.analysis_end_frame
            )
            generate_tracked_video(
                st.session_state.video_path,
                st.session_state.track_coords,
                st.session_state.entry_frame,
                end_f,
                video_export_path,
                st.session_state.circle_center,
                st.session_state.circle_radius
            )

    csv_export_path = os.path.join(export_dir, f"bee_track_{os.path.splitext(video_name)[0]}.csv")
    png_export_path = os.path.join(export_dir, f"trajectory_{os.path.splitext(video_name)[0]}.png")
    
    df.to_csv(csv_export_path, index=False)
    fig.savefig(png_export_path, dpi=200, bbox_inches="tight")
    complete_png_path = os.path.join(export_dir, f"complete_trajectory_{os.path.splitext(video_name)[0]}.png")
    fig.savefig(complete_png_path, dpi=300, bbox_inches="tight")

    trans = df[df["transition_event"].notna()][["frame", "time_sec", "transition_event"]]
    trans_csv_path = os.path.join(export_dir, f"zone_transitions_{os.path.splitext(video_name)[0]}.csv")
    if len(trans):
        trans.to_csv(trans_csv_path, index=False)

    help_tags = df[df["tag_type"] == "help"][['frame', 'time_sec', 'x_mm', 'y_mm']]
    help_csv_path = os.path.join(export_dir, f"help_points_{os.path.splitext(video_name)[0]}.csv")
    if len(help_tags):
        help_tags.to_csv(help_csv_path, index=False)

    if st.session_state.bee_went_back is not None:
        with open(os.path.join(export_dir, "trial_outcome.txt"), "w") as f:
            f.write(f"Bee ID: {bee_id}\n")
            f.write(f"Outcome: {outcome_str}\n")

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
    st.markdown("</div>", unsafe_allow_html=True)

# Sync flat keys back to active slot at the end of execution
sync_flat_to_active_slot()
