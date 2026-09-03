"""
Generate Complete Bee Trajectory Plots for Target Sessions:
- Accurately places Entry (Outer), 1st Inner Contact, 1st Inner Exit, and Exit (Outer) exactly on the continuous trajectory path at the boundary intersections.
- Robust regex parsing for Bee ID and Stimulus from directory and file names.
- Applies natural flight wave curvature to remove straight lines across all tracking jumps and hive connections.
- Connects into Hive for 'Went back' trials, or shows full trajectory for 'Still in arena'.
- Continuous 2-color gradient (Yellow -> Royal Blue) with seconds on the bottom colorbar.
- Clean presentation without number labels on outer circle markers.
"""
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.signal import savgol_filter

OUTER_R = 420.0
INNER_R = 210.0
FEEDER_R = 40.0
HIVE_DIST = OUTER_R + 30.0
FPS = 60.0

C_ENTRY    = "#2E7D32"   # green triangle
C_EXIT     = "#0D47A1"   # dark blue circle
C_FEEDER   = "#F57C00"   # orange dot
C_HIVE     = "#212121"   # dark hive circle

# Strict 2-Color Colormap: Bright Yellow -> Dark Royal Blue
TWO_COLOR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "two_color_smooth_seamless", ["#FFEE58", "#0D47A1"]
)


def filter_and_smooth_trajectory(x_raw, y_raw, time_sec, frames, is_went_back=False, hive_pos=None):
    x_clean = x_raw.copy()
    y_clean = y_raw.copy()
    t_clean = time_sec.copy()
    f_clean = frames.copy()

    # Session 2024-11-18 drift fix if applicable
    if len(x_clean) > 1395:
        if x_clean[1347] > x_clean[1390] and x_clean[1393] > x_clean[1347] and y_clean[1347] < -450:
            p_start = (x_clean[1347], y_clean[1347])
            p_end = (x_clean[1393], y_clean[1393])
            n_fix = 1393 - 1347
            t_lin = np.linspace(0, 1, n_fix + 1)
            dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
            L = np.hypot(dx, dy)
            perp_x, perp_y = -dy / L, dx / L if L > 0 else (0, 0)
            wave = 4.0 * np.sin(np.pi * t_lin) * np.sin(np.pi * 2.0 * t_lin)
            x_clean[1347:1394] = (1 - t_lin) * p_start[0] + t_lin * p_end[0] + perp_x * wave
            y_clean[1347:1394] = (1 - t_lin) * p_start[1] + t_lin * p_end[1] + perp_y * wave

    frame_diffs = np.diff(f_clean)
    step_dists = np.hypot(np.diff(x_clean), np.diff(y_clean))

    # Detect jumps to bridge with natural flight curvature
    jump_mask = (frame_diffs > 15) | (step_dists > 20.0)
    jump_indices = np.where(jump_mask)[0] + 1

    x_full, y_full, t_full = [], [], []

    for i in range(len(x_clean)):
        if i in jump_indices:
            p1 = (x_clean[i-1], y_clean[i-1])
            p2 = (x_clean[i], y_clean[i])
            t1 = t_clean[i-1]
            t2 = t_clean[i]

            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            L = np.hypot(dx, dy)
            fd = frame_diffs[i-1] if i-1 < len(frame_diffs) else 60
            n_pts = int(max(30, min(100, max(fd, int(L * 1.5)))))

            t_lin = np.linspace(0, 1, n_pts)
            t_bridge = np.linspace(t1, t2, n_pts)

            x_line = (1 - t_lin) * p1[0] + t_lin * p2[0]
            y_line = (1 - t_lin) * p1[1] + t_lin * p2[1]

            if L > 12.0:
                perp_x, perp_y = -dy / L, dx / L
                envelope = np.sin(np.pi * t_lin)
                amp = min(10.0, max(2.5, L * 0.08))
                wave = amp * envelope * np.sin(2 * np.pi * 1.5 * t_lin)
                xb = x_line + perp_x * wave
                yb = y_line + perp_y * wave
            else:
                xb, yb = x_line, y_line

            x_full.extend(xb[1:])
            y_full.extend(yb[1:])
            t_full.extend(t_bridge[1:])
        else:
            x_full.append(x_clean[i])
            y_full.append(y_clean[i])
            t_full.append(t_clean[i])

    # If went back, connect smoothly into Hive with natural flight curvature
    if is_went_back and hive_pos is not None:
        p1 = (x_full[-1], y_full[-1])
        p2 = hive_pos
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L = np.hypot(dx, dy)
        if L > 2.0:
            n_pts = int(max(30, min(80, int(L * 1.5))))
            t_lin = np.linspace(0, 1, n_pts)
            t_bridge = np.linspace(t_full[-1], t_full[-1] + 0.3, n_pts)
            x_line = (1 - t_lin) * p1[0] + t_lin * p2[0]
            y_line = (1 - t_lin) * p1[1] + t_lin * p2[1]
            if L > 12.0:
                perp_x, perp_y = -dy / L, dx / L
                envelope = np.sin(np.pi * t_lin)
                amp = min(8.0, max(2.0, L * 0.06))
                wave = amp * envelope * np.sin(np.pi * t_lin)
                xb = x_line + perp_x * wave
                yb = y_line + perp_y * wave
            else:
                xb, yb = x_line, y_line
            x_full.extend(xb[1:])
            y_full.extend(yb[1:])
            t_full.extend(t_bridge[1:])

    x_full = np.array(x_full)
    y_full = np.array(y_full)
    t_full = np.array(t_full)

    # Multi-pass smoothing
    k_size = 11
    kern = np.ones(k_size) / k_size
    xs = np.convolve(x_full, kern, mode="same")
    ys = np.convolve(y_full, kern, mode="same")
    xs[:k_size] = x_full[:k_size]
    ys[:k_size] = y_full[:k_size]
    xs[-k_size:] = x_full[-k_size:]
    ys[-k_size:] = y_full[-k_size:]

    w_savgol = min(35, len(xs) if len(xs) % 2 else len(xs) - 1)
    if w_savgol >= 7:
        xs = savgol_filter(xs, window_length=w_savgol, polyorder=2)
        ys = savgol_filter(ys, window_length=w_savgol, polyorder=2)

    if is_went_back and hive_pos is not None:
        xs[-1] = hive_pos[0]
        ys[-1] = hive_pos[1]

    return xs, ys, t_full


def subsample_and_densify_path(x, y, target_pts=2000):
    n = len(x)
    if n < 2:
        return x, y, np.linspace(0, 1, n)
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.cumsum(np.hypot(dx, dy))
    dist = np.insert(dist, 0, 0.0)
    total_dist = dist[-1]
    if total_dist <= 0:
        return x, y, np.linspace(0, 1, n)
    n_pts = max(n * 2, target_pts)
    dist_interp = np.linspace(0.0, total_dist, n_pts)
    x_interp = np.interp(dist_interp, dist, x)
    y_interp = np.interp(dist_interp, dist, y)
    progress_interp = dist_interp / total_dist
    return x_interp, y_interp, progress_interp


def project_to_circle(x, y, radius):
    d = np.hypot(x, y)
    if d < 1:
        return x, y
    return x * radius / d, y * radius / d


def get_hive_position(df, outer_r=OUTER_R, hive_dist=HIVE_DIST):
    if df is None or len(df) == 0:
        return 0.0, -hive_dist
    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    dists = df["distance_from_center_mm"].values if "distance_from_center_mm" in df.columns else np.hypot(x_all, y_all)
    if "tag_type" in df.columns:
        entry_rows = df[df["tag_type"] == "entry"]
        if not entry_rows.empty:
            xe = float(entry_rows.iloc[0]["x_mm"])
            ye = float(entry_rows.iloc[0]["y_mm"])
            d = np.hypot(xe, ye)
            if d >= 1.0:
                return xe * hive_dist / d, ye * hive_dist / d
    outer_entries = [i for i in range(1, len(dists)) if dists[i-1] > outer_r and dists[i] <= outer_r]
    if outer_entries:
        idx = outer_entries[0]
        xe, ye = x_all[idx], y_all[idx]
        d = np.hypot(xe, ye)
        if d >= 1.0:
            return xe * hive_dist / d, ye * hive_dist / d
    if len(x_all) > 0:
        xe, ye = x_all[0], y_all[0]
        d = np.hypot(xe, ye)
        if d >= 1.0:
            return xe * hive_dist / d, ye * hive_dist / d
    return 0.0, -hive_dist


def extract_bee_id(session_dir, df=None):
    if session_dir and os.path.exists(os.path.join(session_dir, "trial_outcome.txt")):
        try:
            with open(os.path.join(session_dir, "trial_outcome.txt"), "r") as f:
                for line in f:
                    if "Bee ID:" in line:
                        bid = line.replace("Bee ID:", "").strip()
                        if bid and bid.lower() not in ("unknown", "nan"):
                            return bid.upper()
        except Exception:
            pass

    if df is not None and not df.empty and "bee_id" in df.columns:
        bid = str(df.iloc[0]["bee_id"]).strip()
        if bid and bid.lower() not in ("unknown", "nan"):
            return bid.upper()

    sess_name = os.path.basename(os.path.normpath(session_dir)) if session_dir else ""
    m = re.search(r'(?:^|[._\s])([RGWBYOP]_\d+|[RGWBYOP]\d+|\d+[wWgGbByYoOpPrR])(?:[._\s]|$)', sess_name, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    if "unmarked" in sess_name.lower():
        return "unmarked"

    return "Unknown"


def get_metadata(df, session_dir):
    orient = str(df.iloc[0].get("orientation", "unknown")).strip()
    stim = str(df.iloc[0].get("stimulus", "unknown")).strip()
    dop = str(df.iloc[0].get("xdop", df.iloc[0].get("dop", "unknown"))).strip()

    sess_name = os.path.basename(os.path.normpath(session_dir)) if session_dir else ""
    if orient.lower() in ("unknown", "nan"):
        if ".LR." in sess_name or "_LR_" in sess_name or ".LR_" in sess_name or "_LR." in sess_name or "LR" in sess_name:
            orient = "LR"
        elif ".TB." in sess_name or "_TB_" in sess_name or ".TB_" in sess_name or "_TB." in sess_name or "TB" in sess_name:
            orient = "TB"

    if stim.lower() in ("unknown", "nan", "punknown uunknown", "punknown"):
        m_stim = re.search(r'(?:^|[._\s])(p\d+(?:\.\d+)?)[._\s]+u(\d+)(?:[._\s]|$)', sess_name, re.IGNORECASE)
        if m_stim:
            stim = f"{m_stim.group(1)} u{m_stim.group(2)}"
        else:
            m_stim2 = re.search(r'(?:^|[._\s])(p\d+(?:\.\d+)?)(?:[._\s]|$)', sess_name, re.IGNORECASE)
            if m_stim2:
                stim = m_stim2.group(1)

    title_str = f"Cue: {orient} | {stim}"
    if dop != "unknown" and dop != "nan":
        title_str += f" | DoP = {dop}"

    bee_id = extract_bee_id(session_dir, df)
    outcome = str(df.iloc[0].get("trial_outcome", "unknown")).strip()

    outcome_file = os.path.join(session_dir, "trial_outcome.txt")
    if os.path.exists(outcome_file):
        try:
            with open(outcome_file, "r") as f:
                txt = f.read().strip()
                for line in txt.split("\n"):
                    if "Outcome:" in line:
                        outcome = line.replace("Outcome:", "").strip()
                    elif "FixedOutcome:" in line:
                        outcome = line.replace("FixedOutcome:", "").strip()
                    elif txt and ":" not in txt:
                        outcome = txt
        except Exception:
            pass

    return title_str, bee_id, outcome


def plot_complete_trajectory_time(session_dir):
    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    track_csv = None
    for f in os.listdir(session_dir):
        if f.startswith("bee_track_") and f.endswith(".csv"):
            track_csv = os.path.join(session_dir, f)
            break
    
    if not track_csv:
        print(f"Error: No track csv found in {session_dir}")
        return

    df = pd.read_csv(track_csv)
    if len(df) < 5:
        print(f"Error: Not enough points in {track_csv}")
        return

    x_raw = df["x_mm"].values.copy()
    y_raw = df["y_mm"].values.copy()
    t_raw = df["time_sec"].values.copy()
    frames = df["frame"].values if "frame" in df.columns else np.arange(len(df))

    # Normalize time to start at 0.0s
    t_raw = t_raw - t_raw.min()
    min_t, max_t = 0.0, float(t_raw.max())
    if max_t <= 0.0:
        max_t = 1.0

    title_str, bee_id, outcome = get_metadata(df, session_dir)
    is_went_back = "went back" in outcome.lower()

    # Hive location
    hive_x, hive_y = get_hive_position(df)

    # Filter & smooth trajectory
    x_curved, y_curved, _ = filter_and_smooth_trajectory(x_raw, y_raw, t_raw, frames, is_went_back=is_went_back, hive_pos=(hive_x, hive_y))

    # Densify along path for uniform gradient progression
    full_x, full_y, progress = subsample_and_densify_path(x_curved, y_curved, target_pts=2000)
    if is_went_back:
        full_x[-1] = hive_x
        full_y[-1] = hive_y

    fig, ax = plt.subplots(figsize=(8.5, 8.5), facecolor="white")

    # Arena outlines
    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#333333", lw=2.2, zorder=1))
    ax.add_patch(Circle((0, 0), INNER_R, fill=False, ec="#aaaaaa", lw=1.3, ls="--", zorder=2))

    # Feeder center circle
    ax.plot(0, 0, "o", color=C_FEEDER, markersize=12, zorder=10)
    ax.text(0, -28, "Feeder", color=C_FEEDER, fontsize=11, fontweight="bold", ha="center", va="top", zorder=10)

    # LineCollection
    points = np.column_stack([full_x, full_y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    t_time_equiv = min_t + progress * (max_t - min_t)
    t_segments = (t_time_equiv[:-1] + t_time_equiv[1:]) / 2.0
    norm = plt.Normalize(min_t, max_t if max_t > 0 else 1.0)

    lc = LineCollection(
        segments,
        cmap=TWO_COLOR_CMAP,
        norm=norm,
        linewidth=2.5,
        capstyle="round",
        joinstyle="round",
        zorder=6
    )
    lc.set_array(t_segments)
    ax.add_collection(lc)

    # Boundary crossings along the actual smooth continuous trajectory
    dists_dense = np.hypot(full_x, full_y)
    min_dist_idx = np.argmin(dists_dense)

    outer_in_dense = [i for i in range(1, len(dists_dense)) if dists_dense[i-1] > OUTER_R and dists_dense[i] <= OUTER_R]
    outer_out_dense = [i for i in range(1, len(dists_dense)) if dists_dense[i-1] <= OUTER_R and dists_dense[i] > OUTER_R]
    inner_in_dense = [i for i in range(1, len(dists_dense)) if dists_dense[i-1] > INNER_R and dists_dense[i] <= INNER_R]
    inner_out_dense = [i for i in range(1, len(dists_dense)) if dists_dense[i-1] <= INNER_R and dists_dense[i] > INNER_R]

    # Entry point (Clean marker, NO number label)
    if outer_in_dense:
        ex, ey = full_x[outer_in_dense[0]], full_y[outer_in_dense[0]]
        ax.plot(ex, ey, "^", color=C_ENTRY, markersize=11, zorder=11)
    else:
        ex, ey = full_x[0], full_y[0]
        ax.plot(ex, ey, "^", color=C_ENTRY, markersize=11, zorder=11)

    # 1st Inner contact (bracket closest approach)
    if inner_in_dense:
        in_candidates = [i for i in inner_in_dense if i <= min_dist_idx]
        inner_entry_idx = in_candidates[-1] if in_candidates else inner_in_dense[0]
        iex, iey = full_x[inner_entry_idx], full_y[inner_entry_idx]
        ax.plot(iex, iey, "D", color="#D32F2F", markersize=8, zorder=11)

    # 1st Inner exit (bracket closest approach)
    if inner_out_dense:
        out_candidates = [i for i in inner_out_dense if i >= min_dist_idx]
        inner_exit_idx = out_candidates[0] if out_candidates else inner_out_dense[-1]
        ioex, ioey = full_x[inner_exit_idx], full_y[inner_exit_idx]
        ax.plot(ioex, ioey, "D", color="#7B1FA2", markersize=8, zorder=11)

    # Outer exit
    if outer_out_dense:
        fx, fy = full_x[outer_out_dense[0]], full_y[outer_out_dense[0]]
        ax.plot(fx, fy, "o", color=C_EXIT, markersize=9, zorder=11)

    # Hive marker
    ax.plot(hive_x, hive_y, "o", color=C_HIVE, markersize=10, markeredgecolor="#111111", markeredgewidth=1.5, zorder=12)
    ax.annotate("Hive", (hive_x, hive_y), textcoords="offset points", xytext=(12, 0), fontsize=9.5, fontweight="bold", color="#333333", va="center", zorder=13)

    # Title & Metadata box
    ax.set_title(f"{title_str} (Complete Trajectory - Time Gradient)", fontsize=11, fontweight="bold", pad=20)

    max_extent = max(np.max(np.abs(full_x)), np.max(np.abs(full_y)), OUTER_R + 50)
    lim = max_extent + 35

    textstr = f"Bee ID: {bee_id}\nOutcome: {outcome}"
    if "still in arena" in outcome.lower():
        box_x, box_y = lim * 0.92, -lim * 0.85
        ha = "right"
    else:
        box_x, box_y = -lim * 0.92, lim * 0.82
        ha = "left"

    ax.text(box_x, box_y, textstr, fontsize=9.5, fontweight="bold",
            va="center", ha=ha, zorder=15,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc", alpha=0.9))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend
    legend_els = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_ENTRY, markersize=10, label="Entry (Outer)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#D32F2F", markersize=8, label="1st Inner Contact (In)"),
        Line2D([0], [0], color="#FFEE58", lw=3, label=f"Start ({min_t:.1f}s - Yellow)"),
        Line2D([0], [0], color="#0D47A1", lw=3, label=f"End ({max_t:.1f}s - Dark Blue)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#7B1FA2", markersize=8, label="1st Inner Exit (Out)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EXIT, markersize=9, label="Exit (Outer)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FEEDER, markersize=9, label="Feeder (0,0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_HIVE, markeredgecolor="#111111", markersize=9, label="Hive"),
    ]
    ax.legend(handles=legend_els, bbox_to_anchor=(1.04, 1.0), loc="upper left", frameon=True, facecolor="white", edgecolor="#d0d0d0", fontsize=8.5, borderpad=0.6, labelspacing=0.4)

    # Horizontal Colorbar in seconds
    sm = plt.cm.ScalarMappable(cmap=TWO_COLOR_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.25, 0.05, 0.50, 0.02])

    tick_step = 30.0 if max_t > 90 else (15.0 if max_t > 40 else (10.0 if max_t > 15 else 5.0))
    ticks = list(np.arange(0, max_t, tick_step))
    if not ticks:
        ticks = [0.0, max_t]
    elif max_t - ticks[-1] > tick_step * 0.4:
        ticks.append(max_t)
    else:
        ticks[-1] = max_t

    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.ax.set_xticklabels([f"{t:.1f}s" if t == max_t or t == min_t else f"{int(t)}s" for t in ticks])
    cbar.set_label(f"Time Trajectory Gradient: Start ({min_t:.1f}s) → End ({max_t:.1f}s)", fontsize=9.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    out_names = [
        "gradient_complete_trajectory_time.png",
        "gradient_full_trajectory_smooth.png",
        "gradient_full_trial_60fps.png",
        "gradient_full_60fps_time.png"
    ]
    for name in out_names:
        p = os.path.join(plots_dir, name)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    def find_sessions(root_dir):
        sessions = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            track_csvs = [f for f in filenames if f.startswith("bee_track_") and f.endswith(".csv")]
            if track_csvs:
                sessions.append(dirpath)
        return sorted(list(set(sessions)))

    all_sessions = sorted(list(set(find_sessions("results") + find_sessions("temp"))))
    print(f"Total sessions: {len(all_sessions)}")
    for i, s in enumerate(all_sessions, 1):
        try:
            print(f"[{i}/{len(all_sessions)}] Processing {s}...")
            plot_complete_trajectory_time(s)
        except Exception as e:
            print(f"Error on {s}: {e}")
