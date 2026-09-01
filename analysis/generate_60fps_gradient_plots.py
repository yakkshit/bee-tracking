"""
Generate 60fps frame-based two-color gradient trajectory plots for each session:

1. Exit Path Only (gradient_exit_only_60fps.png / gradient_exit_60fps_time.png):
   - Entry path removed completely.
   - Exit path from feeder exit to final arena exit.
   - Time calculated strictly from video frames at 60 FPS: t_sec = frame / 60.0.
   - Rendered using LineCollection with a strict 2-color gradient (Yellow -> Blue).
   - Horizontal colorbar showing Time Progress (0.0s -> T_end s at 60 FPS).
   - Legend contains marker labels only (NO exit duration text).

2. Full Trial Entry to Exit (gradient_full_trial_60fps.png / gradient_full_60fps_time.png):
   - Full trial trajectory from outer entry to feeder to final exit.
   - Time calculated strictly from video frames at 60 FPS: t_sec = frame / 60.0.
   - Rendered using LineCollection with a strict 2-color gradient (Yellow -> Blue).
   - Horizontal colorbar showing Time Progress (0.0s -> T_end s at 60 FPS).
   - Legend contains marker labels only (NO exit duration text).

Outputs:
  - results/<session>/plots/gradient_exit_only_60fps.png
  - results/<session>/plots/gradient_exit_60fps_time.png
  - results/<session>/plots/gradient_full_trial_60fps.png
  - results/<session>/plots/gradient_full_60fps_time.png
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except ImportError:
    HAS_SAVGOL = False

OUTER_R = 420.0
INNER_R = 210.0
FEEDER_R = 40.0
HIVE_DIST = OUTER_R + 30.0
FPS = 60.0  # 60 frames per second

C_ENTRY    = "#2E7D32"   # green triangle
C_EXIT     = "#0D47A1"   # dark blue circle
C_FEEDER   = "#F57C00"   # orange dot

# Strict 2-Color Colormap: Bright Yellow -> Dark Royal Blue
TWO_COLOR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "two_color_60fps", ["#FFEE58", "#0D47A1"]
)


def smooth(x, y, strength=15):
    n = len(x)
    if n < 7:
        return x, y
    if HAS_SAVGOL:
        w = min(strength, n if n % 2 else n - 1)
        if w < 5: w = 5
        if w % 2 == 0: w -= 1
        if 5 <= w <= n:
            try:
                return savgol_filter(x, w, 2), savgol_filter(y, w, 2)
            except Exception:
                pass
    k = min(5, n)
    kern = np.ones(k) / k
    xs = np.convolve(x, kern, mode="same")
    ys = np.convolve(y, kern, mode="same")
    xs[0], ys[0], xs[-1], ys[-1] = x[0], y[0], x[-1], y[-1]
    return xs, ys


def subsample_and_interpolate_frames(x, y, frames, target_pts=600):
    n = len(x)
    if n < 2:
        return x, y, frames / FPS
    n_target = max(n * 3, target_pts)
    f_interp = np.linspace(frames.min(), frames.max(), n_target)
    x_interp = np.interp(f_interp, frames, x)
    y_interp = np.interp(f_interp, frames, y)
    t_sec_interp = (f_interp - frames.min()) / FPS
    return x_interp, y_interp, t_sec_interp


def find_crossings(dists, boundary):
    entries, exits = [], []
    for i in range(1, len(dists)):
        if dists[i - 1] > boundary and dists[i] <= boundary:
            entries.append(i)
        if dists[i - 1] <= boundary and dists[i] > boundary:
            exits.append(i)
    return entries, exits


def project_to_circle(x, y, radius):
    d = np.hypot(x, y)
    if d < 1:
        return x, y
    return x * radius / d, y * radius / d


def get_hive_position(df, outer_r=OUTER_R, hive_dist=HIVE_DIST):
    """
    Compute Hive position.
    For TRex auto sessions (tag_type == 'auto'), Hive is at bottom center (0, -450).
    For manual sessions, projects first entry point or falls back to bottom center.
    """
    if df is not None and len(df) > 0:
        tag = str(df.iloc[0].get("tag_type", "")).strip().lower()
        if tag == "auto":
            return 0.0, -hive_dist

    if df is None or len(df) == 0:
        return 0.0, -hive_dist

    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    dists = df["distance_from_center_mm"].values if "distance_from_center_mm" in df.columns else np.hypot(x_all, y_all)

    outer_entries = [i for i in range(1, len(dists)) if dists[i-1] > outer_r and dists[i] <= outer_r]
    if outer_entries:
        idx = outer_entries[0]
        xe, ye = x_all[idx], y_all[idx]
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
    m = re.search(r'\b(\d+[wWgG])\b', sess_name)
    if m:
        return m.group(1).upper()
    m2 = re.search(r'(\d+[wWgG])', sess_name)
    if m2:
        return m2.group(1).upper()

    if "unmarked" in sess_name.lower():
        return "unmarked"

    m3 = re.search(r'(?:^|[._])([RGWBYOP]_\d+|[RGWBYOP]\d+)(?:[._]|$)', sess_name, re.IGNORECASE)
    if m3:
        return m3.group(1).upper()

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
        except Exception: pass

    return title_str, bee_id, outcome


# ── Plot 1: Exit Path Only (60 FPS Frame-Based 2-Color Gradient) ─────────────
def generate_exit_only_60fps(df, session_dir):
    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if len(df) < 10:
        return

    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    frames_all = df["frame"].values if "frame" in df.columns else np.arange(len(df))
    dists = df["distance_from_center_mm"].values

    outer_entries, outer_exits = find_crossings(dists, OUTER_R)
    inner_entries, inner_exits = find_crossings(dists, INNER_R)

    first_outer_entry = outer_entries[0] if outer_entries else 0
    feeder_idx = np.argmin(dists[first_outer_entry:]) + first_outer_entry

    first_inner_exit = None
    for ix in inner_exits:
        if ix >= feeder_idx:
            first_inner_exit = ix
            break

    first_outer_exit = None
    for ox in outer_exits:
        if ox >= feeder_idx:
            first_outer_exit = ox
            break

    outbound_end = first_outer_exit if first_outer_exit else len(df) - 1
    if outbound_end <= feeder_idx:
        outbound_end = len(df) - 1

    # Extract exit path strictly (No entry path)
    out_x = x_all[feeder_idx : outbound_end + 1]
    out_y = y_all[feeder_idx : outbound_end + 1]
    out_frames = frames_all[feeder_idx : outbound_end + 1]

    if len(out_x) < 2:
        return

    # 60 FPS frame-based interpolation
    out_x, out_y, t_sec = subsample_and_interpolate_frames(out_x, out_y, out_frames, target_pts=600)
    out_x, out_y = smooth(out_x, out_y, strength=15)

    max_t = t_sec.max()

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    # Arena outlines
    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#333333", lw=2.2, zorder=1))
    ax.add_patch(Circle((0, 0), INNER_R, fill=False, ec="#aaaaaa", lw=1.3, ls="--", zorder=2))

    # Feeder center circle
    ax.plot(0, 0, "o", color=C_FEEDER, markersize=12, zorder=10)
    ax.text(0, -28, "Feeder", color=C_FEEDER, fontsize=11, fontweight="bold", ha="center", va="top", zorder=10)

    # ── Exit Path: 2-Color LineCollection mapped to 60fps Time ─────────────
    points = np.column_stack([out_x, out_y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    t_segments = (t_sec[:-1] + t_sec[1:]) / 2.0
    norm = plt.Normalize(0.0, max_t if max_t > 0 else 1.0)

    lc = LineCollection(
        segments,
        cmap=TWO_COLOR_CMAP,
        norm=norm,
        linewidth=2.4,
        capstyle="round",
        joinstyle="round",
        zorder=6
    )
    lc.set_array(t_segments)
    ax.add_collection(lc)

    # ── Static Markers ───────────────────────────────────────────────────
    if first_inner_exit is not None:
        ioex, ioey = project_to_circle(x_all[first_inner_exit], y_all[first_inner_exit], INNER_R)
        ax.plot(ioex, ioey, "D", color="#7B1FA2", markersize=8, zorder=11)

    if first_outer_exit is not None:
        fx, fy = project_to_circle(x_all[first_outer_exit], y_all[first_outer_exit], OUTER_R)
        ax.plot(fx, fy, "o", color=C_EXIT, markersize=9, zorder=11)
        ax.annotate("①", (fx, fy), textcoords="offset points", xytext=(12, -12),
                    fontsize=9, color=C_EXIT, fontweight="bold",
                    ha="center", va="center", zorder=12,
                    bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec=C_EXIT, lw=1.0))

    # Hive marker at original exit position (Red X)
    hive_x, hive_y = get_hive_position(df)
    ax.plot(hive_x, hive_y, "o", color="#333333", markersize=10, markeredgecolor="#111111", markeredgewidth=1.5, zorder=12)
    ax.annotate("Hive", (hive_x, hive_y), textcoords="offset points", xytext=(12, 0), fontsize=9, fontweight="bold", color="#333333", va="center", zorder=13)

    # Title & Metadata box
    title_str, bee_id, outcome = get_metadata(df, session_dir)
    ax.set_title(f"{title_str} (Exit Path - 60 FPS Gradient)", fontsize=11, fontweight="bold", pad=20)

    lim = OUTER_R + 80
    textstr = f"Bee ID: {bee_id}\nOutcome: {outcome}"
    ax.text(-lim * 0.9, -lim * 0.9, textstr, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left", zorder=15,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc", alpha=0.9))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend for elements (NO exit duration text in legend)
    legend_els = [
        Line2D([0], [0], color="#FFEE58", lw=3, label="Time Start (Feeder Exit)"),
        Line2D([0], [0], color="#0D47A1", lw=3, label="Time End (Arena Exit)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#7B1FA2", markersize=8, label="1st Inner Exit (Out)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EXIT, markersize=9, label="Exit (Outer)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FEEDER, markersize=9, label="Feeder (0,0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333", markeredgecolor="#111111", markersize=9, label="Hive"),
    ]
    ax.legend(handles=legend_els, bbox_to_anchor=(1.04, 1.0), loc="upper left", frameon=True, facecolor="white", edgecolor="#d0d0d0", fontsize=8.5, borderpad=0.6, labelspacing=0.4)

    # Horizontal 2-Color Colorbar showing 60 FPS time progress
    sm = plt.cm.ScalarMappable(cmap=TWO_COLOR_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.25, 0.05, 0.50, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(f"Time Progress (60 FPS): 0.0s (Yellow) → {max_t:.2f}s (Dark Blue)", fontsize=9.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(os.path.join(plots_dir, "gradient_exit_only_60fps.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, "gradient_exit_60fps_time.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Plot 2: Full Trial (60 FPS Frame-Based 2-Color Gradient) ───────────────
def generate_full_trial_60fps(df, session_dir):
    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if len(df) < 10:
        return

    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    frames_all = df["frame"].values if "frame" in df.columns else np.arange(len(df))
    dists = df["distance_from_center_mm"].values

    outer_entries, outer_exits = find_crossings(dists, OUTER_R)
    inner_entries, inner_exits = find_crossings(dists, INNER_R)

    first_outer_entry = outer_entries[0] if outer_entries else 0
    feeder_idx = np.argmin(dists[first_outer_entry:]) + first_outer_entry

    first_inner_entry = None
    for ie in inner_entries:
        if first_outer_entry <= ie <= feeder_idx:
            first_inner_entry = ie
            break
    if first_inner_entry is None:
        first_inner_entry = first_outer_entry

    first_inner_exit = None
    for ix in inner_exits:
        if ix >= feeder_idx:
            first_inner_exit = ix
            break

    first_outer_exit = None
    for ox in outer_exits:
        if ox >= feeder_idx:
            first_outer_exit = ox
            break

    final_outer_exit = first_outer_exit if first_outer_exit else len(df) - 1
    if final_outer_exit <= first_outer_entry:
        final_outer_exit = len(df) - 1

    # Extract full trial path
    full_x = x_all[first_outer_entry : final_outer_exit + 1]
    full_y = y_all[first_outer_entry : final_outer_exit + 1]
    full_frames = frames_all[first_outer_entry : final_outer_exit + 1]

    if len(full_x) < 2:
        return

    # 60 FPS frame-based interpolation
    full_x, full_y, t_sec = subsample_and_interpolate_frames(full_x, full_y, full_frames, target_pts=700)
    full_x, full_y = smooth(full_x, full_y, strength=17)

    max_t = t_sec.max()

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    # Arena outlines
    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#333333", lw=2.2, zorder=1))
    ax.add_patch(Circle((0, 0), INNER_R, fill=False, ec="#aaaaaa", lw=1.3, ls="--", zorder=2))

    # Feeder center circle
    ax.plot(0, 0, "o", color=C_FEEDER, markersize=12, zorder=10)
    ax.text(0, -28, "Feeder", color=C_FEEDER, fontsize=11, fontweight="bold", ha="center", va="top", zorder=10)

    # ── Full Path: 2-Color LineCollection mapped to 60fps Time ─────────────
    points = np.column_stack([full_x, full_y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    t_segments = (t_sec[:-1] + t_sec[1:]) / 2.0
    norm = plt.Normalize(0.0, max_t if max_t > 0 else 1.0)

    lc = LineCollection(
        segments,
        cmap=TWO_COLOR_CMAP,
        norm=norm,
        linewidth=2.4,
        capstyle="round",
        joinstyle="round",
        zorder=6
    )
    lc.set_array(t_segments)
    ax.add_collection(lc)

    # ── Static Markers ───────────────────────────────────────────────────
    ex, ey = project_to_circle(x_all[first_outer_entry], y_all[first_outer_entry], OUTER_R)
    ax.plot(ex, ey, "^", color=C_ENTRY, markersize=11, zorder=11)
    ax.annotate("①", (ex, ey), textcoords="offset points", xytext=(-14, 0), fontsize=9, color=C_ENTRY, fontweight="bold", ha="center", va="center", zorder=12)

    if first_inner_entry is not None:
        iex, iey = project_to_circle(x_all[first_inner_entry], y_all[first_inner_entry], INNER_R)
        ax.plot(iex, iey, "D", color="#D32F2F", markersize=8, zorder=11)

    if first_inner_exit is not None:
        ioex, ioey = project_to_circle(x_all[first_inner_exit], y_all[first_inner_exit], INNER_R)
        ax.plot(ioex, ioey, "D", color="#7B1FA2", markersize=8, zorder=11)

    if first_outer_exit is not None:
        fx, fy = project_to_circle(x_all[first_outer_exit], y_all[first_outer_exit], OUTER_R)
        ax.plot(fx, fy, "o", color=C_EXIT, markersize=9, zorder=11)
        ax.annotate("①", (fx, fy), textcoords="offset points", xytext=(12, -12), fontsize=9, color=C_EXIT, fontweight="bold", ha="center", va="center", zorder=12, bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec=C_EXIT, lw=1.0))

    # Hive marker at original exit position (Red X)
    hive_x, hive_y = get_hive_position(df)
    ax.plot(hive_x, hive_y, "o", color="#333333", markersize=10, markeredgecolor="#111111", markeredgewidth=1.5, zorder=12)
    ax.annotate("Hive", (hive_x, hive_y), textcoords="offset points", xytext=(12, 0), fontsize=9, fontweight="bold", color="#333333", va="center", zorder=13)

    # Title & Metadata box
    title_str, bee_id, outcome = get_metadata(df, session_dir)
    ax.set_title(f"{title_str} (Full Trial - 60 FPS Gradient)", fontsize=11, fontweight="bold", pad=20)

    lim = OUTER_R + 80
    textstr = f"Bee ID: {bee_id}\nOutcome: {outcome}"
    ax.text(-lim * 0.9, -lim * 0.9, textstr, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left", zorder=15,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc", alpha=0.9))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend for elements (NO exit duration text in legend)
    legend_els = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_ENTRY, markersize=10, label="Entry (Outer)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#D32F2F", markersize=8, label="1st Inner Contact (In)"),
        Line2D([0], [0], color="#FFEE58", lw=3, label="Trial Start (0.0s - Yellow)"),
        Line2D([0], [0], color="#0D47A1", lw=3, label="Trial End (Dark Blue)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#7B1FA2", markersize=8, label="1st Inner Exit (Out)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EXIT, markersize=9, label="Exit (Outer)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FEEDER, markersize=9, label="Feeder (0,0)"),
    ]
    ax.legend(handles=legend_els, bbox_to_anchor=(1.04, 1.0), loc="upper left", frameon=True, facecolor="white", edgecolor="#d0d0d0", fontsize=8.5, borderpad=0.6, labelspacing=0.4)

    # Horizontal 2-Color Colorbar showing 60 FPS time progress
    sm = plt.cm.ScalarMappable(cmap=TWO_COLOR_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.25, 0.05, 0.50, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(f"Time Progress (60 FPS): 0.0s (Yellow) → {max_t:.2f}s (Dark Blue)", fontsize=9.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(os.path.join(plots_dir, "gradient_full_trial_60fps.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, "gradient_full_60fps_time.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_60fps_gradient_plots(df, session_dir):
    generate_exit_only_60fps(df, session_dir)
    generate_full_trial_60fps(df, session_dir)


def main():
    print("Generating 60fps 2-color gradient plots...")
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("Error: results folder does not exist.")
        return

    for session_name in sorted(os.listdir(results_dir)):
        session_path = os.path.join(results_dir, session_name)
        if not os.path.isdir(session_path) or session_name == "paper_plots":
            continue

        track_csv = None
        for f in os.listdir(session_path):
            if f.startswith("bee_track_") and f.endswith(".csv"):
                track_csv = os.path.join(session_path, f)
                break
        if not track_csv:
            continue

        print(f"  → {session_name}")
        try:
            df = pd.read_csv(track_csv)
            generate_60fps_gradient_plots(df, session_path)
        except Exception as e:
            print(f"    ERROR: {e}")


if __name__ == "__main__":
    main()
