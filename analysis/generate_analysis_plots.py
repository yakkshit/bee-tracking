"""
Prompt 3: Full Trial Trajectory (Both Entry and Exit Phase)

Generates publication-quality full trial bee trajectory plot covering both the Entry phase
(outer boundary to feeder) and Exit phase (feeder to outer boundary).

Requirements fulfilled:
1. Data Handling: Full time-series trajectory (x, y, t) covering inbound and outbound paths.
2. Continuous Gradient Trajectory: Render entire continuous trajectory using LineCollection.
   Smoothly blending Yellow (t=0) -> Green (midway/feeder) -> Blue (t=t_final).
3. Eliminate Color Blocking: Single point-to-point segment mapping with capstyle='round' and joinstyle='round'.
4. Layout & Styling: Circular arena outline with grey shading for inner/outer zones,
   horizontal colorbar at bottom for Time (->), clean minimalist publication aesthetic.

Outputs:
  - results/<session>/plots/complete_trajectory.png
  - results/<session>/plots/approach_and_home_search.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except ImportError:
    HAS_SAVGOL = False

OUTER_R = 420.0   # mm
INNER_R = 210.0   # mm
FEEDER_R = 40.0   # mm
HIVE_DIST = OUTER_R + 30.0  # mm

# Continuous colormap transitioning Yellow -> Green -> Blue
YGB_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "yellow_green_blue", ["#FFEE58", "#66BB6A", "#26C6DA", "#1E88E5", "#0D47A1"]
)

import re

def extract_bee_id(session_dir, df=None):
    sess_name = os.path.basename(os.path.normpath(session_dir)) if session_dir else ""
    m = re.search(r'(\d+[wWgGbByYoOrRpP])', sess_name)
    if m:
        return m.group(1).lower()

    if session_dir and os.path.exists(os.path.join(session_dir, "trial_outcome.txt")):
        try:
            with open(os.path.join(session_dir, "trial_outcome.txt"), "r") as f:
                for line in f:
                    if "Bee ID:" in line:
                        bid = line.replace("Bee ID:", "").strip()
                        if bid and bid.lower() not in ("unknown", "nan", ""):
                            return bid
        except Exception:
            pass

    if df is not None and not df.empty and "bee_id" in df.columns:
        bid = str(df.iloc[0]["bee_id"]).strip()
        if bid and bid.lower() not in ("unknown", "nan", ""):
            return bid

    if "unmarked" in sess_name.lower():
        return "unmarked"

    return "unknown"

    return "Unknown"


def smooth_and_interpolate(x, y, t, target_pts=800):
    n = len(x)
    if n < 3:
        return x, y, t
    
    if HAS_SAVGOL and n >= 7:
        w = min(17, n if n % 2 else n - 1)
        if w >= 5:
            try:
                x = savgol_filter(x, w, 2)
                y = savgol_filter(y, w, 2)
            except Exception:
                pass

    # Interpolate to dense time array to guarantee completely seamless gradient
    n_pts = max(n * 4, target_pts)
    t_interp = np.linspace(t.min(), t.max(), n_pts)
    x_interp = np.interp(t_interp, t, x)
    y_interp = np.interp(t_interp, t, y)
    return x_interp, y_interp, t_interp


def generate_figure(df, session_dir):
    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if len(df) < 10:
        return

    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    t_all = df["time_sec"].values
    dists = df["distance_from_center_mm"].values

    # 1. Clip trajectory between entry at outer boundary and final exit
    outer_entries = [i for i in range(1, len(dists)) if dists[i - 1] > OUTER_R and dists[i] <= OUTER_R]
    first_outer_entry = outer_entries[0] if outer_entries else 0

    outer_exits = [i for i in range(first_outer_entry + 1, len(dists)) if dists[i - 1] <= OUTER_R and dists[i] > OUTER_R]
    final_outer_exit = outer_exits[-1] if outer_exits else len(df) - 1

    x_clip = x_all[first_outer_entry : final_outer_exit + 1]
    y_clip = y_all[first_outer_entry : final_outer_exit + 1]
    t_clip = t_all[first_outer_entry : final_outer_exit + 1]

    if len(x_clip) < 4:
        x_clip, y_clip, t_clip = x_all, y_all, t_all

    # Normalize time slice so t=0 at entry and t=t_final at exit
    t_clip = t_clip - t_clip.min()

    # Smooth & densify for continuous line collection gradient
    x_s, y_s, t_s = smooth_and_interpolate(x_clip, y_clip, t_clip)

    # 2. Point-to-point LineCollection with continuous color mapping
    pts = np.column_stack([x_s, y_s]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    t_segments = (t_s[:-1] + t_s[1:]) / 2.0
    norm = plt.Normalize(t_s.min(), t_s.max())

    lc = LineCollection(
        segs,
        cmap=YGB_CMAP,
        norm=norm,
        linewidth=2.2,
        capstyle="round",
        joinstyle="round",
        zorder=4
    )
    lc.set_array(t_segments)

    # 3. Clean publication-ready figure layout
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    # Grey shaded ring between inner and outer circle
    outer_ring = Circle((0, 0), OUTER_R, facecolor="#f0f0f0", edgecolor="none", zorder=0)
    inner_white = Circle((0, 0), INNER_R, facecolor="white", edgecolor="none", zorder=0.5)
    ax.add_patch(outer_ring)
    ax.add_patch(inner_white)

    # Boundaries
    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#333333", lw=2.2, zorder=1))
    ax.add_patch(Circle((0, 0), INNER_R, fill=False, ec="#b0b0b0", lw=1.2, ls="--", zorder=2))
    ax.add_patch(Circle((0, 0), FEEDER_R, fc="#fff8e1", ec="#ffa000", lw=1.2, zorder=3))

    ax.add_collection(lc)

    # Hive marker outside outer boundary
    tag_type = str(df.iloc[0].get("tag_type", "")).strip().lower()
    orient = str(df.iloc[0].get("orientation", "LR")).strip().upper()
    if tag_type == "auto":
        hive_x, hive_y = 0.0, -HIVE_DIST
    else:
        hive_angle = np.pi / 2 if orient == "TB" else 0.0
        hive_x = HIVE_DIST * np.cos(hive_angle)
        hive_y = HIVE_DIST * np.sin(hive_angle)
    ax.plot(hive_x, hive_y, "o", color="#333333", markersize=9, markeredgecolor="#111111", markeredgewidth=1.2, zorder=10)
    ax.annotate("Hive", (hive_x, hive_y), textcoords="offset points", xytext=(10, 0), fontsize=9, fontweight="bold", color="#333333", va="center", zorder=11)

    # Feeder center marker
    ax.plot(0, 0, "o", color="#ffa000", markersize=8, zorder=10)

    # Metadata & Title
    orient = str(df.iloc[0].get("orientation", "unknown")).strip()
    stim = str(df.iloc[0].get("stimulus", "unknown")).strip()
    dop = str(df.iloc[0].get("xdop", df.iloc[0].get("dop", "unknown"))).strip()

    title_str = f"Cue: {orient} | {stim}"
    if dop != "unknown" and dop != "nan":
        title_str += f" | DoP = {dop}"
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=20)

    # Metadata Overlay (Bee ID & Outcome)
    bee_id = extract_bee_id(session_dir, df)
    outcome = str(df.iloc[0].get("trial_outcome", "unknown")).strip()
    outcome_file = os.path.join(session_dir, "trial_outcome.txt")
    if os.path.exists(outcome_file):
        try:
            with open(outcome_file, "r") as f:
                for line in f:
                    if "Outcome:" in line:
                        outcome = line.replace("Outcome:", "").strip()
        except Exception: pass

    lim = OUTER_R + 60
    textstr = f"Bee ID: {bee_id}\nOutcome: {outcome}"
    ax.text(-lim * 0.9, -lim * 0.9, textstr, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left", zorder=15,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc", alpha=0.9))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    # Horizontal colorbar at the bottom indicating Time (->) running Yellow to Blue
    sm = plt.cm.ScalarMappable(cmap=YGB_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.25, 0.06, 0.50, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Time (→)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    # Save
    fig.savefig(os.path.join(plots_dir, "complete_trajectory.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, "approach_and_home_search.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Generating complete_trajectory.png plots (Prompt 3)...")
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
            generate_figure(df, session_path)
        except Exception as e:
            print(f"    ERROR: {e}")


if __name__ == "__main__":
    main()
