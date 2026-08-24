"""
Prompt 2: Exit from Feeder to Final Exit Trajectory Plot

Generates a continuous time-gradient plot for the complete homeward/egress trajectory
strictly from the Feeder exit all the way until the Final Exit out of the outer arena perimeter.

Requirements fulfilled:
1. Data Slice: Strictly from feeder departure to final outer boundary crossing.
2. Smooth Gradient Rendering: LineCollection with t_segment = (t[:-1] + t[1:]) / 2.0.
3. Colormap & Normalization: Continuous Yellow -> Green -> Blue colormap normalized to slice.
4. Interpolation: Densified timeline via np.interp and savgol_filter to eliminate color-blocking.
   capstyle='round' and joinstyle='round' on LineCollection.
5. Styling: Circular arena boundary and inner region without on-path text annotations or large legend boxes.

Output: results/<session>/plots/feeder_to_final_exit.png
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


def smooth_and_interpolate(x, y, t, target_pts=600):
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

    # 5-10x interpolation along (x, y, t) for seamless color gradient
    n_pts = max(n * 5, target_pts)
    t_interp = np.linspace(t.min(), t.max(), n_pts)
    x_interp = np.interp(t_interp, t, x)
    y_interp = np.interp(t_interp, t, y)
    return x_interp, y_interp, t_interp


def generate_feeder_to_final_exit_plot(df, session_dir):
    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if len(df) < 10:
        return

    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    t_all = df["time_sec"].values
    dists = df["distance_from_center_mm"].values

    # 1. Locate entry into arena and feeder arrival
    outer_entries = [i for i in range(1, len(dists)) if dists[i - 1] > OUTER_R and dists[i] <= OUTER_R]
    first_outer_entry = outer_entries[0] if outer_entries else 0

    feeder_idx = np.argmin(dists[first_outer_entry:]) + first_outer_entry

    # 2. Locate final exit out of outer boundary
    outer_exits = [i for i in range(feeder_idx + 1, len(dists)) if dists[i - 1] <= OUTER_R and dists[i] > OUTER_R]
    if outer_exits:
        final_exit_idx = outer_exits[-1]
    else:
        # Fallback to end of trajectory
        final_exit_idx = len(df) - 1

    if final_exit_idx <= feeder_idx:
        final_exit_idx = len(df) - 1

    x_slice = x_all[feeder_idx : final_exit_idx + 1]
    y_slice = y_all[feeder_idx : final_exit_idx + 1]
    t_slice = t_all[feeder_idx : final_exit_idx + 1]

    if len(x_slice) < 2:
        return

    # Smooth and interpolate 5-10x for seamless gradient
    x_s, y_s, t_s = smooth_and_interpolate(x_slice, y_slice, t_slice)

    # 3. Create LineCollection with per-segment midpoints
    pts = np.column_stack([x_s, y_s]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    t_segments = (t_s[:-1] + t_s[1:]) / 2.0
    norm = plt.Normalize(t_s.min(), t_s.max())

    lc = LineCollection(
        segs,
        cmap=YGB_CMAP,
        norm=norm,
        linewidth=2.0,
        capstyle="round",
        joinstyle="round",
        zorder=4
    )
    lc.set_array(t_segments)

    # 4. Plot layout (clean, no large legend boxes or on-path text annotations)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    outer_ring = Circle((0, 0), OUTER_R, facecolor="#f0f0f0", edgecolor="none", zorder=0)
    inner_white = Circle((0, 0), INNER_R, facecolor="white", edgecolor="none", zorder=0.5)
    ax.add_patch(outer_ring)
    ax.add_patch(inner_white)

    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#333333", lw=2.0, zorder=1))
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

    lim = OUTER_R + 60
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    title_str = f"Feeder Exit to Final Arena Exit ({session_dir.rsplit('/', 1)[-1]})"
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=15)

    fig.savefig(os.path.join(plots_dir, "feeder_to_final_exit.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Generating feeder_to_final_exit.png plots (Prompt 2)...")
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
            generate_feeder_to_final_exit_plot(df, session_path)
        except Exception as e:
            print(f"    ERROR: {e}")


if __name__ == "__main__":
    main()
