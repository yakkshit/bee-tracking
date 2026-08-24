"""
Prompt 1: Exit from Feeder to Inner Circle Trajectory Plot

Generates a continuous time-gradient plot for the bee's trajectory strictly from
the Feeder exit until it first reaches/enters the inner circle boundary.

Requirements fulfilled:
1. Data Slice: Strictly between timestamp leaving feeder and first reaching inner circle threshold.
2. Gradient Line: LineCollection with single point-to-point segments (no flat blocks).
3. Colormap: Custom continuous Yellow (t_start) -> Green -> Blue (t_end) colormap.
   capstyle='round' and joinstyle='round' set on LineCollection.
4. Styling: Clean background with light gray ring overlays (arena layout),
   hive marker outside, no extra event markers on path.

Output: results/<session>/plots/feeder_to_inner_circle.png
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


def smooth_and_interpolate(x, y, t, max_pts=400):
    n = len(x)
    if n < 3:
        return x, y, t
    
    # Smooth with Savitzky-Golay if enough points
    if HAS_SAVGOL and n >= 7:
        w = min(15, n if n % 2 else n - 1)
        if w >= 5:
            try:
                x = savgol_filter(x, w, 2)
                y = savgol_filter(y, w, 2)
            except Exception:
                pass

    # Interpolate to dense time points for continuous gradient
    n_target = max(n * 3, 300)
    t_interp = np.linspace(t.min(), t.max(), n_target)
    x_interp = np.interp(t_interp, t, x)
    y_interp = np.interp(t_interp, t, y)
    return x_interp, y_interp, t_interp


def generate_feeder_to_inner_plot(df, session_dir):
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

    # Index at closest approach to feeder (at or inside feeder_r)
    feeder_idx = np.argmin(dists[first_outer_entry:]) + first_outer_entry

    # 2. Slice strictly from feeder departure until first inner circle contact/crossing
    post_feeder_dists = dists[feeder_idx:]
    inner_contacts = np.where(post_feeder_dists >= INNER_R)[0]

    if len(inner_contacts) > 0:
        inner_idx = feeder_idx + inner_contacts[0]
    else:
        # Fallback to max distance point if exact threshold isn't crossed
        inner_idx = feeder_idx + np.argmax(post_feeder_dists)

    if inner_idx <= feeder_idx:
        inner_idx = min(feeder_idx + 20, len(df) - 1)

    x_slice = x_all[feeder_idx : inner_idx + 1]
    y_slice = y_all[feeder_idx : inner_idx + 1]
    t_slice = t_all[feeder_idx : inner_idx + 1]

    if len(x_slice) < 2:
        return

    # Smooth and interpolate for seamless gradient
    x_s, y_s, t_s = smooth_and_interpolate(x_slice, y_slice, t_slice)

    # 3. Create single point-to-point LineCollection
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

    # 4. Plot over clean polar background with light gray ring overlays
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    # Grey shaded ring between inner and outer circle (arena layout)
    outer_ring = Circle((0, 0), OUTER_R, facecolor="#f0f0f0", edgecolor="none", zorder=0)
    inner_white = Circle((0, 0), INNER_R, facecolor="white", edgecolor="none", zorder=0.5)
    ax.add_patch(outer_ring)
    ax.add_patch(inner_white)

    # Arena boundaries
    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#333333", lw=2.0, zorder=1))
    ax.add_patch(Circle((0, 0), INNER_R, fill=False, ec="#b0b0b0", lw=1.2, ls="--", zorder=2))
    ax.add_patch(Circle((0, 0), FEEDER_R, fc="#fff8e1", ec="#ffa000", lw=1.2, zorder=3))

    # Add LineCollection
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

    # Feeder label at center
    ax.plot(0, 0, "o", color="#ffa000", markersize=8, zorder=10)

    # Formatting
    lim = OUTER_R + 60
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    title_str = f"Feeder Exit to Inner Circle ({session_dir.rsplit('/', 1)[-1]})"
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=15)

    # Save
    fig.savefig(os.path.join(plots_dir, "feeder_to_inner_circle.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Generating feeder_to_inner_circle.png plots (Prompt 1)...")
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
            generate_feeder_to_inner_plot(df, session_path)
        except Exception as e:
            print(f"    ERROR: {e}")


if __name__ == "__main__":
    main()
