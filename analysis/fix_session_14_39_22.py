"""
Fix trajectory plot for session 2024-11-18_14-39-22.R13.TB.P0.1U8:
1. 1st Outer Exit (Yellowish-Green path stage): Marked at (x=-10.4, y=-425.2) -> (x=-10.3, y=-419.9) on boundary.
2. Final Outer Exit (Dark Blue finish stage): Marked at (x=464.1, y=-270.3) -> (x=363.0, y=-211.3) on boundary.
3. Trajectory strictly clipped inside R <= 420mm.
4. Legend placed completely OUTSIDE the arena to avoid any overlap.
"""
import os
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

C_ENTRY  = "#2E7D32"   # green
C_EXIT1  = "#8BC34A"   # yellowish-green (1st exit)
C_EXIT2  = "#0D47A1"   # dark blue (final exit)
C_FEEDER = "#F57C00"   # orange

TWO_COLOR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "two_color_fixed", ["#FFEE58", "#0D47A1"]
)


def smooth(x, y, strength=15):
    n = len(x)
    if n < 7: return x, y
    if HAS_SAVGOL:
        w = min(strength, n if n % 2 else n - 1)
        if w < 5: w = 5
        if w % 2 == 0: w -= 1
        if 5 <= w <= n:
            try: return savgol_filter(x, w, 2), savgol_filter(y, w, 2)
            except Exception: pass
    k = min(5, n)
    kern = np.ones(k) / k
    xs = np.convolve(x, kern, mode="same")
    ys = np.convolve(y, kern, mode="same")
    xs[0], ys[0], xs[-1], ys[-1] = x[0], y[0], x[-1], y[-1]
    return xs, ys


def clip_points_to_arena(x, y, max_r=OUTER_R):
    x_c, y_c = [], []
    for xi, yi in zip(x, y):
        d = np.hypot(xi, yi)
        if d <= max_r:
            x_c.append(xi)
            y_c.append(yi)
        else:
            x_c.append(xi * max_r / d)
            y_c.append(yi * max_r / d)
    return np.array(x_c), np.array(y_c)


def subsample_and_densify(x, y, target_pts=800):
    n = len(x)
    if n < 2: return x, y
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.cumsum(np.hypot(dx, dy))
    dist = np.insert(dist, 0, 0.0)
    total_dist = dist[-1]
    if total_dist <= 0: return x, y
    dist_interp = np.linspace(0.0, total_dist, max(n * 3, target_pts))
    return np.interp(dist_interp, dist, x), np.interp(dist_interp, dist, y)


def get_hive_position(df, outer_r=OUTER_R, hive_dist=HIVE_DIST):
    if df is None or len(df) == 0:
        return hive_dist, 0.0
    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    dists = df["distance_from_center_mm"].values if "distance_from_center_mm" in df.columns else np.hypot(x_all, y_all)
    outer_entries = [i for i in range(1, len(dists)) if dists[i-1] > outer_r and dists[i] <= outer_r]
    idx = outer_entries[0] if outer_entries else 0
    xe, ye = x_all[idx], y_all[idx]
    d = np.hypot(xe, ye)
    if d < 1.0: return hive_dist, 0.0
    return xe * hive_dist / d, ye * hive_dist / d


def project_to_circle(x, y, radius):
    d = np.hypot(x, y)
    if d < 1: return x, y
    return x * radius / d, y * radius / d


def fix_session_14_39_22():
    target_session = "2024-11-18_14-39-22.R13.TB.P0.1U8"
    session_dir = os.path.join("results", target_session)
    if not os.path.exists(session_dir):
        session_dir = os.path.join("results/F", target_session)
    if not os.path.exists(session_dir):
        session_dir = os.path.join("results/maries data", target_session)

    csv_path = None
    if os.path.exists(session_dir):
        for f in os.listdir(session_dir):
            if f.startswith("bee_track_") and f.endswith(".csv"):
                csv_path = os.path.join(session_dir, f)
                break

    if not csv_path or not os.path.exists(csv_path):
        print(f"Skipping fix_session_14_39_22: {csv_path} not found")
        return

    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    dists = df["distance_from_center_mm"].values

    # Find key indices
    feeder_idx = np.argmin(dists) # row 742

    # 1st outer exit (row 827, x=-10.4, y=-425.2)
    first_exit_idx = 827
    # Final outer exit (row 915, x=464.1, y=-270.3)
    final_exit_idx = len(df) - 1

    # Extract trajectory from feeder (or start) to final exit
    full_x = x_all[0 : final_exit_idx + 1]
    full_y = y_all[0 : final_exit_idx + 1]

    # Clip strictly inside arena
    full_x, full_y = clip_points_to_arena(full_x, full_y, max_r=OUTER_R)
    full_x, full_y = subsample_and_densify(full_x, full_y, target_pts=850)
    full_x, full_y = smooth(full_x, full_y, strength=17)
    full_x, full_y = clip_points_to_arena(full_x, full_y, max_r=OUTER_R)

    # Arc-length path progress
    dx = np.diff(full_x)
    dy = np.diff(full_y)
    cum_dist = np.cumsum(np.hypot(dx, dy))
    cum_dist = np.insert(cum_dist, 0, 0.0)
    total_len = cum_dist[-1]
    path_progress = cum_dist / total_len if total_len > 0 else np.linspace(0, 1, len(full_x))

    fig, ax = plt.subplots(figsize=(9.5, 8), facecolor="white")

    # Arena circles
    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#333333", lw=2.2, zorder=1))
    ax.add_patch(Circle((0, 0), INNER_R, fill=False, ec="#aaaaaa", lw=1.3, ls="--", zorder=2))

    # Feeder
    ax.plot(0, 0, "o", color=C_FEEDER, markersize=12, zorder=10)
    ax.text(0, -28, "Feeder", color=C_FEEDER, fontsize=11, fontweight="bold", ha="center", va="top", zorder=10)

    # LineCollection gradient
    points = np.column_stack([full_x, full_y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    prog_segments = (path_progress[:-1] + path_progress[1:]) / 2.0
    norm = plt.Normalize(0.0, 1.0)

    lc = LineCollection(
        segments,
        cmap=TWO_COLOR_CMAP,
        norm=norm,
        linewidth=2.5,
        capstyle="round",
        joinstyle="round",
        zorder=6
    )
    lc.set_array(prog_segments)
    ax.add_collection(lc)

    # Entry marker
    ex, ey = project_to_circle(x_all[0], y_all[0], OUTER_R)
    ax.plot(ex, ey, "^", color=C_ENTRY, markersize=11, zorder=11)
    ax.annotate("Entry", (ex, ey), textcoords="offset points", xytext=(-14, 0), fontsize=9, color=C_ENTRY, fontweight="bold", ha="center", va="center", zorder=12)

    # 1st Exit (Yellowish-Green stage at row 827)
    x1_orig, y1_orig = x_all[first_exit_idx], y_all[first_exit_idx]
    fx1, fy1 = project_to_circle(x1_orig, y1_orig, OUTER_R)
    ax.plot(fx1, fy1, "o", color=C_EXIT1, markersize=10, markeredgecolor="#333333", markeredgewidth=1.2, zorder=11)
    ax.annotate("① 1st Exit (Yellow-Green)", (fx1, fy1), textcoords="offset points", xytext=(-80, -20),
                fontsize=9.0, color="#558B2F", fontweight="bold", va="center", zorder=12,
                arrowprops=dict(arrowstyle="->", color="#558B2F", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#558B2F", lw=1.2, alpha=0.95))

    # Final Exit (Dark Blue stage at row 915)
    x2_orig, y2_orig = x_all[final_exit_idx], y_all[final_exit_idx]
    fx2, fy2 = project_to_circle(x2_orig, y2_orig, OUTER_R)
    ax.plot(fx2, fy2, "o", color=C_EXIT2, markersize=10, markeredgecolor="#ffffff", markeredgewidth=1.2, zorder=11)
    ax.annotate("② Final Exit (Dark Blue)", (fx2, fy2), textcoords="offset points", xytext=(20, 15),
                fontsize=9.0, color=C_EXIT2, fontweight="bold", va="center", zorder=12,
                arrowprops=dict(arrowstyle="->", color=C_EXIT2, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_EXIT2, lw=1.2, alpha=0.95))

    # Hive (original Exit position / Red X)
    hive_x, hive_y = get_hive_position(df)
    ax.plot(hive_x, hive_y, "o", color="#333333", markersize=10, markeredgecolor="#111111", markeredgewidth=1.5, zorder=12)
    ax.annotate("Hive", (hive_x, hive_y), textcoords="offset points", xytext=(12, 0), fontsize=9, fontweight="bold", color="#333333", va="center", zorder=13)

    title_str = "Cue: TB | P0.1U8 | DoP = 0.1 (Session 2024-11-18_14-39-22)"
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=20)

    lim = OUTER_R + 80
    textstr = "Bee ID: R13\nOutcome: Exited Arena"
    ax.text(-lim * 0.9, -lim * 0.9, textstr, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left", zorder=15,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc", alpha=0.9))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_els = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_ENTRY, markersize=10, label="Entry Point"),
        Line2D([0], [0], color="#FFEE58", lw=3, label="Trajectory Start (Yellow)"),
        Line2D([0], [0], color=C_EXIT1, lw=3, label="1st Exit Stage (Yellow-Green)"),
        Line2D([0], [0], color=C_EXIT2, lw=3, label="Final Exit Finish (Dark Blue)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EXIT1, markeredgecolor="#333333", markersize=9, label="① 1st Exit (Row 827)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EXIT2, markeredgecolor="#ffffff", markersize=9, label="② Final Exit (Row 915)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FEEDER, markersize=9, label="Feeder (0,0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333", markeredgecolor="#111111", markersize=9, label="Hive"),
    ]

    ax.legend(
        handles=legend_els,
        bbox_to_anchor=(1.04, 1.0),
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=9.0,
        borderpad=0.7,
        labelspacing=0.5
    )

    sm = plt.cm.ScalarMappable(cmap=TWO_COLOR_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.22, 0.05, 0.48, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0.0, 0.25, 0.50, 0.75, 1.0])
    cbar.ax.set_xticklabels(["0% (Feeder)", "25% (1st Exit)", "50%", "75%", "100% (Final Exit)"])
    cbar.set_label("Trajectory Line Gradient: Start (Yellow) → 1st Exit (Yellow-Green) → Final Exit (Dark Blue)", fontsize=9.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.05, right=0.72, top=0.92, bottom=0.12)

    # Save to all requested plot names in session folder
    fig.savefig(os.path.join(plots_dir, "gradient_full_trajectory_smooth.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, "gradient_full_trial_fixed.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, "gradient_full_trial_60fps.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Fixed plots generated successfully in {plots_dir}!")


if __name__ == "__main__":
    fix_session_14_39_22()
