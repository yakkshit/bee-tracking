"""
Dedicated Chronological Boundary Event Plot for session 2024-11-18_14-39-22.R13.TB.P0.1U8:

Chronological Boundary Events:
1. ① 1st Entry (t=0.50s): Outside -> Inside at (344.1, -215.6) [Green ▲]
2. ② 1st Exit (t=118.73s): Inside -> Outside at (-373.8, -194.7) [Yellow-Green ●]
3. ③ 2nd Entry (t=181.07s): Outside -> Inside at (169.0, 378.4) [Cyan ▲]
4. ④ 2nd Exit (t=227.68s): Inside -> Outside at (423.1, -22.4) [Royal Blue ●]
5. ⑤ 3rd / Final Exit (t=249.35s): Inside -> Outside at (-10.4, -425.2) [Dark Blue ●]

Fixes:
- Trajectory clipped strictly inside R <= 420mm.
- All events labeled in chronological order with exact timestamps and directional markers.
- Legend positioned completely OUTSIDE the circular arena layout.
- Saved as results/2024-11-18_14-39-22.R13.TB.P0.1U8/plots/gradient_full_chronological_events_fixed.png
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

# 2-Color Colormap: Yellow -> Dark Royal Blue
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


def subsample_and_densify(x, y, target_pts=900):
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


def generate_chronological_fixed_plot():
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
        print(f"Skipping generate_chronological_fixed_plot: {csv_path} not found")
        return

    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    dists = df["distance_from_center_mm"].values
    times = df["time_sec"].values

    feeder_idx = np.argmin(dists)

    # Boundary Events (R = 420mm)
    # Event 1: Entry 1 (row 3, t=0.50s)
    # Event 2: Exit 1 (row 175, t=118.73s)
    # Event 3: Entry 2 (row 540, t=181.07s)
    # Event 4: Exit 2 (row 697, t=227.68s)
    # Event 5: Final Exit 3 (row 827, t=249.35s)
    e1_idx, e1_t = 3, times[3]
    x1_exit_idx, x1_t = 175, times[175]
    e2_idx, e2_t = 540, times[540]
    x2_exit_idx, x2_t = 697, times[697]
    x3_exit_idx, x3_t = 827, times[827]

    # Full trial path to final exit (row 827)
    full_x = x_all[e1_idx : x3_exit_idx + 1]
    full_y = y_all[e1_idx : x3_exit_idx + 1]

    # Truncate / Clip trajectory points strictly to inside arena boundary
    full_x, full_y = clip_points_to_arena(full_x, full_y, max_r=OUTER_R)
    full_x, full_y = subsample_and_densify(full_x, full_y, target_pts=900)
    full_x, full_y = smooth(full_x, full_y, strength=17)
    full_x, full_y = clip_points_to_arena(full_x, full_y, max_r=OUTER_R)

    # Path-length progress from 0.0 to 1.0
    dx = np.diff(full_x)
    dy = np.diff(full_y)
    cum_dist = np.cumsum(np.hypot(dx, dy))
    cum_dist = np.insert(cum_dist, 0, 0.0)
    total_len = cum_dist[-1]
    path_progress = cum_dist / total_len if total_len > 0 else np.linspace(0, 1, len(full_x))

    fig, ax = plt.subplots(figsize=(9.5, 8), facecolor="white")

    # Arena outlines
    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#333333", lw=2.2, zorder=1))
    ax.add_patch(Circle((0, 0), INNER_R, fill=False, ec="#aaaaaa", lw=1.3, ls="--", zorder=2))

    # Feeder center
    ax.plot(0, 0, "o", color="#F57C00", markersize=12, zorder=10)
    ax.text(0, -28, "Feeder", color="#F57C00", fontsize=11, fontweight="bold", ha="center", va="top", zorder=10)

    # Smooth LineCollection
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

    # ── Chronological Event Markers ─────────────────────────────────────
    # Event 1: 1st Entry (Green ▲)
    px1, py1 = project_to_circle(x_all[e1_idx], y_all[e1_idx], OUTER_R)
    ax.plot(px1, py1, "^", color="#2E7D32", markersize=11, zorder=12)
    ax.annotate(f"① 1st Entry ({e1_t:.1f}s)", (px1, py1), textcoords="offset points", xytext=(-25, -18),
                fontsize=8.5, color="#2E7D32", fontweight="bold", zorder=13,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2E7D32", lw=1.0, alpha=0.95))

    # Event 2: 1st Exit (Yellow-Green ●)
    px2, py2 = project_to_circle(x_all[x1_exit_idx], y_all[x1_exit_idx], OUTER_R)
    ax.plot(px2, py2, "o", color="#689F38", markersize=10, markeredgecolor="#333333", zorder=12)
    ax.annotate(f"② 1st Exit ({x1_t:.1f}s)", (px2, py2), textcoords="offset points", xytext=(-75, 12),
                fontsize=8.5, color="#689F38", fontweight="bold", zorder=13,
                arrowprops=dict(arrowstyle="->", color="#689F38", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#689F38", lw=1.0, alpha=0.95))

    # Event 3: 2nd Entry (Cyan ▲)
    px3, py3 = project_to_circle(x_all[e2_idx], y_all[e2_idx], OUTER_R)
    ax.plot(px3, py3, "^", color="#00838F", markersize=11, zorder=12)
    ax.annotate(f"③ 2nd Entry ({e2_t:.1f}s)", (px3, py3), textcoords="offset points", xytext=(15, 12),
                fontsize=8.5, color="#00838F", fontweight="bold", zorder=13,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#00838F", lw=1.0, alpha=0.95))

    # Event 4: 2nd Exit (Royal Blue ●)
    px4, py4 = project_to_circle(x_all[x2_exit_idx], y_all[x2_exit_idx], OUTER_R)
    ax.plot(px4, py4, "o", color="#1565C0", markersize=10, markeredgecolor="#ffffff", zorder=12)
    ax.annotate(f"④ 2nd Exit ({x2_t:.1f}s)", (px4, py4), textcoords="offset points", xytext=(15, -12),
                fontsize=8.5, color="#1565C0", fontweight="bold", zorder=13,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1565C0", lw=1.0, alpha=0.95))

    # Event 5: 3rd / Final Exit (Dark Blue ●)
    px5, py5 = project_to_circle(x_all[x3_exit_idx], y_all[x3_exit_idx], OUTER_R)
    ax.plot(px5, py5, "o", color="#0D47A1", markersize=11, markeredgecolor="#ffffff", markeredgewidth=1.3, zorder=12)
    ax.annotate(f"⑤ 3rd Exit / Final ({x3_t:.1f}s)", (px5, py5), textcoords="offset points", xytext=(-90, -22),
                fontsize=8.5, color="#0D47A1", fontweight="bold", zorder=13,
                arrowprops=dict(arrowstyle="->", color="#0D47A1", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#0D47A1", lw=1.2, alpha=0.95))

    # Hive (original Exit position / Red X)
    hive_x, hive_y = get_hive_position(df)
    ax.plot(hive_x, hive_y, "o", color="#333333", markersize=10, markeredgecolor="#111111", markeredgewidth=1.5, zorder=12)
    ax.annotate("Hive", (hive_x, hive_y), textcoords="offset points", xytext=(12, 0), fontsize=9, fontweight="bold", color="#333333", va="center", zorder=13)

    title_str = "Cue: TB | P0.1U8 | Chronological Entry & Exit Events"
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=20)

    lim = OUTER_R + 80
    textstr = "Bee ID: R13\nTotal Duration: 264.0s"
    ax.text(-lim * 0.9, -lim * 0.9, textstr, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left", zorder=15,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc", alpha=0.9))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_els = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2E7D32", markersize=10, label=f"① 1st Entry ({e1_t:.1f}s)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#689F38", markeredgecolor="#333333", markersize=9, label=f"② 1st Exit ({x1_t:.1f}s)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#00838F", markersize=10, label=f"③ 2nd Entry ({e2_t:.1f}s)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565C0", markeredgecolor="#ffffff", markersize=9, label=f"④ 2nd Exit ({x2_t:.1f}s)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0D47A1", markeredgecolor="#ffffff", markersize=10, label=f"⑤ 3rd Exit / Final ({x3_t:.1f}s)"),
        Line2D([0], [0], color="#FFEE58", lw=3, label="Start (0s - Yellow)"),
        Line2D([0], [0], color="#0D47A1", lw=3, label="Finish (264s - Dark Blue)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#F57C00", markersize=9, label="Feeder (0,0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333", markeredgecolor="#111111", markersize=9, label="Hive"),
    ]

    ax.legend(
        handles=legend_els,
        bbox_to_anchor=(1.04, 1.0),
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=8.5,
        borderpad=0.7,
        labelspacing=0.5
    )

    sm = plt.cm.ScalarMappable(cmap=TWO_COLOR_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.20, 0.05, 0.50, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0.0, 0.25, 0.50, 0.75, 1.0])
    cbar.ax.set_xticklabels(["0.5s (1st Entry)", "118.7s (1st Exit)", "181.1s (2nd Entry)", "227.7s (2nd Exit)", "249.4s (3rd Exit)"])
    cbar.set_label("Chronological Gradient: ① 1st Entry → ② 1st Exit → ③ 2nd Entry → ④ 2nd Exit → ⑤ 3rd Exit", fontsize=9.0, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.05, right=0.72, top=0.92, bottom=0.12)

    # Save fixed plots in session directory
    fig.savefig(os.path.join(plots_dir, "gradient_full_chronological_events_fixed.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, "gradient_full_60fps_time.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, "gradient_full_trial_60fps.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, "gradient_full_trajectory_smooth.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, "gradient_full_trial_fixed.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Successfully generated chronological fixed plot in {plots_dir}!")


if __name__ == "__main__":
    generate_chronological_fixed_plot()
