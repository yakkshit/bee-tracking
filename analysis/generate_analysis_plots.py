"""
Publication-quality bee trajectory with time-based colour gradient.

Style matched to reference: thin path line coloured yellow→green→cyan→blue
by time, grey shaded ring between inner and outer circle, hive marker
OUTSIDE the outer boundary, inner entry/exit in distinct colours.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge

try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except ImportError:
    HAS_SAVGOL = False

OUTER_R = 420.0   # mm
INNER_R = 210.0   # mm
FEEDER_R = 40.0   # mm

# Hive is OUTSIDE outer boundary (on the rim edge)
HIVE_DIST = OUTER_R + 30.0  # 30 mm outside the outer circle

# Time gradient: yellow (early) → green → cyan → dark blue (late)
TIME_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "time_grad", ["#FFA726", "#FFEE58", "#66BB6A", "#26C6DA", "#42A5F5", "#1565C0", "#1A237E"]
)


def smooth(x, y, strength=21):
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


def subsample(x, y, t, max_pts=500):
    n = len(x)
    if n <= max_pts:
        return x, y, t
    idx = np.linspace(0, n - 1, max_pts, dtype=int)
    return x[idx], y[idx], t[idx]


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


def generate_figure(df, session_dir):
    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if len(df) < 10:
        return

    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    t_all = df["time_sec"].values
    dists = df["distance_from_center_mm"].values

    # ── Find key boundary crossings ──────────────────────────────────────
    outer_entries, outer_exits = find_crossings(dists, OUTER_R)
    inner_entries, inner_exits = find_crossings(dists, INNER_R)

    first_outer_entry = outer_entries[0] if outer_entries else 0
    feeder_idx = np.argmin(dists[first_outer_entry:]) + first_outer_entry

    first_inner_entry = None
    for ie in inner_entries:
        if first_outer_entry <= ie <= feeder_idx:
            first_inner_entry = ie
            break

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

    # ── Clip trajectory to inside arena only ─────────────────────────────
    margin = 15.0
    arena_mask = dists <= (OUTER_R + margin)
    # Also skip points inside feeder
    feeder_mask = dists >= FEEDER_R
    valid_mask = arena_mask & feeder_mask
    
    x_clip = x_all[valid_mask]
    y_clip = y_all[valid_mask]
    t_clip = t_all[valid_mask]

    if len(x_clip) < 4:
        x_clip, y_clip, t_clip = x_all, y_all, t_all

    # Subsample then smooth
    x_sub, y_sub, t_sub = subsample(x_clip, y_clip, t_clip, max_pts=500)
    x_s, y_s = smooth(x_sub, y_sub, strength=17)
    n = len(x_s)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    # Grey shaded ring between inner and outer circle (like reference)
    outer_ring = Circle((0, 0), OUTER_R, facecolor="#e8e8e8", edgecolor="none", zorder=0)
    inner_white = Circle((0, 0), INNER_R, facecolor="white", edgecolor="none", zorder=0.5)
    ax.add_patch(outer_ring)
    ax.add_patch(inner_white)

    # Outer boundary (thick black)
    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#222222", lw=2.5, zorder=1))
    # Inner boundary (thin grey)
    ax.add_patch(Circle((0, 0), INNER_R, fill=False, ec="#aaaaaa", lw=1.2, zorder=2))

    # Feeder circle (center)
    ax.add_patch(Circle((0, 0), FEEDER_R, fc="white", ec="#555555", lw=1.2, zorder=6))

    # ── Time-gradient trajectory (LineCollection) ────────────────────────
    pts = np.column_stack([x_s, y_s]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(t_sub.min(), t_sub.max())
    lc = LineCollection(segs, cmap=TIME_CMAP, norm=norm,
                        linewidth=1.6, capstyle="round", joinstyle="round", zorder=4)
    lc.set_array(t_sub[:-1])
    ax.add_collection(lc)

    # ── Hive marker OUTSIDE the outer boundary ───────────────────────────
    orient = str(df.iloc[0].get("orientation", "LR")).strip().upper()
    if orient == "TB":
        hive_angle = np.pi / 2   # top
    else:
        hive_angle = 0.0          # right
    
    hive_x = HIVE_DIST * np.cos(hive_angle)
    hive_y = HIVE_DIST * np.sin(hive_angle)
    ax.plot(hive_x, hive_y, "o", color="#333333", markersize=10,
            markeredgecolor="#111111", markeredgewidth=1.5, zorder=12)
    ax.annotate("Hive", (hive_x, hive_y), textcoords="offset points",
                xytext=(12, 0), fontsize=9, fontweight="bold", color="#333333",
                va="center", zorder=13)

    # ── Entry point (green ▲ on outer circle) ────────────────────────────
    ex, ey = project_to_circle(x_all[first_outer_entry], y_all[first_outer_entry], OUTER_R)
    ax.plot(ex, ey, "^", color="#2E7D32", markersize=10, zorder=11)

    # ── 1st Inner Contact – inbound (red ●) ─────────────────────────────
    if first_inner_entry is not None:
        ix_pt, iy_pt = project_to_circle(x_all[first_inner_entry], y_all[first_inner_entry], INNER_R)
        ax.plot(ix_pt, iy_pt, "o", color="#D32F2F", markersize=8, zorder=11)

    # ── 1st Inner Exit – outbound (magenta ■) ───────────────────────────
    if first_inner_exit is not None:
        ox_pt, oy_pt = project_to_circle(x_all[first_inner_exit], y_all[first_inner_exit], INNER_R)
        ax.plot(ox_pt, oy_pt, "s", color="#7B1FA2", markersize=8, zorder=11)

    # ── Exit point (blue ● on outer circle) ──────────────────────────────
    if first_outer_exit is not None:
        fx, fy = project_to_circle(x_all[first_outer_exit], y_all[first_outer_exit], OUTER_R)
        ax.plot(fx, fy, "o", color="#0D47A1", markersize=9, zorder=11)

    # ── Title & Axis formatting ──────────────────────────────────────────
    orient = str(df.iloc[0].get("orientation", "unknown")).strip()
    stim = str(df.iloc[0].get("stimulus", "unknown")).strip()
    dop = str(df.iloc[0].get("xdop", df.iloc[0].get("dop", "unknown"))).strip()
    
    title_str = f"Cue: {orient} | {stim}"
    if dop != "unknown" and dop != "nan":
        title_str += f" | DoP = {dop}"
    title_str += " (n = 1 trials)"
    
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=25)

    # ── Metadata Overlay (Bee ID & Outcome) ─────────────────────────────
    bee_id = str(df.iloc[0].get("bee_id", "unknown")).strip()
    outcome = str(df.iloc[0].get("trial_outcome", "unknown")).strip()
    
    outcome_file = os.path.join(session_dir, "trial_outcome.txt")
    if os.path.exists(outcome_file):
        try:
            with open(outcome_file, "r") as f:
                txt = f.read().strip()
                for line in txt.split("\n"):
                    if "Bee ID:" in line:
                        bee_id = line.replace("Bee ID:", "").strip()
                    elif "Outcome:" in line:
                        outcome = line.replace("Outcome:", "").strip()
                    elif txt and ":" not in txt:
                        outcome = txt
        except Exception: pass

    lim = OUTER_R + 80
    textstr = f"Bee ID: {bee_id}\nOutcome: {outcome}"
    ax.text(-lim * 0.9, -lim * 0.9, textstr, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left", zorder=15,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc", alpha=0.9))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Time gradient colorbar at bottom ─────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=TIME_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.25, 0.06, 0.50, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Time →", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    # ── Legend below the arena ───────────────────────────────────────────
    legend_els = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2E7D32",
               markersize=9, label="Entry (Outer)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D32F2F",
               markersize=8, label="1st Inner Contact (Inbound)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#7B1FA2",
               markersize=8, label="1st Inner Exit (Outbound)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0D47A1",
               markersize=8, label="Exit (Outer)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333",
               markeredgecolor="#111111", markersize=9, label="Hive"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#555555", markersize=8, label="Feeder"),
    ]
    ax.legend(handles=legend_els, loc="upper right", frameon=True,
              facecolor="white", edgecolor="#d0d0d0", fontsize=8,
              borderpad=0.5, labelspacing=0.35, handletextpad=0.4)

    # ── Save ─────────────────────────────────────────────────────────────
    fig.savefig(os.path.join(plots_dir, "complete_trajectory.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, "approach_and_home_search.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Scanning results folder for video sessions...")
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("Error: results folder does not exist.")
        return

    for session_name in sorted(os.listdir(results_dir)):
        session_path = os.path.join(results_dir, session_name)
        if not os.path.isdir(session_path):
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
