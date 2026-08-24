"""
Generate fixed, clean 2-color gradient trajectory plots for each session:

Fixes & Enhancements:
1. Trajectory Truncation: Removes/clips all tracking points outside the outer circle (R > 420mm).
2. Multiple Exits Display: Detects and places distinct markers for 1st Outer Exit (1st Exit) and 2nd/Final Outer Exit (2nd Exit) if present.
3. External Legend Placement: Positions the legend completely OUTSIDE the circular arena to eliminate any overlap with trajectory or arena visuals.
4. Smooth 2-Color Gradient: Uses path-length normalized continuous 2-color gradient (Light Yellow -> Dark Royal Blue).

Outputs:
  - results/<session>/plots/gradient_full_trial_fixed.png
  - results/<session>/plots/gradient_exit_only_fixed.png
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

C_ENTRY    = "#2E7D32"   # green triangle
C_EXIT1    = "#0D47A1"   # dark blue circle (1st exit)
C_EXIT2    = "#1565C0"   # royal blue circle (2nd exit)
C_FEEDER   = "#F57C00"   # orange dot

# Strict 2-Color Colormap: Bright Yellow -> Dark Royal Blue
TWO_COLOR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "two_color_fixed", ["#FFEE58", "#0D47A1"]
)


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


def clip_points_to_arena(x, y, max_r=OUTER_R):
    """Clip trajectory points strictly to inside or on the arena boundary circle (R <= max_r)."""
    x_clipped, y_clipped = [], []
    for xi, yi in zip(x, y):
        d = np.hypot(xi, yi)
        if d <= max_r:
            x_clipped.append(xi)
            y_clipped.append(yi)
        else:
            # Project onto exact boundary circle
            x_clipped.append(xi * max_r / d)
            y_clipped.append(yi * max_r / d)
    return np.array(x_clipped), np.array(y_clipped)


def get_hive_position(df, outer_r=OUTER_R, hive_dist=HIVE_DIST):
    """
    Compute Hive position dynamically:
    Projects the first outer entry angle (or first tracked point angle) onto radius R = hive_dist (450mm).
    """
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

    # Fallback to the first available coordinate point
    if len(x_all) > 0:
        xe, ye = x_all[0], y_all[0]
        d = np.hypot(xe, ye)
        if d >= 1.0:
            return xe * hive_dist / d, ye * hive_dist / d

    return 0.0, -hive_dist


def subsample_and_densify(x, y, target_pts=700):
    n = len(x)
    if n < 2:
        return x, y
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.cumsum(np.hypot(dx, dy))
    dist = np.insert(dist, 0, 0.0)

    total_dist = dist[-1]
    if total_dist <= 0:
        return x, y

    dist_interp = np.linspace(0.0, total_dist, max(n * 3, target_pts))
    x_interp = np.interp(dist_interp, dist, x)
    y_interp = np.interp(dist_interp, dist, y)

    return x_interp, y_interp


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


# ── Plot 1: Full Trial Fixed Plot (Clean, External Legend, Both Exits Marked) ──
def generate_full_trial_fixed(df, session_dir):
    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if len(df) < 10:
        return

    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
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

    # Collect outer exits after feeder
    outer_exits_after_feeder = [ox for ox in outer_exits if ox >= feeder_idx]

    first_outer_exit = outer_exits_after_feeder[0] if outer_exits_after_feeder else None
    second_outer_exit = outer_exits_after_feeder[-1] if len(outer_exits_after_feeder) > 1 else None

    final_exit_idx = outer_exits_after_feeder[-1] if outer_exits_after_feeder else len(df) - 1
    if final_exit_idx <= first_outer_entry:
        final_exit_idx = len(df) - 1

    # Extract full trial path
    full_x = x_all[first_outer_entry : final_exit_idx + 1]
    full_y = y_all[first_outer_entry : final_exit_idx + 1]

    if len(full_x) < 2:
        return

    # Truncate / Clip trajectory points strictly to inside arena boundary
    full_x, full_y = clip_points_to_arena(full_x, full_y, max_r=OUTER_R)

    # Densify and smooth
    full_x, full_y = subsample_and_densify(full_x, full_y, target_pts=800)
    full_x, full_y = smooth(full_x, full_y, strength=17)

    # Re-clip after smoothing to ensure zero spillage
    full_x, full_y = clip_points_to_arena(full_x, full_y, max_r=OUTER_R)

    # Arc-length path progress from 0.0 to 1.0
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

    # Feeder center circle
    ax.plot(0, 0, "o", color=C_FEEDER, markersize=12, zorder=10)
    ax.text(0, -28, "Feeder", color=C_FEEDER, fontsize=11, fontweight="bold", ha="center", va="top", zorder=10)

    # ── Path-Length Smooth 2-Color LineCollection ───────────────────────
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

    # ── Static Markers & Exits Display ──────────────────────────────────
    # Entry marker (green ▲ on outer circle)
    ex, ey = project_to_circle(x_all[first_outer_entry], y_all[first_outer_entry], OUTER_R)
    ax.plot(ex, ey, "^", color=C_ENTRY, markersize=11, zorder=11)
    ax.annotate("Entry", (ex, ey), textcoords="offset points", xytext=(-14, 0), fontsize=9, color=C_ENTRY, fontweight="bold", ha="center", va="center", zorder=12)

    # 1st Inner contact (red ◆)
    if first_inner_entry is not None:
        iex, iey = project_to_circle(x_all[first_inner_entry], y_all[first_inner_entry], INNER_R)
        ax.plot(iex, iey, "D", color="#D32F2F", markersize=8, zorder=11)

    # 1st Inner exit (purple ◆)
    if first_inner_exit is not None:
        ioex, ioey = project_to_circle(x_all[first_inner_exit], y_all[first_inner_exit], INNER_R)
        ax.plot(ioex, ioey, "D", color="#7B1FA2", markersize=8, zorder=11)

    # 1st Outer Exit (blue ●)
    if first_outer_exit is not None:
        fx1, fy1 = project_to_circle(x_all[first_outer_exit], y_all[first_outer_exit], OUTER_R)
        ax.plot(fx1, fy1, "o", color=C_EXIT1, markersize=9, zorder=11)
        ax.annotate("① 1st Exit", (fx1, fy1), textcoords="offset points", xytext=(14, -10),
                    fontsize=8.5, color=C_EXIT1, fontweight="bold", va="center", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_EXIT1, lw=1.0, alpha=0.9))

    # 2nd / Final Outer Exit (royal blue ●) if distinct from 1st exit
    if second_outer_exit is not None and abs(second_outer_exit - first_outer_exit) > 5:
        fx2, fy2 = project_to_circle(x_all[second_outer_exit], y_all[second_outer_exit], OUTER_R)
        ax.plot(fx2, fy2, "o", color=C_EXIT2, markersize=9, zorder=11)
        ax.annotate("② 2nd Exit", (fx2, fy2), textcoords="offset points", xytext=(14, 10),
                    fontsize=8.5, color=C_EXIT2, fontweight="bold", va="center", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_EXIT2, lw=1.0, alpha=0.9))

    # Hive marker at original exit position (Red X)
    hive_x, hive_y = get_hive_position(df)
    ax.plot(hive_x, hive_y, "o", color="#333333", markersize=10, markeredgecolor="#111111", markeredgewidth=1.5, zorder=12)
    ax.annotate("Hive", (hive_x, hive_y), textcoords="offset points", xytext=(12, 0), fontsize=9, fontweight="bold", color="#333333", va="center", zorder=13)

    # Title & Metadata box
    title_str, bee_id, outcome = get_metadata(df, session_dir)
    ax.set_title(f"{title_str} (Full Trial - Clean Fixed)", fontsize=11, fontweight="bold", pad=20)

    lim = OUTER_R + 80
    textstr = f"Bee ID: {bee_id}\nOutcome: {outcome}"
    ax.text(-lim * 0.9, -lim * 0.9, textstr, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left", zorder=15,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc", alpha=0.9))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── External Legend (Completely Outside the Arena) ──────────────────────
    legend_els = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_ENTRY, markersize=10, label="Entry (Outer)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#D32F2F", markersize=8, label="1st Inner Contact"),
        Line2D([0], [0], color="#FFEE58", lw=3, label="Trajectory Start"),
        Line2D([0], [0], color="#0D47A1", lw=3, label="Trajectory Finish"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#7B1FA2", markersize=8, label="1st Inner Exit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EXIT1, markersize=9, label="1st Outer Exit"),
    ]
    if second_outer_exit is not None and abs(second_outer_exit - first_outer_exit) > 5:
        legend_els.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EXIT2, markersize=9, label="2nd Outer Exit"))
    legend_els.extend([
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FEEDER, markersize=9, label="Feeder (0,0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333", markeredgecolor="#111111", markersize=9, label="Hive"),
    ])

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

    # Horizontal 2-Color Colorbar showing Path Progress
    sm = plt.cm.ScalarMappable(cmap=TWO_COLOR_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.22, 0.05, 0.48, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0.0, 0.25, 0.50, 0.75, 1.0])
    cbar.ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.set_label("Trajectory Line Gradient: Start (Yellow) → 50% → Finish (Dark Blue)", fontsize=9.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.05, right=0.72, top=0.92, bottom=0.12)
    fig.savefig(os.path.join(plots_dir, "gradient_full_trial_fixed.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Plot 2: Exit Path Fixed Plot (Clean, External Legend, Both Exits Marked) ──
def generate_exit_only_fixed(df, session_dir):
    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if len(df) < 10:
        return

    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
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

    outer_exits_after_feeder = [ox for ox in outer_exits if ox >= feeder_idx]
    first_outer_exit = outer_exits_after_feeder[0] if outer_exits_after_feeder else None
    second_outer_exit = outer_exits_after_feeder[-1] if len(outer_exits_after_feeder) > 1 else None

    outbound_end = outer_exits_after_feeder[-1] if outer_exits_after_feeder else len(df) - 1
    if outbound_end <= feeder_idx:
        outbound_end = len(df) - 1

    out_x = x_all[feeder_idx : outbound_end + 1]
    out_y = y_all[feeder_idx : outbound_end + 1]

    if len(out_x) < 2:
        return

    out_x, out_y = clip_points_to_arena(out_x, out_y, max_r=OUTER_R)
    out_x, out_y = subsample_and_densify(out_x, out_y, target_pts=700)
    out_x, out_y = smooth(out_x, out_y, strength=15)
    out_x, out_y = clip_points_to_arena(out_x, out_y, max_r=OUTER_R)

    dx = np.diff(out_x)
    dy = np.diff(out_y)
    cum_dist = np.cumsum(np.hypot(dx, dy))
    cum_dist = np.insert(cum_dist, 0, 0.0)
    total_len = cum_dist[-1]
    path_progress = cum_dist / total_len if total_len > 0 else np.linspace(0, 1, len(out_x))

    fig, ax = plt.subplots(figsize=(9.5, 8), facecolor="white")

    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#333333", lw=2.2, zorder=1))
    ax.add_patch(Circle((0, 0), INNER_R, fill=False, ec="#aaaaaa", lw=1.3, ls="--", zorder=2))

    ax.plot(0, 0, "o", color=C_FEEDER, markersize=12, zorder=10)
    ax.text(0, -28, "Feeder", color=C_FEEDER, fontsize=11, fontweight="bold", ha="center", va="top", zorder=10)

    points = np.column_stack([out_x, out_y]).reshape(-1, 1, 2)
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

    if first_inner_exit is not None:
        ioex, ioey = project_to_circle(x_all[first_inner_exit], y_all[first_inner_exit], INNER_R)
        ax.plot(ioex, ioey, "D", color="#7B1FA2", markersize=8, zorder=11)

    if first_outer_exit is not None:
        fx1, fy1 = project_to_circle(x_all[first_outer_exit], y_all[first_outer_exit], OUTER_R)
        ax.plot(fx1, fy1, "o", color=C_EXIT1, markersize=9, zorder=11)
        ax.annotate("① 1st Exit", (fx1, fy1), textcoords="offset points", xytext=(14, -10),
                    fontsize=8.5, color=C_EXIT1, fontweight="bold", va="center", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_EXIT1, lw=1.0, alpha=0.9))

    if second_outer_exit is not None and abs(second_outer_exit - first_outer_exit) > 5:
        fx2, fy2 = project_to_circle(x_all[second_outer_exit], y_all[second_outer_exit], OUTER_R)
        ax.plot(fx2, fy2, "o", color=C_EXIT2, markersize=9, zorder=11)
        ax.annotate("② 2nd Exit", (fx2, fy2), textcoords="offset points", xytext=(14, 10),
                    fontsize=8.5, color=C_EXIT2, fontweight="bold", va="center", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_EXIT2, lw=1.0, alpha=0.9))

    # Hive marker at original exit position (Red X)
    hive_x, hive_y = get_hive_position(df)
    ax.plot(hive_x, hive_y, "o", color="#333333", markersize=10, markeredgecolor="#111111", markeredgewidth=1.5, zorder=12)
    ax.annotate("Hive", (hive_x, hive_y), textcoords="offset points", xytext=(12, 0), fontsize=9, fontweight="bold", color="#333333", va="center", zorder=13)

    title_str, bee_id, outcome = get_metadata(df, session_dir)
    ax.set_title(f"{title_str} (Exit Path - Clean Fixed)", fontsize=11, fontweight="bold", pad=20)

    lim = OUTER_R + 80
    textstr = f"Bee ID: {bee_id}\nOutcome: {outcome}"
    ax.text(-lim * 0.9, -lim * 0.9, textstr, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left", zorder=15,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc", alpha=0.9))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_els = [
        Line2D([0], [0], color="#FFEE58", lw=3, label="Feeder Exit Start"),
        Line2D([0], [0], color="#0D47A1", lw=3, label="Arena Exit Finish"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#7B1FA2", markersize=8, label="1st Inner Exit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EXIT1, markersize=9, label="1st Outer Exit"),
    ]
    if second_outer_exit is not None and abs(second_outer_exit - first_outer_exit) > 5:
        legend_els.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EXIT2, markersize=9, label="2nd Outer Exit"))
    legend_els.extend([
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FEEDER, markersize=9, label="Feeder (0,0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333", markeredgecolor="#111111", markersize=9, label="Hive"),
    ])

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
    cbar.ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.set_label("Exit Line Gradient: Feeder Exit (Yellow) → 50% → Finish (Dark Blue)", fontsize=9.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.05, right=0.72, top=0.92, bottom=0.12)
    fig.savefig(os.path.join(plots_dir, "gradient_exit_only_fixed.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_fixed_clean_plots(df, session_dir):
    generate_full_trial_fixed(df, session_dir)
    generate_exit_only_fixed(df, session_dir)


def main():
    print("Generating fixed clean trajectory plots (gradient_full_trial_fixed.png)...")
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
            generate_fixed_clean_plots(df, session_path)
        except Exception as e:
            print(f"    ERROR: {e}")


if __name__ == "__main__":
    main()
