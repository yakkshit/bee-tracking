"""
Generate a clean 'single colour' trajectory plot for each session.

Style (matching reference image):
  - Inbound (Approach):  grey dotted line with direction arrows
  - Outbound (Home Search): blue solid line with direction arrows
  - Entry: green triangle ▲ with ① on outer boundary
  - Exit: dark blue circle ● with ① on outer boundary
  - Feeder: orange dot at centre
  - Hive: outside the outer boundary
  - Inner boundary: dashed grey circle
  - Outer boundary: solid dark circle

Output: results/<session>/plots/single_colour.png
Does NOT modify any existing plot files.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

C_INBOUND  = "#888888"   # grey
C_OUTBOUND = "#1E88E5"   # blue
C_ENTRY    = "#2E7D32"   # green
C_EXIT     = "#0D47A1"   # dark blue
C_FEEDER   = "#F57C00"   # orange


def smooth(x, y, strength=17):
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


def subsample(x, y, max_pts=300):
    n = len(x)
    if n <= max_pts:
        return x, y
    idx = np.linspace(0, n - 1, max_pts, dtype=int)
    return x[idx], y[idx]


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


def add_arrows(ax, x, y, color, num=3, zorder=7):
    """Add small triangular direction arrows along a path."""
    n = len(x)
    if n < 10:
        return
    positions = np.linspace(int(n * 0.15), int(n * 0.85), num, dtype=int)
    for idx in positions:
        if idx < 2 or idx >= n:
            continue
        dx = x[idx] - x[idx - 2]
        dy = y[idx] - y[idx - 2]
        dist = np.hypot(dx, dy)
        if dist > 1.0:
            ax.annotate("", xy=(x[idx], y[idx]),
                        xytext=(x[idx - 2], y[idx - 2]),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=0, mutation_scale=10),
                        zorder=zorder)


def clip_and_prepare(x, y, start, end):
    """Extract segment, clip to arena, subsample, smooth."""
    seg_x = x[start:end + 1].copy()
    seg_y = y[start:end + 1].copy()
    d = np.hypot(seg_x, seg_y)
    mask = (d <= OUTER_R + 10) & (d >= FEEDER_R)
    if mask.sum() >= 3:
        seg_x, seg_y = seg_x[mask], seg_y[mask]
    seg_x, seg_y = subsample(seg_x, seg_y, max_pts=250)
    seg_x, seg_y = smooth(seg_x, seg_y, strength=15)
    return seg_x, seg_y


def generate_single_colour(df, session_dir):
    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if len(df) < 10:
        return

    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    dists = df["distance_from_center_mm"].values

    # Key crossings
    outer_entries, outer_exits = find_crossings(dists, OUTER_R)
    inner_entries, inner_exits = find_crossings(dists, INNER_R)

    first_outer_entry = outer_entries[0] if outer_entries else 0
    feeder_idx = np.argmin(dists[first_outer_entry:]) + first_outer_entry

    # Inner boundary crossings
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

    # Prepare inbound & outbound segments
    inbound_end = feeder_idx
    outbound_end = first_outer_exit if first_outer_exit else len(df) - 1

    in_x, in_y = clip_and_prepare(x_all, y_all, first_outer_entry, inbound_end)
    out_x, out_y = clip_and_prepare(x_all, y_all, feeder_idx, outbound_end)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    # Outer boundary (solid dark)
    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#333333", lw=2.2, zorder=1))

    # Inner boundary (dashed grey)
    ax.add_patch(Circle((0, 0), INNER_R, fill=False, ec="#aaaaaa",
                         lw=1.3, ls="--", zorder=2))

    # Feeder (orange dot)
    ax.plot(0, 0, "o", color=C_FEEDER, markersize=12, zorder=10)
    ax.text(0, -28, "Feeder", color=C_FEEDER, fontsize=11,
            fontweight="bold", ha="center", va="top", zorder=10)

    # ── Inbound path: grey dotted ────────────────────────────────────────
    if len(in_x) > 1:
        ax.plot(in_x, in_y, linestyle=":", color=C_INBOUND, lw=1.5, zorder=4)
        add_arrows(ax, in_x, in_y, color=C_INBOUND, num=2, zorder=5)

    # ── Outbound path: blue solid ────────────────────────────────────────
    if len(out_x) > 1:
        ax.plot(out_x, out_y, linestyle="-", color=C_OUTBOUND, lw=1.5, zorder=4)
        add_arrows(ax, out_x, out_y, color=C_OUTBOUND, num=2, zorder=5)

    # ── Entry marker (green ▲ on outer circle) ──────────────────────────
    ex, ey = project_to_circle(x_all[first_outer_entry],
                                y_all[first_outer_entry], OUTER_R)
    ax.plot(ex, ey, "^", color=C_ENTRY, markersize=11, zorder=11)
    ax.annotate("①", (ex, ey), textcoords="offset points", xytext=(-14, 0),
                fontsize=9, color=C_ENTRY, fontweight="bold",
                ha="center", va="center", zorder=12)

    # ── 1st Inner entry marker (red ◆ on inner circle) ────────────────────
    if first_inner_entry is not None:
        iex, iey = project_to_circle(x_all[first_inner_entry],
                                      y_all[first_inner_entry], INNER_R)
        ax.plot(iex, iey, "D", color="#D32F2F", markersize=8, zorder=11)

    # ── 1st Inner exit marker (purple ◆ on inner circle) ─────────────────
    if first_inner_exit is not None:
        ioex, ioey = project_to_circle(x_all[first_inner_exit],
                                        y_all[first_inner_exit], INNER_R)
        ax.plot(ioex, ioey, "D", color="#7B1FA2", markersize=8, zorder=11)

    # ── Exit marker (blue ● on outer circle) ─────────────────────────────
    if first_outer_exit is not None:
        fx, fy = project_to_circle(x_all[first_outer_exit],
                                    y_all[first_outer_exit], OUTER_R)
        ax.plot(fx, fy, "o", color=C_EXIT, markersize=9, zorder=11)
        ax.annotate("①", (fx, fy), textcoords="offset points", xytext=(12, -12),
                    fontsize=9, color=C_EXIT, fontweight="bold",
                    ha="center", va="center", zorder=12,
                    bbox=dict(boxstyle="circle,pad=0.1", fc="white",
                              ec=C_EXIT, lw=1.0))

    # ── Hive marker OUTSIDE the outer boundary ──────────────────────────
    orient = str(df.iloc[0].get("orientation", "LR")).strip().upper()
    hive_angle = np.pi / 2 if orient == "TB" else 0.0
    hive_x = HIVE_DIST * np.cos(hive_angle)
    hive_y = HIVE_DIST * np.sin(hive_angle)
    ax.plot(hive_x, hive_y, "o", color="#333333", markersize=10,
            markeredgecolor="#111111", markeredgewidth=1.5, zorder=12)
    ax.annotate("Hive", (hive_x, hive_y), textcoords="offset points",
                xytext=(12, 0), fontsize=9, fontweight="bold",
                color="#333333", va="center", zorder=13)

    # ── Title & Axis formatting ──────────────────────────────────────────
    orient = str(df.iloc[0].get("orientation", "unknown")).strip()
    stim = str(df.iloc[0].get("stimulus", "unknown")).strip()
    dop = str(df.iloc[0].get("xdop", df.iloc[0].get("dop", "unknown"))).strip()
    
    title_str = f"Cue: {orient} | {stim}"
    if dop != "unknown" and dop != "nan":
        title_str += f" | DoP = {dop}"
    title_str += " (n = 1 trials)"
    
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=20)

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

    # ── Legend ────────────────────────────────────────────────────────────
    legend_els = [
        Line2D([0], [0], color=C_INBOUND, ls=":", lw=2,
               label="Inbound (Approach Path)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_ENTRY,
               markersize=10, label="Entry (Outer)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#D32F2F",
               markersize=8, label="1st Inner Contact (In)"),
        Line2D([0], [0], color=C_OUTBOUND, ls="-", lw=2,
               label="Outbound (Home Search)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#7B1FA2",
               markersize=8, label="1st Inner Exit (Out)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EXIT,
               markersize=9, label="Exit (Outer)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FEEDER,
               markersize=9, label="Feeder (0,0)"),
    ]
    ax.legend(handles=legend_els, loc="upper right", frameon=True,
              facecolor="white", edgecolor="#d0d0d0", fontsize=8.5,
              borderpad=0.6, labelspacing=0.4, handletextpad=0.5)

    # ── Save as single_colour.png only ───────────────────────────────────
    fig.savefig(os.path.join(plots_dir, "single_colour.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Generating single_colour.png plots...")
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
            generate_single_colour(df, session_path)
        except Exception as e:
            print(f"    ERROR: {e}")


if __name__ == "__main__":
    main()
