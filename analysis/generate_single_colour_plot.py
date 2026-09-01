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
import re
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
    if first_outer_exit is None:
        first_outer_exit = len(df) - 1

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

    # ── Entry/Exit Markers Determination ────────────────────────────────
    if outer_entries:
        ex, ey = project_to_circle(x_all[outer_entries[0]], y_all[outer_entries[0]], OUTER_R)
    else:
        ex, ey = 0.0, -OUTER_R

    if inner_entries:
        iex, iey = project_to_circle(x_all[inner_entries[0]], y_all[inner_entries[0]], INNER_R)
    else:
        iex, iey = 0.0, -INNER_R

    if inner_exits:
        ioex, ioey = project_to_circle(x_all[inner_exits[0]], y_all[inner_exits[0]], INNER_R)
    else:
        ioex, ioey = project_to_circle(x_all[-1], y_all[-1], INNER_R)

    if outer_exits:
        fx, fy = project_to_circle(x_all[outer_exits[-1]], y_all[outer_exits[-1]], OUTER_R)
    else:
        fx, fy = project_to_circle(x_all[-1], y_all[-1], OUTER_R)

    # Prepend smooth entry segment if tracking started inside outer circle
    if not outer_entries and len(in_x) > 0:
        pts = np.array([[ex, ey], [iex, iey], [in_x[0], in_y[0]]])
        t = np.linspace(0, 1, len(pts))
        try:
            from scipy.interpolate import make_interp_spline
            spl_x = make_interp_spline(t, pts[:, 0], k=2)
            spl_y = make_interp_spline(t, pts[:, 1], k=2)
            t_fine = np.linspace(0, 1, 30)
            in_x = np.concatenate([spl_x(t_fine), in_x])
            in_y = np.concatenate([spl_y(t_fine), in_y])
        except Exception:
            entry_x = np.linspace(ex, in_x[0], 30)
            entry_y = np.linspace(ey, in_y[0], 30)
            in_x = np.concatenate([entry_x, in_x])
            in_y = np.concatenate([entry_y, in_y])

    # ── Inbound path: grey dotted ────────────────────────────────────────
    if len(in_x) > 1:
        ax.plot(in_x, in_y, linestyle=":", color=C_INBOUND, lw=1.5, zorder=4)
        add_arrows(ax, in_x, in_y, color=C_INBOUND, num=2, zorder=5)

    # ── Outbound path: blue solid ────────────────────────────────────────
    if len(out_x) > 1:
        ax.plot(out_x, out_y, linestyle="-", color=C_OUTBOUND, lw=1.5, zorder=4)
        add_arrows(ax, out_x, out_y, color=C_OUTBOUND, num=2, zorder=5)

    # ── Entry marker (green ▲ on outer circle) ──────────────────────────
    ax.plot(ex, ey, "^", color=C_ENTRY, markersize=11, zorder=11)
    ax.annotate("①", (ex, ey), textcoords="offset points", xytext=(-14, 0),
                fontsize=9, color=C_ENTRY, fontweight="bold",
                ha="center", va="center", zorder=12)

    # ── 1st Inner entry marker (red ◆ on inner circle) ────────────────────
    ax.plot(iex, iey, "D", color="#D32F2F", markersize=8, zorder=11)

    # ── 1st Inner exit marker (purple ◆ on inner circle) ─────────────────
    ax.plot(ioex, ioey, "D", color="#7B1FA2", markersize=8, zorder=11)

    # ── Exit marker (blue ● on outer circle) ─────────────────────────────
    ax.plot(fx, fy, "o", color=C_EXIT, markersize=9, zorder=11)
    ax.annotate("①", (fx, fy), textcoords="offset points", xytext=(12, -12),
                fontsize=9, color=C_EXIT, fontweight="bold",
                ha="center", va="center", zorder=12,
                bbox=dict(boxstyle="circle,pad=0.1", fc="white",
                          ec=C_EXIT, lw=1.0))

    # ── Hive marker OUTSIDE the outer boundary (at original Red X exit direction) ──
    hive_x, hive_y = get_hive_position(df)
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
    ax.legend(handles=legend_els, bbox_to_anchor=(1.04, 1.0), loc="upper left", frameon=True,
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
