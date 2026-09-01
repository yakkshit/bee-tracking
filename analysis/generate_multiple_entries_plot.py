"""
Generate a 'multiple entries and exits' trajectory plot for each session.

Style:
  - Detects all entry/exit passes across outer and inner boundaries.
  - Plots each individual entry-exit pass with light distinct colours and numbered markers (①, ②, ③...).
  - Computes and plots the Mean Trajectory across passes as a bold line.
  - Hive marker outside the outer boundary.
  - Output: results/<session>/plots/multiple_entries_exits.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

PASS_COLORS = ["#1E88E5", "#D81B60", "#004D40", "#F57C00", "#7CB342", "#8E24AA", "#00ACC1"]


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


def subsample(x, y, max_pts=200):
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


def get_hive_position(df, outer_r=OUTER_R, hive_dist=HIVE_DIST):
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


def generate_multiple_entries_plot(df, session_dir):
    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if len(df) < 10:
        return

    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    dists = df["distance_from_center_mm"].values

    outer_entries, outer_exits = find_crossings(dists, OUTER_R)
    inner_entries, inner_exits = find_crossings(dists, INNER_R)

    # Segment passes based on inner circle entries and exits
    passes = []
    all_entries = sorted(list(set(outer_entries + inner_entries)))
    
    if len(inner_entries) > 0:
        for idx, i_entry in enumerate(inner_entries):
            # Find next exit (inner or outer)
            next_exits = [e for e in (inner_exits + outer_exits) if e > i_entry]
            i_exit = next_exits[0] if next_exits else len(df) - 1
            passes.append((i_entry, i_exit))
    else:
        # Fallback to single trajectory if no distinct inner entries
        first_entry = outer_entries[0] if outer_entries else 0
        last_exit = outer_exits[-1] if outer_exits else len(df) - 1
        passes.append((first_entry, last_exit))

    # Limits max 5 passes for clarity
    passes = passes[:5]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    # Arena boundaries
    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#333333", lw=2.2, zorder=1))
    ax.add_patch(Circle((0, 0), INNER_R, fill=False, ec="#aaaaaa", lw=1.3, ls="--", zorder=2))
    ax.plot(0, 0, "o", color="#F57C00", markersize=12, zorder=10)
    ax.text(0, -28, "Feeder", color="#F57C00", fontsize=11, fontweight="bold", ha="center", va="top", zorder=10)

    resampled_paths = []
    circled_numbers = ["①", "②", "③", "④", "⑤"]

    for i, (p_start, p_end) in enumerate(passes):
        px = x_all[p_start:p_end + 1]
        py = y_all[p_start:p_end + 1]
        pd_val = dists[p_start:p_end + 1]

        mask = (pd_val <= OUTER_R + 15) & (pd_val >= FEEDER_R)
        if mask.sum() >= 3:
            px, py = px[mask], py[mask]

        px, py = subsample(px, py, max_pts=150)
        px, py = smooth(px, py, strength=11)

        c = PASS_COLORS[i % len(PASS_COLORS)]
        num_str = circled_numbers[i % len(circled_numbers)]

        # Plot individual pass
        ax.plot(px, py, linestyle="--", color=c, alpha=0.6, lw=1.4, zorder=4, label=f"Pass {i+1} {num_str}")

        # Mark entry / exit of pass on inner boundary
        if len(px) > 0:
            iex, iey = project_to_circle(px[0], py[0], INNER_R)
            ax.plot(iex, iey, "o", color=c, markersize=6, alpha=0.8, zorder=9)
            ax.annotate(num_str, (iex, iey), textcoords="offset points", xytext=(-8, 8),
                        fontsize=8, color=c, fontweight="bold", zorder=10)

        # Store for mean trajectory calculation
        if len(px) >= 20:
            # Resample to 100 points for aligned mean calculation
            t_orig = np.linspace(0, 1, len(px))
            t_norm = np.linspace(0, 1, 100)
            mx = np.interp(t_norm, t_orig, px)
            my = np.interp(t_norm, t_orig, py)
            resampled_paths.append((mx, my))

    # Plot Mean Trajectory if multiple passes exist
    if len(resampled_paths) > 1:
        mean_x = np.mean([p[0] for p in resampled_paths], axis=0)
        mean_y = np.mean([p[1] for p in resampled_paths], axis=0)
        mean_x, mean_y = smooth(mean_x, mean_y, strength=15)
        ax.plot(mean_x, mean_y, color="#000000", lw=2.5, zorder=6, label="Mean Trajectory")

    # Hive Marker OUTSIDE outer boundary
    hive_x, hive_y = get_hive_position(df)
    ax.plot(hive_x, hive_y, "o", color="#333333", markersize=10, markeredgecolor="#111111", markeredgewidth=1.5, zorder=12)
    ax.annotate("Hive", (hive_x, hive_y), textcoords="offset points", xytext=(12, 0), fontsize=9, fontweight="bold", color="#333333", va="center", zorder=13)

    # Title formatting
    orient = str(df.iloc[0].get("orientation", "unknown")).strip()
    stim = str(df.iloc[0].get("stimulus", "unknown")).strip()
    dop = str(df.iloc[0].get("xdop", df.iloc[0].get("dop", "unknown"))).strip()
    
    title_str = f"Cue: {orient} | {stim}"
    if dop != "unknown" and dop != "nan":
        title_str += f" | DoP = {dop}"
    title_str += " (n = 1 trials)"
    
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=20)

import re

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

    ax.legend(bbox_to_anchor=(1.04, 1.0), loc="upper left", frameon=True, facecolor="white", edgecolor="#d0d0d0", fontsize=8.5, borderpad=0.5, labelspacing=0.35)

    fig.savefig(os.path.join(plots_dir, "multiple_entries_exits.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Generating multiple_entries_exits.png plots...")
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
            generate_multiple_entries_plot(df, session_path)
        except Exception as e:
            print(f"    ERROR: {e}")


if __name__ == "__main__":
    main()
