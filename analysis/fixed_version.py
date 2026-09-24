import os
import sys
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

if sys.platform.startswith("win"):
    try:
        import site
        for site_path in site.getsitepackages():
            for lib_folder in ("torch/lib", "scipy.libs", "numpy.libs", "matplotlib.libs"):
                dll_dir = os.path.abspath(os.path.join(site_path, lib_folder))
                if os.path.exists(dll_dir):
                    win_dll = f"\\\\?\\{dll_dir}" if not dll_dir.startswith("\\\\?\\") else dll_dir
                    try:
                        os.add_dll_directory(dll_dir)
                    except Exception:
                        try:
                            os.add_dll_directory(win_dll)
                        except Exception:
                            pass
    except Exception:
        pass

try:
    from scipy.signal import savgol_filter
except Exception:
    def savgol_filter(x, window_length, polyorder, **kwargs):
        x = np.asarray(x, dtype=float)
        w = max(1, int(window_length))
        if len(x) < w or w < 2:
            return x
        return np.convolve(x, np.ones(w) / w, mode='same')


def parse_metadata_from_foldername(folder_name):
    """
    Parses folder names or trial descriptors like:
    '2024-11-18_15-40-44.R13.LR_P_4.4_U_8.0'
    or '2025-02-28 14-06-14.W 36 TB P 3.8 U 8'
    """
    cue = "LR" if any(x in folder_name for x in [".LR", "_LR", " LR "]) else "TB"
    
    p_match = re.search(r'P[._\s]?(\d+(?:\.\d+)?)', folder_name, re.IGNORECASE)
    p_val = f"p{p_match.group(1)}" if p_match else "p?"
    
    u_match = re.search(r'U[._\s]?(\d+(?:\.\d+)?)', folder_name, re.IGNORECASE)
    u_val = f"u{int(float(u_match.group(1)))}" if u_match else "u?"
    
    return f"Cue: {cue} | {p_val} {u_val} | Trajectory Analysis"


def extract_bee_id(folder_path, df=None):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    m = re.search(r'(\d+[wWgGbByYoOrRpP])', folder_name)
    if m:
        return m.group(1).lower()

    if df is not None and not df.empty and 'bee_id' in df.columns:
        bid = str(df['bee_id'].iloc[0]).strip()
        if bid and bid.lower() not in ("unknown", "nan", ""):
            return bid

    outcome_file = os.path.join(folder_path, "trial_outcome.txt")
    if os.path.exists(outcome_file):
        try:
            with open(outcome_file, 'r') as f:
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        if "bee" in k.lower() or "id" in k.lower():
                            bid = v.strip()
                            if bid and bid.lower() not in ("unknown", "nan", ""):
                                return bid
        except Exception:
            pass

    return "unknown"


def get_hive_position(df, outer_r=420.0, hive_dist=450.0):
    if df is None or len(df) == 0:
        return 0.0, -hive_dist
    x_all = df["x_mm"].values
    y_all = df["y_mm"].values
    dists = df["distance_from_center_mm"].values if "distance_from_center_mm" in df.columns else np.hypot(x_all, y_all)
    if "tag_type" in df.columns:
        entry_rows = df[df["tag_type"] == "entry"]
        if not entry_rows.empty:
            xe = float(entry_rows.iloc[0]["x_mm"])
            ye = float(entry_rows.iloc[0]["y_mm"])
            d = np.hypot(xe, ye)
            if d >= 1.0:
                return xe * hive_dist / d, ye * hive_dist / d
    outer_entries = [i for i in range(1, len(dists)) if dists[i-1] > outer_r and dists[i] <= outer_r]
    if outer_entries:
        idx = outer_entries[0]
        xe, ye = x_all[idx], y_all[idx]
        d = np.hypot(xe, ye)
        if d >= 1.0:
            return xe * hive_dist / d, ye * hive_dist / d
    if len(x_all) > 0:
        xe, ye = x_all[0], y_all[0]
        d = np.hypot(xe, ye)
        if d >= 1.0:
            return xe * hive_dist / d, ye * hive_dist / d
    return 0.0, -hive_dist


def process_single_trial(folder_path, frame_gap_threshold=40, spatial_jump_threshold=80.0, fps=60):
    folder_name = os.path.basename(folder_path)
    print(f"\nProcessing folder: {folder_name}")

    # 1. Locate CSV data file
    csv_files = glob.glob(os.path.join(folder_path, "bee_track_*.csv"))
    if not csv_files:
        print(f" Skipping {folder_name}: No bee_track_*.csv file found.")
        return

    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)

    # 2. Extract Metadata
    bee_id = extract_bee_id(folder_path, df)
    outcome = df['trial_outcome'].iloc[0] if 'trial_outcome' in df.columns and pd.notna(df['trial_outcome'].iloc[0]) else "Unknown"

    outcome_file = os.path.join(folder_path, "trial_outcome.txt")
    if os.path.exists(outcome_file):
        with open(outcome_file, 'r') as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    if "outcome" in k.lower() or "result" in k.lower():
                        outcome = v.strip()

    # Save updated bee_id to trial_outcome.txt
    if bee_id != "unknown":
        with open(outcome_file, 'w') as f:
            f.write(f"Bee ID: {bee_id}\nOutcome: {outcome}\n")
        df['bee_id'] = bee_id
        df.to_csv(csv_path, index=False)

    # 3. Read Coordinates & Temporal Info
    x_raw = df['x_mm'].values
    y_raw = df['y_mm'].values
    time_sec = df['time_sec'].values

    if 'frame' in df.columns:
        frames = df['frame'].values
        frame_diffs = np.diff(frames)
    else:
        frame_diffs = np.diff(time_sec) * fps

    step_dists = np.hypot(np.diff(x_raw), np.diff(y_raw))

    # 4. Detect Gaps & Bridge with realistic path segments (>80 frames)
    frame_gap_threshold = 80
    spatial_jump_threshold = 80.0
    jump_mask = (frame_diffs > frame_gap_threshold) | (step_dists > spatial_jump_threshold)
    jump_indices = np.where(jump_mask)[0] + 1

    x_full, y_full, t_full, is_gen_point = [], [], [], []

    for i in range(len(x_raw)):
        if i in jump_indices:
            p1 = (x_raw[i-1], y_raw[i-1])
            p2 = (x_raw[i], y_raw[i])
            t1 = time_sec[i-1]
            t2 = time_sec[i]

            n_pts = int(max(80, frame_diffs[i-1] if i-1 < len(frame_diffs) else 80))
            t_lin = np.linspace(0, 1, n_pts)
            t_bridge = np.linspace(t1, t2, n_pts)

            x_line = (1 - t_lin) * p1[0] + t_lin * p2[0]
            y_line = (1 - t_lin) * p1[1] + t_lin * p2[1]

            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            L = np.hypot(dx, dy)
            if L > 0:
                perp_x, perp_y = -dy / L, dx / L
                envelope = np.sin(np.pi * t_lin)
                wave = 20.0 * envelope * np.sin(2 * np.pi * 2.5 * t_lin)
                xb = x_line + perp_x * wave
                yb = y_line + perp_y * wave
            else:
                xb, yb = x_line, y_line

            x_full.extend(xb[1:])
            y_full.extend(yb[1:])
            t_full.extend(t_bridge[1:])
            is_gen_point.extend([True] * (len(xb) - 1))
        else:
            x_full.append(x_raw[i])
            y_full.append(y_raw[i])
            t_full.append(time_sec[i])
            is_gen_point.append(False)

    x_full = np.array(x_full)
    y_full = np.array(y_full)
    t_full = np.array(t_full)
    is_gen_point = np.array(is_gen_point)

    # 5. Smooth Trajectory
    window_len = 11
    if len(x_full) > window_len:
        x_smooth = savgol_filter(x_full, window_length=window_len, polyorder=3)
        y_smooth = savgol_filter(y_full, window_length=window_len, polyorder=3)
    else:
        x_smooth, y_smooth = x_full, y_full

    # 6. Detect Zone Entry & Exit Events (Chronological relative to Feeder visit)
    r_inner, r_outer = 210.0, 420.0
    r_feeder = 40.0
    r_all = np.hypot(x_smooth, y_smooth)

    # Find feeder visit index (first contact with feeder)
    feeder_candidates = np.where(r_all <= r_feeder)[0]
    if len(feeder_candidates) > 0:
        feeder_idx = feeder_candidates[0]
    else:
        feeder_idx = np.argmin(r_all)

    # 1st Outer Entry: Initial entry point into arena (or start of tracking)
    outer_entry_idx = 0
    for i in range(1, feeder_idx + 1):
        if r_all[i-1] > r_outer and r_all[i] <= r_outer:
            outer_entry_idx = i
            break

    # 1st Inner Entry: Entry into inner circle prior to feeder visit
    inner_entry_idx = None
    for i in range(outer_entry_idx + 1, feeder_idx + 1):
        if r_all[i-1] > r_inner and r_all[i] <= r_inner:
            inner_entry_idx = i
            break
    if inner_entry_idx is None:
        # Fallback to first frame inside inner boundary before feeder visit
        inside_inner_pre = np.where((np.arange(len(r_all)) <= feeder_idx) & (r_all <= r_inner))[0]
        if len(inside_inner_pre) > 0:
            inner_entry_idx = inside_inner_pre[0]
        else:
            inner_entry_idx = feeder_idx

    # 1st Inner Exit: Chronologically the FIRST departure from inner circle after feeder visit
    inner_exit_idx = None
    for i in range(feeder_idx + 1, len(x_smooth)):
        if r_all[i-1] <= r_inner and r_all[i] > r_inner:
            inner_exit_idx = i
            break

    if inner_exit_idx is None:
        post_feeder_crossings = np.where((np.arange(len(r_all)) > feeder_idx) & (r_all > r_inner))[0]
        if len(post_feeder_crossings) > 0:
            inner_exit_idx = post_feeder_crossings[0]
        else:
            inner_exit_idx = len(x_smooth) - 1

    # 1st Outer Exit: First departure from outer circle after feeder visit (or arrival at hive exit)
    outer_exit_idx = None
    for i in range(feeder_idx + 1, len(x_smooth)):
        if r_all[i-1] <= r_outer and r_all[i] > r_outer:
            outer_exit_idx = i
            break

    if outer_exit_idx is None:
        post_feeder_outer = np.where((np.arange(len(r_all)) > feeder_idx) & ((r_all >= r_outer - 10.0) | (y_smooth <= -380.0)))[0]
        if len(post_feeder_outer) > 0:
            outer_exit_idx = post_feeder_outer[-1] if y_smooth[-1] <= -300.0 else post_feeder_outer[0]
        else:
            outer_exit_idx = len(x_smooth) - 1

    def proj_point(px, py, r_target):
        d = np.hypot(px, py)
        if d < 1e-3:
            return (0.0, -r_target)
        return (px * r_target / d, py * r_target / d)

    first_events = {
        'outer_entry': {'type': 'outer_entry', 'pos': proj_point(x_smooth[outer_entry_idx], y_smooth[outer_entry_idx], r_outer)},
        'inner_entry': {'type': 'inner_entry', 'pos': proj_point(x_smooth[inner_entry_idx], y_smooth[inner_entry_idx], r_inner)},
        'inner_exit':  {'type': 'inner_exit',  'pos': proj_point(x_smooth[inner_exit_idx], y_smooth[inner_exit_idx], r_inner)},
        'outer_exit':  {'type': 'outer_exit',  'pos': proj_point(x_smooth[outer_exit_idx], y_smooth[outer_exit_idx], r_outer)},
    }

    # Angles setup
    entry_px, entry_py = first_events['outer_entry']['pos']
    exit_px, exit_py = first_events['outer_exit']['pos']

    # DYNAMIC HIVE LOCATION: matching get_hive_position exactly
    hive_x, hive_y = get_hive_position(df, outer_r=r_outer, hive_dist=450.0)
    hive_pos = (hive_x, hive_y)

    # 7. Crop Outside Arena
    inside_arena_mask = r_all <= r_outer
    x_cropped = np.where(inside_arena_mask, x_smooth, np.nan)
    y_cropped = np.where(inside_arena_mask, y_smooth, np.nan)

    # 8. Build Line Segments
    points = np.array([x_cropped, y_cropped]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    valid_seg_mask = ~np.isnan(segments).any(axis=(1, 2))
    is_gen_seg = is_gen_point[:-1] | is_gen_point[1:]

    raw_indices = np.where(~is_gen_seg & valid_seg_mask)[0]
    gen_indices = np.where(is_gen_seg & valid_seg_mask)[0]

    cmap_two_color = mcolors.LinearSegmentedColormap.from_list("yellow_to_blue", ["#F1C40F", "#2E86C1", "#1B4F72"])
    norm = plt.Normalize(0, t_full.max() if len(t_full) > 0 else 1)

    # 9. Render Plot
    fig, ax = plt.subplots(figsize=(10, 11), dpi=300)

    # Arena Circle Boundaries
    ax.add_patch(plt.Circle((0, 0), r_outer, fill=False, color='black', lw=2.2, zorder=2))
    ax.add_patch(plt.Circle((0, 0), r_inner, fill=False, color='gray', linestyle='--', lw=1.5, zorder=2))

    # PLOT DYNAMIC HIVE LOCATION
    ax.scatter(hive_pos[0], hive_pos[1], color='black', marker='o', s=180, zorder=9, edgecolors='white', linewidth=1.0)
    ax.annotate('Hive', (hive_pos[0], hive_pos[1]), textcoords="offset points",
                xytext=(15, 16) if hive_pos[0] > 200 else (-15, 16) if hive_pos[0] < -200 else (0, -25),
                fontsize=11, fontweight='bold', color='black',
                va='center', ha='left' if hive_pos[0] >= 0 else 'right', zorder=9)

    # Draw Raw Segments
    if len(raw_indices) > 0:
        lc_raw = LineCollection(segments[raw_indices], cmap=cmap_two_color, norm=norm, linewidth=2.0, alpha=0.95, zorder=3)
        lc_raw.set_array(t_full[:-1][raw_indices])
        ax.add_collection(lc_raw)

    # Draw Generated Gap Segments with Exact Gradient Colors
    if len(gen_indices) > 0:
        lc_gen = LineCollection(segments[gen_indices], cmap=cmap_two_color, norm=norm, linestyle='--', linewidth=2.5, alpha=0.9, zorder=4)
        lc_gen.set_array(t_full[:-1][gen_indices])
        ax.add_collection(lc_gen)

    # 10. Event Annotations & Legend
    event_details = {
        'outer_entry': {'name': 'First Outer Entry', 'color': '#27AE60', 'marker': '^'},
        'inner_entry': {'name': 'First Inner Entry', 'color': '#C0392B', 'marker': 'D'},
        'inner_exit':  {'name': 'First Inner Exit',  'color': '#8E44AD', 'marker': 'D'},
        'outer_exit':  {'name': 'First Outer Exit',  'color': '#D35400', 'marker': '^'},
    }
    legend_elements = []

    # Hive Legend Entry
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Hive', markerfacecolor='black', markersize=9))

    for key in ['outer_entry', 'inner_entry', 'inner_exit', 'outer_exit']:
        if key in first_events:
            ev = first_events[key]
            px, py = ev['pos']
            info = event_details[key]
            col, m, name = info['color'], info['marker'], info['name']

            # Dynamic offset calculation based on position relative to center
            ox = 25 if px >= 0 else -135
            oy = 25 if py >= 0 else -25
            if key == 'inner_exit':
                if py > 0:
                    ox, oy = (-135 if px < 0 else 25), 30
                else:
                    ox, oy = (-135 if px < 0 else 25), -30
            elif key == 'inner_entry' and px >= 0 and py < 0:
                ox, oy = 25, -20

            ax.scatter(px, py, color=col, marker=m, s=140, zorder=8, edgecolors='white', linewidth=0.8)
            ax.annotate(f"① {name}", xy=(px, py), xytext=(px + ox, py + oy),
                        fontsize=9.5, fontweight='bold', color=col, zorder=10,
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=col, alpha=0.9))
            legend_elements.append(Line2D([0], [0], marker=m, color='w', label=name, markerfacecolor=col, markersize=10))

    # Legend entries for trajectory line types
    legend_elements.append(Line2D([0], [0], color='#2E86C1', lw=2.0, linestyle='-', label='Tracked Path (Solid Gradient)'))
    legend_elements.append(Line2D([0], [0], color='#2E86C1', lw=2.2, linestyle='--', label='Generated Path / Gap Bridge (Dashed ---)'))

    # Feeder Center Icon
    ax.scatter(0, 0, color='#E67E22', marker='o', s=200, zorder=6)
    ax.text(0, -35, 'Feeder', fontsize=12, fontweight='bold', color='#E67E22', ha='center', zorder=7)

    # Metadata Box
    info_box = f"Bee ID: {bee_id}\nOutcome: {outcome}"
    ax.text(-440, -420, info_box, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.9))

    # Axis Limits & Formatting
    ax.set_xlim(-500, 500)
    ax.set_ylim(-480, 480)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(parse_metadata_from_foldername(folder_name), fontsize=12, fontweight='bold', pad=15)
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.04, 1.0), loc='upper left', frameon=True, fontsize=8.5)

    # Colorbar
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    cbar = fig.colorbar(lc_raw if 'lc_raw' in locals() else lc_gen, cax=cbar_ax, orientation='horizontal')
    cbar_ax.set_xlabel("Time Gradient (seconds): Start (Yellow) → End (Dark Blue)", fontsize=10, fontweight='bold', labelpad=5)

    # Output Plot (Save as both trajectory_generated_paths.png and final_trajectories.png)
    plots_dir = os.path.join(folder_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_path_gen = os.path.join(plots_dir, "trajectory_generated_paths.png")
    output_path_final = os.path.join(plots_dir, "final_trajectories.png")
    
    plt.savefig(output_path_gen, bbox_inches='tight')
    plt.savefig(output_path_final, bbox_inches='tight')
    plt.close(fig)
    print(f" Saved plots to:\n  - {output_path_gen}\n  - {output_path_final}")


def main():
    import sys
    if len(sys.argv) > 1:
        trial_folders = sys.argv[1:]
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(base_dir, "results")
        if not os.path.exists(results_dir):
            results_dir = "."

        trial_folders = [os.path.join(results_dir, d) for d in os.listdir(results_dir)
                         if os.path.isdir(os.path.join(results_dir, d))]

        if not trial_folders:
            trial_folders = ["."]

    for folder in sorted(trial_folders):
        try:
            process_single_trial(folder)
        except Exception as e:
            print(f" Error processing {folder}: {e}")


if __name__ == "__main__":
    main()