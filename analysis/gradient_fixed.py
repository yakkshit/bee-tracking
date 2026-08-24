import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.signal import savgol_filter


def parse_metadata_from_foldername(folder_name):
    cue = "LR" if any(x in folder_name for x in [".LR", "_LR", " LR "]) else "TB"
    
    p_match = re.search(r'P[._\s]?(\d+(?:\.\d+)?)', folder_name, re.IGNORECASE)
    p_val = f"p{p_match.group(1)}" if p_match else "p?"
    
    u_match = re.search(r'U[._\s]?(\d+(?:\.\d+)?)', folder_name, re.IGNORECASE)
    u_val = f"u{int(float(u_match.group(1)))}" if u_match else "u?"
    
    return f"Center: polarizer offset 0° ({cue} | {p_val} {u_val})"


def extract_bee_id(folder_path, df=None):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    m = re.search(r'(\d+[wWgGbByYoOrRpP])', folder_name)
    if m:
        return m.group(1).lower()

    if df is not None and not df.empty and 'bee_id' in df.columns:
        bid = str(df['bee_id'].iloc[0]).strip()
        if bid and bid.lower() not in ("unknown", "nan", ""):
            return bid

    return "unknown"


def process_single_trial(folder_path, fps=60):
    folder_name = os.path.basename(folder_path)
    print(f"\nProcessing smooth gradient for folder: {folder_name}")

    # 1. Locate CSV data file
    csv_files = glob.glob(os.path.join(folder_path, "bee_track_*.csv"))
    if not csv_files:
        print(f" Skipping {folder_name}: No bee_track_*.csv file found.")
        return

    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)

    # 2. Extract Metadata
    bee_id = extract_bee_id(folder_path, df)

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

    # 4. Detect Gaps & Bridge (>80 frames)
    frame_gap_threshold = 80
    spatial_jump_threshold = 80.0
    jump_mask = (frame_diffs > frame_gap_threshold) | (step_dists > spatial_jump_threshold)
    jump_indices = np.where(jump_mask)[0] + 1

    x_full, y_full, t_full = [], [], []

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
        else:
            x_full.append(x_raw[i])
            y_full.append(y_raw[i])
            t_full.append(time_sec[i])

    x_full = np.array(x_full)
    y_full = np.array(y_full)
    t_full = np.array(t_full)

    # 5. Smooth Trajectory
    window_len = 11
    if len(x_full) > window_len:
        x_smooth = savgol_filter(x_full, window_length=window_len, polyorder=3)
        y_smooth = savgol_filter(y_full, window_length=window_len, polyorder=3)
    else:
        x_smooth, y_smooth = x_full, y_full

    # 6. Normalize coordinates for arena boundary (Outer = 420mm)
    r_outer = 420.0
    r_inner = 210.0
    r_feeder = 40.0

    # Crop points inside outer circle
    r_all = np.hypot(x_smooth, y_smooth)
    inside_mask = r_all <= r_outer

    x_crop = x_smooth[inside_mask]
    y_crop = y_smooth[inside_mask]
    t_crop = t_full[inside_mask]

    # Map time for colormap (0 to 1)
    if len(t_crop) > 1:
        t_norm = (t_crop - t_crop.min()) / (t_crop.max() - t_crop.min() + 1e-8)
    else:
        t_norm = np.zeros_like(t_crop)

    # Build line segments for LineCollection
    points = np.array([x_crop, y_crop]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 7. Render Minimalist Gradient Plot matching reference screenshot C
    fig, ax = plt.subplots(figsize=(7, 8.5), dpi=300)

    # Outer gray arena ring (background fill circle: outer 460mm gray, inner 420mm white)
    outer_arena_bg = plt.Circle((0, 0), r_outer + 50, color='#EAEAEA', zorder=1)
    inner_arena_white = plt.Circle((0, 0), r_outer, color='white', ec='black', lw=1.5, zorder=2)
    outer_boundary_line = plt.Circle((0, 0), r_outer + 50, color='none', ec='black', lw=1.5, zorder=2)

    # Inner feeder center circle
    inner_feeder_circle = plt.Circle((0, 0), r_feeder, color='white', ec='black', lw=1.2, zorder=4)

    ax.add_patch(outer_arena_bg)
    ax.add_patch(inner_arena_white)
    ax.add_patch(outer_boundary_line)
    ax.add_patch(inner_feeder_circle)

    # Target orientation orientation dots (top and bottom)
    ax.plot([0, 0], [r_outer + 25, -(r_outer + 25)], 'o', mfc='white', mec='black', ms=6, mew=1.2, zorder=5)

    # LineCollection with turbo_r colormap (Yellow -> Teal -> Blue)
    lc = LineCollection(segments, cmap='turbo_r', zorder=3)
    lc.set_array(t_norm[:-1])
    lc.set_linewidth(2.2)
    ax.add_collection(lc)

    # Axes limits & formatting
    limit = r_outer + 70
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axis('off')
    ax.set_aspect('equal')

    # Subtitle text matching figure style
    ax.text(0, -limit - 30, f"Center: polarizer offset 0°\n(Bee ID: {bee_id})", fontsize=14, fontweight='bold', ha='center', va='top')
    ax.text(0, -limit - 90, "Homeward\nPath", fontsize=14, fontweight='bold', ha='center', va='top')

    # Colorbar matching reference screenshot
    cbar_ax = fig.add_axes([0.25, 0.12, 0.25, 0.015])
    cbar = fig.colorbar(lc, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([])
    cbar_ax.text(0.5, -1.8, "Time", fontsize=10, ha='center', va='top', transform=cbar_ax.transAxes)
    cbar_ax.annotate('', xy=(0.85, -1.2), xytext=(0.15, -1.2),
                      xycoords='axes fraction', textcoords='axes fraction',
                      arrowprops=dict(arrowstyle="->", lw=1.2, color='black'))

    # Save to plots directory
    plots_dir = os.path.join(folder_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, "gradient_fixed_trajectory.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f" Saved smooth gradient plot to: {output_path}")


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
