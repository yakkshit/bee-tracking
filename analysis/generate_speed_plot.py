"""
Generate Speed of Movement comparison plot for tracked videos:
  - 'results/maries data'
  - 'results/F'

Output saved to:
  - results/paper_plots/fig24_speed_of_movement_analysis.png
  - results/speed_of_movement_summary.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_speed_plots(search_dirs=None):
    output_dir = os.environ.get("OUTPUT_DIR", "results/paper_plots")
    os.makedirs(output_dir, exist_ok=True)
    target_dir = os.environ.get("TARGET_DIR", None)

    if search_dirs is None:
        if target_dir:
            search_dirs = [target_dir]
        else:
            search_dirs = ['results/maries data', 'results/F']
        
    all_instantaneous = []
    session_summaries = []
    
    print("Extracting speed data from tracked videos...")
    
    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            print(f"Warning: Directory {s_dir} does not exist.")
            continue
            
        folder_label = "Marie's Data" if "maries" in s_dir.lower() else "F Dataset"
        
        for session_name in sorted(os.listdir(s_dir)):
            session_path = os.path.join(s_dir, session_name)
            if not os.path.isdir(session_path):
                continue
                
            csv_files = [f for f in os.listdir(session_path) if f.startswith('bee_track_') and f.endswith('.csv')]
            if not csv_files:
                continue
                
            csv_path = os.path.join(session_path, csv_files[0])
            try:
                df = pd.read_csv(csv_path)
                if 'x_mm' not in df.columns or 'y_mm' not in df.columns or 'time_sec' not in df.columns:
                    continue
                    
                df = df.dropna(subset=['x_mm', 'y_mm', 'time_sec']).sort_values('time_sec')
                df['dist_center_mm'] = np.hypot(df['x_mm'], df['y_mm'])
                
                # Calculate instantaneous speed (mm/s)
                dt = df['time_sec'].diff()
                dx = df['x_mm'].diff()
                dy = df['y_mm'].diff()
                dist_mm = np.hypot(dx, dy)
                speed_mm_s = dist_mm / dt
                
                df['speed_mm_s'] = speed_mm_s
                
                # Filter valid points (within arena R <= 420mm, realistic speed <= 1200 mm/s, dt > 0)
                valid = df[(dt > 0) & (df['speed_mm_s'].notna()) & (df['speed_mm_s'] >= 0) & 
                           (df['speed_mm_s'] <= 1200) & (df['dist_center_mm'] <= 420)].copy()
                           
                if len(valid) < 10:
                    continue
                    
                # Classify radial arena zones
                valid['zone'] = pd.cut(
                    valid['dist_center_mm'],
                    bins=[-1, 70, 210, 420],
                    labels=['Feeder (0-70mm)', 'Inner (70-210mm)', 'Outer (210-420mm)']
                )
                
                # Experimental condition classification
                orient = str(df.iloc[0].get('orientation', 'LR')).strip().upper()
                if orient not in ['LR', 'TB']:
                    orient = 'TB' if 'TB' in session_name else 'LR'
                    
                stim = str(df.iloc[0].get('stimulus', '')).lower()
                sess_lower = session_name.lower()
                
                is_weak = ('u8' in stim or 'u8' in sess_lower or 'u2' in stim or 'u2' in sess_lower or
                           'weak' in stim or 'p0.' in stim or 'p0_' in sess_lower or
                           'p0.1' in sess_lower or 'p0.3' in sess_lower or 'p0.4' in sess_lower)
                is_train = ('train' in stim or 'train' in sess_lower or 'p8.0 u0.0' in stim or 'p0.0 u0.0' in stim)
                
                if is_train:
                    cond = 'Training'
                elif is_weak:
                    cond = 'Weak LR' if orient == 'LR' else 'Weak TB'
                else:
                    cond = 'Strong LR' if orient == 'LR' else 'Strong TB'
                    
                valid['folder'] = folder_label
                valid['session_name'] = session_name
                valid['condition'] = cond
                
                all_instantaneous.append(valid[['folder', 'session_name', 'condition', 'time_sec', 'dist_center_mm', 'speed_mm_s', 'zone']])
                
                session_summaries.append({
                    'folder': folder_label,
                    'session_name': session_name,
                    'condition': cond,
                    'mean_speed_mm_s': valid['speed_mm_s'].mean(),
                    'median_speed_mm_s': valid['speed_mm_s'].median(),
                    'max_speed_mm_s': valid['speed_mm_s'].quantile(0.99),
                    'total_dist_meters': dist_mm.sum() / 1000.0,
                    'duration_sec': df['time_sec'].iloc[-1] - df['time_sec'].iloc[0]
                })
            except Exception as e:
                print(f"Skipping {session_name}: {e}")
                
    full_df = pd.concat(all_instantaneous, ignore_index=True)
    stats_df = pd.DataFrame(session_summaries)
    
    # Save CSV summary
    csv_output = os.path.join(output_dir, "speed_of_movement_summary.csv")
    stats_df.to_csv(csv_output, index=False)
    print(f"Saved speed summary table to {csv_output}")
    
    # ── Create 4-Subplot Speed of Movement Figure ────────────────────────────
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    
    colors = {"Marie's Data": "#2E7D32", "F Dataset": "#1976D2"}
    cond_order = ["Training", "Strong LR", "Weak LR", "Strong TB", "Weak TB"]

    # 1. Subplot A: Overall Speed Distribution KDE
    ax_a = axes[0, 0]
    for label, color in colors.items():
        data_s = full_df[full_df['folder'] == label]['speed_mm_s'].values
        data_s = data_s[(data_s >= 0) & (data_s <= 800)]
        if len(data_s) > 2:
            kde = gaussian_kde(data_s)
            x_grid = np.linspace(0, 800, 200)
            ax_a.plot(x_grid, kde(x_grid), color=color, lw=2.2, label=f"{label} Distribution Density")
            ax_a.fill_between(x_grid, kde(x_grid), color=color, alpha=0.25)
            
            med_val = np.median(data_s)
            ax_a.axvline(med_val, color=color, linestyle="--", linewidth=1.5, label=f"{label} Median Line ({med_val:.1f} mm/s)")

    ax_a.set_title("A) Instantaneous Speed Distribution Density", fontsize=11, fontweight="bold", pad=8)
    ax_a.set_xlabel("Instantaneous Speed (mm/s)", fontsize=10, fontweight="bold")
    ax_a.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
    ax_a.set_xlim(0, 800)
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(fontsize=8.5, loc="upper right")

    # 2. Subplot B: Mean Speed per Session Boxplot across Experimental Conditions
    ax_b = axes[0, 1]
    datasets = ["Marie's Data", "F Dataset"]
    b_conds = ["Strong LR", "Weak LR", "Strong TB", "Weak TB"]
    x_b = np.arange(len(b_conds))
    width = 0.35

    for idx, ds in enumerate(datasets):
        box_data = []
        for c in b_conds:
            vals = stats_df[(stats_df['folder'] == ds) & (stats_df['condition'] == c)]['mean_speed_mm_s'].values
            box_data.append(vals if len(vals) > 0 else np.array([0]))
            
        positions = x_b + (idx - 0.5) * width
        bpb = ax_b.boxplot(box_data, positions=positions, widths=0.28, patch_artist=True, showfliers=False,
                           showmeans=True, meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=5))
        for patch in bpb['boxes']:
            patch.set_facecolor(colors[ds])
            patch.set_alpha(0.7)

        # Scatter individual session points
        for c_idx, c in enumerate(b_conds):
            y_vals = stats_df[(stats_df['folder'] == ds) & (stats_df['condition'] == c)]['mean_speed_mm_s'].values
            if len(y_vals) > 0:
                pos_x = positions[c_idx]
                x_vals = np.random.normal(pos_x, 0.03, size=len(y_vals))
                ax_b.scatter(x_vals, y_vals, color='black', alpha=0.5, s=18, zorder=5)

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(b_conds, fontsize=9, fontweight="bold")
    ax_b.set_title("B) Mean Walking Speed per Session by Condition", fontsize=11, fontweight="bold", pad=8)
    ax_b.set_xlabel("Polarization Condition", fontsize=10, fontweight="bold")
    ax_b.set_ylabel("Session Mean Speed (mm/s)", fontsize=10, fontweight="bold")
    ax_b.set_ylim(0, 310)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    
    # Legend for B
    from matplotlib.lines import Line2D
    legend_b_elements = [
        Line2D([0], [0], color='#2E7D32', lw=6, alpha=0.7, label="Marie's Data (IQR Box)"),
        Line2D([0], [0], color='#1976D2', lw=6, alpha=0.7, label="F Dataset (IQR Box)"),
        Line2D([0], [0], color='orange', lw=1.5, label='Median Line (Center Bar Line)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='black', markersize=6, label='Mean Point Indicator'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=5, label='Individual Session Means')
    ]
    ax_b.legend(handles=legend_b_elements, fontsize=7.5, loc="upper right")

    # 3. Subplot C: Arena Zone Speed Comparison
    ax_c = axes[1, 0]
    zones = ['Feeder (0-70mm)', 'Inner (70-210mm)', 'Outer (210-420mm)']
    zone_labels = ['Feeder\n(0-70mm)', 'Inner Arena\n(70-210mm)', 'Outer Arena\n(210-420mm)']
    x_indices = np.arange(len(zones))
    width = 0.35
    
    for idx, ds in enumerate(datasets):
        ds_df = full_df[full_df['folder'] == ds]
        zone_data = [ds_df[ds_df['zone'] == z]['speed_mm_s'].dropna().values for z in zones]
        positions = x_indices + (idx - 0.5) * width
        bpc = ax_c.boxplot(zone_data, positions=positions, widths=0.28, patch_artist=True, showfliers=False)
        for patch in bpc['boxes']:
            patch.set_facecolor(colors[ds])
            patch.set_alpha(0.7)
            
    ax_c.set_xticks(x_indices)
    ax_c.set_xticklabels(zone_labels, fontsize=9, fontweight="bold")
    ax_c.set_title("C) Movement Speed Across Arena Zones", fontsize=11, fontweight="bold", pad=8)
    ax_c.set_xlabel("Arena Zone (Radial Distance from Feeder)", fontsize=10, fontweight="bold")
    ax_c.set_ylabel("Speed (mm/s)", fontsize=10, fontweight="bold")
    ax_c.set_ylim(0, 450)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    
    # Detailed element legend for C describing bar lines
    legend_c_elements = [
        Line2D([0], [0], color='#2E7D32', lw=6, alpha=0.7, label="Marie's Data (IQR Box)"),
        Line2D([0], [0], color='#1976D2', lw=6, alpha=0.7, label="F Dataset (IQR Box)"),
        Line2D([0], [0], color='orange', lw=1.5, label='Median Line (Bar Center)'),
        Line2D([0], [0], color='black', lw=1.0, label='Whisker Extremes (Min/Max)')
    ]
    ax_c.legend(handles=legend_c_elements, fontsize=8.0, loc="upper left")

    # 4. Subplot D: Condition Breakdown
    ax_d = axes[1, 1]
    avail_conds = [c for c in cond_order if c in stats_df['condition'].unique()]
    x_c = np.arange(len(avail_conds))
    
    for idx, ds in enumerate(datasets):
        means = []
        stds = []
        for c in avail_conds:
            c_vals = stats_df[(stats_df['folder'] == ds) & (stats_df['condition'] == c)]['mean_speed_mm_s'].values
            means.append(np.mean(c_vals) if len(c_vals) > 0 else 0)
            stds.append(np.std(c_vals) if len(c_vals) > 0 else 0)
            
        pos = x_c + (idx - 0.5) * width
        ax_d.bar(pos, means, yerr=stds, width=width, color=colors[ds], alpha=0.75, label=ds, capsize=4)

    ax_d.set_xticks(x_c)
    ax_d.set_xticklabels(avail_conds, rotation=15, fontsize=9, fontweight="bold")
    ax_d.set_title("D) Mean Speed by Polarization Condition & Dataset", fontsize=11, fontweight="bold", pad=8)
    ax_d.set_xlabel("Experimental Condition", fontsize=10, fontweight="bold")
    ax_d.set_ylabel("Mean Speed (mm/s)", fontsize=10, fontweight="bold")
    ax_d.grid(True, linestyle=":", alpha=0.6)
    
    legend_d_elements = [
        Line2D([0], [0], color='#2E7D32', lw=6, alpha=0.75, label="Marie's Data Mean Bar"),
        Line2D([0], [0], color='#1976D2', lw=6, alpha=0.75, label="F Dataset Mean Bar"),
        Line2D([0], [0], color='black', lw=1.5, label='Error Bar Line (Standard Deviation)')
    ]
    ax_d.legend(handles=legend_d_elements, fontsize=8.0, loc="upper right")

    plt.suptitle("Speed of Movement Analysis across Tracked Datasets ('results/maries data' vs 'results/F')", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(output_dir, "fig24_speed_of_movement_analysis.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 24: {output_path}")

    # ── Export individual standalone subplots for Figure 24 ────────────────
    # Subplot A
    fig_a, ax_sa = plt.subplots(figsize=(7, 5), facecolor='white')
    for label, color in colors.items():
        data_s = full_df[full_df['folder'] == label]['speed_mm_s'].values
        data_s = data_s[(data_s >= 0) & (data_s <= 800)]
        if len(data_s) > 2:
            kde = gaussian_kde(data_s)
            x_grid = np.linspace(0, 800, 200)
            ax_sa.plot(x_grid, kde(x_grid), color=color, lw=2.2, label=f"{label} Distribution Density")
            ax_sa.fill_between(x_grid, kde(x_grid), color=color, alpha=0.25)
            med_val = np.median(data_s)
            ax_sa.axvline(med_val, color=color, linestyle="--", linewidth=1.5, label=f"{label} Median Line ({med_val:.1f} mm/s)")
    ax_sa.set_title("A) Instantaneous Speed Distribution Density", fontsize=11, fontweight="bold", pad=8)
    ax_sa.set_xlabel("Instantaneous Speed (mm/s)", fontsize=10, fontweight="bold")
    ax_sa.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
    ax_sa.set_xlim(0, 800)
    ax_sa.grid(True, linestyle=":", alpha=0.6)
    ax_sa.legend(fontsize=8.5, loc="upper right")
    fig_a.tight_layout()
    fig_a.savefig(os.path.join(output_dir, "fig24a_speed_kde_density.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_a)

    # Subplot B
    fig_b, ax_sb = plt.subplots(figsize=(7, 5), facecolor='white')
    for idx, ds in enumerate(datasets):
        box_data = []
        for c in b_conds:
            vals = stats_df[(stats_df['folder'] == ds) & (stats_df['condition'] == c)]['mean_speed_mm_s'].values
            box_data.append(vals if len(vals) > 0 else np.array([0]))
        positions = x_b + (idx - 0.5) * width
        bpb = ax_sb.boxplot(box_data, positions=positions, widths=0.28, patch_artist=True, showfliers=False,
                           showmeans=True, meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=5))
        for patch in bpb['boxes']:
            patch.set_facecolor(colors[ds])
            patch.set_alpha(0.7)
        for c_idx, c in enumerate(b_conds):
            y_vals = stats_df[(stats_df['folder'] == ds) & (stats_df['condition'] == c)]['mean_speed_mm_s'].values
            if len(y_vals) > 0:
                pos_x = positions[c_idx]
                x_vals = np.random.normal(pos_x, 0.03, size=len(y_vals))
                ax_sb.scatter(x_vals, y_vals, color='black', alpha=0.5, s=18, zorder=5)
    ax_sb.set_xticks(x_b)
    ax_sb.set_xticklabels(b_conds, fontsize=9, fontweight="bold")
    ax_sb.set_title("B) Mean Walking Speed per Session by Condition", fontsize=11, fontweight="bold", pad=8)
    ax_sb.set_xlabel("Polarization Condition", fontsize=10, fontweight="bold")
    ax_sb.set_ylabel("Session Mean Speed (mm/s)", fontsize=10, fontweight="bold")
    ax_sb.set_ylim(0, 310)
    ax_sb.grid(True, linestyle=":", alpha=0.6)
    ax_sb.legend(handles=legend_b_elements, fontsize=7.5, loc="upper right")
    fig_b.tight_layout()
    fig_b.savefig(os.path.join(output_dir, "fig24b_mean_speed_by_condition.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_b)

    # Subplot C
    fig_c, ax_sc = plt.subplots(figsize=(7, 5), facecolor='white')
    for idx, ds in enumerate(datasets):
        ds_df = full_df[full_df['folder'] == ds]
        zone_data = [ds_df[ds_df['zone'] == z]['speed_mm_s'].dropna().values for z in zones]
        positions = x_indices + (idx - 0.5) * width
        bpc = ax_sc.boxplot(zone_data, positions=positions, widths=0.28, patch_artist=True, showfliers=False)
        for patch in bpc['boxes']:
            patch.set_facecolor(colors[ds])
            patch.set_alpha(0.7)
    ax_sc.set_xticks(x_indices)
    ax_sc.set_xticklabels(zone_labels, fontsize=9, fontweight="bold")
    ax_sc.set_title("C) Movement Speed Across Arena Zones", fontsize=11, fontweight="bold", pad=8)
    ax_sc.set_xlabel("Arena Zone (Radial Distance from Feeder)", fontsize=10, fontweight="bold")
    ax_sc.set_ylabel("Speed (mm/s)", fontsize=10, fontweight="bold")
    ax_sc.set_ylim(0, 450)
    ax_sc.grid(True, linestyle=":", alpha=0.6)
    ax_sc.legend(handles=legend_c_elements, fontsize=8.0, loc="upper left")
    fig_c.tight_layout()
    fig_c.savefig(os.path.join(output_dir, "fig24c_speed_across_zones.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_c)

    # Subplot D
    fig_d, ax_sd = plt.subplots(figsize=(7, 5), facecolor='white')
    for idx, ds in enumerate(datasets):
        means = []
        stds = []
        for c in avail_conds:
            c_vals = stats_df[(stats_df['folder'] == ds) & (stats_df['condition'] == c)]['mean_speed_mm_s'].values
            means.append(np.mean(c_vals) if len(c_vals) > 0 else 0)
            stds.append(np.std(c_vals) if len(c_vals) > 0 else 0)
        pos = x_c + (idx - 0.5) * width
        ax_sd.bar(pos, means, yerr=stds, width=width, color=colors[ds], alpha=0.75, label=ds, capsize=4)
    ax_sd.set_xticks(x_c)
    ax_sd.set_xticklabels(avail_conds, rotation=15, fontsize=9, fontweight="bold")
    ax_sd.set_title("D) Mean Speed by Polarization Condition & Dataset", fontsize=11, fontweight="bold", pad=8)
    ax_sd.set_xlabel("Experimental Condition", fontsize=10, fontweight="bold")
    ax_sd.set_ylabel("Mean Speed (mm/s)", fontsize=10, fontweight="bold")
    ax_sd.grid(True, linestyle=":", alpha=0.6)
    ax_sd.legend(handles=legend_d_elements, fontsize=8.0, loc="upper right")
    fig_d.tight_layout()
    fig_d.savefig(os.path.join(output_dir, "fig24d_mean_speed_bar_chart.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_d)

if __name__ == "__main__":
    generate_speed_plots()
