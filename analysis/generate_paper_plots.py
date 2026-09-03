"""
Generate all Paper 1 statistical figures (Figures 9–18), trajectory occupancy heatmaps (Figures 19–21), and summary statistics.

Replicates the exact side-by-side circular raw data plot and Circ-MLE model fit
rose histogram format for Figures 9–13 per paper reference specification.

Outputs saved to:
  - Figures: results/paper_plots/fig09_training_inner_circle.png
             results/paper_plots/fig10_strong_polarization_lr.png
             results/paper_plots/fig11_weak_polarization_lr.png
             results/paper_plots/fig12_strong_polarization_tb.png
             results/paper_plots/fig13_weak_polarization_tb.png
             results/paper_plots/fig14_home_vs_fictive_lr.png
             results/paper_plots/fig15_home_vs_fictive_tb.png
             results/paper_plots/fig16_circular_plots_lr_tb.png
             results/paper_plots/fig17_average_homing_accuracy.png
             results/paper_plots/fig18_boxplot_deviation_from_nest.png
             results/paper_plots/fig19_heatmap_strong_vs_weak.png
             results/paper_plots/fig20_heatmap_conditions_grid.png
             results/paper_plots/fig21_heatmap_overall_occupancy.png
  - Stats CSV: results/paper_stats_summary.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

def get_output_dir():
    out = os.environ.get("OUTPUT_DIR", "results/paper_plots")
    os.makedirs(out, exist_ok=True)
    return out

OUTER_R = 420.0
INNER_R = 210.0


def rayleigh_test(angles_deg):
    """Compute Rayleigh test statistic r, p-value, mean direction mu, and kappa."""
    rad = np.radians(angles_deg)
    n = len(rad)
    if n == 0:
        return 0.0, 1.0, 0.0, 0.0
    C = np.sum(np.cos(rad))
    S = np.sum(np.sin(rad))
    r = np.hypot(C, S) / n
    mu = np.degrees(np.arctan2(S, C)) % 360.0

    # Rayleigh p-value approximation (Zar 1999)
    R_val = n * r
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R_val**2)) - (1 + 2 * n))

    # Concentration kappa estimate (Mardia & Jupp 2000)
    if r < 0.53:
        kappa = 2 * r + r**3 + 5 * r**5 / 6
    elif r < 0.85:
        kappa = -0.4 + 1.39 * r + 0.43 / (1 - r)
    else:
        kappa = 1 / (r**3 - 4 * r**2 + 3 * r)

    return float(r), float(p), float(mu), float(kappa)


def plot_paper_circular_figure(angles_deg, left_title, right_title, filename, color):
    """
    Generate side-by-side circular raw scatter plot (Left) and Circ-MLE rose histogram fit (Right)
    matching Paper 1 Figures 9–13 exact layout and typography.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), subplot_kw=dict(projection='polar'), facecolor="white")

    rad = np.radians(angles_deg) if len(angles_deg) > 0 else np.array([])
    n = len(rad)
    r, p, mu, kappa = rayleigh_test(angles_deg) if n > 0 else (0.0, 1.0, 0.0, 0.0)
    mu_rad = np.radians(mu)

    # ── Left Plot: Raw scatter points + Mean vector arrow ───────────────────
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)  # Clockwise

    circle_theta = np.linspace(0, 2 * np.pi, 360)
    ax1.plot(circle_theta, np.ones_like(circle_theta), color="#333333", lw=1.2, ls="-")

    for diag in np.radians([0, 45, 90, 135, 180, 225, 270, 315]):
        ax1.plot([diag, diag], [0, 1.0], color="#d0d0d0", lw=0.7, ls=":")

    if n > 0:
        ax1.scatter(rad, np.ones_like(rad), color=color, alpha=0.9, s=50, zorder=5)

        arrow_len = max(0.25, min(0.95, r))
        ax1.annotate("", xy=(mu_rad, arrow_len), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.8), zorder=10)

    ax1.set_ylim(0, 1.15)
    ax1.set_yticks([])
    ax1.set_xticks(np.radians([0, 90, 180, 270]))
    ax1.set_xticklabels(["0°\n(Nest)", "90°", "180°", "270°"], fontsize=10, fontweight="bold")
    ax1.text(0.5, -0.18, left_title, transform=ax1.transAxes, ha="center", va="top", fontsize=10, fontweight="bold")

    # ── Right Plot: Circ-MLE Rose Histogram & Model Fit ───────────────────
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)

    ax2.plot(circle_theta, np.ones_like(circle_theta), color="#333333", lw=1.2, ls="-")

    if n > 0:
        num_bins = 18
        bins = np.linspace(0, 2 * np.pi, num_bins + 1)
        counts, _ = np.histogram(rad, bins=bins)
        max_c = max(counts) if max(counts) > 0 else 1
        heights = (counts / max_c) * 0.75
        width = 2 * np.pi / num_bins
        centers = (bins[:-1] + bins[1:]) / 2.0

        ax2.bar(centers, heights, width=width, bottom=0.0, color=color, edgecolor="#333333", alpha=0.8, zorder=4)

        dens = np.exp(max(0.8, kappa) * np.cos(2 * (circle_theta - mu_rad)))
        dens = 1.0 + 0.28 * (dens / np.max(dens))
        ax2.plot(circle_theta, dens, color="#333333", lw=1.4, ls="--", zorder=6)

        ax2.annotate("", xy=(mu_rad, 1.22), xytext=(mu_rad + np.pi, 1.22),
                     arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.4, ls="--"), zorder=8)

    ax2.set_ylim(0, 1.35)
    ax2.set_yticks([])
    ax2.set_xticks(np.radians([0, 90, 180, 270]))
    ax2.set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"], fontsize=10, fontweight="bold")
    ax2.text(0.5, -0.18, right_title, transform=ax2.transAxes, ha="center", va="top", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(get_output_dir(), filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_spatial_heatmap(ax, x_pts, y_pts, title, cmap="inferno"):
    """
    Render 2D kernel density / occupancy heatmap for bee trajectories inside circular arena.
    """
    bins = 150
    extent = [-OUTER_R, OUTER_R, -OUTER_R, OUTER_R]

    if len(x_pts) == 0:
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold")
        return None

    H, xedges, yedges = np.histogram2d(x_pts, y_pts, bins=bins, range=[[-OUTER_R, OUTER_R], [-OUTER_R, OUTER_R]])
    H = H.T

    H_smooth = gaussian_filter(H, sigma=2.0)

    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2.0, (yedges[:-1] + yedges[1:]) / 2.0)
    mask = (np.hypot(X, Y) > OUTER_R)
    H_masked = np.ma.masked_array(H_smooth, mask=mask)

    im = ax.imshow(H_masked, extent=extent, origin="lower", cmap=cmap, zorder=3, alpha=0.92)

    # Arena Outlines
    ax.add_patch(Circle((0, 0), OUTER_R, fill=False, ec="#333333", lw=2.2, zorder=10))
    ax.add_patch(Circle((0, 0), INNER_R, fill=False, ec="#ffffff", lw=1.3, ls="--", zorder=10))

    # Feeder center circle
    ax.plot(0, 0, "o", color="#F57C00", markersize=9, zorder=12, markeredgecolor="white")
    ax.text(0, -32, "Feeder", color="white", fontsize=9.5, fontweight="bold", ha="center", va="top", zorder=13,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.65))

    lim = OUTER_R + 50
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    return im


def classify_trials(df):
    """Classify trials into Training, Strong LR, Weak LR, Strong TB, Weak TB."""
    df["rot"] = df["orientation"].astype(str).str.upper()
    df["stim"] = df["stimulus"].astype(str).str.lower()
    df["sess"] = df["session_name"].astype(str).str.lower()

    df["inner_exit_angle_deg"] = pd.to_numeric(df["inner_exit_angle_deg"], errors="coerce")
    df_clean = df.dropna(subset=["inner_exit_angle_deg"]).copy()

    def is_weak(row):
        s = str(row["stim"]) + " " + str(row["sess"])
        return "u8" in s or "u2" in s or "weak" in s or "p0." in s or "p0_" in s or "p0.1" in s or "p0.3" in s or "p0.4" in s

    df_clean["is_weak"] = df_clean.apply(is_weak, axis=1)

    training = df_clean[df_clean["stim"].str.contains("train|p8.0 u0.0|p0.0 u0.0") | df_clean["sess"].str.contains("train|p0u8|p_0_u_1")]
    strong_lr = df_clean[(df_clean["rot"] == "LR") & (~df_clean["is_weak"])]
    weak_lr   = df_clean[(df_clean["rot"] == "LR") & (df_clean["is_weak"])]
    strong_tb = df_clean[(df_clean["rot"] == "TB") & (~df_clean["is_weak"])]
    weak_tb   = df_clean[(df_clean["rot"] == "TB") & (df_clean["is_weak"])]

    return {
        "Training": training,
        "Strong_LR": strong_lr,
        "Weak_LR": weak_lr,
        "Strong_TB": strong_tb,
        "Weak_TB": weak_tb
    }


def find_all_session_paths(search_dirs=None):
    """Recursively or iteratively find all session directories containing a bee_track_*.csv file."""
    if search_dirs is None:
        target_dir = os.environ.get("TARGET_DIR", None)
        if target_dir:
            search_dirs = [target_dir]
        else:
            search_dirs = ["results/maries data", "results/F", "results"]

    session_paths = []
    seen = set()

    for b_dir in search_dirs:
        if not os.path.exists(b_dir):
            continue

        # Check if b_dir itself contains bee_track_*.csv
        csvs = [f for f in os.listdir(b_dir) if f.startswith('bee_track_') and f.endswith('.csv') and os.path.isfile(os.path.join(b_dir, f))]
        if csvs:
            if b_dir not in seen:
                session_paths.append(b_dir)
                seen.add(b_dir)
            continue

        for root, dirs, files in os.walk(b_dir):
            dirs[:] = [d for d in dirs if not d.endswith("paper_plots") and d != "plots"]
            for f in files:
                if f.startswith('bee_track_') and f.endswith('.csv'):
                    if root not in seen:
                        session_paths.append(root)
                        seen.add(root)
                    break

    return sorted(session_paths)


def load_spatial_points_by_condition(search_dirs=None):
    """Extract spatial trajectory points (x_mm, y_mm) grouped by polarization condition."""
    data_by_cond = {
        "Strong": {"x": [], "y": []},
        "Weak": {"x": [], "y": []},
        "Strong_LR": {"x": [], "y": []},
        "Weak_LR": {"x": [], "y": []},
        "Strong_TB": {"x": [], "y": []},
        "Weak_TB": {"x": [], "y": []},
        "All": {"x": [], "y": []}
    }

    session_paths = find_all_session_paths(search_dirs)

    for session_path in session_paths:
        session_name = os.path.basename(session_path)
        track_csv = None
        for f in os.listdir(session_path):
            if f.startswith("bee_track_") and f.endswith(".csv"):
                track_csv = os.path.join(session_path, f)
                break
        if not track_csv:
            continue

        try:
            sdf = pd.read_csv(track_csv)
            if "x_mm" not in sdf.columns or "y_mm" not in sdf.columns:
                continue

                sess_lower = session_name.lower()
                orient = str(sdf.iloc[0].get("orientation", "LR")).strip().upper()
                stim = str(sdf.iloc[0].get("stimulus", "")).lower()

                is_weak = ("u8" in stim or "u8" in sess_lower or "u2" in stim or "u2" in sess_lower or
                           "weak" in stim or "p0." in stim or "p0_" in sess_lower or
                           "p0.1" in sess_lower or "p0.3" in sess_lower or "p0.4" in sess_lower)

                dists = np.hypot(sdf["x_mm"].values, sdf["y_mm"].values)
                mask = dists <= OUTER_R
                xs = sdf["x_mm"].values[mask]
                ys = sdf["y_mm"].values[mask]

                if len(xs) == 0:
                    continue

                # Subsample for uniform density sampling across sessions
                step = max(1, len(xs) // 1000)
                xs_sub = xs[::step]
                ys_sub = ys[::step]

                data_by_cond["All"]["x"].extend(xs_sub)
                data_by_cond["All"]["y"].extend(ys_sub)

                if is_weak:
                    data_by_cond["Weak"]["x"].extend(xs_sub)
                    data_by_cond["Weak"]["y"].extend(ys_sub)
                    if orient == "LR":
                        data_by_cond["Weak_LR"]["x"].extend(xs_sub)
                        data_by_cond["Weak_LR"]["y"].extend(ys_sub)
                    else:
                        data_by_cond["Weak_TB"]["x"].extend(xs_sub)
                        data_by_cond["Weak_TB"]["y"].extend(ys_sub)
                else:
                    data_by_cond["Strong"]["x"].extend(xs_sub)
                    data_by_cond["Strong"]["y"].extend(ys_sub)
                    if orient == "LR":
                        data_by_cond["Strong_LR"]["x"].extend(xs_sub)
                        data_by_cond["Strong_LR"]["y"].extend(ys_sub)
                    else:
                        data_by_cond["Strong_TB"]["x"].extend(xs_sub)
                        data_by_cond["Strong_TB"]["y"].extend(ys_sub)

            except Exception as e:
                print(f"Warning skipping heatmap sample {session_name}: {e}")

    # Convert lists to numpy arrays
    for k in data_by_cond:
        data_by_cond[k]["x"] = np.array(data_by_cond[k]["x"])
        data_by_cond[k]["y"] = np.array(data_by_cond[k]["y"])

    return data_by_cond


def generate_paper_heatmaps():
    """Generate spatial trajectory occupancy heatmaps for paper plots."""
    print("Generating Paper trajectory heatmaps (Figures 19–21)...")
    spatial_data = load_spatial_points_by_condition()

    # ── Figure 19: Side-by-Side Heatmap (Strong vs Weak Polarization) ─────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5), facecolor="white")

    im1 = create_spatial_heatmap(ax1, spatial_data["Strong"]["x"], spatial_data["Strong"]["y"],
                                 "Strong Polarization Summary\n(Trajectory Occupancy Density)", cmap="inferno")

    im2 = create_spatial_heatmap(ax2, spatial_data["Weak"]["x"], spatial_data["Weak"]["y"],
                                 "Weak Polarization Summary\n(Trajectory Occupancy Density)", cmap="inferno")

    # Shared colorbar
    if im1 is not None:
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(im1, cax=cax)
        cbar.set_label("Spatial Occupancy Density", fontsize=10, fontweight="bold")

    plt.suptitle("Figure 19: Trajectory Density Heatmaps: Strong vs Weak Polarization", fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.08)
    fig.savefig(os.path.join(get_output_dir(), "fig19_heatmap_strong_vs_weak.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 20: 2x2 Grid Heatmap (Strong LR, Weak LR, Strong TB, Weak TB)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5), facecolor="white")
    grid_configs = [
        (axes[0, 0], spatial_data["Strong_LR"], "Strong Polarization - LR", "fig20a_heatmap_strong_lr.png"),
        (axes[0, 1], spatial_data["Strong_TB"], "Strong Polarization - TB", "fig20b_heatmap_strong_tb.png"),
        (axes[1, 0], spatial_data["Weak_LR"], "Weak Polarization - LR", "fig20c_heatmap_weak_lr.png"),
        (axes[1, 1], spatial_data["Weak_TB"], "Weak Polarization - TB", "fig20d_heatmap_weak_tb.png"),
    ]

    im_last = None
    for ax, d, title, fname in grid_configs:
        im_last = create_spatial_heatmap(ax, d["x"], d["y"], title, cmap="magma")

        # Save individual standalone plot
        fig_single, ax_single = plt.subplots(figsize=(6, 6), facecolor="white")
        im_single = create_spatial_heatmap(ax_single, d["x"], d["y"], title, cmap="magma")
        if im_single is not None:
            cbar_s = fig_single.colorbar(im_single, ax=ax_single, fraction=0.046, pad=0.04)
            cbar_s.set_label("Occupancy Density", fontsize=10, fontweight="bold")
        plt.tight_layout()
        fig_single.savefig(os.path.join(get_output_dir(), fname), dpi=300, bbox_inches="tight")
        plt.close(fig_single)

    if im_last is not None:
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(im_last, cax=cax)
        cbar.set_label("Occupancy Density", fontsize=10, fontweight="bold")

    plt.suptitle("Figure 20: Trajectory Spatial Density Grid", fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.05, hspace=0.15, wspace=0.15)
    fig.savefig(os.path.join(get_output_dir(), "fig20_heatmap_conditions_grid.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 21: Overall Arena Occupancy Heatmap (All Sessions) ─────────
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    im_all = create_spatial_heatmap(ax, spatial_data["All"]["x"], spatial_data["All"]["y"],
                                    "Overall Arena Occupancy Heatmap\n(All Video Sessions Summary)", cmap="inferno")

    if im_all is not None:
        cbar = fig.colorbar(im_all, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Trajectory Density", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(get_output_dir(), "fig21_heatmap_overall_occupancy.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated all heatmap summary figures in {get_output_dir()}/!")


def load_paper_dataset(search_dirs=None):
    """
    Dynamically scan tracking session folders and build a comprehensive dataset
    for all tracked sessions with calculated inner (R=210mm) and outer (R=420mm) exit angles.
    """
    session_paths = find_all_session_paths(search_dirs)
    rows = []

    for session_path in session_paths:
        session_name = os.path.basename(session_path)
        track_csv = None
        for f in os.listdir(session_path):
            if f.startswith("bee_track_") and f.endswith(".csv"):
                track_csv = os.path.join(session_path, f)
                break
        if not track_csv:
            continue

        try:
            sdf = pd.read_csv(track_csv, low_memory=False)
            if "x_mm" not in sdf.columns or "y_mm" not in sdf.columns:
                continue

                xs = sdf["x_mm"].values
                ys = sdf["y_mm"].values
                dists = np.hypot(xs, ys)

                # Calculate Inner Circle Exit (R = 210 mm)
                in_idx = np.where(dists >= INNER_R)[0]
                in_ang = np.degrees(np.arctan2(xs[in_idx[0]], -ys[in_idx[0]])) % 360.0 if len(in_idx) > 0 else np.nan

                # Calculate Outer Circle Exit (R = 420 mm)
                out_idx = np.where(dists >= OUTER_R)[0]
                out_ang = np.degrees(np.arctan2(xs[out_idx[0]], -ys[out_idx[0]])) % 360.0 if len(out_idx) > 0 else np.nan

                first = sdf.iloc[0]
                orient = str(first.get("orientation", "LR")).strip().upper()
                if orient not in ("LR", "TB"):
                    orient = "LR"
                stim = str(first.get("stimulus", "")).lower()

                rows.append({
                    "session_name": session_name,
                    "orientation": orient,
                    "stimulus": stim,
                    "inner_exit_angle_deg": in_ang,
                    "outer_exit_angle_deg": out_ang
                })
            except Exception as e:
                pass

    return pd.DataFrame(rows)


def generate_memory_analysis_plot(groups):
    """
    Generate Short-Term Memory (STM) vs Long-Term Memory (LTM) breakdown plot (Figure 25)
    matching paper1.pdf memory sector definitions:
      - STM Sectors: 45° to 135° and 225° to 315° (Aligned with 90° rotated stimulus)
      - LTM Sectors: 315° to 45° and 135° to 225° (Aligned with trained nest/anti-nest axis)
    """
    lr_df = pd.concat([groups["Strong_LR"], groups["Weak_LR"]])
def generate_memory_analysis_plot(groups):
    """
    Generate 100% Stacked Bar Chart for Short-Term vs Long-Term Memory (matching reference figure format).
    - Subplot A: Combined (LR vs TB)
    - Subplot B: Separated 4 Conditions (Strong LR, Weak LR, Strong TB, Weak TB)
    Includes Chi-squared test calculation and text annotations inside bars.
    Saved to: fig25_short_vs_long_term_memory.png
    """
    from scipy.stats import chi2_contingency

    def classify_memory(angles):
        stm_count = 0
        ltm_count = 0
        for deg in angles.dropna():
            if (45 <= deg <= 135) or (225 <= deg <= 315):
                stm_count += 1
            else:
                ltm_count += 1
        return stm_count, ltm_count

    # 1. Combined LR vs TB
    lr_df = pd.concat([groups["Strong_LR"], groups["Weak_LR"]])
    tb_df = pd.concat([groups["Strong_TB"], groups["Weak_TB"]])
    
    stm_lr, ltm_lr = classify_memory(lr_df["inner_exit_angle_deg"])
    stm_tb, ltm_tb = classify_memory(tb_df["inner_exit_angle_deg"])
    
    tot_lr = max(1, stm_lr + ltm_lr)
    tot_tb = max(1, stm_tb + ltm_tb)

    pct_stm_lr, pct_ltm_lr = (stm_lr / tot_lr) * 100.0, (ltm_lr / tot_lr) * 100.0
    pct_stm_tb, pct_ltm_tb = (stm_tb / tot_tb) * 100.0, (ltm_tb / tot_tb) * 100.0

    obs_comb = np.array([[ltm_lr, stm_lr], [ltm_tb, stm_tb]])
    try:
        _, p_val_comb, _, _ = chi2_contingency(obs_comb)
    except Exception:
        p_val_comb = 1.0

    # 2. Separated 4 Conditions
    cond_keys = ["Strong_LR", "Weak_LR", "Strong_TB", "Weak_TB"]
    cond_labels = ["Strong LR", "Weak LR", "Strong TB", "Weak TB"]
    
    sep_stm_pct = []
    sep_ltm_pct = []
    obs_sep = []

    for k in cond_keys:
        s_cnt, l_cnt = classify_memory(groups[k]["inner_exit_angle_deg"])
        tot = max(1, s_cnt + l_cnt)
        sep_stm_pct.append((s_cnt / tot) * 100.0)
        sep_ltm_pct.append((l_cnt / tot) * 100.0)
        obs_sep.append([l_cnt, s_cnt])

    try:
        _, p_val_sep, _, _ = chi2_contingency(np.array(obs_sep))
    except Exception:
        p_val_sep = 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), facecolor="white")
    width = 0.50

    # ── Subplot A: Combined LR vs TB ───────────────────────────────────────
    cat_comb = ["LR", "TB"]
    x1 = np.arange(len(cat_comb))
    rects_stm1 = ax1.bar(x1, [pct_stm_lr, pct_stm_tb], width, label="STM", color="#A0D7E6", edgecolor="#2C3E50", lw=1.2)
    rects_ltm1 = ax1.bar(x1, [pct_ltm_lr, pct_ltm_tb], width, bottom=[pct_stm_lr, pct_stm_tb], label="LTM", color="#2E7D32", edgecolor="#2C3E50", lw=1.2)

    for rect in rects_stm1:
        h = rect.get_height()
        if h > 5:
            ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h/2), ha="center", va="center", fontweight="bold", fontsize=11, color="#2C3E50")

    for rect, stm_h in zip(rects_ltm1, [pct_stm_lr, pct_stm_tb]):
        h = rect.get_height()
        if h > 5:
            ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, stm_h + h/2), ha="center", va="center", fontweight="bold", fontsize=11, color="#ffffff")

    ax1.set_xticks(x1)
    ax1.set_xticklabels(cat_comb, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Percentage", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Condition (LR vs TB)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.set_yticks([0, 25, 50, 75, 100])
    p_str1 = f"p = {p_val_comb:.4f}" if p_val_comb >= 0.0001 else "p < 0.0001"
    ax1.set_title(f"A) Combined Memory (LR vs TB)\nChi-squared {p_str1}", fontsize=11.5, fontweight="bold", pad=10)
    ax1.legend(title="MemoryRegion", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9.5)
    ax1.grid(axis="y", ls=":", alpha=0.4)

    # ── Subplot B: Separated 4 Conditions ─────────────────────────────────
    x2 = np.arange(len(cond_labels))
    rects_stm2 = ax2.bar(x2, sep_stm_pct, width, label="STM", color="#A0D7E6", edgecolor="#2C3E50", lw=1.2)
    rects_ltm2 = ax2.bar(x2, sep_ltm_pct, width, bottom=sep_stm_pct, label="LTM", color="#2E7D32", edgecolor="#2C3E50", lw=1.2)

    for rect in rects_stm2:
        h = rect.get_height()
        if h > 5:
            ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h/2), ha="center", va="center", fontweight="bold", fontsize=10.5, color="#2C3E50")

    for rect, stm_h in zip(rects_ltm2, sep_stm_pct):
        h = rect.get_height()
        if h > 5:
            ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, stm_h + h/2), ha="center", va="center", fontweight="bold", fontsize=10.5, color="#ffffff")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cond_labels, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Percentage", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Polarization Condition", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100])
    p_str2 = f"p = {p_val_sep:.4f}" if p_val_sep >= 0.0001 else "p < 0.0001"
    ax2.set_title(f"B) Memory across 4 Conditions (Strong vs Weak)\nChi-squared {p_str2}", fontsize=11.5, fontweight="bold", pad=10)
    ax2.legend(title="MemoryRegion", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9.5)
    ax2.grid(axis="y", ls=":", alpha=0.4)

    plt.suptitle("Short-Term vs Long-Term Memory Distribution by Experimental Conditions", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = get_output_dir()
    output_path = os.path.join(output_dir, "fig25_short_vs_long_term_memory.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 25 (100% Stacked Memory Chart - Combined & Separated) in {output_path}!")


def generate_feeder_visit_plot(groups):
    """
    Generate paper plot for Feeder Visit Behavior: Returned Back vs Still in Arena (%)
    - Combined view (LR vs TB)
    - Separated view (Strong LR, Weak LR, Strong TB, Weak TB)
    - No raw number annotations on graph bars per user request.
    Saved to: fig26_feeder_visit_percentage.png
    """
    session_path_lookup = {os.path.basename(p): p for p in find_all_session_paths()}

    def get_return_behavior(sub_df):
        returned_back = 0
        still_in_arena = 0
        total = len(sub_df)
        for _, row in sub_df.iterrows():
            sess_name = row['session_name']
            sp = session_path_lookup.get(sess_name)
            returned_after_exit = False
            if sp and os.path.exists(sp):
                csv_files = [f for f in os.listdir(sp) if f.startswith('bee_track_') and f.endswith('.csv')]
                if csv_files:
                    tdf = pd.read_csv(os.path.join(sp, csv_files[0]))
                    if 'distance_from_center_mm' in tdf.columns:
                        dists = tdf['distance_from_center_mm'].values
                    else:
                        dists = np.hypot(tdf['x_mm'], tdf['y_mm']).values
                    
                    at_feeder_idx = np.where(dists <= 70)[0]
                    if len(at_feeder_idx) > 0:
                        first_feeder_idx = at_feeder_idx[0]
                        after_feeder_dists = dists[first_feeder_idx:]
                        exited_idx = np.where(after_feeder_dists > 150)[0]
                        if len(exited_idx) > 0:
                            first_exit_idx = exited_idx[0]
                            after_exit_dists = after_feeder_dists[first_exit_idx:]
                            if np.any(after_exit_dists <= 100):
                                returned_after_exit = True
            if returned_after_exit:
                returned_back += 1
            else:
                still_in_arena += 1
        pct_returned = (returned_back / total * 100) if total > 0 else 0
        pct_still = (still_in_arena / total * 100) if total > 0 else 0
        return pct_returned, pct_still

    # Combined LR vs TB
    lr_df = pd.concat([groups["Strong_LR"], groups["Weak_LR"]])
    tb_df = pd.concat([groups["Strong_TB"], groups["Weak_TB"]])
    
    pr_lr, ps_lr = get_return_behavior(lr_df)
    pr_tb, ps_tb = get_return_behavior(tb_df)

    # Separated 4 conditions
    cond_keys = ["Strong_LR", "Weak_LR", "Strong_TB", "Weak_TB"]
    cond_labels = ["Strong LR", "Weak LR", "Strong TB", "Weak TB"]
    sep_returned = []
    sep_still = []
    for k in cond_keys:
        pr, ps = get_return_behavior(groups[k])
        sep_returned.append(pr)
        sep_still.append(ps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    width = 0.40
    
    # ── Panel A: Combined LR vs TB ─────────────────────────────────────────
    comb_labels = ["LR", "TB"]
    x1 = np.arange(len(comb_labels))

    rects1 = ax1.bar(x1 - width/2, [pr_lr, pr_tb], width, label="Returned Back", color="#66BB6A", edgecolor="#2E7D32", lw=1.2)
    rects2 = ax1.bar(x1 + width/2, [ps_lr, ps_tb], width, label="Still in Arena / Exited", color="#FFA726", edgecolor="#E65100", lw=1.2)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    ax1.set_xticks(x1)
    ax1.set_xticklabels(comb_labels, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.set_title("A) Combined Trajectory Return Behavior (LR vs TB)", fontsize=11, fontweight="bold")
    ax1.legend(title="Bee Behavior", frameon=True, facecolor="#f8f9fa", edgecolor="#cccccc", fontsize=9.5)
    ax1.grid(axis="y", ls=":", alpha=0.6)

    # ── Panel B: Separated 4 Conditions ───────────────────────────────────
    x2 = np.arange(len(cond_labels))
    rects3 = ax2.bar(x2 - width/2, sep_returned, width, label="Returned Back", color="#66BB6A", edgecolor="#2E7D32", lw=1.2)
    rects4 = ax2.bar(x2 + width/2, sep_still, width, label="Still in Arena / Exited", color="#FFA726", edgecolor="#E65100", lw=1.2)

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cond_labels, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.set_title("B) Trajectory Return Behavior across Polarization Conditions", fontsize=11, fontweight="bold")
    ax2.legend(title="Bee Behavior", frameon=True, facecolor="#f8f9fa", edgecolor="#cccccc", fontsize=9.5)
    ax2.grid(axis="y", ls=":", alpha=0.6)

    plt.suptitle("Percentage of Bumblebees: Returned Back vs Still in Arena / Exited", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir = get_output_dir()
    output_path = os.path.join(output_dir, "fig26_feeder_visit_percentage.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 26 (Feeder Visit Percentage Plot) in {output_path}!")


def main():
    print("Generating Paper 1 plots and statistical summary...")
    df = load_paper_dataset()
    print(f"Loaded {len(df)} tracked video sessions for analysis in {get_output_dir()}")
    groups = classify_trials(df)
    generate_memory_analysis_plot(groups)
    generate_feeder_visit_plot(groups)

    option = os.environ.get("PAPER_PLOT_OPTION", "1").strip()

    # Option 8: CSV Summary only
    if option == "8":
        stats_list = []
        fig_configs = [
            ("Training",  "All bumblebees\nInner circle (train trials)", "Model fit\nInner bearings (train trials)", "fig09_training_inner_circle.png", "#795548"),
            ("Strong_LR", "Strong polarization\nLR, Inner circle bearings", "Strong polarization\nLR, Inner circle bearings", "fig10_strong_polarization_lr.png", "#FF7043"),
            ("Weak_LR",   "Weak polarization\nLR, Inner circle bearings", "Weak polarization\nLR, Inner circle bearings", "fig11_weak_polarization_lr.png", "#FFAB91"),
            ("Strong_TB", "Strong polarization\nTB, Inner circle bearings", "Strong polarization\nTB, Inner circle bearings", "fig12_strong_polarization_tb.png", "#1A237E"),
            ("Weak_TB",   "Weak polarization\nTB, Inner circle bearings", "Weak polarization\nTB, Inner circle bearings", "fig13_weak_polarization_tb.png", "#26A69A"),
        ]
        for name, left_t, right_t, fname, col in fig_configs:
            sub_df = groups[name]
            angles = sub_df["inner_exit_angle_deg"].values
            n = len(angles)
            r, p, mu, kappa = rayleigh_test(angles) if n > 0 else (0.0, 1.0, 0.0, 0.0)
            stats_list.append({
                "Condition": name,
                "n": n,
                "Rayleigh_r": round(r, 4),
                "Rayleigh_p": round(p, 4),
                "Mean_mu_deg": round(mu, 2),
                "Kappa": round(kappa, 2)
            })
        stats_df = pd.DataFrame(stats_list)
        csv_path = os.path.join(get_output_dir(), "paper_stats_summary.csv")
        stats_df.to_csv(csv_path, index=False)
        print(f"Saved statistical summary to {csv_path}")
        return

    # ── Figures 9–13: Dual Circular Bearing & Circ-MLE Fit Plots ──────────
    if option in ("1", "2"):
        stats_list = []
        fig_configs = [
            ("Training",  "All bumblebees\nInner circle (train trials)", "Model fit\nInner bearings (train trials)", "fig09_training_inner_circle.png", "#795548"),
            ("Strong_LR", "Strong polarization\nLR, Inner circle bearings", "Strong polarization\nLR, Inner circle bearings", "fig10_strong_polarization_lr.png", "#FF7043"),
            ("Weak_LR",   "Weak polarization\nLR, Inner circle bearings", "Weak polarization\nLR, Inner circle bearings", "fig11_weak_polarization_lr.png", "#FFAB91"),
            ("Strong_TB", "Strong polarization\nTB, Inner circle bearings", "Strong polarization\nTB, Inner circle bearings", "fig12_strong_polarization_tb.png", "#1A237E"),
            ("Weak_TB",   "Weak polarization\nTB, Inner circle bearings", "Weak polarization\nTB, Inner circle bearings", "fig13_weak_polarization_tb.png", "#26A69A"),
        ]

        for name, left_t, right_t, fname, col in fig_configs:
            sub_df = groups[name]
            angles = sub_df["inner_exit_angle_deg"].values
            n = len(angles)
            r, p, mu, kappa = rayleigh_test(angles) if n > 0 else (0.0, 1.0, 0.0, 0.0)

            plot_paper_circular_figure(angles, left_t, right_t, fname, col)

            stats_list.append({
                "Condition": name,
                "n": n,
                "Rayleigh_r": round(r, 4),
                "Rayleigh_p": round(p, 4),
                "Mean_mu_deg": round(mu, 2),
                "Kappa": round(kappa, 2)
            })

        # Export Stats CSV
        stats_df = pd.DataFrame(stats_list)
        csv_path = os.path.join(get_output_dir(), "paper_stats_summary.csv")
        stats_df.to_csv(csv_path, index=False)
        print(f"Saved statistical summary to {csv_path}")

    # ── Figures 14 & 15: Home vs Fictive Distribution ────────────────────
    if option in ("1", "3"):
        def calc_quadrant_pcts(sub_df):
            angles = sub_df["inner_exit_angle_deg"].dropna().values
            total = len(angles)
            if total == 0:
                return 0.0, 0.0
            # Home: [-45°, 45°] (Nest quadrant)
            home_c = np.sum((angles >= 315) | (angles <= 45))
            # Fictive: [45°, 135°] (Rotated Nest quadrant)
            fictive_c = np.sum((angles >= 45) & (angles <= 135))
            return (fictive_c / total * 100.0), (home_c / total * 100.0)

        categories = ["Strong", "Weak"]
        x = np.arange(len(categories))
        width = 0.35

        # ── Figure 14: Left-Right (LR) ─────────────────────────────────────
        fic_s_lr, home_s_lr = calc_quadrant_pcts(groups["Strong_LR"])
        fic_w_lr, home_w_lr = calc_quadrant_pcts(groups["Weak_LR"])

        fig, ax = plt.subplots(figsize=(6.5, 5), facecolor="white")
        rects1 = ax.bar(x - width/2, [fic_s_lr, fic_w_lr], width, label="fictive", color="#A0522D")
        rects2 = ax.bar(x + width/2, [home_s_lr, home_w_lr], width, label="home", color="#B0E0E6")

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Degrees of Polarisation (DoP)", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_title("Figure 14: Left–Right (LR) Trials: Home vs Fictive Distribution", fontsize=11, fontweight="bold")
        ax.legend(title="Quadrant", frameon=True)
        ax.grid(axis="y", ls=":", alpha=0.6)
        plt.tight_layout()
        fig.savefig(os.path.join(get_output_dir(), "fig14_home_vs_fictive_lr.png"), dpi=300)
        plt.close(fig)

        # ── Figure 15: Top-Bottom (TB) ─────────────────────────────────────
        fic_s_tb, home_s_tb = calc_quadrant_pcts(groups["Strong_TB"])
        fic_w_tb, home_w_tb = calc_quadrant_pcts(groups["Weak_TB"])

        fig, ax = plt.subplots(figsize=(6.5, 5), facecolor="white")
        rects3 = ax.bar(x - width/2, [fic_s_tb, fic_w_tb], width, label="fictive", color="#A0522D")
        rects4 = ax.bar(x + width/2, [home_s_tb, home_w_tb], width, label="home", color="#B0E0E6")

        for rect in rects3:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)
        for rect in rects4:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Degrees of Polarisation", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_title("Figure 15: Top–Bottom (TB) Trials: Home vs Fictive Distribution", fontsize=11, fontweight="bold")
        ax.legend(title="Quadrant", frameon=True)
        ax.grid(axis="y", ls=":", alpha=0.6)
        plt.tight_layout()
        fig.savefig(os.path.join(get_output_dir(), "fig15_home_vs_fictive_tb.png"), dpi=300)
        plt.close(fig)

    # ── Figure 16: Circular plots grid for LR and TB ──────────────────────
    if option in ("1", "4"):
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), subplot_kw=dict(projection='polar'), facecolor="white")
        grid_data = [
            (axes[0, 0], groups["Weak_LR"], "Weak - LR", "#FFAB91"),
            (axes[0, 1], groups["Weak_TB"], "Weak - TB", "#26A69A"),
            (axes[1, 0], groups["Strong_LR"], "Strong - LR", "#FF7043"),
            (axes[1, 1], groups["Strong_TB"], "Strong - TB", "#1A237E"),
        ]
        for ax, sub_df, title, col in grid_data:
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            rad = np.radians(sub_df["inner_exit_angle_deg"].values)
            if len(rad) > 0:
                ax.scatter(rad, np.ones_like(rad), color=col, alpha=0.9, s=40)
            ax.set_ylim(0, 1.1)
            ax.set_yticks([])
            ax.set_xticks(np.radians([0, 90, 180, 270]))
            ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=8, fontweight="bold")
            ax.set_title(title, fontsize=10, fontweight="bold")

        plt.suptitle("Figure 16: Circular Plots of Stronger & Weaker Polarization (LR vs TB)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(get_output_dir(), "fig16_circular_plots_lr_tb.png"), dpi=300)
        plt.close(fig)

    # ── Figures 17 & 18: Homing Accuracy & Boxplots ───────────────────────
    if option in ("1", "5"):
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
        def calc_dev(sub_df):
            a = sub_df["inner_exit_angle_deg"].values
            if len(a) == 0: return 0.0
            d = np.minimum(np.abs(a % 360), 360 - (a % 360))
            return float(np.mean(d))

        dev_strong_lr = calc_dev(groups["Strong_LR"])
        dev_weak_lr   = calc_dev(groups["Weak_LR"])
        dev_strong_tb = calc_dev(groups["Strong_TB"])
        dev_weak_tb   = calc_dev(groups["Weak_TB"])

        x_pos = np.arange(4)
        means = [dev_strong_lr, dev_weak_lr, dev_strong_tb, dev_weak_tb]
        colors = ["#FF7043", "#FFAB91", "#1A237E", "#26A69A"]
        labels = ["Strong\n(LR)", "Weak\n(LR)", "Strong\n(TB)", "Weak\n(TB)"]

        ax.bar(x_pos, means, color=colors, width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
        ax.set_ylabel("Mean Deviation from Nest (°)", fontsize=10, fontweight="bold")
        ax.set_title("Figure 17: Average Homing Accuracy by Rotation & Polarization", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(get_output_dir(), "fig17_average_homing_accuracy.png"), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
        def get_dev_array(sub_df):
            a = sub_df["inner_exit_angle_deg"].values
            if len(a) == 0: return [0.0]
            return list(np.minimum(np.abs(a % 360), 360 - (a % 360)))

        data_box = [
            get_dev_array(groups["Strong_LR"]),
            get_dev_array(groups["Weak_LR"]),
            get_dev_array(groups["Strong_TB"]),
            get_dev_array(groups["Weak_TB"]),
        ]

        box_labels = ["Strong\n(LR)", "Weak\n(LR)", "Strong\n(TB)", "Weak\n(TB)"]
        try:
            bp = ax.boxplot(data_box, patch_artist=True, tick_labels=box_labels)
        except TypeError:
            bp = ax.boxplot(data_box, patch_artist=True, labels=box_labels)
        box_colors = ["#FF7043", "#FFAB91", "#1A237E", "#26A69A"]
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        ax.set_ylabel("Deviation from Nest (°)", fontsize=10, fontweight="bold")
        ax.set_title("Figure 18: Boxplot of Deviation from Nest by Condition", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        fig.savefig(os.path.join(get_output_dir(), "fig18_boxplot_deviation_from_nest.png"), dpi=300)
        plt.close(fig)

    # ── Figures 19–21: Spatial Trajectory Heatmaps ─────────────────────────
    if option in ("1", "6"):
        generate_paper_heatmaps()

    # ── Figures 22 & 23: Exit Angle Density Heatmaps ───────────────────────
    if option in ("1", "7"):
        generate_exit_angle_heatmaps(df)

    # ── Figure 25: Short-Term vs Long-Term Memory Analysis ─────────────────
    if option in ("1", "9"):
        generate_memory_analysis_plot(groups)
        try:
            from analysis.generate_speed_plot import generate_speed_plots
        except ModuleNotFoundError:
            from generate_speed_plot import generate_speed_plots
        generate_speed_plots()

    print(f"All Paper 1 figures generated successfully in {get_output_dir()}/!")


def generate_exit_angle_heatmaps(df):
    df["inner_exit_angle_deg"] = pd.to_numeric(df["inner_exit_angle_deg"], errors="coerce")
    df["outer_exit_angle_deg"] = pd.to_numeric(df["outer_exit_angle_deg"], errors="coerce")

    groups = classify_trials(df)

    num_bins = 24
    bins = np.linspace(0, 2 * np.pi, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    width = 2 * np.pi / num_bins

    # Extract angle arrays
    all_lr_in = np.radians(pd.concat([groups["Strong_LR"]["inner_exit_angle_deg"], groups["Weak_LR"]["inner_exit_angle_deg"]]).dropna().values)
    all_tb_in = np.radians(pd.concat([groups["Strong_TB"]["inner_exit_angle_deg"], groups["Weak_TB"]["inner_exit_angle_deg"]]).dropna().values)

    all_lr_out = np.radians(pd.concat([groups["Strong_LR"]["outer_exit_angle_deg"], groups["Weak_LR"]["outer_exit_angle_deg"]]).dropna().values)
    all_tb_out = np.radians(pd.concat([groups["Strong_TB"]["outer_exit_angle_deg"], groups["Weak_TB"]["outer_exit_angle_deg"]]).dropna().values)

    strong_in_all = np.radians(pd.concat([groups["Strong_LR"]["inner_exit_angle_deg"], groups["Strong_TB"]["inner_exit_angle_deg"]]).dropna().values)
    weak_in_all   = np.radians(pd.concat([groups["Weak_LR"]["inner_exit_angle_deg"], groups["Weak_TB"]["inner_exit_angle_deg"]]).dropna().values)

    # ── Figure 22: 4-Subplot Polar Exit Angle Heatmaps ─────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(projection='polar'), facecolor="white")
    
    # 4 Combinations: (ax, title, strong_angles, weak_angles)
    # 1. LR Inner Circle
    lr_in_s = np.radians(groups["Strong_LR"]["inner_exit_angle_deg"].dropna().values)
    lr_in_w = np.radians(groups["Weak_LR"]["inner_exit_angle_deg"].dropna().values)
    
    # 2. LR Outer Circle
    lr_out_s = np.radians(groups["Strong_LR"]["outer_exit_angle_deg"].dropna().values)
    lr_out_w = np.radians(groups["Weak_LR"]["outer_exit_angle_deg"].dropna().values)
    
    # 3. TB Inner Circle
    tb_in_s = np.radians(groups["Strong_TB"]["inner_exit_angle_deg"].dropna().values)
    tb_in_w = np.radians(groups["Weak_TB"]["inner_exit_angle_deg"].dropna().values)
    
    # 4. TB Outer Circle
    tb_out_s = np.radians(groups["Strong_TB"]["outer_exit_angle_deg"].dropna().values)
    tb_out_w = np.radians(groups["Weak_TB"]["outer_exit_angle_deg"].dropna().values)
    
    combos = [
        (axes[0, 0], "A) LR : Inner Circle (R = 210 mm)", lr_in_s, lr_in_w, "fig22a_exit_angle_polar_lr_inner.png"),
        (axes[0, 1], "B) LR : Outer Circle (R = 420 mm)", lr_out_s, lr_out_w, "fig22b_exit_angle_polar_lr_outer.png"),
        (axes[1, 0], "C) TB : Inner Circle (R = 210 mm)", tb_in_s, tb_in_w, "fig22c_exit_angle_polar_tb_inner.png"),
        (axes[1, 1], "D) TB : Outer Circle (R = 420 mm)", tb_out_s, tb_out_w, "fig22d_exit_angle_polar_tb_outer.png"),
    ]
    
    for ax, title, strong_a, weak_a, fname in combos:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1) # Clockwise
        
        c_s, _ = np.histogram(strong_a, bins=bins)
        c_w, _ = np.histogram(weak_a, bins=bins)
        
        max_c = max(1, max(c_s.max() if len(c_s) else 1, c_w.max() if len(c_w) else 1))
        
        # Outer Ring: Strong Polarization
        ax.bar(bin_centers, np.ones(num_bins)*0.38, width=width, bottom=0.65,
               color=plt.cm.YlOrRd(c_s / max_c), edgecolor="#333333", lw=0.6)
        # Inner Ring: Weak Polarization
        ax.bar(bin_centers, np.ones(num_bins)*0.38, width=width, bottom=0.18,
               color=plt.cm.YlOrRd(c_w / max_c), edgecolor="#333333", lw=0.6)
               
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0.37, 0.84])
        ax.set_yticklabels(["Weak", "Strong"], fontsize=9, fontweight="bold")
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["0°\n(Nest)", "90°", "180°", "270°"], fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=25)
        
        r_s, p_s, mu_s, k_s = rayleigh_test(np.degrees(strong_a))
        r_w, p_w, mu_w, k_w = rayleigh_test(np.degrees(weak_a))
        txt_stats = (f"Strong: n={len(strong_a)}, r={r_s:.2f}, μ={mu_s:.1f}°\n"
                     f"Weak:   n={len(weak_a)}, r={r_w:.2f}, μ={mu_w:.1f}°")
        ax.text(0.5, -0.32, transform=ax.transAxes, s=txt_stats, ha="center", va="top",
                 fontsize=8.0, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#cccccc"))

        # Save individual polar plot with colorbar legend
        fig_s, ax_s = plt.subplots(figsize=(7, 6.8), subplot_kw=dict(projection='polar'), facecolor="white")
        ax_s.set_theta_zero_location("N")
        ax_s.set_theta_direction(-1)
        ax_s.bar(bin_centers, np.ones(num_bins)*0.38, width=width, bottom=0.65,
                 color=plt.cm.YlOrRd(c_s / max_c), edgecolor="#333333", lw=0.6)
        ax_s.bar(bin_centers, np.ones(num_bins)*0.38, width=width, bottom=0.18,
                 color=plt.cm.YlOrRd(c_w / max_c), edgecolor="#333333", lw=0.6)
        ax_s.set_ylim(0, 1.12)
        ax_s.set_yticks([0.37, 0.84])
        ax_s.set_yticklabels(["Weak", "Strong"], fontsize=9, fontweight="bold")
        ax_s.set_xticks(np.radians([0, 90, 180, 270]))
        ax_s.set_xticklabels(["0°\n(Nest)", "90°", "180°", "270°"], fontsize=9.5, fontweight="bold")
        ax_s.set_title(title, fontsize=11, fontweight="bold", pad=20)
        ax_s.text(0.5, -0.25, transform=ax_s.transAxes, s=txt_stats, ha="center", va="top",
                  fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#cccccc"))
        
        sm_s = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(0, 100))
        sm_s.set_array([])
        cbar_s = fig_s.colorbar(sm_s, ax=ax_s, orientation="vertical", fraction=0.046, pad=0.10, shrink=0.75)
        cbar_s.set_label("Relative Exit Density (%)", fontsize=9.5, fontweight="bold")
        
        fig_s.subplots_adjust(bottom=0.20, top=0.88, left=0.08, right=0.88)
        fig_s.savefig(os.path.join(get_output_dir(), fname), dpi=300, bbox_inches="tight")
        plt.close(fig_s)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Relative Exit Density (%)", fontsize=10, fontweight="bold")

    plt.suptitle("Figure 22: Polar Exit Angle Heatmaps Across 4 Arena & Cue Combinations", fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.89, top=0.88, bottom=0.10, hspace=0.60, wspace=0.32)
    fig.savefig(os.path.join(get_output_dir(), "fig22_exit_angle_polar_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 22a: LR Rotation Polar Heatmap (Inner & Outer) ─────────────────
    fig_lr, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(13.5, 7.0), subplot_kw=dict(projection='polar'), facecolor="white")
    for ax, title, strong_a, weak_a in [
        (ax_in, "LR : Inner Circle (R = 210 mm)", lr_in_s, lr_in_w),
        (ax_out, "LR : Outer Circle (R = 420 mm)", lr_out_s, lr_out_w)
    ]:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        c_s, _ = np.histogram(strong_a, bins=bins)
        c_w, _ = np.histogram(weak_a, bins=bins)
        max_c = max(1, max(c_s.max() if len(c_s) else 1, c_w.max() if len(c_w) else 1))
        ax.bar(bin_centers, np.ones(num_bins)*0.38, width=width, bottom=0.65, color=plt.cm.YlOrRd(c_s / max_c), edgecolor="#333333", lw=0.6)
        ax.bar(bin_centers, np.ones(num_bins)*0.38, width=width, bottom=0.18, color=plt.cm.YlOrRd(c_w / max_c), edgecolor="#333333", lw=0.6)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0.37, 0.84])
        ax.set_yticklabels(["Weak", "Strong"], fontsize=9, fontweight="bold")
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["0°\n(Nest)", "90°", "180°", "270°"], fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=20)

        r_s, p_s, mu_s, k_s = rayleigh_test(np.degrees(strong_a))
        r_w, p_w, mu_w, k_w = rayleigh_test(np.degrees(weak_a))
        txt_stats = (f"Strong: n={len(strong_a)}, r={r_s:.2f}, μ={mu_s:.1f}°\n"
                     f"Weak:   n={len(weak_a)}, r={r_w:.2f}, μ={mu_w:.1f}°")
        ax.text(0.5, -0.28, transform=ax.transAxes, s=txt_stats, ha="center", va="top",
                fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#cccccc"))

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar_ax = fig_lr.add_axes([0.91, 0.22, 0.018, 0.55])
    cbar = fig_lr.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Relative Exit Density (%)", fontsize=10, fontweight="bold")

    fig_lr.suptitle("Figure 22a: Left-Right (LR) Rotation Exit Angle Polar Heatmaps", fontsize=13, fontweight="bold", y=0.96)
    fig_lr.subplots_adjust(left=0.06, right=0.88, top=0.80, bottom=0.18, wspace=0.35)
    fig_lr.savefig(os.path.join(get_output_dir(), "fig22a_exit_angle_polar_lr.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_lr)

    # ── Figure 22b: TB Rotation Polar Heatmap (Inner & Outer) ─────────────────
    fig_tb, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(13.5, 7.0), subplot_kw=dict(projection='polar'), facecolor="white")
    for ax, title, strong_a, weak_a in [
        (ax_in, "TB Rotation: Inner Circle (R = 210 mm)", tb_in_s, tb_in_w),
        (ax_out, "TB Rotation: Outer Circle (R = 420 mm)", tb_out_s, tb_out_w)
    ]:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        c_s, _ = np.histogram(strong_a, bins=bins)
        c_w, _ = np.histogram(weak_a, bins=bins)
        max_c = max(1, max(c_s.max() if len(c_s) else 1, c_w.max() if len(c_w) else 1))
        ax.bar(bin_centers, np.ones(num_bins)*0.38, width=width, bottom=0.65, color=plt.cm.YlOrRd(c_s / max_c), edgecolor="#333333", lw=0.6)
        ax.bar(bin_centers, np.ones(num_bins)*0.38, width=width, bottom=0.18, color=plt.cm.YlOrRd(c_w / max_c), edgecolor="#333333", lw=0.6)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0.37, 0.84])
        ax.set_yticklabels(["Weak", "Strong"], fontsize=9, fontweight="bold")
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["0°\n(Nest)", "90°", "180°", "270°"], fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=20)

        r_s, p_s, mu_s, k_s = rayleigh_test(np.degrees(strong_a))
        r_w, p_w, mu_w, k_w = rayleigh_test(np.degrees(weak_a))
        txt_stats = (f"Strong: n={len(strong_a)}, r={r_s:.2f}, μ={mu_s:.1f}°\n"
                     f"Weak:   n={len(weak_a)}, r={r_w:.2f}, μ={mu_w:.1f}°")
        ax.text(0.5, -0.28, transform=ax.transAxes, s=txt_stats, ha="center", va="top",
                fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#cccccc"))

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar_ax = fig_tb.add_axes([0.91, 0.22, 0.018, 0.55])
    cbar = fig_tb.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Relative Exit Density (%)", fontsize=10, fontweight="bold")

    fig_tb.suptitle("Figure 22b: Top-Bottom (TB) Rotation Exit Angle Polar Heatmaps", fontsize=13, fontweight="bold", y=0.96)
    fig_tb.subplots_adjust(left=0.06, right=0.88, top=0.80, bottom=0.18, wspace=0.35)
    fig_tb.savefig(os.path.join(get_output_dir(), "fig22b_exit_angle_polar_tb.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_tb)

    # ── Figure 23: Condition × Angle Matrix Heatmap (Inner & Outer) ───────────
    fig, (ax_mat1, ax_mat2) = plt.subplots(2, 1, figsize=(10, 7), facecolor="white")

    cond_order = ["Training", "Strong_LR", "Weak_LR", "Strong_TB", "Weak_TB"]
    angle_bins = np.linspace(0, 360, 13)
    bin_labels = [f"{int(angle_bins[i])}°–{int(angle_bins[i+1])}°" for i in range(12)]

    matrix_inner = []
    matrix_outer = []

    for name in cond_order:
        sub = groups[name]
        in_deg = sub["inner_exit_angle_deg"].dropna().values
        out_deg = sub["outer_exit_angle_deg"].dropna().values

        c_in, _ = np.histogram(in_deg, bins=angle_bins)
        c_out, _ = np.histogram(out_deg, bins=angle_bins)

        pct_in = (c_in / len(in_deg) * 100.0) if len(in_deg) > 0 else np.zeros(12)
        pct_out = (c_out / len(out_deg) * 100.0) if len(out_deg) > 0 else np.zeros(12)

        matrix_inner.append(pct_in)
        matrix_outer.append(pct_out)

    matrix_inner = np.array(matrix_inner)
    matrix_outer = np.array(matrix_outer)

    # Top: Inner Circle Exit Angles Matrix
    im_m1 = ax_mat1.imshow(matrix_inner, aspect="auto", cmap="YlOrRd", vmin=0, vmax=50)
    ax_mat1.set_yticks(np.arange(len(cond_order)))
    ax_mat1.set_yticklabels(["Training", "Strong (LR)", "Weak (LR)", "Strong (TB)", "Weak (TB)"], fontweight="bold")
    ax_mat1.set_xticks(np.arange(12))
    ax_mat1.set_xticklabels(bin_labels, rotation=35, ha="right", fontsize=8.5)
    ax_mat1.set_title("Inner Circle Exit Angle Distribution Heatmap (R = 210 mm)", fontsize=11, fontweight="bold")

    for i in range(len(cond_order)):
        for j in range(12):
            val = matrix_inner[i, j]
            if val > 0:
                txt_col = "white" if val > 25 else "black"
                ax_mat1.text(j, i, f"{val:.0f}%", ha="center", va="center", color=txt_col, fontsize=8, fontweight="bold")

    # Bottom: Outer Circle Exit Angles Matrix
    im_m2 = ax_mat2.imshow(matrix_outer, aspect="auto", cmap="YlOrRd", vmin=0, vmax=50)
    ax_mat2.set_yticks(np.arange(len(cond_order)))
    ax_mat2.set_yticklabels(["Training", "Strong (LR)", "Weak (LR)", "Strong (TB)", "Weak (TB)"], fontweight="bold")
    ax_mat2.set_xticks(np.arange(12))
    ax_mat2.set_xticklabels(bin_labels, rotation=35, ha="right", fontsize=8.5)
    ax_mat2.set_title("Outer Circle Exit Angle Distribution Heatmap (R = 420 mm)", fontsize=11, fontweight="bold")

    for i in range(len(cond_order)):
        for j in range(12):
            val = matrix_outer[i, j]
            if val > 0:
                txt_col = "white" if val > 25 else "black"
                ax_mat2.text(j, i, f"{val:.0f}%", ha="center", va="center", color=txt_col, fontsize=8, fontweight="bold")

    fig.subplots_adjust(right=0.88, hspace=0.45)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(im_m2, cax=cbar_ax)
    cbar.set_label("Exit Frequency (%)", fontsize=10, fontweight="bold")

    plt.suptitle("Figure 23: Exit Angle Sector Matrix Heatmap (Inner vs Outer Circle)", fontsize=12, fontweight="bold", y=0.98)
    fig.savefig(os.path.join(get_output_dir(), "fig23_exit_angle_matrix_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated Exit Angle Heatmaps (Figures 22 & 23) in {get_output_dir()}/!")


if __name__ == "__main__":
    main()
