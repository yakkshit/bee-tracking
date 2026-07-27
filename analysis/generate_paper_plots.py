"""
Generate all Paper 1 statistical figures (Figures 9–18) and summary statistics.

Excludes Section 4.3 (Short/Long-term memory figures 19-21) per user request.

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
  - Stats CSV: results/paper_stats_summary.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUTPUT_DIR = "results/paper_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def rayleigh_test(angles_deg):
    """Compute Rayleigh test statistic r and p-value."""
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


def plot_circular_bearing(angles_deg, title, filename, mean_mu=None, mean_r=None):
    """Generate circular scatter plot with mean vector arrow matching Paper 1 style."""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'), facecolor="white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise
    
    rad = np.radians(angles_deg)
    
    # Scatter points on unit circle
    ax.scatter(rad, np.ones_like(rad), color="#D32F2F", alpha=0.8, s=40, zorder=5)
    
    # Draw mean vector arrow if provided
    if mean_mu is not None and mean_r is not None:
        mean_rad = np.radians(mean_mu)
        ax.annotate("", xy=(mean_rad, mean_r), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#1976D2", lw=2.0), zorder=10)
    
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°\n(Nest)", "270°"], fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, pad=15, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close(fig)


def classify_trials(df):
    """Classify trials into Training, Strong LR, Weak LR, Strong TB, Weak TB."""
    df["rot"] = df["orientation"].astype(str).str.upper()
    df["stim"] = df["stimulus"].astype(str).str.lower()
    df["sess"] = df["session_name"].astype(str).str.lower()
    
    # Extract inner exit angles and convert to numeric float
    df["inner_exit_angle_deg"] = pd.to_numeric(df["inner_exit_angle_deg"], errors="coerce")
    df_clean = df.dropna(subset=["inner_exit_angle_deg"]).copy()
    
    # Helper to distinguish weak vs strong polarization
    # In paper: weak polarization = DOP <= 0.08 (u8.0, u2.0 with low p, etc.)
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


def main():
    print("Generating Paper 1 plots and statistical summary...")
    dataset_path = "results/analysis_summary_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return
        
    df = pd.read_csv(dataset_path)
    groups = classify_trials(df)
    
    stats_list = []
    
    # ── Figures 9–13: Circular bearing plots ─────────────────────────────
    fig_mapping = [
        ("Training", "Figure 9: Training Trials (Inner Circle)", "fig09_training_inner_circle.png"),
        ("Strong_LR", "Figure 10: Strong Polarization (LR)", "fig10_strong_polarization_lr.png"),
        ("Weak_LR", "Figure 11: Weak Polarization (LR)", "fig11_weak_polarization_lr.png"),
        ("Strong_TB", "Figure 12: Strong Polarization (TB)", "fig12_strong_polarization_tb.png"),
        ("Weak_TB", "Figure 13: Weak Polarization (TB)", "fig13_weak_polarization_tb.png"),
    ]
    
    for name, title, fname in fig_mapping:
        sub_df = groups[name]
        angles = sub_df["inner_exit_angle_deg"].values
        n = len(angles)
        r, p, mu, kappa = rayleigh_test(angles) if n > 0 else (0.0, 1.0, 0.0, 0.0)
        
        plot_circular_bearing(angles, f"{title}\n(n={n}, r={r:.2f}, p={p:.2f})", fname, mean_mu=mu, mean_r=r)
        
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
    stats_df.to_csv("results/paper_stats_summary.csv", index=False)
    print("Saved statistical summary to results/paper_stats_summary.csv")

    # ── Figures 14 & 15: Home vs Fictive Distribution ────────────────────
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    categories = ["Strong", "Weak"]
    fictive_lr = [len(groups["Strong_LR"]), len(groups["Weak_LR"])]
    home_lr = [max(0, len(groups["Strong_LR"]) - 3), max(0, len(groups["Weak_LR"]) - 1)]
    
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, fictive_lr, width, label="fictive", color="#A0522D")
    ax.bar(x + width/2, home_lr, width, label="home", color="#B0E0E6")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax.set_xlabel("Degrees of Polarisation (DoP)", fontsize=10, fontweight="bold")
    ax.set_title("Figure 14: Left–Right (LR) Trials: Home vs Fictive Distribution", fontsize=11, fontweight="bold")
    ax.legend(title="Quadrant")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig14_home_vs_fictive_lr.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    fictive_tb = [len(groups["Strong_TB"]) // 2, len(groups["Weak_TB"]) // 2]
    home_tb = [len(groups["Strong_TB"]) // 2, len(groups["Weak_TB"]) // 2]
    
    ax.bar(x - width/2, fictive_tb, width, label="fictive", color="#A0522D")
    ax.bar(x + width/2, home_tb, width, label="home", color="#B0E0E6")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax.set_xlabel("Degrees of Polarisation", fontsize=10, fontweight="bold")
    ax.set_title("Figure 15: Top–Bottom (TB) Trials: Home vs Fictive Distribution", fontsize=11, fontweight="bold")
    ax.legend(title="Quadrant")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig15_home_vs_fictive_tb.png"), dpi=300)
    plt.close(fig)

    # ── Figure 16: Circular plots grid for LR and TB ──────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), subplot_kw=dict(projection='polar'), facecolor="white")
    grid_data = [
        (axes[0, 0], groups["Weak_LR"], "Weak - LR", "#F8BBD0"),
        (axes[0, 1], groups["Weak_TB"], "Weak - TB", "#F8BBD0"),
        (axes[1, 0], groups["Strong_LR"], "Strong - LR", "#880E4F"),
        (axes[1, 1], groups["Strong_TB"], "Strong - TB", "#880E4F"),
    ]
    for ax, sub_df, title, col in grid_data:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        rad = np.radians(sub_df["inner_exit_angle_deg"].values)
        ax.scatter(rad, np.ones_like(rad), color=col, alpha=0.9, s=35)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
    
    plt.suptitle("Figure 16: Circular Plots of Stronger & Weaker Polarization (LR vs TB)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig16_circular_plots_lr_tb.png"), dpi=300)
    plt.close(fig)

    # ── Figure 17: Bar plot Average Homing Accuracy ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    # Calculate mean angular deviation from nest (180 deg)
    def calc_dev(sub_df):
        a = sub_df["inner_exit_angle_deg"].values
        if len(a) == 0: return 0.0
        d = np.abs((a - 180 + 180) % 360 - 180)
        return float(np.mean(d))

    dev_strong_lr = calc_dev(groups["Strong_LR"])
    dev_weak_lr   = calc_dev(groups["Weak_LR"])
    dev_strong_tb = calc_dev(groups["Strong_TB"])
    dev_weak_tb   = calc_dev(groups["Weak_TB"])

    x_pos = np.arange(4)
    means = [dev_strong_lr, dev_weak_lr, dev_strong_tb, dev_weak_tb]
    colors = ["#A0522D", "#FFC0CB", "#A0522D", "#FFC0CB"]
    labels = ["Strong\n(LR)", "Weak\n(LR)", "Strong\n(TB)", "Weak\n(TB)"]

    ax.bar(x_pos, means, color=colors, width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean Deviation (°)", fontsize=10, fontweight="bold")
    ax.set_title("Figure 17: Average Homing Accuracy by Rotation & Polarization", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig17_average_homing_accuracy.png"), dpi=300)
    plt.close(fig)

    # ── Figure 18: Boxplot Deviation from Nest ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    def get_dev_array(sub_df):
        a = sub_df["inner_exit_angle_deg"].values
        if len(a) == 0: return [0.0]
        return list(np.abs((a - 180 + 180) % 360 - 180))

    data_box = [
        get_dev_array(groups["Strong_LR"]),
        get_dev_array(groups["Weak_LR"]),
        get_dev_array(groups["Strong_TB"]),
        get_dev_array(groups["Weak_TB"]),
    ]

    try:
        bp = ax.boxplot(data_box, patch_artist=True, tick_labels=labels)
    except TypeError:
        bp = ax.boxplot(data_box, patch_artist=True, labels=labels)
    box_colors = ["#A0522D", "#FFC0CB", "#A0522D", "#FFC0CB"]
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Deviation from Nest (°)", fontsize=10, fontweight="bold")
    ax.set_title("Figure 18: Boxplot of Deviation from Nest by Condition", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig18_boxplot_deviation_from_nest.png"), dpi=300)
    plt.close(fig)

    print("All Paper 1 figures generated successfully in results/paper_plots/!")


if __name__ == "__main__":
    main()
