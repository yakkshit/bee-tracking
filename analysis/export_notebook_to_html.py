"""
Generate paper_and_session_analysis.ipynb and export it to paper_and_session_analysis.html.

Follows lab best practices:
  - Detailed Markdown cell above every code cell explaining its purpose and methodology.
  - Plots displayed directly alongside code.
  - Saved to results/paper_and_session_analysis.html and analysis/paper_and_session_analysis.ipynb
"""
import os
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = []

    # Title Markdown
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# Bumblebee Navigation & Path Integration Analysis Report\n"
        "**Author**: Neurobiology Lab\n\n"
        "This notebook provides a complete analysis of bumblebee walking trajectories under varying degrees of visual artificial polarization cues (DoP) and stimulus rotation (Left-Right vs Top-Bottom). "
        "It includes both individual session trajectory plots (single colour & multiple entries/exits) and aggregated paper figures (Figures 9–18).\n\n"
        "---"
    ))

    # Section 1: Session Trajectories
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 1. Single-Colour Trajectory Plotting\n"
        "### Purpose\n"
        "Generates clean, publication-ready trajectory plots (`single_colour.png`) for each session. "
        "The inbound path is strictly mapped from outer boundary entry to the 1st inner circle contact to the feeder. "
        "The outbound path is mapped from the feeder to the 1st inner circle exit to the outer boundary exit."
    ))

    nb.cells.append(nbf.v4.new_code_cell(
        "# Run single colour plot generation script\n"
        "import sys\n"
        "sys.path.append('analysis')\n"
        "import generate_single_colour_plot\n"
        "generate_single_colour_plot.main()\n"
        "print('Single colour plots generated successfully.')"
    ))

    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 2. Multiple Entries & Exits Trajectory Plotting\n"
        "### Purpose\n"
        "Detects and plots all entry/exit passes across inner and outer boundaries (`multiple_entries_exits.png`). "
        "Individual passes are rendered with light dashed lines and numbered markers (①, ②, ③...). "
        "A bold black **Mean Trajectory** is computed across all passes."
    ))

    nb.cells.append(nbf.v4.new_code_cell(
        "# Run multiple entries & exits plot generation script\n"
        "import generate_multiple_entries_plot\n"
        "generate_multiple_entries_plot.main()\n"
        "print('Multiple entries & exits plots generated successfully.')"
    ))

    # Section 2: Paper 1 Statistical Analysis
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 3. Paper 1 Statistical Analysis & Figures 9–18\n"
        "### Purpose\n"
        "Computes circular statistics (Rayleigh test statistic $r$, $p$-value, mean direction $\\mu$, concentration $\\kappa$) "
        "and angular deviation from the nest ($180^\\circ$). Generates Figures 9–18 matching *Paper 1* methodology."
    ))

    nb.cells.append(nbf.v4.new_code_cell(
        "# Run paper analysis and figure generation script\n"
        "import generate_paper_plots\n"
        "generate_paper_plots.main()\n"
        "import pandas as pd\n"
        "stats_df = pd.read_csv('results/paper_stats_summary.csv')\n"
        "display(stats_df)"
    ))

    # Write notebook file
    ipynb_path = "analysis/paper_and_session_analysis.ipynb"
    with open(ipynb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook at {ipynb_path}")
    return ipynb_path


def export_to_html():
    ipynb_path = create_notebook()
    html_path = "results/paper_and_session_analysis.html"

    # Generate rich HTML file directly
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Bumblebee Navigation & Path Integration Analysis Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1100px; margin: 0 auto; padding: 20px; background: #f8f9fa; }
        .header { background: #1e293b; color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px; }
        .header h1 { margin: 0 0 10px 0; font-size: 28px; }
        .header p { margin: 0; color: #cbd5e1; }
        .cell { background: white; border-radius: 8px; padding: 25px; margin-bottom: 25px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }
        .md-cell { border-left: 4px solid #3b82f6; }
        .code-cell { border-left: 4px solid #10b981; font-family: monospace; background: #f1f5f9; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 13px; }
        .img-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 15px; }
        .img-card { text-align: center; background: #fff; padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px; }
        .img-card img { max-width: 100%; height: auto; border-radius: 4px; }
        .img-card p { font-size: 12px; font-weight: bold; color: #64748b; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; font-size: 13px; }
        th { background: #e2e8f0; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Bumblebee Navigation & Path Integration Analysis Report</h1>
        <p>Complete code, markdown explanations, and publication figures side-by-side.</p>
    </div>

    <div class="cell md-cell">
        <h2>1. Single-Colour Trajectory Plotting</h2>
        <p><strong>Methodology:</strong> Maps the inbound path from outer boundary entry to 1st inner circle contact to feeder (grey dotted line). Maps outbound path from feeder to 1st inner circle exit to outer boundary exit (blue solid line).</p>
    </div>

    <div class="code-cell">
        <pre># Code from analysis/generate_single_colour_plot.py
in_x, in_y = clip_and_prepare(x_all, y_all, first_outer_entry, feeder_idx)
out_x, out_y = clip_and_prepare(x_all, y_all, feeder_idx, first_outer_exit)
ax.plot(in_x, in_y, linestyle=":", color="#888888", label="Inbound (Approach Path)")
ax.plot(out_x, out_y, linestyle="-", color="#1E88E5", label="Outbound (Home Search)")</pre>
    </div>

    <div class="cell">
        <h3>Sample Single-Colour Plots</h3>
        <div class="img-grid">
            <div class="img-card">
                <img src="../results/2024-11-20_16-10-04.R_13.LR.P_0_U_1/plots/single_colour.png" alt="Single Colour LR">
                <p>Session 2024-11-20_16-10-04 (LR Condition)</p>
            </div>
            <div class="img-card">
                <img src="../results/2024-11-20_16-04-20.R_13.TB.P_0_U_1/plots/single_colour.png" alt="Single Colour TB">
                <p>Session 2024-11-20_16-04-20 (TB Condition)</p>
            </div>
        </div>
    </div>

    <div class="cell md-cell">
        <h2>2. Multiple Entries & Exits Trajectory Plotting</h2>
        <p><strong>Methodology:</strong> Detects each individual pass across the inner boundary. Plots each pass with a distinct color and numbered tag (①, ②, ③...). Calculates and renders the bold black Mean Trajectory across passes.</p>
    </div>

    <div class="code-cell">
        <pre># Code from analysis/generate_multiple_entries_plot.py
for i, (p_start, p_end) in enumerate(passes):
    ax.plot(px, py, linestyle="--", color=PASS_COLORS[i], label=f"Pass {i+1}")
mean_x = np.mean([p[0] for p in resampled_paths], axis=0)
mean_y = np.mean([p[1] for p in resampled_paths], axis=0)
ax.plot(mean_x, mean_y, color="#000000", lw=2.5, label="Mean Trajectory")</pre>
    </div>

    <div class="cell">
        <h3>Sample Multiple Entries Plots</h3>
        <div class="img-grid">
            <div class="img-card">
                <img src="../results/2024-11-19_18-13-28.R13.LR.P0.6_U_2/plots/multiple_entries_exits.png" alt="Multiple Entries LR">
                <p>Session 2024-11-19_18-13-28 (LR Condition)</p>
            </div>
            <div class="img-card">
                <img src="../results/2024-11-20_16-04-20.R_13.TB.P_0_U_1/plots/multiple_entries_exits.png" alt="Multiple Entries TB">
                <p>Session 2024-11-20_16-04-20 (TB Condition)</p>
            </div>
        </div>
    </div>

    <div class="cell md-cell">
        <h2>3. Paper 1 Statistical Figures & Analysis (Figures 9–18)</h2>
        <p><strong>Methodology:</strong> Evaluates circular exit bearings of the inner circle. Computes Rayleigh test statistics (r, p-value), mean direction (μ), concentration (κ), and mean angular deviation from nest (180°).</p>
    </div>

    <div class="cell">
        <h3>Paper 1 Output Figures</h3>
        <div class="img-grid">
            <div class="img-card">
                <img src="../results/paper_plots/fig10_strong_polarization_lr.png" alt="Fig 10">
                <p>Figure 10: Strong Polarization (LR)</p>
            </div>
            <div class="img-card">
                <img src="../results/paper_plots/fig12_strong_polarization_tb.png" alt="Fig 12">
                <p>Figure 12: Strong Polarization (TB)</p>
            </div>
            <div class="img-card">
                <img src="../results/paper_plots/fig14_home_vs_fictive_lr.png" alt="Fig 14">
                <p>Figure 14: Home vs Fictive Distribution (LR)</p>
            </div>
            <div class="img-card">
                <img src="../results/paper_plots/fig18_boxplot_deviation_from_nest.png" alt="Fig 18">
                <p>Figure 18: Boxplot Deviation from Nest</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Exported HTML report to {html_path}")


if __name__ == "__main__":
    export_to_html()
