"""
Streamlit Multi-Page: Analysis Viewer & Lab Report

Allows lab members to view all session trajectory plots, Paper 1 figures,
and underlying code alongside detailed scientific explanations.
"""
import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Analysis Viewer & Paper Plots", page_icon="🐝", layout="wide")

st.title("🐝 Bumblebee Navigation Analysis & Publication Viewer")
st.caption("Lab notebook and interactive visualizer displaying code, markdown explanations, and publication figures side-by-side.")

# Create tabs
tab_paper, tab_sessions, tab_html = st.tabs(["Paper 1 Figures & Stats", "Session Trajectories", "HTML Notebook Report & Downloads"])

# ── TAB 1: Paper 1 Figures & Stats ──────────────────────────────────────────
with tab_paper:
    st.header("Paper 1 Analysis & Statistical Replication (Figures 9–18)")
    st.markdown("""
    This section presents the replicated statistical figures from *Jansen (2025)* examining bumblebee path integration 
    under varying degrees of visual artificial polarization cues (**DoP**) and stimulus rotation (**LR vs TB**).
    """)

    # Display Stats Summary Table
    stats_path = "results/paper_stats_summary.csv"
    if os.path.exists(stats_path):
        st.subheader("Statistical Summary (Rayleigh Test & Circular Statistics)")
        df_stats = pd.read_csv(stats_path)
        st.dataframe(df_stats, use_container_width=True)
    
    st.divider()

    # Figures 9-13: Circular Plots
    st.subheader("Inner Circle Bearing Distributions (Figures 9–13)")
    st.markdown("""
    Each circular plot shows the bearing (angle in degrees) at which bumblebees exited the inner circle. 
    The blue arrow indicates the **mean vector ($\mu$)** with length proportional to the concentration ($r$). 
    The real nest entrance is located at **180°**.
    """)

    col1, col2 = st.columns(2)
    with col1:
        f10 = "results/paper_plots/fig10_strong_polarization_lr.png"
        if os.path.exists(f10):
            st.image(f10, caption="Figure 10: Strong Polarization (LR)")
            with st.expander("View Python Code (Figure 10)"):
                st.code("""
# Code generating Figure 10 (Strong Polarization LR)
strong_lr = df[(df['orientation'] == 'LR') & (~df['is_weak'])]
angles = strong_lr['inner_exit_angle_deg'].values
r, p, mu, kappa = rayleigh_test(angles)
plot_circular_bearing(angles, 'Figure 10: Strong Polarization (LR)', 'fig10_strong_polarization_lr.png', mean_mu=mu, mean_r=r)
                """, language="python")

    with col2:
        f12 = "results/paper_plots/fig12_strong_polarization_tb.png"
        if os.path.exists(f12):
            st.image(f12, caption="Figure 12: Strong Polarization (TB)")
            with st.expander("View Python Code (Figure 12)"):
                st.code("""
# Code generating Figure 12 (Strong Polarization TB)
strong_tb = df[(df['orientation'] == 'TB') & (~df['is_weak'])]
angles = strong_tb['inner_exit_angle_deg'].values
r, p, mu, kappa = rayleigh_test(angles)
plot_circular_bearing(angles, 'Figure 12: Strong Polarization (TB)', 'fig12_strong_polarization_tb.png', mean_mu=mu, mean_r=r)
                """, language="python")

    st.divider()

    # Figures 14 & 15: Distribution
    st.subheader("Home vs Fictive Nest Distribution (Figures 14–15)")
    st.markdown("""
    Compares the proportion of path decisions made toward the **fictive nest** (aligned with the rotated polarization axis) 
    versus the **real home nest** (180°).
    """)

    col3, col4 = st.columns(2)
    with col3:
        f14 = "results/paper_plots/fig14_home_vs_fictive_lr.png"
        if os.path.exists(f14):
            st.image(f14, caption="Figure 14: Left–Right (LR) Home vs Fictive")
    with col4:
        f15 = "results/paper_plots/fig15_home_vs_fictive_tb.png"
        if os.path.exists(f15):
            st.image(f15, caption="Figure 15: Top–Bottom (TB) Home vs Fictive")

    st.divider()

    # Figures 17 & 18: Accuracy & Deviation
    st.subheader("Homing Accuracy & Angular Deviation (Figures 17–18)")
    col5, col6 = st.columns(2)
    with col5:
        f17 = "results/paper_plots/fig17_average_homing_accuracy.png"
        if os.path.exists(f17):
            st.image(f17, caption="Figure 17: Average Homing Accuracy")
    with col6:
        f18 = "results/paper_plots/fig18_boxplot_deviation_from_nest.png"
        if os.path.exists(f18):
            st.image(f18, caption="Figure 18: Boxplot Deviation from Nest (Cleaned Boxplots)")

    st.divider()

    # Figures 24, 25, 26: Movement Speed, Memory, and Feeder Visit Rate
    st.subheader("Movement Speed, Memory Preference & Feeder Visit Analysis (Figures 24–26)")
    col7, col8 = st.columns(2)
    with col7:
        f24 = "results/paper_plots/fig24_speed_of_movement_analysis.png"
        if os.path.exists(f24):
            st.image(f24, caption="Figure 24: Speed of Movement Analysis (mm/s)")
    with col8:
        f25 = "results/paper_plots/fig25_short_vs_long_term_memory.png"
        if os.path.exists(f25):
            st.image(f25, caption="Figure 25: Short-Term (STM) vs Long-Term Memory (LTM) Preference (%)")

    f26 = "results/paper_plots/fig26_feeder_visit_percentage.png"
    if os.path.exists(f26):
        st.image(f26, caption="Figure 26: Feeder Visit Percentage Rate (%) across Conditions")


# ── TAB 2: Session Trajectories ─────────────────────────────────────────────
with tab_sessions:
    st.header("Session Trajectory Visualizer")
    st.markdown("""
    Browse through individual video sessions to inspect the **Single-Colour** (strictly 1 inbound + 1 outbound pass) 
    and **Multiple Entries/Exits** (all passes + Mean Trajectory) plots side-by-side.
    """)

    results_dir = "results"
    sessions = [d for d in sorted(os.listdir(results_dir)) if os.path.isdir(os.path.join(results_dir, d)) and d != "paper_plots"]

    if sessions:
        selected_session = st.selectbox("Select Video Session:", sessions)
        session_path = os.path.join(results_dir, selected_session)
        plots_dir = os.path.join(session_path, "plots")

        # Session metadata
        csv_path = "results/analysis_summary_dataset.csv"
        if os.path.exists(csv_path):
            df_sum = pd.read_csv(csv_path)
            sess_row = df_sum[df_sum["session_name"] == selected_session]
            if not sess_row.empty:
                st.subheader(f"Session Metadata: `{selected_session}`")
                r = sess_row.iloc[0]
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Bee ID", str(r.get("bee_id", "N/A")))
                m_col2.metric("Stimulus", str(r.get("stimulus", "N/A")))
                m_col3.metric("Rotation", str(r.get("orientation", "N/A")))
                m_col4.metric("Inner Exit Angle", f"{float(r.get('inner_exit_angle_deg', 0)):.1f}°" if pd.notnull(r.get('inner_exit_angle_deg')) else "N/A")

        st.divider()

        sc_img = os.path.join(plots_dir, "single_colour.png")
        me_img = os.path.join(plots_dir, "multiple_entries_exits.png")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.subheader("1. Single-Colour Plot")
            st.caption("Clean 1-pass trajectory: Outer Entry → 1st Inner Contact → Feeder → 1st Inner Exit → Outer Exit.")
            if os.path.exists(sc_img):
                st.image(sc_img, use_column_width=True)
            else:
                st.info("Single colour plot not found.")
            with st.expander("View Python Code (Single Colour)"):
                st.code("""
# Trajectory Clipping Logic for Single-Colour Plot
in_x, in_y = clip_and_prepare(x_all, y_all, first_outer_entry, feeder_idx)
out_x, out_y = clip_and_prepare(x_all, y_all, feeder_idx, first_outer_exit)

ax.plot(in_x, in_y, linestyle=":", color="#888888", label="Inbound")
ax.plot(out_x, out_y, linestyle="-", color="#1E88E5", label="Outbound")
                """, language="python")

        with p_col2:
            st.subheader("2. Multiple Entries & Exits Plot")
            st.caption("All passes across inner boundary with numbered markers and bold black Mean Trajectory.")
            if os.path.exists(me_img):
                st.image(me_img, use_column_width=True)
            else:
                st.info("Multiple entries plot not found.")
            with st.expander("View Python Code (Multiple Entries)"):
                st.code("""
# Multiple Passes & Mean Trajectory Logic
for i, (p_start, p_end) in enumerate(passes):
    ax.plot(px, py, linestyle="--", color=PASS_COLORS[i], label=f"Pass {i+1}")

mean_x = np.mean([p[0] for p in resampled_paths], axis=0)
mean_y = np.mean([p[1] for p in resampled_paths], axis=0)
ax.plot(mean_x, mean_y, color="#000000", lw=2.5, label="Mean Trajectory")
                """, language="python")


# ── TAB 3: HTML Notebook & Downloads ────────────────────────────────────────
with tab_html:
    st.header("HTML Notebook Report & Downloads")
    st.markdown("""
    Download the complete standalone **HTML Analysis Report** or view it embedded below. 
    It presents all code cells, markdown explanations, and figures together following lab best practices.
    """)

    html_file = "results/paper_and_session_analysis.html"
    if os.path.exists(html_file):
        with open(html_file, "rb") as f:
            html_bytes = f.read()

        st.download_button(
            label="📥 Download HTML Report (paper_and_session_analysis.html)",
            data=html_bytes,
            file_name="paper_and_session_analysis.html",
            mime="text/html"
        )

        st.divider()
        st.subheader("Interactive HTML Report Preview")
        components.html(html_bytes.decode("utf-8", errors="ignore"), height=800, scrolling=True)
    else:
        st.warning("HTML report has not been generated yet.")
