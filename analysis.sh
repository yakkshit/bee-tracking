#!/bin/bash

# Cross-platform interactive script to run analysis and plot generation for Bee Arena Tracker

echo "=========================================================="
echo "         Bee Arena Tracker - Plot & Analysis Tool         "
echo "=========================================================="

# Determine the virtual environment activation script path
ACTIVATE_PATH=""
if [ -d ".venv" ]; then
    if [ -f ".venv/bin/activate" ]; then
        ACTIVATE_PATH=".venv/bin/activate"
    elif [ -f ".venv/Scripts/activate" ]; then
        ACTIVATE_PATH=".venv/Scripts/activate"
    fi
fi

if [ -n "$ACTIVATE_PATH" ]; then
    echo "Activating virtual environment: $ACTIVATE_PATH"
    source "$ACTIVATE_PATH"
fi

# Determine python command
if command -v python &>/dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python executable not found."
    exit 1
fi

echo "Using Python executable: $(which $PYTHON_CMD 2>/dev/null || echo $PYTHON_CMD)"

# Ensure requirements are satisfied if needed
if [ -f "requirements.txt" ]; then
    echo "Checking dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt --quiet
fi

echo ""
echo "Select Analysis Mode:"
echo "  1) Individual Session Analysis Plots"
echo "  2) Paper Plots & Statistical Summaries"
echo ""
read -p "Enter choice [1 or 2]: " MAIN_CHOICE

echo ""
read -p "Enter path to Videos/Data folder (Press Enter for default: '.'): " VIDEOS_PATH
if [ -z "$VIDEOS_PATH" ]; then
    VIDEOS_PATH="."
fi

if [ ! -d "$VIDEOS_PATH" ]; then
    echo "Warning: Path '$VIDEOS_PATH' does not exist. Using current directory '.' instead."
    VIDEOS_PATH="."
fi

export VIDEOS_PATH

echo ""
read -p "Enter path to SAVE outcome/plots folder (Press Enter for default: 'results'): " OUTPUT_PATH
if [ -z "$OUTPUT_PATH" ]; then
    OUTPUT_PATH="results"
fi
mkdir -p "$OUTPUT_PATH"
export OUTPUT_DIR="$OUTPUT_PATH"
echo "Outcome will be saved to: $OUTPUT_DIR"

if [ "$MAIN_CHOICE" == "1" ]; then
    echo ""
    echo "=== Running Individual Trajectory Analysis Plots ==="
    
    echo ""
    echo "Select plot type to run:"
    echo "  1) All Individual Trajectory Plots"
    echo "  2) Full Trial Trajectories (complete_trajectory.png)"
    echo "  3) Feeder Exit to Inner Circle (feeder_to_inner_circle.png)"
    echo "  4) Feeder Exit to Final Exit (feeder_to_final_exit.png)"
    echo "  5) Single-colour Trajectories (single_colour.png)"
    echo "  6) Gradient Coloured Trajectories (gradient_coloured.png)"
    echo "  7) 60fps Gradient Plots"
    echo "  8) Smooth Path-Length Gradient Plots"
    echo "  9) Multiple Entries & Exits Plots"
    echo " 10) Minimalist Clean Fixed Trajectory Plots"
    echo " 11) Notebook & HTML Analysis Report"
    echo ""
    read -p "Enter choice [1-11]: " INDIV_CHOICE

    case $INDIV_CHOICE in
        1)
            $PYTHON_CMD analysis/generate_analysis_plots.py
            $PYTHON_CMD analysis/generate_feeder_to_inner_circle_plot.py
            $PYTHON_CMD analysis/generate_feeder_to_final_exit_plot.py
            $PYTHON_CMD analysis/generate_single_colour_plot.py
            $PYTHON_CMD analysis/generate_gradient_coloured_plot.py
            $PYTHON_CMD analysis/generate_60fps_gradient_plots.py
            $PYTHON_CMD analysis/generate_smooth_path_gradient_plots.py
            $PYTHON_CMD analysis/generate_multiple_entries_plot.py
            $PYTHON_CMD analysis/generate_fixed_clean_trajectory_plot.py
            $PYTHON_CMD analysis/fixed_version.py
            $PYTHON_CMD analysis/gradient_fixed.py
            $PYTHON_CMD analysis/export_notebook_to_html.py
            ;;
        2) $PYTHON_CMD analysis/generate_analysis_plots.py ;;
        3) $PYTHON_CMD analysis/generate_feeder_to_inner_circle_plot.py ;;
        4) $PYTHON_CMD analysis/generate_feeder_to_final_exit_plot.py ;;
        5) $PYTHON_CMD analysis/generate_single_colour_plot.py ;;
        6) $PYTHON_CMD analysis/generate_gradient_coloured_plot.py ;;
        7) $PYTHON_CMD analysis/generate_60fps_gradient_plots.py ;;
        8) $PYTHON_CMD analysis/generate_smooth_path_gradient_plots.py ;;
        9) $PYTHON_CMD analysis/generate_multiple_entries_plot.py ;;
        10)
            $PYTHON_CMD analysis/generate_fixed_clean_trajectory_plot.py
            $PYTHON_CMD analysis/fixed_version.py
            $PYTHON_CMD analysis/gradient_fixed.py
            ;;
        11) $PYTHON_CMD analysis/export_notebook_to_html.py ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac

elif [ "$MAIN_CHOICE" == "2" ]; then
    echo ""
    echo "=== Paper Plots & Statistical Summaries Menu ==="
    echo "Select Dataset Scope:"
    echo "  1) Combined Analysis (All Tracked Datasets -> 'results/f+m_paper_plots')"
    echo "  2) Individual Tracking Analysis: F Dataset -> 'results/F/F_Paper_plots'"
    echo "  3) Individual Tracking Analysis: Marie's Data -> 'results/maries data/m_paper_plots'"
    echo "  4) Custom Output Path Analysis"
    echo ""
    read -p "Enter scope choice [1-4]: " SCOPE_CHOICE

    case $SCOPE_CHOICE in
        1)
            export OUTPUT_DIR="results/f+m_paper_plots"
            unset TARGET_DIR
            ;;
        2)
            export OUTPUT_DIR="results/F/F_Paper_plots"
            export TARGET_DIR="results/F"
            ;;
        3)
            export OUTPUT_DIR="results/maries data/m_paper_plots"
            export TARGET_DIR="results/maries data"
            ;;
        4)
            read -p "Enter custom SAVE directory path: " CUSTOM_OUT
            if [ -n "$CUSTOM_OUT" ]; then
                export OUTPUT_DIR="$CUSTOM_OUT"
            fi
            read -p "Enter specific input dataset directory path (leave blank for all): " CUSTOM_IN
            if [ -n "$CUSTOM_IN" ]; then
                export TARGET_DIR="$CUSTOM_IN"
            fi
            ;;
        *)
            echo "Using default combined output path: $OUTPUT_DIR"
            ;;
    esac

    mkdir -p "$OUTPUT_DIR"
    echo "Outcome plots will be saved to: $OUTPUT_DIR"

    echo ""
    echo "Select figure options to generate:"
    echo "  1) ALL Paper Figures, Heatmaps & Memory Plots (Figs 9-26)"
    echo "  2) Circular Bearings & Circ-MLE Fit Plots (Figs 9-13: Training, Strong/Weak LR & TB)"
    echo "  3) Home vs Fictive Distribution Bar Charts (Figs 14-15)"
    echo "  4) Multi-Condition Circular Grid Plot (Fig 16)"
    echo "  5) Homing Accuracy & Deviation Boxplots (Figs 17-18)"
    echo "  6) Trajectory Occupancy Heatmaps (Figs 19-21: Strong vs Weak, 2x2 Grid, Overall)"
    echo "  7) Exit Angle Density Polar & Matrix Heatmaps (Figs 22-23)"
    echo "  8) Generate Statistical Summary CSV only (paper_stats_summary.csv)"
    echo ""
    read -p "Enter figure choice [1-8]: " PAPER_CHOICE

    export PAPER_PLOT_OPTION="$PAPER_CHOICE"
    $PYTHON_CMD analysis/generate_paper_plots.py
    if [ "$PAPER_CHOICE" == "1" ]; then
        $PYTHON_CMD analysis/generate_speed_plot.py
    fi

else
    echo "Invalid choice. Please run script again and select option 1 or 2."
    exit 1
fi

echo ""
echo "=== Processing Completed Successfully! ==="