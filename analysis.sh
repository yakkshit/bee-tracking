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
elif [ -d ".venv_new" ]; then
    if [ -f ".venv_new/bin/activate" ]; then
        ACTIVATE_PATH=".venv_new/bin/activate"
    elif [ -f ".venv_new/Scripts/activate" ]; then
        ACTIVATE_PATH=".venv_new/Scripts/activate"
    fi
fi

if [ -n "$ACTIVATE_PATH" ]; then
    echo "Activating virtual environment: $ACTIVATE_PATH"
    source "$ACTIVATE_PATH"
fi

# Determine python command
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
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
echo "Select Option:"
echo "  1) Individual Analysis"
echo "  2) Complete Paper Plots"
echo ""
read -p "Enter choice [1 or 2]: " MAIN_CHOICE

if [ "$MAIN_CHOICE" == "1" ]; then
    echo ""
    echo "=== 1. Individual Analysis ==="
    read -p "Enter path to folder where tracking sessions are located (Press Enter for 'results'): " TRACKING_PATH
    if [ -z "$TRACKING_PATH" ]; then
        TRACKING_PATH="results"
    fi

    if [ ! -d "$TRACKING_PATH" ]; then
        echo "Error: Directory '$TRACKING_PATH' does not exist."
        exit 1
    fi

    export TARGET_DIR="$TRACKING_PATH"
    export VIDEOS_PATH="$TRACKING_PATH"
    echo ""
    echo "Running individual analysis for tracking sessions in: $TARGET_DIR"
    $PYTHON_CMD analysis/run_individual_analysis.py "$TARGET_DIR"

elif [ "$MAIN_CHOICE" == "2" ]; then
    echo ""
    echo "=== 2. Complete Paper Plots ==="
    read -p "Enter path where individual tracking is located (Press Enter for 'results'): " TRACKING_PATH
    if [ -z "$TRACKING_PATH" ]; then
        TRACKING_PATH="results"
    fi

    if [ ! -d "$TRACKING_PATH" ]; then
        echo "Error: Directory '$TRACKING_PATH' does not exist."
        exit 1
    fi

    export TARGET_DIR="$TRACKING_PATH"
    export OUTPUT_DIR="${TRACKING_PATH%/}/paper_plots"
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "Input dataset directory: $TARGET_DIR"
    echo "Output paper plots directory: $OUTPUT_DIR"
    echo ""

    export PAPER_PLOT_OPTION="1"
    $PYTHON_CMD analysis/generate_paper_plots.py
    $PYTHON_CMD analysis/generate_speed_plot.py

else
    echo "Invalid choice. Please enter 1 or 2."
    exit 1
fi

echo ""
echo "=== Processing Completed Successfully! ==="