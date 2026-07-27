#!/bin/bash

# Cross-platform script to run all analysis and plot generation scripts for Bee Arena Tracker

echo "=== Running Bee Arena Tracker Plot & Analysis Scripts ==="

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
echo "[1/5] Generating analysis plots (complete trajectory & approach)..."
$PYTHON_CMD analysis/generate_analysis_plots.py

echo ""
echo "[2/5] Generating single-colour trajectory plots..."
$PYTHON_CMD analysis/generate_single_colour_plot.py

echo ""
echo "[3/5] Generating multiple entries & exits plots..."
$PYTHON_CMD analysis/generate_multiple_entries_plot.py

echo ""
echo "[4/5] Generating paper statistical plots (Figures 9-18)..."
$PYTHON_CMD analysis/generate_paper_plots.py

echo ""
echo "[5/5] Exporting notebook & HTML analysis report..."
$PYTHON_CMD analysis/export_notebook_to_html.py

echo ""
echo "=== All plot generation scripts completed successfully! ==="
