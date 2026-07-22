#!/bin/bash

# Cross-platform startup script for Bee Arena Tracker (macOS, Linux, and Windows Git Bash/WSL)

echo "=== Starting Bee Arena Tracker ==="

# Detect OS
OS_TYPE="Unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="Linux"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    OS_TYPE="Windows"
fi
echo "Detected OS: $OS_TYPE"

# Determine the virtual environment activation script path
ACTIVATE_PATH=""
if [ -d ".venv" ]; then
    if [ -f ".venv/bin/activate" ]; then
        ACTIVATE_PATH=".venv/bin/activate"
    elif [ -f ".venv/Scripts/activate" ]; then
        ACTIVATE_PATH=".venv/Scripts/activate"
    fi
fi

# Activate virtual environment if found
if [ -n "$ACTIVATE_PATH" ]; then
    echo "Activating virtual environment: $ACTIVATE_PATH"
    source "$ACTIVATE_PATH"
else
    echo "No .venv found. Checking Python/Pip..."
    # Fallback to system Python
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "Error: Python is not installed or not in PATH."
        exit 1
    fi
    
    echo "Creating virtual environment using $PYTHON_CMD..."
    $PYTHON_CMD -m venv .venv
    
    if [ -f ".venv/bin/activate" ]; then
        ACTIVATE_PATH=".venv/bin/activate"
    elif [ -f ".venv/Scripts/activate" ]; then
        ACTIVATE_PATH=".venv/Scripts/activate"
    fi
    
    if [ -n "$ACTIVATE_PATH" ]; then
        source "$ACTIVATE_PATH"
    else
        echo "Error: Could not activate virtual environment."
        exit 1
    fi
fi

# Ensure dependencies are installed
if [ -f "requirements.txt" ]; then
    echo "Checking dependencies..."
    pip install -r requirements.txt
fi

# Start Streamlit application
echo "Launching Streamlit application..."
streamlit run app.py
