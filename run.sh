#!/bin/bash

# Cross-platform interactive launcher for Bee Arena Tracker

# If already inside a Docker container, execute app directly
if [ -f "/.dockerenv" ]; then
    echo "=== Starting Bee Arena Tracker inside Docker container ==="
    streamlit run app.py
    exit 0
fi

echo "================================================="
echo "          🐝 BEE ARENA TRACKER LAUNCHER          "
echo "================================================="
echo "How would you like to run the application?"
echo "  [1] Launch Locally (Use existing .venv if healthy)"
echo "  [2] Fresh Reinstall Environment (Remove & recreate .venv)"
echo "  [3] Docker Containerized"
echo "================================================="
read -p "Select option (1, 2, or 3) [Default: 1]: " CHOICE

CHOICE=${CHOICE:-1}

if [ "$1" == "--clean" ] || [ "$1" == "clean" ]; then
    CHOICE="2"
fi

if [ "$CHOICE" == "3" ]; then
    echo ""
    echo "=== Launching with Docker ==="
    
    if ! command -v docker &>/dev/null; then
        echo "Error: Docker command not found. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &>/dev/null; then
        echo "Error: Docker daemon is not running. Please start Docker Desktop and try again."
        exit 1
    fi

    if command -v docker-compose &>/dev/null; then
        docker-compose down
        docker-compose up -d --build
    else
        docker compose down
        docker compose up -d --build
    fi
    echo "Docker container started in detached mode. Access the app at http://localhost:8501"
    exit 0
fi

echo ""
echo "=== Launching Locally ==="

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

# Check if clean reinstall requested
if [ "$CHOICE" == "2" ]; then
    echo "🧹 Removing existing .venv environment..."
    rm -rf .venv
fi

# Test existing .venv python if present
if [ -d ".venv" ]; then
    VENV_PYTHON=""
    if [ -f ".venv/bin/python" ]; then
        VENV_PYTHON=".venv/bin/python"
    elif [ -f ".venv/Scripts/python.exe" ]; then
        VENV_PYTHON=".venv/Scripts/python.exe"
    fi

    if [ -n "$VENV_PYTHON" ]; then
        if ! "$VENV_PYTHON" -c "import streamlit" &>/dev/null; then
            echo "⚠️ Existing .venv appears broken or incomplete. Recreating fresh .venv..."
            rm -rf .venv
        fi
    fi
fi

# Determine virtual environment activation path
ACTIVATE_PATH=""
if [ -d ".venv" ]; then
    if [ -f ".venv/bin/activate" ]; then
        ACTIVATE_PATH=".venv/bin/activate"
    elif [ -f ".venv/Scripts/activate" ]; then
        ACTIVATE_PATH=".venv/Scripts/activate"
    fi
fi

if [ -z "$ACTIVATE_PATH" ]; then
    echo "No valid .venv found. Checking Python..."
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "Error: Python is not installed or not in PATH."
        exit 1
    fi
    
    echo "Creating fresh virtual environment using $PYTHON_CMD..."
    $PYTHON_CMD -m venv .venv
    
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

if [ -f "requirements.txt" ]; then
    echo "Installing/verifying dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

echo "Launching Streamlit application..."
streamlit run app.py
