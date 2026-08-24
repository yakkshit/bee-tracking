#!/bin/bash

# webapp.sh - Single-command launcher for Bee Arena Tracker web application via Docker

echo "=========================================================="
echo "          🐝 BEE ARENA TRACKER - WEB APPLICATION          "
echo "=========================================================="

# Check if Docker is installed
if ! command -v docker &>/dev/null; then
    echo "❌ Error: Docker is not installed or not in system PATH."
    echo "   Please install Docker Desktop (https://www.docker.com/products/docker-desktop) and try again."
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &>/dev/null; then
    echo "❌ Error: Docker daemon is not running."
    echo "   Please start Docker Desktop and run ./webapp.sh again."
    exit 1
fi

echo "🐳 Docker detected and running."
echo "🚀 Building and starting containerized web application..."
echo ""

# Ensure output directory exists
mkdir -p results

# Function to auto-open browser
open_browser() {
    local url="http://localhost:8501"
    echo "🌐 Waiting for web application to start at $url..."
    
    # Wait up to 30 seconds for server health check
    for i in {1..30}; do
        if curl -s http://localhost:8501/_stcore/health &>/dev/null || curl -s http://localhost:8501 &>/dev/null; then
            echo "✅ Application is live!"
            echo "🔗 Opening $url in your browser..."
            if [[ "$OSTYPE" == "darwin"* ]]; then
                open "$url"
            elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                xdg-open "$url" &>/dev/null || true
            elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
                start "$url" &>/dev/null || true
            fi
            return 0
        fi
        sleep 1
    done
}

# Run browser opener in background
open_browser &

# Launch container via docker-compose or docker compose
if command -v docker-compose &>/dev/null; then
    docker-compose up --build
else
    docker compose up --build
fi
