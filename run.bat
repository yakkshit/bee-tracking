@echo off
title Bee Arena Tracker Startup
echo =================================================
echo           🐝 BEE ARENA TRACKER LAUNCHER          
echo =================================================
echo How would you like to run the application?
echo   [1] Launch Locally (Use existing .venv if healthy)
echo   [2] Clean Reinstall Environment (Remove & recreate .venv)
echo   [3] Docker Containerized
echo =================================================
set /p choice="Select option (1, 2, or 3) [Default: 1]: "
if "%choice%"=="" set choice=1

if "%choice%"=="3" (
    echo.
    echo === Launching with Docker ===
    docker-compose down 2>nul
    docker-compose up -d --build
    if %errorlevel% neq 0 (
        docker compose down 2>nul
        docker compose up -d --build
    )
    echo Docker container started in detached mode. Access the app at http://localhost:8501
    pause
    exit /b 0
)

if "%choice%"=="2" (
    echo.
    echo Removing existing .venv environment...
    if exist .venv rmdir /s /q .venv
)

echo.
echo === Launching Locally ===

set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1

if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        echo Error: Python is not installed or not in PATH.
        pause
        exit /b 1
    )
    python -m venv .venv
    call .venv\Scripts\activate.bat
)

if exist requirements.txt (
    echo Checking dependencies...
    pip install -r requirements.txt
)

echo Launching Streamlit application...
streamlit run app.py

pause
