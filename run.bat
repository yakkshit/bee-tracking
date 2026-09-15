@echo off
title Bee Arena Tracker Startup
echo =================================================
echo           🐝 BEE ARENA TRACKER LAUNCHER          
echo =================================================
echo How would you like to run the application?
echo   [1] Locally (Python virtualenv)
echo   [2] Docker (Containerized)
echo =================================================
set /p choice="Select option (1 or 2) [Default: 1]: "
if "%choice%"=="" set choice=1

if "%choice%"=="2" (
    echo.
    echo === Launching with Docker ===
    docker-compose down
    docker-compose up -d --build
    if %errorlevel% neq 0 (
        docker compose down
        docker compose up -d --build
    )
    echo Docker container started in detached mode. Access the app at http://localhost:8501
    pause
    exit /b 0
)

echo.
echo === Launching Locally ===

rem Check if virtual environment directory exists
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else if exist .venv\bin\activate (
    echo Activating virtual environment...
    call .venv\bin\activate
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

rem Install requirements
if exist requirements.txt (
    echo Checking dependencies...
    pip install -r requirements.txt
)

rem Launch streamlit
echo Launching Streamlit application...
streamlit run app.py

pause

