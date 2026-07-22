@echo off
title Bee Arena Tracker Startup
echo === Starting Bee Arena Tracker ===

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
