@echo off
rem Double-click to open the NFL DFS Optimizer in your browser.
rem The server runs in this window, minimized to the taskbar; close it to stop the app.
if not "%~1"=="--minimized" (
    start "NFL DFS Optimizer (close to stop)" /min "%~f0" --minimized
    exit /b
)
cd /d "%~dp0"
uv run streamlit run src\nfl_dfs_optimizer\app.py
if errorlevel 1 pause
