@echo off
rem Double-click to open the NFL DFS Optimizer in your browser.
rem Close this window to stop the app.
cd /d "%~dp0"
uv run streamlit run src\nfl_dfs_optimizer\app.py
if errorlevel 1 pause
