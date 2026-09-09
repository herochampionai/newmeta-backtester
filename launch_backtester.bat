@echo off
title Newmeta Backtester
echo ============================================
echo   NEWMETA BACKTESTER - Universal Strategy Tester
echo ============================================
echo.

REM Set working directory to the backtester folder
cd /d "%~dp0"

REM Verify Python is available
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    echo Install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

REM Set PYTHONPATH so imports work
set PYTHONPATH=%cd%

REM Open browser after 3 seconds (gives Streamlit time to start)
start /min cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:8501"

REM Launch Streamlit (auto-detect port)
echo Starting Streamlit... browser will open automatically.
echo Close this window or press Ctrl+C to stop.
echo.
python -m streamlit run frontend\app.py --server.headless false --server.port 8501 --browser.gatherUsageStats false --theme.base dark --theme.primaryColor "#00d4aa" --theme.backgroundColor "#0e1117" --theme.secondaryBackgroundColor "#1a1f2e" --theme.textColor "#e8eaf0"

pause