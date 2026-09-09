@echo off
title Newmeta Backtester
echo ============================================
echo   NEWMETA BACKTESTER - Universal Strategy Tester
echo ============================================
echo.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    pause
    exit /b 1
)

set PYTHONPATH=%cd%

REM Open browser after 4 seconds (gives Streamlit time to start)
start /min cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:8501"

REM Use pythonw.exe if available (no console window for the launcher)
where pythonw >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_BIN=pythonw"
) else (
    set "PYTHON_BIN=python"
)

echo.
echo Starting Streamlit on http://localhost:8501
echo Browser will open automatically.
echo.

REM Pipe empty line to skip the email prompt
(echo.| %PYTHON_BIN% -m streamlit run frontend\app.py)

pause