@echo off
title Newmeta Backtester API
cd /d "%~dp0"
set PYTHONPATH=%cd%
echo Starting Newmeta Backtester API on http://127.0.0.1:8765
echo This serves the Python harness to the NewMeta Terminal widget.
echo Close this window or press Ctrl+C to stop.
echo.
python -m uvicorn backtester.api:app --host 127.0.0.1 --port 8765 --log-level info
pause
