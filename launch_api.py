"""Launches the Newmeta Backtester HTTP API in the background.
Used by the NewMeta Terminal widget to call backtests.

Also launches the Streamlit UI for direct browser use.

Adds three desktop shortcuts:
  1. Newmeta Backtester        (Streamlit UI)
  2. Newmeta Backtester API    (HTTP API for widget integration)
  3. Newmeta Terminal         (your widget tool, if not already linked)
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API_URL = "http://127.0.0.1:8765/health"


def launch_api():
    """Start the FastAPI server on port 8765."""
    project_root = Path(__file__).parent
    print(f"[API] Starting on http://127.0.0.1:8765")
    print(f"[API] Project root: {project_root}")
    # Run uvicorn as a subprocess so it persists
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backtester.api:app",
         "--host", "127.0.0.1", "--port", "8765"],
        cwd=project_root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    # Wait for health check
    for _ in range(20):
        try:
            urllib.request.urlopen(API_URL, timeout=1)
            print(f"[API] Healthy at {API_URL}")
            return proc
        except Exception:
            time.sleep(0.5)
    print("[API] Warning: did not respond within 10s. Check logs.")
    return proc


def create_api_shortcut():
    """Create a desktop shortcut for the API launcher."""
    import os
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    lnk_path = os.path.join(desktop, "Newmeta Backtester API.lnk")
    bat_path = Path(__file__).parent / "launch_api.bat"

    bat_content = f"""@echo off
title Newmeta Backtester API
cd /d "%~dp0"
set PYTHONPATH=%cd%
echo Starting Newmeta Backtester API on port 8765...
echo This serves the Python harness to the NewMeta Terminal widget.
echo Close this window or press Ctrl+C to stop.
echo.
python -c "from backtester.api import launch_api; launch_api(); import time; time.sleep(999999)"
pause
"""
    bat_path.write_text(bat_content)

    import win32com.client  # type: ignore
    ws = win32com.client.Dispatch("WScript.Shell")
    shortcut = ws.CreateShortcut(lnk_path)
    shortcut.TargetPath = str(bat_path)
    shortcut.WorkingDirectory = str(Path(__file__).parent)
    shortcut.IconLocation = "shell32.dll,238"
    shortcut.Description = "Newmeta Backtester API (serves Python backtester to widget)"
    shortcut.Save()
    print(f"[API] Shortcut created: {lnk_path}")


if __name__ == "__main__":
    proc = launch_api()
    print("[API] Press Ctrl+C to stop.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()