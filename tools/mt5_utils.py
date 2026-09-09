"""MT5 utilities — compile .mq5 to .ex5, list MQL5 source folder, etc."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import Optional, List


def find_metatrader() -> Optional[Path]:
    """Find MetaEditor.exe in the standard MT5 install locations."""
    candidates = [
        r"C:\Program Files\MetaTrader 5\MetaEditor.exe",
        r"D:\MT5_EuroPrinter\metaeditor64.exe",
        r"D:\MT5_Bybit\metaeditor64.exe",
        r"D:\MT5_EuroPrinter\MetaEditor.exe",
    ]
    for path in candidates:
        if Path(path).exists():
            return Path(path)
    return None


def list_mql5_files(source_root: str | Path, extensions: tuple = (".mq5", ".mqh")) -> List[Path]:
    """List all MQL5 source files recursively under source_root/MQL5."""
    root = Path(source_root)
    if not root.exists():
        return []
    mql5_dir = root / "MQL5"
    if not mql5_dir.exists():
        mql5_dir = root  # fallback to root
    return sorted([p for p in mql5_dir.rglob("*") if p.suffix in extensions])


def compile_mq5(mq5_path: str | Path, metatrader: str | Path | None = None,
                  timeout: int = 60) -> dict:
    """Compile a .mq5 file into .ex5 using MetaEditor.
    Returns dict: {success, ex5_path, log_output, errors}."""
    mq5_path = Path(mq5_path).resolve()
    if not mq5_path.exists():
        return {"success": False, "errors": f"File not found: {mq5_path}"}

    if metatrader is None:
        metatrader = find_metatrader()
    if metatrader is None or not Path(metatrader).exists():
        return {"success": False, "errors": "MetaEditor.exe not found in standard locations"}

    metatrader = Path(metatrader)
    log_file = mq5_path.with_suffix(".log")

    # /compile:<file> — compile this file
    # /log — write log to file
    # /include:<path> — extra include path
    # /s — silent
    cmd = [
        str(metatrader),
        f"/compile:{mq5_path}",
        f"/log:{log_file}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        log_content = ""
        if log_file.exists():
            log_content = log_file.read_text(encoding="utf-8", errors="replace")
        # Parse for errors
        errors = []
        for line in log_content.splitlines():
            if "error" in line.lower() or "fatal" in line.lower():
                errors.append(line)
        # Determine ex5 output path (MT5 places .ex5 in MQL5/Indicators or MQL5/Experts)
        ex5_path = None
        for candidate in mq5_path.parent.rglob(mq5_path.stem + ".ex5"):
            ex5_path = candidate
            break
        return {
            "success": len(errors) == 0 and result.returncode == 0,
            "returncode": result.returncode,
            "ex5_path": str(ex5_path) if ex5_path else None,
            "log": log_content[-2000:] if log_content else "",
            "errors": errors[:20],
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "errors": f"Compile timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "errors": str(e)}