"""MT5 → Parquet export and live data access with persistent connection pooling."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5

CACHE_DIR = Path(__file__).parent / "cache"
ROOT = Path(__file__).parent.parent
SETTINGS_FILE = ROOT / "config" / "settings.yaml"

TERMINAL_CANDIDATES = [
    r"D:\MT5_EuroPrinter\terminal64.exe",
    r"D:\MT5_Bybit\terminal64.exe",
    r"D:\work D\mt5-mcp\MT5_EuroPrinter\terminal64.exe",
]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def resolve_terminal(cli_value: str | None = None) -> str | None:
    if cli_value:
        p = Path(cli_value)
        return str(p) if p.exists() else None
    env = os.environ.get("MT5_TERMINAL_PATH")
    if env and Path(env).exists():
        return env
    s = _load_yaml(SETTINGS_FILE).get("mt5_terminal")
    if s and Path(s).exists():
        return s
    for c in TERMINAL_CANDIDATES:
        if Path(c).exists():
            return c
    return None

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3,
    "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def find_terminal() -> str | None:
    return resolve_terminal()


def init_mt5(terminal: str | None = None) -> bool:
    try:
        info = mt5.terminal_info()
        if info is not None and info.connected:
            return True
    except Exception:
        pass
        
    init_kwargs = {}
    if terminal:
        init_kwargs["path"] = terminal
    ok = mt5.initialize(**init_kwargs)
    if not ok:
        return False
    return True


def fetch_bars(symbol: str, timeframe: str, start: str, end: str | None = None, n_bars: int | None = None) -> pd.DataFrame:
    tf = TF_MAP.get(timeframe.upper(), mt5.TIMEFRAME_H1)
    
    # Ensure symbol is selected in Market Watch
    mt5.symbol_select(symbol, True)
    
    # If symbol not found, try common alias resolution
    s_info = mt5.symbol_info(symbol)
    if s_info is None:
        aliases = {
            "NAS100": "USTEC", "US100": "USTEC", "NASDAQ": "USTEC",
            "US30": "US30", "DJ30": "US30", "DOW": "US30",
            "GOLD": "XAUUSD", "SILVER": "XAGUSD",
        }
        alias = aliases.get(symbol.upper())
        if alias:
            symbol = alias
            mt5.symbol_select(symbol, True)
            
    if n_bars is not None:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    else:
        date_from = datetime.fromisoformat(start)
        date_to = datetime.fromisoformat(end) if end else datetime.now()
        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
        if (rates is None or len(rates) == 0) and start:
            # Fallback to copy recent bars
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, 5000)
            
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"[MT5] no data for {symbol} {timeframe}")
        
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def fingerprint(df: pd.DataFrame, meta: dict) -> str:
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes() + json.dumps(meta, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]
