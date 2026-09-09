"""Data fetcher with priority chain: LIVE MT5 → Yahoo Finance → cached Parquet → synthetic.
Used by the Streamlit front-end and run_pipeline."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent


def _import_yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        return None


def load_settings() -> dict:
    path = ROOT / "config" / "settings.yaml"
    if not path.exists():
        return {}
    yaml = _import_yaml()
    if not yaml:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def try_live_mt5(symbol: str, timeframe: str, start: str, end: str | None,
                 terminal_override: str | None = None) -> tuple[pd.DataFrame | None, dict]:
    """Attempt live MT5 fetch. Returns (df, info) or (None, error_info)."""
    try:
        from data.mt5_export import resolve_terminal, init_mt5, fetch_bars
        terminal = resolve_terminal(terminal_override)
        if not terminal:
            return None, {"error": "no_terminal_found",
                          "candidates": ["D:\\MT5_EuroPrinter\\terminal64.exe",
                                         "D:\\MT5_Bybit\\terminal64.exe"]}
        ok = init_mt5(terminal)
        if not ok:
            return None, {"error": "init_failed", "terminal": terminal}
        df = fetch_bars(symbol, timeframe, start, end)
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except Exception:
            pass
        return df, {"source": "mt5_live", "terminal": terminal,
                    "rows": len(df), "first": str(df.index[0]), "last": str(df.index[-1])}
    except Exception as e:
        return None, {"error": str(e), "type": type(e).__name__}


def try_yahoo(symbol: str, timeframe: str, start: str, end: str | None) -> tuple[pd.DataFrame | None, dict]:
    """Yahoo Finance fallback. Lower quality for FX."""
    try:
        from data.yahoo_fallback import fetch
        df = fetch(symbol, timeframe, start, end)
        return df, {"source": "yahoo", "rows": len(df),
                    "first": str(df.index[0]), "last": str(df.index[-1])}
    except Exception as e:
        return None, {"error": str(e), "type": type(e).__name__}


def try_cache(symbol: str, timeframe: str) -> tuple[pd.DataFrame | None, dict]:
    """Look for cached Parquet."""
    try:
        from data.cache import load
        df, meta = load(symbol, timeframe)
        return df, {"source": "cache", "rows": len(df),
                    "first": str(df.index[0]), "last": str(df.index[-1]),
                    "sha": meta.get("sha", "?")}
    except FileNotFoundError as e:
        return None, {"error": "no_cache", "detail": str(e)}


def synthetic(n_bars: int = 8000, seed: int = 42, drift: float = 0.0001,
              vol: float = 0.0003) -> tuple[pd.DataFrame, dict]:
    """Last-resort synthetic data with mild uptrend."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_bars, freq="h", tz="UTC")
    price = 1.10 * np.exp(np.cumsum(rng.normal(drift, vol, n_bars)))
    high = price * (1 + np.abs(rng.normal(0, 0.0005, n_bars)))
    low = price * (1 - np.abs(rng.normal(0, 0.0005, n_bars)))
    opn = np.roll(price, 1); opn[0] = price[0]
    vol_arr = rng.integers(50, 5000, n_bars)
    df = pd.DataFrame({"open": opn, "high": high, "low": low,
                       "close": price, "volume": vol_arr}, index=idx)
    return df, {"source": "synthetic", "rows": n_bars,
                "first": str(idx[0]), "last": str(idx[-1])}


def fetch_with_priority(symbol: str, timeframe: str = "H1",
                       start: str = "2022-01-01", end: str | None = None,
                       terminal_override: str | None = None,
                       allow_synthetic: bool = True) -> tuple[pd.DataFrame, dict]:
    """Live-first data fetch chain. Returns (df, chain_info).
    chain_info contains 'source' and any error from each prior attempt."""
    chain = []

    # 1. Live MT5
    df, info = try_live_mt5(symbol, timeframe, start, end, terminal_override)
    chain.append(info)
    if df is not None:
        info["chain"] = chain
        return df, info

    # 2. Yahoo
    df, info2 = try_yahoo(symbol, timeframe, start, end)
    chain.append(info2)
    if df is not None:
        info2["chain"] = chain
        return df, info2

    # 3. Cache
    df, info3 = try_cache(symbol, timeframe)
    chain.append(info3)
    if df is not None:
        info3["chain"] = chain
        return df, info3

    # 4. Synthetic fallback
    if allow_synthetic:
        df, info4 = synthetic()
        chain.append(info4)
        info4["chain"] = chain
        return df, info4

    raise RuntimeError(f"all data sources failed: {chain}")