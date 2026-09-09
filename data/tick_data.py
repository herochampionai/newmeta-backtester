"""Tick data fetcher — pulls real ticks from MT5, falls back to synthetic tick-from-bar.
Tick data enables realistic execution simulation (path-dependent fills, spread variability,
intra-bar order triggers)."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd


def fetch_ticks_mt5(symbol: str, start: str, end: str | None = None,
                    terminal: str | None = None) -> tuple[pd.DataFrame | None, dict]:
    """Try to fetch real ticks from MT5.
    Returns (df, info) where df has columns: time, bid, ask, last, volume, flags.
    """
    try:
        from data.mt5_export import resolve_terminal, init_mt5
        from datetime import datetime
        t = terminal or resolve_terminal()
        if not t:
            return None, {"error": "no_terminal"}
        if not init_mt5(t):
            return None, {"error": "init_failed"}
        import MetaTrader5 as mt5
        date_from = datetime.fromisoformat(start)
        date_to = datetime.fromisoformat(end) if end else datetime.now()
        # MT5 caps tick requests per call. Use copy_ticks_range for short windows
        ticks = mt5.copy_ticks_range(symbol, date_from, date_to, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            mt5.shutdown()
            return None, {"error": "no_ticks", "symbol": symbol}
        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        df = df[["bid", "ask", "last", "volume", "flags"]]
        info = {"source": "mt5_ticks", "rows": len(df),
                "first": str(df.index[0]), "last": str(df.index[-1])}
        mt5.shutdown()
        return df, info
    except Exception as e:
        return None, {"error": str(e), "type": type(e).__name__}


def synthesize_ticks_from_bars(df: pd.DataFrame, ticks_per_bar: int = 20,
                                seed: int = 42) -> pd.DataFrame:
    """Synthesize intra-bar tick walk from OHLC bars.

    Method (Geometric Brownian Motion between OHLC):
      - Each bar generates `ticks_per_bar` ticks uniformly spaced in time
      - Tick prices follow: open → high (with random walks up), then low (walk down),
        then close (walk back)
      - For each segment: random walk with sigma scaled to fit within range

    Returns DataFrame indexed by datetime with columns: bid, ask, last, volume, flags
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i, bar in df.iterrows():
        o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
        # Determine path: open -> high -> low -> close (zig-zag walk)
        # This gives intrabar volatility with realistic high/low excursions
        path_prices = []
        # Phase 1: open → high (random walk up)
        n1 = ticks_per_bar // 4
        for j in range(n1):
            progress = (j + 1) / n1
            price = o + (h - o) * progress + rng.normal(0, (h - o) * 0.1)
            path_prices.append(price)
        # Phase 2: high → low
        n2 = ticks_per_bar // 4
        for j in range(n2):
            progress = (j + 1) / n2
            price = h + (l - h) * progress + rng.normal(0, (h - l) * 0.1)
            path_prices.append(price)
        # Phase 3: low → close
        n3 = ticks_per_bar - n1 - n2
        for j in range(n3):
            progress = (j + 1) / n3
            price = l + (c - l) * progress + rng.normal(0, (c - l) * 0.1)
            path_prices.append(price)
        # Volume per tick (rough estimate)
        base_vol = float(bar.get("volume", 100)) / ticks_per_bar
        # Spread (typical FX: 0.5-2 pips, but we use 1 pip = 0.0001)
        spread = 0.0001
        for k, price in enumerate(path_prices):
            rows.append({
                "time": i + pd.Timedelta(milliseconds=int(k * 60000 / ticks_per_bar)),
                "bid": price - spread / 2,
                "ask": price + spread / 2,
                "last": price,
                "volume": max(1, int(rng.normal(base_vol, base_vol * 0.3))),
                "flags": 0,
            })
    out = pd.DataFrame(rows).set_index("time")
    return out


def fetch_ticks_with_priority(symbol: str, start: str, end: str | None = None,
                               prefer_ticks: bool = True,
                               ticks_per_bar: int = 20) -> tuple[pd.DataFrame, str, dict]:
    """Try real MT5 ticks → synthetic ticks from cached bars.

    Returns (df, source, info).
    source: 'mt5_ticks' | 'synthetic_ticks'
    """
    if prefer_ticks:
        df, info = fetch_ticks_mt5(symbol, start, end)
        if df is not None:
            return df, "mt5_ticks", info
    # Fallback to cached bars → synthesize ticks
    from data.cache import load as load_cache
    try:
        bars, _ = load_cache(symbol, "H1")
    except FileNotFoundError:
        # Last resort: tiny synthetic
        idx = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
        bars = pd.DataFrame({"open": 1.10, "high": 1.11, "low": 1.09, "close": 1.105,
                              "volume": 1000}, index=idx)
    ticks = synthesize_ticks_from_bars(bars, ticks_per_bar=ticks_per_bar)
    info = {"source": "synthetic_ticks", "rows": len(ticks),
            "from_bars": len(bars), "ticks_per_bar": ticks_per_bar}
    return ticks, "synthetic_ticks", info


def aggregate_ticks_to_bars(ticks: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Reverse: aggregate ticks back into OHLCV bars (sanity check)."""
    agg = ticks.resample(freq).agg({
        "bid": "last", "ask": "last", "last": ["first", "max", "min", "last"],
        "volume": "sum",
    })
    agg.columns = ["bid_close", "ask_close", "open", "high", "low", "close", "volume"]
    return agg.dropna()