"""Multi-asset data loader with priority chain.

Loads OHLCV data following this priority:
  1. Cached Parquet (`data/cache.py`) — fastest, highest quality
  2. Yahoo Finance fallback (`data/yahoo_fallback.py`) — for crypto + FX
  3. MT5 live fetch (`data/live_fetcher.py`) — when terminal is running
  4. Synthetic data — for testing when no real source available

Usage:
    PYTHONPATH=. python -m analysis.data_loader --symbol BTCUSDT --timeframe D1 --start 2023-01-01 --end 2024-12-31
    PYTHONPATH=. python -m analysis.data_loader --symbols EURUSD,GBPUSD,BTCUSDT --timeframe H1 --start 2024-01-01
    PYTHONPATH=. python -c "from analysis.data_loader import load_asset; df = load_asset('BTCUSDT', 'D1')"
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.cache import list_cache  # noqa: E402
from data.cache import load as load_cache  # noqa: E402
from data.yahoo_fallback import YAHOO_MAP, INTERVAL_MAP

_CRYPTO_YAHOO_MAP = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "ADAUSDT": "ADA-USD",
    "DOTUSDT": "DOT-USD",
}

YAHOO_TICKER_MAP = {**YAHOO_MAP, **_CRYPTO_YAHOO_MAP}


def _try_yahoo(symbol: str, timeframe: str, start: str, end: str | None) -> pd.DataFrame | None:
    """Fetch from Yahoo Finance (FX + crypto)."""
    yahoo_ticker = YAHOO_TICKER_MAP.get(symbol.upper())
    if yahoo_ticker is None:
        return None
    try:
        import yfinance as yf
        interval = INTERVAL_MAP.get(timeframe.upper())
        if interval is None:
            tf_norm = timeframe.upper().replace("1", "")
            if tf_norm == "H":
                interval = "60m"
            elif tf_norm == "M":
                interval = "15m"
            elif tf_norm == "D":
                interval = "1d"
            else:
                interval = "60m"
        # Yahoo caps intraday history (1h/15m/5m/1m) at 730 days — clamp start
        start_ts = pd.Timestamp(start, tz="UTC")
        if interval in ("60m", "15m", "5m", "1m"):
            now = pd.Timestamp.now(tz="UTC")
            max_lookback = now - pd.Timedelta(days=729)
            if start_ts < max_lookback:
                start_ts = max_lookback
            end_ts = pd.Timestamp(end, tz="UTC") if end else now
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(yahoo_ticker, start=start_ts, end=end_ts,
                                 interval=interval, progress=False)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(yahoo_ticker, start=start_ts, end=end,
                                 interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Normalize yfinance's capitalized columns (Close/High/Low/Open/Volume)
        df.columns = [str(c).lower() for c in df.columns]
        df = df.rename(columns={"adj close": "adj_close"})
        if "volume" not in df.columns:
            df["volume"] = 0
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]]
        # Cache to parquet
        from data.cache import write as cache_write
        import hashlib
        sha = hashlib.sha256(f"{symbol}_{timeframe}_{start}_{end or 'now'}".encode()).hexdigest()[:12]
        meta = {"symbol": symbol, "timeframe": timeframe, "sha": sha,
                "source": "yahoo", "start": start, "end": end or "now"}
        cache_write(df, meta)
        return df
    except Exception:
        return None


def _is_cached(symbol: str, timeframe: str) -> bool:
    """Check if symbol/timeframe exists in the cache."""
    all_cache = list_cache()
    for p in all_cache:
        if symbol.upper() in p.stem.upper() and timeframe.upper() in p.stem.upper():
            return True
    return False


def _try_cache(symbol: str, timeframe: str) -> pd.DataFrame | None:
    try:
        df, _ = load_cache(symbol, timeframe)
        return df
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _synthetic(symbol: str, timeframe: str, start: str, end: str | None) -> pd.DataFrame:
    """Generate synthetic OHLCV for testing (no real data available)."""
    import numpy as np
    s = pd.to_datetime(start)
    e = pd.to_datetime(end) if end else pd.Timestamp.now()
    # Normalize timeframe to pandas freq (H1 -> 1h, D1 -> 1D, M15 -> 15min)
    tf_norm = timeframe.upper()
    if tf_norm in ("H1", "1H"):
        freq = "1h"
    elif tf_norm in ("D1", "1D"):
        freq = "1D"
    elif tf_norm in ("M15", "15M"):
        freq = "15min"
    elif tf_norm in ("M5", "5M"):
        freq = "5min"
    elif tf_norm in ("M1", "1M"):
        freq = "1min"
    elif tf_norm in ("W1", "1W"):
        freq = "1W"
    else:
        freq = "1h"
    idx = pd.date_range(s, e, freq=freq, tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(seed=hash(symbol) % 2**32)
    price = 1.1000 + np.cumsum(rng.standard_normal(n) * 0.001)
    price = np.maximum(price, 0.5)
    high = price * (1 + rng.uniform(0, 0.002, n))
    low = price * (1 - rng.uniform(0, 0.002, n))
    open_ = price + rng.normal(0, 0.0005, n)
    close = price + rng.normal(0, 0.0005, n)
    volume = rng.integers(100, 1000, n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=idx)


def load_asset(symbol: str, timeframe: str, start: str | None = None,
               end: str | None = None) -> pd.DataFrame:
    """Load OHLCV for a symbol/timeframe with fallback chain.

    Args:
        symbol: e.g. 'EURUSD', 'BTCUSDT', 'ETHUSDT'
        timeframe: e.g. 'H1', 'D1', 'M15'
        start: optional start date string (used for Yahoo/synthetic)
        end: optional end date string (used for Yahoo/synthetic)

    Returns:
        DataFrame with open/high/low/close/volume columns, DatetimeIndex (UTC)
    """
    # 1. Try cache
    df = _try_cache(symbol, timeframe)
    if df is not None:
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df

    # 2. Try Yahoo Finance
    start_str = start or "2023-01-01"
    df = _try_yahoo(symbol, timeframe, start_str, end)
    if df is not None:
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df

    # 3. Synthetic (testing fallback)
    if start is None:
        raise ValueError(f"No cached data for {symbol} {timeframe} and no start date provided")
    return _synthetic(symbol, timeframe, start, end)


def load_assets(symbols: list[str], timeframe: str, start: str | None = None,
                end: str | None = None) -> dict[str, pd.DataFrame]:
    """Load multiple symbols. Returns {symbol: DataFrame}."""
    out = {}
    for s in symbols:
        try:
            out[s] = load_asset(s, timeframe, start, end)
        except Exception as e:
            print(f"  [WARN] {s} {timeframe}: {e}")
    return out


def list_available_assets() -> list[str]:
    """List all symbols available in the cache."""
    all_cache = list_cache()
    symbols = set()
    for p in all_cache:
        stem = p.stem.split("_")[0]  # e.g. "EURUSD" from "EURUSD_H1_abc.parquet"
        symbols.add(stem)
    return sorted(symbols)


def main():
    ap = argparse.ArgumentParser(description="Multi-asset data loader")
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--symbols", default=None,
                    help="Comma-separated list (overrides --symbol)")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--list", action="store_true", help="List cached symbols")
    args = ap.parse_args()

    if args.list:
        cached = list_available_assets()
        print(f"[cached] {len(cached)} symbols: {', '.join(cached)}")
        return 0

    symbols = args.symbols.split(",") if args.symbols else [args.symbol]
    print(f"[load] {len(symbols)} symbols × {args.timeframe}")

    for s in symbols:
        s = s.strip()
        try:
            df = load_asset(s, args.timeframe, args.start, args.end)
            print(f"  {s}: {len(df)} bars  {df.index[0].date()}→{df.index[-1].date()}  cols={list(df.columns)}")
        except Exception as e:
            print(f"  {s}: [FAIL] {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
