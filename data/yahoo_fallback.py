"""Yahoo Finance fallback when MT5 is unavailable. Lower-quality but works for
sanity checks. Note: Yahoo may not have all FX pairs beyond majors."""
from __future__ import annotations
import pandas as pd

# Yahoo ticker mapping for FX pairs
YAHOO_MAP = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X", "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X",
}

# Yahoo interval mapping
INTERVAL_MAP = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "60m", "H4": "60m",    # Yahoo doesn't have 4h
    "D1": "1d", "W1": "1wk",
}


def fetch(symbol: str, timeframe: str, start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf
    ticker = YAHOO_MAP.get(symbol.upper(), f"{symbol}=X")
    interval = INTERVAL_MAP.get(timeframe.upper(), "60m")
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError(f"yahoo returned empty for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Adj Close": "adj_close"})
    if "volume" not in df.columns:
        df["volume"] = 0
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["open", "high", "low", "close", "volume"]]