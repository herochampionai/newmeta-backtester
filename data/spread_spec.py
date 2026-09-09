"""Realistic spread model — load broker spreads from MT5, fall back to default.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def get_spreads_mt5(symbol: str, n_days: int = 30) -> pd.Series | None:
    """Fetch recent spread history from MT5 (real broker spreads).
    Returns Series indexed by datetime, or None if unavailable."""
    try:
        from data.mt5_export import resolve_terminal, init_mt5
        from datetime import datetime, timedelta
        import MetaTrader5 as mt5
        terminal = resolve_terminal()
        if not terminal or not init_mt5(terminal):
            return None
        end = datetime.now()
        start = end - timedelta(days=n_days)
        ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            mt5.shutdown()
            return None
        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["spread_pips"] = (df["ask"] - df["bid"]) / 0.0001
        # Resample to 1-min average spread
        df = df.set_index("time")
        spread_1m = df["spread_pips"].resample("1min").mean()
        mt5.shutdown()
        return spread_1m.dropna()
    except Exception:
        return None


def default_spread_series(index: pd.DatetimeIndex,
                            base_pips: float = 1.0,
                            night_multiplier: float = 1.5,
                            weekend_pips: float = 50.0,
                            seed: int = 42) -> pd.Series:
    """Generate a realistic spread series when broker data isn't available.
    - Base spread during London/NY overlap
    - Higher (multiplier) during Asian session (night)
    - Very wide on weekends
    """
    rng = np.random.default_rng(seed)
    # Vectorized for speed
    hours = pd.Series(index.hour, index=index)
    weekdays = pd.Series(index.weekday, index=index)
    base = np.where(weekdays >= 5, weekend_pips,
                    np.where((hours < 7) | (hours >= 20),
                             base_pips * night_multiplier,
                             base_pips))
    # Add random noise
    noise = rng.normal(0, base * 0.1, size=len(index))
    out = np.maximum(0.1, base + noise)
    return pd.Series(out, index=index)