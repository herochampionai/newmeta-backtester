"""Swap + holiday + weekend awareness for accurate P&L.

FX swap rules:
  - Standard swap is applied once per business day at 17:00 NY (EOD rollover)
  - Wednesday: 3x swap (covers Sat + Sun)
  - Friday close: weekend swap already collected on Wed
  - Monday: no swap (covered by Wed)
  - Bank holidays: no swap (depends on currency pair)

Positive swap = earn carry (long high-yielding / short low-yielding).
For EURUSD: typically both directions have negative swap (broker charges).
For AUDJPY, NZDJPY: long positions earn positive swap.

This module computes the swap cost/credit per bar based on:
  - Trade direction
  - Position size (lots)
  - Date (Wed = 3x, holiday = 0)
"""
from __future__ import annotations
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


# Standard FX swap holidays (approx — broker-specific)
SWAP_HOLIDAYS_2024_2025 = {
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
}


def is_holiday(date: pd.Timestamp) -> bool:
    return date.strftime("%Y-%m-%d") in SWAP_HOLIDAYS_2024_2025


def is_weekend(date: pd.Timestamp) -> bool:
    return date.weekday() >= 5  # Sat=5, Sun=6


def swap_multiplier(date: pd.Timestamp) -> float:
    """Returns swap multiplier for a given date.
    Wed = 3x (covers weekend), other business days = 1x, weekends = 0, holidays = 0.
    """
    if is_weekend(date) or is_holiday(date):
        return 0.0
    if date.weekday() == 2:  # Wednesday
        return 3.0
    return 1.0


def compute_swap_series(df: pd.DataFrame, long_swap_pips: float, short_swap_pips: float,
                         position_sizes: pd.Series | None = None,
                         pip_size: float = 0.0001, contract_size: float = 100_000
                         ) -> pd.Series:
    """Compute swap cost/credit per bar.

    Args:
        df: OHLCV df with DatetimeIndex
        long_swap_pips: positive = earn, negative = pay (per standard lot per day)
        short_swap_pips: same for short positions
        position_sizes: lots held per bar (defaults to 1.0 if None)
    Returns:
        pd.Series of swap $ per bar.
    """
    n = len(df)
    if position_sizes is None:
        position_sizes = pd.Series(1.0, index=df.index)
    # Determine swap at each bar from date index
    swaps = pd.Series(0.0, index=df.index)
    for i, ts in enumerate(df.index):
        mult = swap_multiplier(ts)
        if mult == 0.0:
            continue
        pos = position_sizes.iloc[i]
        # Long pays long_swap, short pays short_swap (positive = receive)
        # Average: use abs(pos) * average swap as estimate
        avg_swap_pips = (long_swap_pips + short_swap_pips) / 2
        swap_dollar = avg_swap_pips * mult * pip_size * contract_size * abs(pos)
        swaps.iloc[i] = swap_dollar
    return swaps


def apply_weekend_filter(df: pd.DataFrame) -> pd.Series:
    """Returns bool Series — True if bar is a tradable session (not weekend/holiday)."""
    return pd.Series([not (is_weekend(ts) or is_holiday(ts))
                       for ts in df.index], index=df.index)


def monday_open_indicator(df: pd.DataFrame, lookback_hours: int = 4) -> pd.Series:
    """Detect Monday opening hours (first N hours of Monday session, NY 17:00 close Sun)."""
    out = pd.Series(False, index=df.index)
    for i, ts in enumerate(df.index):
        if ts.weekday() == 0 and ts.hour < lookback_hours:
            out.iloc[i] = True
    return out