"""Futures/options expiry roll + auto-roll for continuous contracts.

Handles: expiry calendar, volume/OI roll trigger, price adjustment (back-adjust),
continuous contract stitching. Works with portfolio_runner for multi-asset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EXPIRY_PATH = Path(__file__).parent.parent / "config" / "expiry_calendar.json"


@dataclass
class FutureSpec:
    symbol: str
    expiry: str  # YYYY-MM-DD
    tick_size: float = 0.01
    tick_value: float = 1.0
    contract_size: float = 1.0
    currency: str = "USD"
    roll_rule: str = "volume"  # volume | oi | date


def load_expiry_calendar() -> dict:
    try:
        if EXPIRY_PATH.exists():
            return json.loads(EXPIRY_PATH.read_text())
    except Exception:
        pass
    return {}


def save_expiry_calendar(cal: dict) -> None:
    EXPIRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPIRY_PATH.write_text(json.dumps(cal, indent=2))


def get_future_spec(symbol: str) -> FutureSpec | None:
    cal = load_expiry_calendar()
    return FutureSpec(**cal.get(symbol.upper(), {})) if symbol.upper() in cal else None


def find_roll_date(df: pd.DataFrame, spec: FutureSpec, lookback_days: int = 5) -> pd.Timestamp | None:
    """Find optimal roll date: max volume/OI in lookback window before expiry."""
    try:
        exp = pd.Timestamp(spec.expiry)
        window = df[(df.index >= exp - pd.Timedelta(days=lookback_days)) & (df.index < exp)]
        if window.empty:
            return exp - pd.Timedelta(days=1)
        if spec.roll_rule == "volume" and "volume" in window.columns:
            return window["volume"].idxmax()
        if spec.roll_rule == "oi" and "open_interest" in window.columns:
            return window["open_interest"].idxmax()
        return exp - pd.Timedelta(days=1)
    except Exception:
        return None


def back_adjust(near: pd.DataFrame, far: pd.DataFrame, roll_date: pd.Timestamp) -> pd.DataFrame:
    """Stitch far contract to near at roll_date with price adjustment (ratio method)."""
    near = near.copy()
    far = far.copy()
    # Align on roll_date
    near_px = near.loc[roll_date, "close"] if roll_date in near.index else near["close"].iloc[-1]
    far_px = far.loc[roll_date, "close"] if roll_date in far.index else far["close"].iloc[0]
    if far_px == 0:
        return near
    ratio = near_px / far_px
    # Adjust far prices before roll_date
    far_adj = far[far.index < roll_date].copy()
    for col in ("open", "high", "low", "close"):
        far_adj[col] *= ratio
    # Combine: far_adj (before roll) + near (on/after roll)
    combined = pd.concat([far_adj, near[near.index >= roll_date]]).sort_index()
    return combined


def auto_roll_continuous(symbol: str, contracts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build continuous contract from dict of {contract_symbol: df} sorted by expiry."""
    cal = load_expiry_calendar()
    if symbol.upper() not in cal:
        return pd.DataFrame()
    ordered = sorted(cal[symbol.upper()].get("contracts", []), key=lambda x: x["expiry"])
    if len(ordered) < 2:
        return contracts.get(ordered[0]["symbol"], pd.DataFrame()) if ordered else pd.DataFrame()
    continuous = None
    for i in range(len(ordered) - 1):
        near_sym = ordered[i]["symbol"]
        far_sym = ordered[i + 1]["symbol"]
        near_df = contracts.get(near_sym)
        far_df = contracts.get(far_sym)
        if near_df is None or far_df is None:
            continue
        spec = FutureSpec(**ordered[i])
        roll = find_roll_date(near_df, spec)
        if roll is None:
            continue
        continuous = back_adjust(near_df, far_df, roll) if continuous is None else back_adjust(continuous, far_df, roll)
    return continuous if continuous is not None else pd.DataFrame()
