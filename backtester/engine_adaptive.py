"""Adaptive position sizing overlay — runs after pure or grid backtest.

Tracks win/loss streaks and adjusts lot size:
  - 3 consecutive losses → drop to 0.5× lot
  - 5 consecutive wins → bump to 1.2× lot
  - 10% rolling drawdown → pause (return lot = 0)

This is a POST-PROCESS overlay — it doesn't change the trade log, but it annotates
the result with sizer state (paused flag, current lot, consecutive counts) and
records the lot sizes that WOULD have been used.

Note: in the pure engine, lots are fixed (vectorbt). In the grid engine, lots
are already computed by GridRecoveryManager. Adaptive primarily serves the grid
engine for realistic martingale-aware sizing.
"""
from __future__ import annotations
import pandas as pd

from backtester.adaptive import AdaptiveSizer, AdaptiveConfig


def apply_adaptive(result: dict,
                    adaptive_config: AdaptiveConfig | None = None,
                    base_lot: float = 0.1) -> dict:
    """Apply adaptive sizing overlay to a backtest result.

    Reads result['trades'] for win/loss, updates sizer state, annotates result.
    """
    sizer = AdaptiveSizer(adaptive_config or AdaptiveConfig(base_lot=base_lot))
    trades = result.get("trades")
    if trades is None or trades.empty:
        # No trades — still record sizer state
        result["sizer_state"] = sizer.get_state()
        result.setdefault("active_overlays", []).append("adaptive")
        return result
    pnl_col = "pnl" if "pnl" in trades.columns else None
    if pnl_col is None:
        result["sizer_state"] = sizer.get_state()
        result.setdefault("active_overlays", []).append("adaptive")
        return result
    # Feed each trade to sizer
    lots_per_trade = []
    for _, tr in trades.iterrows():
        lots_per_trade.append(sizer.get_lot())
        sizer.on_trade_close(float(tr[pnl_col]))
    # Annotate trades with adaptive lot
    trades = trades.copy()
    trades["adaptive_lot"] = lots_per_trade
    result["trades"] = trades
    result["sizer_state"] = sizer.get_state()
    result.setdefault("active_overlays", []).append("adaptive")
    return result