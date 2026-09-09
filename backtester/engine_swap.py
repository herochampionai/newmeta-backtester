"""Swap overlay — adds Wed-3× / holiday-aware swap to equity curve.

This is a POST-PROCESS overlay. Takes a base backtest result, computes the swap
series from each closed trade's direction and lot, and applies it to the equity
curve.

Honest semantics: only runs when swap_enabled=True. The flag in app.py / CLI
controls whether swap is applied at all.
"""
from __future__ import annotations
import pandas as pd

from backtester.swaps import compute_swap_series


def apply_swap(result: dict,
                df: pd.DataFrame,
                long_swap_pips: float = -0.5,
                short_swap_pips: float = 0.2,
                pip_size: float = 0.0001,
                contract_size: float = 100_000) -> dict:
    """Apply swap overlay to a backtest result.

    Estimate avg position size from closed trades, compute daily swap,
    add to equity curve.
    """
    trades = result.get("trades")
    equity = result.get("equity")
    if trades is None or trades.empty or equity is None:
        result["swap_total"] = 0.0
        result.setdefault("active_overlays", []).append("swap")
        return result
    # Estimate avg lots
    if "lots" in trades.columns:
        lots = trades["lots"]
        if isinstance(lots.iloc[0], list):
            avg_lots = lots.apply(sum).mean()
        else:
            avg_lots = lots.mean()
    else:
        avg_lots = 0.1
    position_sizes = pd.Series(avg_lots, index=df.index)
    swap = compute_swap_series(df,
                                long_swap_pips=long_swap_pips,
                                short_swap_pips=short_swap_pips,
                                position_sizes=position_sizes,
                                pip_size=pip_size, contract_size=contract_size)
    # Add swap to equity
    new_equity = equity + swap.cumsum()
    result["equity"] = new_equity
    result["swap_total"] = float(swap.sum())
    # Recompute metrics with swap-adjusted equity
    from backtester.metrics_v2 import compute_all
    returns = new_equity.pct_change().fillna(0)
    metrics = compute_all(returns, trades, new_equity, periods_per_year=252 * 24)
    result["metrics"] = metrics
    result.setdefault("active_overlays", []).append("swap")
    return result