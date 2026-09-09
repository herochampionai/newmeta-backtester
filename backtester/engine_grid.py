"""Backtest engine that wraps vectorbt with grid/recovery simulation.
Use this for accurate EA-equivalent backtests that include grid + recovery modes."""
from __future__ import annotations
import pandas as pd
import numpy as np
import vectorbt as vbt

from backtester.metrics_v2 import compute_all
from backtester.grid_recovery import GridRecoveryManager, GRID_LOSS_AND_PROFIT, RECOVERY_NONE


def run_with_grid_recovery(df: pd.DataFrame, signals_by_strategy: dict[str, tuple],
                            mgr: GridRecoveryManager,
                            init_cash: float = 10_000.0,
                            commission_pips: float = 0.7,
                            slippage_pips: float = 0.3,
                            periods_per_year: int = 252 * 24,
                            ) -> dict:
    """Run backtest with grid + recovery simulation.

    signals_by_strategy: {strategy_name: (entries, direction)} where entries/direction
    can be pd.Series or np.ndarray aligned with df.index.

    Returns dict with:
      - 'mgr': the manager (with .summary() + .to_trades_df())
      - 'metrics': computed metrics from grid trades
      - 'per_strategy': per-strategy metrics
    """
    # Normalize to np arrays for fast access
    sig_arrays = {}
    for name, (entries, direction) in signals_by_strategy.items():
        if isinstance(entries, pd.Series):
            entries = entries.values
        if isinstance(direction, pd.Series):
            direction = direction.values
        sig_arrays[name] = (entries.astype(bool), direction.astype(int))
    # Iterate bars
    for bar_idx in range(len(df)):
        bar = df.iloc[bar_idx]
        for strat_name, (entries_arr, direction_arr) in sig_arrays.items():
            sig_dir = 0
            if entries_arr[bar_idx]:
                sig_dir = int(direction_arr[bar_idx])
            closed = mgr.on_bar_close(
                strategy=strat_name, signal_direction=sig_dir,
                bar_high=float(bar["high"]), bar_low=float(bar["low"]),
                bar_close=float(bar["close"]), bar_index=bar_idx,
                commission_pips=commission_pips, slippage_pips=slippage_pips,
            )
    # Build aggregate trade log
    trades_df = mgr.to_trades_df()
    # Build equity curve from manager's equity_curve list
    if mgr.equity_curve:
        eq_df = pd.DataFrame(mgr.equity_curve, columns=["bar_index", "pnl", "strategy"])
        equity = pd.Series(0.0, index=df.index, name="equity")
        for _, row in eq_df.iterrows():
            equity.iloc[int(row["bar_index"]):] += float(row["pnl"])
        equity = init_cash + equity.cumsum()
    else:
        equity = pd.Series(init_cash, index=df.index)
    # Compute metrics
    returns = equity.pct_change().fillna(0)
    metrics = compute_all(returns, trades_df, equity, periods_per_year=periods_per_year)
    # Per-strategy metrics
    per_strat = {}
    if not trades_df.empty:
        for s, grp in trades_df.groupby("strategy"):
            s_rets = pd.Series(0.0, index=df.index)
            for _, row in grp.iterrows():
                s_rets.iloc[int(row["exit_bar"]):] += float(row["pnl"])
            s_eq = init_cash + s_rets.cumsum()
            s_metrics = compute_all(s_eq.pct_change().fillna(0), grp, s_eq,
                                     periods_per_year=periods_per_year)
            per_strat[s] = s_metrics
    return {
        "mgr": mgr,
        "trades": trades_df,
        "metrics": metrics,
        "equity": equity,
        "per_strategy": per_strat,
    }