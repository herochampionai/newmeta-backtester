"""Pure strategy backtest — signals → trades → metrics via vectorbt.
No grid, no adaptive, no swap. Strategy-only.

This is the BASE engine. All overlays (grid / adaptive / swap) compose on top
via engine_full.run_full()."""
from __future__ import annotations
import pandas as pd
import numpy as np
import vectorbt as vbt

from backtester.metrics_v2 import compute_all
from backtester.analytics import streak_stats, annual_trade_count, strategy_scoreboard


def run_pure(df: pd.DataFrame,
              signals_by_strategy: dict[str, tuple],
              init_cash: float = 10_000.0,
              commission_pips: float = 0.7,
              slippage_pips: float = 0.3,
              periods_per_year: int = 252 * 24,
              ) -> dict:
    """Pure vectorized backtest. No overlays.

    Args:
        df: OHLCV DataFrame
        signals_by_strategy: {name: (entries, direction)} where direction ∈ {-1, 0, +1}
        init_cash: starting capital
        commission_pips / slippage_pips: per-trade costs in pips

    Returns dict with: equity, trades, metrics, per_strategy, streak_stats, etc.
    """
    # Multi-strategy: union of all entries; direction takes first non-zero
    if len(signals_by_strategy) == 1:
        first_name = next(iter(signals_by_strategy))
        entries, direction = signals_by_strategy[first_name]
        entries = pd.Series(entries).fillna(False).astype(bool).values
        direction = pd.Series(direction).fillna(0).astype(int).values
    else:
        entries_arrays = []
        direction_arrays = []
        for name, (e, d) in signals_by_strategy.items():
            entries_arrays.append(pd.Series(e).fillna(False).astype(bool).values)
            direction_arrays.append(pd.Series(d).fillna(0).astype(int).values)
        # Union: any strategy fires an entry → trigger
        entries = np.zeros(len(df), dtype=bool)
        for e in entries_arrays:
            entries |= e
        # Direction: first non-zero per bar
        direction = np.zeros(len(df), dtype=int)
        for d in direction_arrays:
            mask = (direction == 0) & (d != 0)
            direction[mask] = d[mask]
        first_name = "multi"

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entries,
        direction=direction,
        init_cash=init_cash,
        fees=commission_pips * 1e-4,
        slippage=slippage_pips * 1e-4,
        freq="h",
    )
    returns = pf.returns().dropna()
    equity = (1 + returns).cumprod() * init_cash
    # Trade log
    try:
        trades_raw = pf.trades.records_readable
        if isinstance(trades_raw, pd.DataFrame) and not trades_raw.empty:
            trades = trades_raw.copy()
            trades["strategy"] = first_name if len(signals_by_strategy) == 1 else "primary"
            if "Entry Timestamp" in trades.columns:
                trades["entry_bar"] = pd.to_datetime(trades["Entry Timestamp"]).map(
                    lambda t: df.index.get_indexer([t], method="nearest")[0]
                    if hasattr(df.index, "get_indexer") else 0)
            if "Exit Timestamp" in trades.columns:
                trades["exit_bar"] = pd.to_datetime(trades["Exit Timestamp"]).map(
                    lambda t: df.index.get_indexer([t], method="nearest")[0]
                    if hasattr(df.index, "get_indexer") else 0)
        else:
            trades = pd.DataFrame()
    except Exception:
        trades = pd.DataFrame()

    metrics = compute_all(returns, trades, equity, periods_per_year=periods_per_year)
    streak = streak_stats(trades, "pnl") if not trades.empty and "pnl" in trades.columns else {}
    scoreboard = strategy_scoreboard({first_name: metrics},
                                       rank_by="sharpe")
    n_bars = len(df)
    tpy = annual_trade_count(len(trades), n_bars, periods_per_year) if not trades.empty else 0.0
    return {
        "equity": equity,
        "trades": trades,
        "metrics": metrics,
        "per_strategy": {first_name: metrics},
        "streak_stats": streak,
        "annual_trades": tpy,
        "scoreboard": scoreboard,
        "n_bars": n_bars,
        "grid_summary": {"n_grid_trades": 0, "note": "pure engine — no grid"},
        "swap_total": 0.0,
        "sizer_state": None,
        "active_overlays": [],
    }