"""Full backtest engine — integrates grid + recovery + adaptive sizing + swaps.
Single entry point: `run_full(df, signals, config)` returns complete result dict."""
from __future__ import annotations
import pandas as pd
import numpy as np

from backtester.grid_recovery import GridRecoveryManager, GRID_NONE, GRID_LOSS_AND_PROFIT
from backtester.adaptive import AdaptiveSizer, AdaptiveConfig
from backtester.swaps import compute_swap_series
from backtester.metrics_v2 import compute_all
from backtester.analytics import streak_stats, annual_trade_count, strategy_scoreboard


def run_full(df: pd.DataFrame,
              signals_by_strategy: dict[str, tuple],
              init_cash: float = 10_000.0,
              commission_pips: float = 0.7,
              slippage_pips: float = 0.3,
              pip_size: float = 0.0001,
              contract_size: float = 100_000,
              grid_mode: int = GRID_NONE,
              pips_between_orders: float = 30.0,
              grid_lot_multiplier: float = 1.5,
              grid_take_profit: float = 50.0,
              grid_stop_loss: float = 200.0,
              max_grid_layers: int = 4,
              recovery_mode: int = 0,
              recovery_lot_multiplier: float = 2.0,
              base_lot: float = 0.1,
              adaptive_enabled: bool = False,
              adaptive_config: AdaptiveConfig | None = None,
              swap_enabled: bool = False,
              long_swap_pips: float = -0.5,
              short_swap_pips: float = 0.2,
              ) -> dict:
    """Full-featured backtest.

    Returns dict with:
      - 'equity': pd.Series — equity curve (with grid + swap)
      - 'trades': pd.DataFrame — closed trades
      - 'metrics': comprehensive metrics dict
      - 'per_strategy': per-strategy metrics
      - 'grid_summary': from GridRecoveryManager
      - 'swap_total': total $ earned/paid in swap
      - 'streak_stats': longest win/loss streaks
      - 'annual_trades': trades per year
      - 'scoreboard': ranked per-strategy metrics
    """
    # 1. Build grid + recovery manager
    mgr = GridRecoveryManager(
        grid_mode=grid_mode,
        pips_between_orders=pips_between_orders,
        grid_lot_multiplier=grid_lot_multiplier,
        grid_take_profit=grid_take_profit,
        grid_stop_loss=grid_stop_loss,
        max_layers=max_grid_layers,
        recovery_mode=recovery_mode,
        recovery_lot_multiplier=recovery_lot_multiplier,
        base_lot=base_lot,
        pip_size=pip_size,
        contract_size=contract_size,
    )
    # 2. Build adaptive sizer
    sizer = AdaptiveSizer(adaptive_config or AdaptiveConfig(base_lot=base_lot))
    # 3. Normalize signals to ndarrays
    sig_arrays = {}
    for name, (entries, direction) in signals_by_strategy.items():
        if isinstance(entries, pd.Series):
            entries = entries.values
        if isinstance(direction, pd.Series):
            direction = direction.values
        sig_arrays[name] = (entries.astype(bool), direction.astype(int))
    # 4. Bar-by-bar loop
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
            # Feed adaptive sizer
            if closed and adaptive_enabled:
                for tr in closed:
                    sizer.on_trade_close(tr.pnl)
    # 5. Build trade log
    trades = mgr.to_trades_df()
    # 6. Equity curve from grid trades
    if mgr.equity_curve:
        eq_df = pd.DataFrame(mgr.equity_curve, columns=["bar_index", "pnl", "strategy"])
        equity = pd.Series(init_cash, index=df.index, dtype=float)
        for _, row in eq_df.iterrows():
            equity.iloc[int(row["bar_index"]):] += float(row["pnl"])
    else:
        equity = pd.Series(init_cash, index=df.index, dtype=float)
    # 7. Apply swap if enabled
    swap_total = 0.0
    if swap_enabled:
        # Estimate average position size from grid trades
        if not trades.empty:
            avg_lots = trades["lots"].apply(lambda l: sum(l) if isinstance(l, list) else l).mean()
        else:
            avg_lots = base_lot
        position_sizes = pd.Series(avg_lots, index=df.index)
        swap = compute_swap_series(df, long_swap_pips=long_swap_pips,
                                    short_swap_pips=short_swap_pips,
                                    position_sizes=position_sizes,
                                    pip_size=pip_size, contract_size=contract_size)
        equity = equity + swap.cumsum()
        swap_total = float(swap.sum())
    # 8. Metrics
    returns = equity.pct_change().fillna(0)
    metrics = compute_all(returns, trades, equity, periods_per_year=252 * 24)
    # 9. Per-strategy
    per_strat = {}
    if not trades.empty:
        for s, grp in trades.groupby("strategy"):
            s_rets = pd.Series(0.0, index=df.index)
            for _, row in grp.iterrows():
                s_rets.iloc[int(row["exit_bar"]):] += float(row["pnl"])
            s_eq = init_cash + s_rets.cumsum()
            s_metrics = compute_all(s_eq.pct_change().fillna(0), grp, s_eq,
                                     periods_per_year=252 * 24)
            per_strat[s] = s_metrics
    # 10. Streak / scoreboard
    streak = streak_stats(trades, "pnl") if not trades.empty else {}
    scoreboard = strategy_scoreboard(per_strat, rank_by="sharpe")
    n_bars = len(df)
    tpy = annual_trade_count(len(trades), n_bars, 252 * 24) if not trades.empty else 0.0
    return {
        "equity": equity,
        "trades": trades,
        "metrics": metrics,
        "per_strategy": per_strat,
        "grid_summary": mgr.summary(),
        "swap_total": swap_total,
        "streak_stats": streak,
        "annual_trades": tpy,
        "scoreboard": scoreboard,
        "sizer_state": sizer.get_state(),
        "n_bars": n_bars,
    }