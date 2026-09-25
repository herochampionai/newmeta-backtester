"""Backtest engine that wraps vectorbt with grid/recovery simulation.
Use this for accurate EA-equivalent backtests that include grid + recovery modes."""
from __future__ import annotations

import pandas as pd

from backtester.analytics import annual_trade_count, strategy_scoreboard, streak_stats
from backtester.grid_recovery import (
    GRID_LOSS_AND_PROFIT,
    RECOVERY_NONE,
    GridRecoveryManager,
)
from backtester.metrics_v2 import compute_all
from core.surgical_features import get_feature_params, is_enabled


def run_grid(df: pd.DataFrame,
              signals_by_strategy: dict[str, tuple],
              init_cash: float = 10_000.0,
              commission_pips: float = 0.7,
              slippage_pips: float = 0.3,
              spread_pips: float = 0.0,
              commission_pct: float = 0.0,
              pip_size: float = 0.0001,
              contract_size: float = 100_000,
              grid_mode: int = GRID_LOSS_AND_PROFIT,
              pips_between_orders: float = 30.0,
              grid_lot_multiplier: float = 1.5,
              grid_take_profit: float = 50.0,
              grid_stop_loss: float = 200.0,
              max_grid_layers: int = 4,
              recovery_mode: int = RECOVERY_NONE,
              recovery_lot_multiplier: float = 2.0,
              base_lot: float = 0.1,
              params: dict | None = None,
              ) -> dict:
    """Run grid + recovery backtest. Builds the GridRecoveryManager internally.

    Returns dict with: equity, trades, metrics, per_strategy, grid_summary, etc.
    """
    # Resolve surgical feature params from strategy params dict
    rr_p = get_feature_params(params, "recovery_restart")
    bmt_p = get_feature_params(params, "basket_money_tp")
    plt_p = get_feature_params(params, "profit_lock_trail")
    car_p = get_feature_params(params, "carry_adjusted_tp")

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
        recovery_restart_enabled=is_enabled(params, "recovery_restart"),
        recovery_restart_size_mult=rr_p.get("size_multiplier", 0.5),
        recovery_restart_strictness_bonus=rr_p.get("strictness_bonus", 1),
        recovery_restart_cooldown_bars=rr_p.get("cooldown_bars", 12),
        basket_money_tp_enabled=is_enabled(params, "basket_money_tp"),
        basket_take_profit_usd=bmt_p.get("basket_take_profit_usd", 50.0),
        profit_lock_trail_enabled=is_enabled(params, "profit_lock_trail"),
        profit_lock_pct=plt_p.get("profit_lock_pct", 60.0),
        carry_adjusted_tp_enabled=is_enabled(params, "carry_adjusted_tp"),
        rollover_window_hours=car_p.get("rollover_window_hours", 8),
        tp_extension_pct=car_p.get("tp_extension_pct", 25.0),
    )
    result = run_with_grid_recovery(
        df, signals_by_strategy, mgr,
        init_cash=init_cash,
        commission_pips=commission_pips, slippage_pips=slippage_pips,
        spread_pips=spread_pips, commission_pct=commission_pct,
    )
    # Enrich with grid_summary + streak + scoreboard + n_bars
    result["grid_summary"] = mgr.summary()
    trades = result.get("trades")
    result["streak_stats"] = streak_stats(trades, "pnl") if trades is not None and not trades.empty else {}
    result["scoreboard"] = strategy_scoreboard(result.get("per_strategy", {}), rank_by="sharpe")
    result["n_bars"] = len(df)
    result["annual_trades"] = annual_trade_count(len(trades) if trades is not None else 0,
                                                 len(df), 252 * 24)
    result["swap_total"] = 0.0
    result["sizer_state"] = None
    result["active_overlays"] = ["grid"]
    return result



def run_with_grid_recovery(df: pd.DataFrame, signals_by_strategy: dict[str, tuple],
                            mgr: GridRecoveryManager,
                            init_cash: float = 10_000.0,
                            commission_pips: float = 0.7,
                            slippage_pips: float = 0.3,
                            spread_pips: float = 0.0,
                            commission_pct: float = 0.0,
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
                spread_pips=spread_pips, commission_pct=commission_pct,
                bar_timestamp=df.index[bar_idx] if hasattr(df.index, '__getitem__') else None,
            )
    # Build aggregate trade log
    trades_df = mgr.to_trades_df()
    # Build equity from closed-trade PnL events. Each trade PnL is applied once
    # at its close bar, then accumulated forward through time.
    if mgr.equity_curve:
        eq_df = pd.DataFrame(mgr.equity_curve, columns=["bar_index", "pnl", "strategy"])
        pnl_events = pd.Series(0.0, index=df.index, name="pnl")
        for _, row in eq_df.iterrows():
            bar_index = int(row["bar_index"])
            if 0 <= bar_index < len(pnl_events):
                pnl_events.iloc[bar_index] += float(row["pnl"])
        equity = (init_cash + pnl_events.cumsum()).rename("equity")
    else:
        equity = pd.Series(init_cash, index=df.index, name="equity")
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

