"""Backtest engine FULL — thin orchestrator that composes pure + grid + adaptive + swap.

Pipeline:
  1. If grid_mode != GRID_NONE → run_grid()  (uses GridRecoveryManager)
     Else                   → run_pure()   (uses vectorbt)
  2. If adaptive_enabled    → apply_adaptive()
  3. If swap_enabled        → apply_swap()

Each overlay is independently optional. The flag name means what it says.
"""
from __future__ import annotations
import pandas as pd

from backtester.adaptive import AdaptiveConfig
from backtester.grid_recovery import GridRecoveryManager, GRID_NONE, GRID_LOSS, GRID_PROFIT, GRID_LOSS_AND_PROFIT
from backtester.engine_grid import run_grid
from backtester.engine_pure import run_pure
from backtester.engine_adaptive import apply_adaptive
from backtester.engine_swap import apply_swap


GRID_LABELS = {GRID_NONE: "None", GRID_LOSS: "Loss", GRID_PROFIT: "Profit",
                GRID_LOSS_AND_PROFIT: "Loss+Profit"}


def run_full(df: pd.DataFrame,
              signals_by_strategy: dict[str, tuple],
              init_cash: float = 10_000.0,
              commission_pips: float = 0.7,
              slippage_pips: float = 0.3,
              pip_size: float = 0.0001,
              contract_size: float = 100_000,
              # Grid overlay
              grid_mode: int = GRID_NONE,
              pips_between_orders: float = 30.0,
              grid_lot_multiplier: float = 1.5,
              grid_take_profit: float = 50.0,
              grid_stop_loss: float = 200.0,
              max_grid_layers: int = 4,
              recovery_mode: int = 0,
              recovery_lot_multiplier: float = 2.0,
              base_lot: float = 0.1,
              # Adaptive overlay
              adaptive_enabled: bool = False,
              adaptive_config: AdaptiveConfig | None = None,
              # Swap overlay
              swap_enabled: bool = False,
              long_swap_pips: float = -0.5,
              short_swap_pips: float = 0.2,
              ) -> dict:
    """Run backtest with optional overlays.

    Each overlay (grid / adaptive / swap) runs ONLY when its flag is True.
    The flag name means what it says — no hidden behavior.

    Returns dict with all the result keys.
    """
    # Stage 1: base engine (pure OR grid)
    if grid_mode != GRID_NONE:
        result = run_grid(
            df, signals_by_strategy,
            init_cash=init_cash,
            commission_pips=commission_pips, slippage_pips=slippage_pips,
            pip_size=pip_size, contract_size=contract_size,
            grid_mode=grid_mode,
            pips_between_orders=pips_between_orders,
            grid_lot_multiplier=grid_lot_multiplier,
            grid_take_profit=grid_take_profit,
            grid_stop_loss=grid_stop_loss,
            max_grid_layers=max_grid_layers,
            recovery_mode=recovery_mode,
            recovery_lot_multiplier=recovery_lot_multiplier,
            base_lot=base_lot,
        )
        result["active_overlays"] = ["grid"]
    else:
        result = run_pure(
            df, signals_by_strategy,
            init_cash=init_cash,
            commission_pips=commission_pips, slippage_pips=slippage_pips,
        )
        result["active_overlays"] = []

    # Stage 2: adaptive overlay (opt-in)
    if adaptive_enabled:
        result = apply_adaptive(result, adaptive_config=adaptive_config, base_lot=base_lot)

    # Stage 3: swap overlay (opt-in)
    if swap_enabled:
        result = apply_swap(result, df,
                            long_swap_pips=long_swap_pips,
                            short_swap_pips=short_swap_pips,
                            pip_size=pip_size,
                            contract_size=contract_size)

    return result