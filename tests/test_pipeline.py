"""Test the refactored engine pipeline: each overlay is independently optional."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from data.cache import load as load_cache
from backtester.engine_pure import run_pure
from backtester.engine_grid import run_with_grid_recovery as run_grid
from backtester.engine_adaptive import apply_adaptive
from backtester.engine_swap import apply_swap
from backtester.engine_full import run_full, GRID_LABELS
from backtester.grid_recovery import GRID_NONE, GRID_LOSS_AND_PROFIT
from strategies import STRATEGY_REGISTRY

df, _ = load_cache("EURUSD", "H1")
print(f"Data: {len(df)} bars EURUSD H1")

# Build FBB signals
strat = STRATEGY_REGISTRY["fbb"](params={"open_orders_type_1": 1, "open_orders_type_2": 0,
                                          "level_open_orders_1": 0, "level_open_orders_2": 50,
                                          "bars_calculate": 20, "deviation": 1.8})
sig = strat.generate(df)
entries = sig.entries.fillna(False).astype(bool)
direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
signals = {"fbb": (entries, direction)}

print("\n=== Test 1: Pure engine (no overlays) ===")
r = run_pure(df, signals)
print(f"  active_overlays: {r['active_overlays']}")
print(f"  trades: {len(r['trades'])}")
print(f"  equity: ${r['equity'].iloc[-1]:.0f}")
print(f"  sharpe: {r['metrics']['sharpe']:+.2f}")
print(f"  swap_total: ${r['swap_total']:.2f}")

print("\n=== Test 2: Pure + Adaptive ===")
r = apply_adaptive(r.copy())
print(f"  active_overlays: {r['active_overlays']}")
print(f"  sizer_state: paused={r['sizer_state']['paused']}, lot={r['sizer_state']['current_lot']}")

print("\n=== Test 3: Pure + Swap ===")
r = apply_swap(r.copy(), df, long_swap_pips=-0.5, short_swap_pips=0.2)
print(f"  active_overlays: {r['active_overlays']}")
print(f"  swap_total: ${r['swap_total']:.2f}")

print("\n=== Test 4: Grid alone (via run_full with grid_mode set) ===")
r_grid = run_full(df, signals,
                   grid_mode=GRID_LOSS_AND_PROFIT,
                   pips_between_orders=30, grid_take_profit=50,
                   grid_stop_loss=200, base_lot=0.1)
print(f"  active_overlays: {r_grid['active_overlays']}")
print(f"  grid trades: {r_grid['grid_summary']['n_grid_trades']}")
print(f"  equity: ${r_grid['equity'].iloc[-1]:.0f}")

print("\n=== Test 5: Grid + Adaptive (NO swap) ===")
r_full_no_swap = run_full(df, signals,
                            grid_mode=GRID_LOSS_AND_PROFIT,
                            adaptive_enabled=True, swap_enabled=False,
                            base_lot=0.1)
print(f"  active_overlays: {r_full_no_swap['active_overlays']}")
print(f"  swap_total: ${r_full_no_swap['swap_total']:.2f}  (should be 0)")

print("\n=== Test 6: Grid + Adaptive + Swap (all on) ===")
r_all = run_full(df, signals,
                  grid_mode=GRID_LOSS_AND_PROFIT,
                  adaptive_enabled=True, swap_enabled=True,
                  long_swap_pips=-0.5, short_swap_pips=0.2,
                  base_lot=0.1)
print(f"  active_overlays: {r_all['active_overlays']}")
print(f"  swap_total: ${r_all['swap_total']:.2f}  (should be non-zero)")
print(f"  equity: ${r_all['equity'].iloc[-1]:.0f}")
print(f"  sharpe: {r_all['metrics']['sharpe']:+.2f}")

print("\n=== Test 7: run_full with NO overlays (default) ===")
r_none = run_full(df, signals)  # all defaults: no grid, no adaptive, no swap
print(f"  active_overlays: {r_none['active_overlays']}  (should be [])")
print(f"  swap_total: ${r_none['swap_total']:.2f}  (should be 0)")
print(f"  equity: ${r_none['equity'].iloc[-1]:.0f}")

print("\n=== Test 8: Verify each overlay changes results ===")
r_pure = run_full(df, signals)
r_swap_only = run_full(df, signals, swap_enabled=True)
delta_swap = r_swap_only['equity'].iloc[-1] - r_pure['equity'].iloc[-1]
print(f"  Pure equity: ${r_pure['equity'].iloc[-1]:.0f}")
print(f"  Pure+Swap equity: ${r_swap_only['equity'].iloc[-1]:.0f}")
print(f"  Delta from swap alone: ${delta_swap:+.0f}  (should NOT be 0)")

print("\n=== ALL OVERLAY TESTS COMPLETE ===")