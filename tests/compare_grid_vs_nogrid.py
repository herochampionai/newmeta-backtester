"""Compare backtest with vs without grid/recovery on real EURUSD data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from data.cache import load as load_cache
from strategies import STRATEGY_REGISTRY
from backtester.engine import run_direction
from backtester.grid_recovery import (
    GridRecoveryManager, GRID_NONE, GRID_LOSS_AND_PROFIT,
    RECOVERY_NONE, RECOVERY_LAST_CLOSING,
)
from backtester.engine_grid import run_with_grid_recovery

df, meta = load_cache("EURUSD", "H1")
print(f"Data: {len(df)} bars EURUSD H1 ({meta['first'][:10]} to {meta['last'][:10]})")

# Generate signals from all 6 strategies
signals = {}
for name, cls in STRATEGY_REGISTRY.items():
    if name == "ms":
        params = dict(open_orders_type_1=8, open_orders_type_2=0,
                      level_open_orders_1=20, level_open_orders_2=80,
                      use_confluence_filter=False)
    elif name == "fbb":
        params = dict(open_orders_type_1=1, open_orders_type_2=0,
                      level_open_orders_1=0, level_open_orders_2=50,
                      bars_calculate=20, deviation=1.8)
    elif name == "mfi":
        params = dict(open_orders_type=3, level_open_orders=70,
                      level_close_orders=70, bars_calculate=12)
    elif name == "dem":
        params = dict(open_orders_type=3, level_open_orders=75, bars_calculate=20)
    elif name == "adx":
        params = dict(open_orders_type=1, level_open_orders_1=55,
                      level_open_orders_2=15, use_di_crossover=True)
    else:  # ac_ao
        params = dict(open_orders_type=1, level_open_orders=80)
    s = cls(params=params)
    sig = s.generate(df)
    signals[name] = (sig.entries, sig.direction)

# 1) Baseline: vectorbt backtest, no grid/recovery (current default)
print("\n=== A) Vectorized baseline (no grid/recovery) ===")
total_pnl_baseline = 0
for name, (entries, direction) in signals.items():
    pf, summary = run_direction(df, entries, direction, init_cash=10000)
    print(f"  {name:8s} trades={summary['trades']:>4d}  sharpe={summary['sharpe']:+.2f}  "
          f"final=${summary['final_equity']:.0f}")
    total_pnl_baseline += summary['final_equity'] - 10000
print(f"  Total PnL (sum of independent): ${total_pnl_baseline:.2f}")

# 2) Grid + Recovery simulation (EA-equivalent)
print("\n=== B) Grid + Recovery mode (EA default) ===")
mgr = GridRecoveryManager(
    grid_mode=GRID_LOSS_AND_PROFIT,
    pips_between_orders=30, grid_lot_multiplier=1.5,
    grid_take_profit=50.0, grid_stop_loss=200.0, max_layers=4,
    recovery_mode=RECOVERY_LAST_CLOSING, recovery_lot_multiplier=2.0,
    base_lot=0.1, pip_size=0.0001, contract_size=100_000,
)
result = run_with_grid_recovery(df, signals, mgr, init_cash=10000)
s = mgr.summary()
print(f"  Total grid trades: {s['n_grid_trades']}")
print(f"  Total PnL:         ${s['grid_total_pnl']:.2f}")
print(f"  Win rate:          {s['grid_win_rate']:.1%}")
print(f"  Avg win / loss:    ${s['grid_avg_win']:.2f} / ${s['grid_avg_loss']:.2f}")
print(f"  Largest win/loss:  ${s['grid_largest_win']:.2f} / ${s['grid_largest_loss']:.2f}")
print(f"  Max layers:        {s['grid_max_layers']}")

# Per-strategy
print("\nPer-strategy grid breakdown:")
for strat, sm in result["per_strategy"].items():
    n_tr = sm.get("n_trades", 0)
    pnl = sm.get("net_pnl", 0)
    wr = sm.get("win_rate", 0)
    pf_v = sm.get("profit_factor", 0)
    print(f"  {strat:8s}  trades={n_tr:>3d}  PnL=${pnl:>7.0f}  win_rate={wr:.1%}  PF={pf_v:.2f}")

print("\n=== C) No grid, just recovery (recovery_lot_multiplier only) ===")
mgr = GridRecoveryManager(
    grid_mode=GRID_NONE,
    grid_take_profit=80.0, grid_stop_loss=400.0,
    recovery_mode=RECOVERY_LAST_CLOSING, recovery_lot_multiplier=2.0,
    base_lot=0.1, pip_size=0.0001, contract_size=100_000,
)
result = run_with_grid_recovery(df, signals, mgr, init_cash=10000)
s = mgr.summary()
print(f"  Total trades: {s['n_grid_trades']}, PnL=${s['grid_total_pnl']:.2f}, "
      f"WR={s['grid_win_rate']:.1%}")