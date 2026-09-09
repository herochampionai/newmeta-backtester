"""Smoke test: GridRecoveryManager + run_with_grid_recovery."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from backtester.grid_recovery import (
    GridRecoveryManager, GRID_LOSS_AND_PROFIT, GRID_NONE, GRID_LOSS, GRID_PROFIT,
    RECOVERY_LAST_CLOSING, RECOVERY_NONE,
)
from backtester.engine_grid import run_with_grid_recovery

# Synthetic trending data (mild uptrend, occasional dips)
np.random.seed(42)
n = 2000
idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
price = 1.10 * np.exp(np.cumsum(np.random.normal(0, 0.0005, n)))
high = price * (1 + np.abs(np.random.normal(0, 0.0007, n)))
low = price * (1 - np.abs(np.random.normal(0, 0.0007, n)))
opn = np.roll(price, 1); opn[0] = price[0]
df = pd.DataFrame({"open": opn, "high": high, "low": low, "close": price, "volume": 1000}, index=idx)

# Fake signals: long every 100 bars
entries = pd.Series(False, index=df.index)
entries.iloc[::100] = True
direction = pd.Series(0, index=df.index)
direction.iloc[::100] = 1

print("=" * 60)
print("Test 1: GRID_NONE (no grid logic, just initial trades)")
mgr = GridRecoveryManager(grid_mode=GRID_NONE, base_lot=0.1,
                           grid_take_profit=50.0, grid_stop_loss=200.0)
result = run_with_grid_recovery(df, {"fbb": (entries, direction)}, mgr)
s = mgr.summary()
print(f"  grid trades: {s['n_grid_trades']}, "
      f"total PnL: ${s['grid_total_pnl']:.2f}, "
      f"win rate: {s['grid_win_rate']:.1%}")

print("\nTest 2: GRID_LOSS_AND_PROFIT + recovery (default EA mode)")
mgr = GridRecoveryManager(
    grid_mode=GRID_LOSS_AND_PROFIT,
    pips_between_orders=20, grid_lot_multiplier=1.5,
    grid_take_profit=50.0, grid_stop_loss=200.0, max_layers=5,
    recovery_mode=RECOVERY_LAST_CLOSING, recovery_lot_multiplier=2.0,
    base_lot=0.1, pip_size=0.0001, contract_size=100_000,
)
result = run_with_grid_recovery(df, {"fbb": (entries, direction)}, mgr)
s = mgr.summary()
m = result["metrics"]
print(f"  grid trades:  {s['n_grid_trades']}")
print(f"  total PnL:    ${s['grid_total_pnl']:.2f}")
print(f"  win rate:     {s['grid_win_rate']:.1%}")
print(f"  avg win/loss: ${s['grid_avg_win']:.2f} / ${s['grid_avg_loss']:.2f}")
print(f"  max layers:   {s['grid_max_layers']}")
print(f"  portfolio Sharpe: {m['sharpe']:+.2f}, Max DD: {m['max_drawdown']:.2%}")

print("\nTest 3: GRID_PROFIT only (pyramiding winners)")
mgr = GridRecoveryManager(
    grid_mode=GRID_PROFIT,
    pips_between_orders=15, grid_lot_multiplier=1.0,
    grid_take_profit=80.0, grid_stop_loss=500.0, max_layers=4,
    base_lot=0.1,
)
result = run_with_grid_recovery(df, {"fbb": (entries, direction)}, mgr)
s = mgr.summary()
print(f"  grid trades:  {s['n_grid_trades']}, max layers: {s['grid_max_layers']}, "
      f"PnL: ${s['grid_total_pnl']:.2f}")

print("\nTest 4: 6-strategy multi-strategy grid")
entries_per = {}
for name in ["ac_ao", "adx", "dem", "fbb", "mfi", "ms"]:
    e = pd.Series(False, index=df.index)
    e.iloc[::(80 + 10 * hash(name) % 30)] = True
    d = pd.Series(0, index=df.index)
    d.iloc[::(80 + 10 * hash(name) % 30)] = 1 if hash(name) % 2 else -1
    entries_per[name] = (e, d)

mgr = GridRecoveryManager(
    grid_mode=GRID_LOSS_AND_PROFIT, pips_between_orders=20,
    grid_lot_multiplier=1.5, grid_take_profit=50.0, grid_stop_loss=200.0,
    max_layers=4, recovery_mode=RECOVERY_LAST_CLOSING,
    recovery_lot_multiplier=2.0, base_lot=0.1,
)
result = run_with_grid_recovery(df, entries_per, mgr, init_cash=10000)
s = mgr.summary()
m = result["metrics"]
print(f"  grid trades (6 strategies): {s['n_grid_trades']}")
print(f"  total PnL:    ${s['grid_total_pnl']:.2f}")
print(f"  portfolio Sharpe: {m['sharpe']:+.2f}, Max DD: {m['max_drawdown']:.2%}")
print()
print("Per-strategy breakdown:")
for strat, sm in result["per_strategy"].items():
    n_trades = sm.get("n_trades", 0)
    pnl = sm.get("net_pnl", 0)
    wr = sm.get("win_rate", 0)
    print(f"  {strat:8s}  trades={n_trades:3d}  PnL=${pnl:>8.0f}  win_rate={wr:.1%}")

print("\nFirst 3 grid-closed trades:")
print(result["trades"].head(3).to_string())