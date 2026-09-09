"""Quick smoke test for tick + spread (no MT5 needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

from data.spread_spec import default_spread_series
from data.tick_data import synthesize_ticks_from_bars, aggregate_ticks_to_bars

idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
spreads = default_spread_series(idx)
print(f"Spread series: {len(spreads)} bars")
print(f"  Mean:    {spreads.mean():.2f} pips")
print(f"  Median:  {spreads.median():.2f} pips")
print(f"  Std:     {spreads.std():.2f} pips")
print(f"  Min:     {spreads.min():.2f}")
print(f"  Max:     {spreads.max():.2f}")

np.random.seed(42)
n = 100
idx2 = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
price = 1.10 * np.exp(np.cumsum(np.random.normal(0, 0.0003, n)))
bars = pd.DataFrame({"open": price, "high": price * 1.0005, "low": price * 0.9995,
                     "close": price, "volume": 1000}, index=idx2)
ticks = synthesize_ticks_from_bars(bars, ticks_per_bar=10)
print(f"\nTick synthesis: {len(ticks)} ticks from {len(bars)} bars")
print(f"  Mean bid:    {ticks['bid'].mean():.5f}")
print(f"  Mean ask:    {ticks['ask'].mean():.5f}")
print(f"  Mean spread: {(ticks['ask'] - ticks['bid']).mean() * 10000:.2f} pips")
agg = aggregate_ticks_to_bars(ticks, freq="1h")
print(f"  Aggregated back: {len(agg)} bars (matches input: {len(agg) == len(bars)})")

# Final check: can we run a full backtest using ticks?
print("\n=== Tick-level backtest integration ===")
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from strategies import STRATEGY_REGISTRY

# Use bars (since our engine runs on bars) but show that tick data
# is available for future tick-level execution simulation
strat = STRATEGY_REGISTRY["fbb"](params={"open_orders_type_1": 1, "level_open_orders_1": 0,
                                          "level_open_orders_2": 50, "bars_calculate": 20,
                                          "deviation": 1.8})
sig = strat.generate(bars)
result = run_full(bars, {"fbb": (sig.entries.fillna(False).astype(bool),
                                 pd.Series(sig.direction, index=bars.index).fillna(0).astype(int))},
                   grid_mode=GRID_NONE, base_lot=0.1)
m = result["metrics"]
print(f"  Test backtest: {m.get('n_trades', 0)} trades, Sharpe {m['sharpe']:+.2f}, equity ${m['final_equity']:.0f}")
print(f"\n[OK] Tick data + spread modules functional")