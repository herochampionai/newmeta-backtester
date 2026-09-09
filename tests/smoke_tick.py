"""Phase C1 smoke test: tick data + realistic spread."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.tick_data import synthesize_ticks_from_bars, fetch_ticks_with_priority, aggregate_ticks_to_bars
from data.spread_spec import default_spread_series, get_spreads_mt5

print("=== Tick data smoke test ===\n")

# Synthesize ticks from synthetic bars
import numpy as np
import pandas as pd

np.random.seed(42)
n = 100
idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
price = 1.10 * np.exp(np.cumsum(np.random.normal(0, 0.0003, n)))
bars = pd.DataFrame({
    "open": price, "high": price * 1.0005, "low": price * 0.9995,
    "close": price, "volume": 1000,
}, index=idx)

ticks = synthesize_ticks_from_bars(bars, ticks_per_bar=20)
print(f"Synthesized {len(ticks)} ticks from {len(bars)} bars ({len(ticks)/len(bars):.0f} ticks/bar)")
print(f"  Time range: {ticks.index[0]} to {ticks.index[-1]}")
print(f"  Price range: bid min={ticks['bid'].min():.5f}, ask max={ticks['ask'].max():.5f}")

# Verify we can aggregate back to bars
agg = aggregate_ticks_to_bars(ticks, freq="1h")
print(f"\nAggregated back to {len(agg)} bars")
print(f"  Bar[0]: open={agg.iloc[0]['open']:.5f} high={agg.iloc[0]['high']:.5f} "
      f"low={agg.iloc[0]['low']:.5f} close={agg.iloc[0]['close']:.5f}")
print(f"  Original bar[0]: open={bars.iloc[0]['open']:.5f} high={bars.iloc[0]['high']:.5f} "
      f"low={bars.iloc[0]['low']:.5f} close={bars.iloc[0]['close']:.5f}")

# Real MT5 ticks (skip if MT5 unavailable — slow)
print("\n=== Real MT5 tick fetch (skipped — slow) ===")
print("  Use data.tick_data.fetch_ticks_with_priority() when needed")

# Realistic spread series
print("\n=== Default spread series (no MT5 needed) ===")
spreads = default_spread_series(idx)
print(f"  Mean spread (London/NY hours): "
      f"{spreads[(idx.hour >= 7) & (idx.hour < 20) & (idx.weekday < 5)].mean():.2f} pips")
print(f"  Mean spread (Asian/night): "
      f"{spreads[(idx.hour < 7) | (idx.hour >= 20) & (idx.weekday < 5)].mean():.2f} pips")
print(f"  Weekend: not in test window (Mon-Fri only)")

# Real MT5 spreads if available
print("\n=== Real MT5 spread fetch ===")
real_spreads = get_spreads_mt5("EURUSD", n_days=7)
if real_spreads is not None and len(real_spreads) > 0:
    print(f"  Got {len(real_spreads)} 1-min spread samples")
    print(f"  Mean: {real_spreads.mean():.2f} pips")
    print(f"  Median: {real_spreads.median():.2f} pips")
    print(f"  Max: {real_spreads.max():.2f} pips")
else:
    print("  No real data (MT5 unavailable or no ticks)")

print("\n=== Tick-data + spread integration complete ===")