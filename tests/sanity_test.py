"""Sanity check: confirm AC+AO triggers on trending data and shows proper range."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from strategies.ac_ao import AC_AO_Strategy
from strategies.ms import MS_Strategy
from strategies import indicators as ind
from backtester.engine import run_direction

np.random.seed(42)
n = 8000
idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")

# Trending data: drift + noise
drift = np.cumsum(np.random.normal(0.0001, 0.0003, n))  # mild uptrend
price = 1.10 * np.exp(drift)
high = price * (1 + np.abs(np.random.normal(0, 0.0005, n)))
low = price * (1 - np.abs(np.random.normal(0, 0.0005, n)))
opn = np.roll(price, 1); opn[0] = price[0]
vol = np.random.randint(50, 5000, n)
df = pd.DataFrame({"open": opn, "high": high, "low": low, "close": price, "volume": vol}, index=idx)

# Check indicator ranges
ac_raw = ind.ac(df["high"], df["low"])
ao_raw = ind.ao(df["high"], df["low"])
ac_scaled = ac_raw * 40000
print(f"AC raw range:   [{ac_raw.min():.5f}, {ac_raw.max():.5f}]")
print(f"AC×40000 range: [{ac_scaled.min():.1f}, {ac_scaled.max():.1f}]")
print(f"AO raw range:   [{ao_raw.min():.5f}, {ao_raw.max():.5f}]")
print(f"Bars with |AC×40000| > 80: {(ac_scaled.abs() > 80).sum()} / {len(ac_scaled)}")
print()

# AC: type 1 default (continuation)
params_ac = dict(open_orders_type=1, close_orders_type=0, level_open_orders=80,
                 level_close_orders=70, use_acceleration_filter=False,
                 use_ao_synchronization=False, acceleration_bars=3,
                 min_acceleration=0.0005, min_ao_synchronization=0.0003)
sig = AC_AO_Strategy(params=params_ac).generate(df)
pf, m = run_direction(df, sig.entries, sig.direction)
print(f"AC+AO type 1, level 80:  trades={m['trades']}  sharpe={m['sharpe']:+.2f}  equity=${m['final_equity']:.0f}")

# AC: type 5 (continuation down — symmetric)
params_ac5 = dict(params_ac); params_ac5["open_orders_type"] = 5
sig = AC_AO_Strategy(params=params_ac5).generate(df)
pf, m = run_direction(df, sig.entries, sig.direction)
print(f"AC+AO type 5, level 80:  trades={m['trades']}  sharpe={m['sharpe']:+.2f}  equity=${m['final_equity']:.0f}")

# AC: type 2 (cross up)
params_ac2 = dict(params_ac); params_ac2["open_orders_type"] = 2
sig = AC_AO_Strategy(params=params_ac2).generate(df)
pf, m = run_direction(df, sig.entries, sig.direction)
print(f"AC+AO type 2, level 80:  trades={m['trades']}  sharpe={m['sharpe']:+.2f}  equity=${m['final_equity']:.0f}")

# MS: type 8 with confluence
params_ms = dict(open_orders_type_1=8, open_orders_type_2=0,
                 level_open_orders_1=60, level_open_orders_2=80,
                 close_orders_type_1=0, close_orders_type_2=0,
                 level_close_orders_1=50, level_close_orders_2=65,
                 fast_ema_period=3, slow_ema_period=9, signal_period=2,
                 k_period=5, d_period=3, slowing_period=12,
                 use_confluence_filter=True, use_macd_divergence=False,
                 use_stoch_divergence=False, use_histogram_divergence=False)
sig = MS_Strategy(params=params_ms).generate(df)
pf, m = run_direction(df, sig.entries, sig.direction)
print(f"MS type 8, conf on:     trades={m['trades']}  sharpe={m['sharpe']:+.2f}  equity=${m['final_equity']:.0f}")

# MS: type 0 special (always allow Type_1)
params_ms0 = dict(params_ms); params_ms0["open_orders_type_1"] = 0
sig = MS_Strategy(params=params_ms0).generate(df)
pf, m = run_direction(df, sig.entries, sig.direction)
print(f"MS type 0 (always):     trades={m['trades']}  sharpe={m['sharpe']:+.2f}  equity=${m['final_equity']:.0f}")