"""Full smoke test: synthetic trending data, all 6 strategies, EA-equivalent defaults."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from strategies import STRATEGY_REGISTRY
from backtester.engine import run_direction

np.random.seed(42)
n = 8000
idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")

# Trending data: mild uptrend + noise
drift = np.cumsum(np.random.normal(0.0001, 0.0003, n))
price = 1.10 * np.exp(drift)
high = price * (1 + np.abs(np.random.normal(0, 0.0005, n)))
low = price * (1 - np.abs(np.random.normal(0, 0.0005, n)))
opn = np.roll(price, 1); opn[0] = price[0]
vol = np.random.randint(50, 5000, n)
df = pd.DataFrame({"open": opn, "high": high, "low": low, "close": price, "volume": vol}, index=idx)

# Per-strategy defaults matching MQL5 inputs
DEFAULT = {
    # AC+AO
    "open_orders_type": 1, "close_orders_type": 0,
    "level_open_orders": 80, "level_close_orders": 70,
    "use_acceleration_filter": False, "acceleration_bars": 3, "min_acceleration": 0.0005,
    "use_ao_synchronization": False, "min_ao_synchronization": 0.0003,
    # ADX
    "use_di_crossover": True, "crossover_lookback": 3, "min_crossover_gap": 5,
    # MFI
    "use_slope_filter": False, "min_slope_strength": 3, "slope_lookback": 5,
    "use_divergence": False, "use_hidden_divergence": False, "divergence_bars": 10,
    # FBB
    "deviation": 1.8,
    # MS
    "use_confluence_filter": False, "use_macd_divergence": False,
    "use_stoch_divergence": False, "use_histogram_divergence": False,
    # common
    "bars_calculate": 20,
    "level_open_orders_1": 0, "level_open_orders_2": 50,
    "level_close_orders_1": 40, "level_close_orders_2": 40,
    "open_orders_type_1": 1, "open_orders_type_2": 0,
    "close_orders_type_1": 0, "close_orders_type_2": 0,
    "fast_ema_period": 3, "slow_ema_period": 9, "signal_period": 2,
    "k_period": 5, "d_period": 3, "slowing_period": 12,
}

# Per-strategy overrides for known defaults
DEFAULTS_PER_STRAT = {
    "ac_ao": {"open_orders_type": 1, "level_open_orders": 80, "use_acceleration_filter": False,
              "use_ao_synchronization": False, "close_orders_type": 0},
    "adx":   {"open_orders_type": 1, "close_orders_type": 4,
              "level_open_orders_1": 55, "level_open_orders_2": 15,
              "level_close_orders_1": 15, "level_close_orders_2": 5,
              "use_di_crossover": True, "crossover_lookback": 3, "min_crossover_gap": 5},
    "dem":   {"open_orders_type": 3, "close_orders_type": 0,
              "level_open_orders": 75, "level_close_orders": 70},
    "fbb":   {"open_orders_type_1": 1, "open_orders_type_2": 0,
              "level_open_orders_1": 0, "level_open_orders_2": 50,
              "close_orders_type_1": 0, "close_orders_type_2": 0,
              "level_close_orders_1": 40, "level_close_orders_2": 40},
    "mfi":   {"open_orders_type": 3, "close_orders_type": 0,
              "level_open_orders": 90, "level_close_orders": 90,
              "use_slope_filter": False, "use_divergence": False, "use_hidden_divergence": False},
    "ms":    {"open_orders_type_1": 8, "open_orders_type_2": 0,
              "level_open_orders_1": 60, "level_open_orders_2": 80,
              "close_orders_type_1": 0, "close_orders_type_2": 0,
              "level_close_orders_1": 50, "level_close_orders_2": 65,
              "use_confluence_filter": False},
}

print(f"=== Synthetic trending H1 ({n} bars) ===")
for name, cls in STRATEGY_REGISTRY.items():
    try:
        params = {**DEFAULT, **DEFAULTS_PER_STRAT.get(name, {})}
        s = cls(params=params)
        sig = s.generate(df)
        pf, m = run_direction(df, sig.entries, sig.direction)
        print(f"  {name:8s} trades={m['trades']:4d}  sharpe={m['sharpe']:+.2f}  "
              f"calmar={m['calmar']:+.2f}  mdd={m['max_drawdown']:.2%}  "
              f"equity=${m['final_equity']:.0f}")
    except Exception as e:
        import traceback
        print(f"  {name:8s} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()