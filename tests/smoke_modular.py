"""Quick smoke test of new modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

print("=== Strictness slider ===")
from core.strictness import apply_strictness, apply_tp_sl_widening, RISK_PROFILES

base = {"open_orders_type": 1, "level_open_orders": 80, "use_acceleration_filter": False}
for s in [0, 3, 5, 7, 10]:
    out = apply_strictness("ac_ao", s, base)
    print(f"  ac_ao strictness={s:>2d}: level={out['level_open_orders']}, "
          f"close={out['level_close_orders']}, accel={out.get('use_acceleration_filter')}")

print("\n=== TP/SL widening ===")
base = {"take_profit": 50, "stop_loss": 150}
for w in [0, 5, 10]:
    out = apply_tp_sl_widening(base, w)
    print(f"  widening={w:>2d}: TP={out['take_profit']}, SL={out['stop_loss']}")

print("\n=== Risk profiles ===")
for name, p in RISK_PROFILES.items():
    print(f"  {name:14s}: strictness={p['strictness']}, tp_widening={p['tp_widening']}, "
          f"lot={p['base_lot']}, grid={p['grid_mode']}, recovery={p['recovery_mode']}")

print("\n=== Composite criterion ===")
from analysis.composite_criterion import composite_score, CRITERION_PRESETS
test_metrics = dict(sharpe=1.5, calmar=2.0, profit_factor=1.8, max_drawdown=-0.08)
for name, fn in CRITERION_PRESETS.items():
    s = fn(test_metrics)
    print(f"  {name:30s} -> score={s:.1f}")

print("\n=== Modular features ===")
from core.features import RegimeFilter, CorrelationGuard, DailyPnLCap, SessionFilter
# Build synthetic data + signals
np.random.seed(42)
idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
df = pd.DataFrame({
    "open": 1.10, "high": 1.1005, "low": 1.0995, "close": 1.10,
    "volume": 1000,
}, index=idx)
signals = pd.DataFrame({"entries": [True] * 200, "direction": [1] * 200}, index=idx)
trades = pd.DataFrame({"pnl": [-10, -20, 5, -50, -30], "exit_bar": [10, 30, 50, 100, 150]})

from core.features import FeatureContext
ctx = FeatureContext(params={}, signals=signals, equity=pd.Series([10000]*200, index=idx),
                     trades=trades, df=df)

regime = RegimeFilter(min_adx=25).apply(ctx)
print(f"  RegimeFilter: {regime.notes[-1] if regime.notes else 'no effect'}")

session = SessionFilter(sessions=["london"]).apply(ctx)
print(f"  SessionFilter: {session.notes[-1] if session.notes else 'no effect'}")

daily = DailyPnLCap(max_loss=100).apply(ctx)
print(f"  DailyPnLCap: paused={daily.paused}, reason={daily.pause_reason}")

print("\n=== All new modules functional ===")