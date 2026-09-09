"""Real-data backtest across all 6 strategies with MQL5-equivalent defaults."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.cache import load as load_cache
from strategies import STRATEGY_REGISTRY
from backtester.engine import run_direction

df, meta = load_cache("EURUSD", "H1")
print(f"Loaded {len(df)} bars EURUSD H1 ({meta['first']} -> {meta['last']})")

REAL_DEFAULT = {
    "ac_ao": dict(open_orders_type=1, close_orders_type=0, level_open_orders=80,
                  level_close_orders=70, use_acceleration_filter=False,
                  use_ao_synchronization=False, acceleration_bars=3,
                  min_acceleration=0.0005, min_ao_synchronization=0.0003),
    "adx":   dict(open_orders_type=1, close_orders_type=4, level_open_orders_1=55,
                  level_open_orders_2=15, level_close_orders_1=15,
                  level_close_orders_2=5, use_di_crossover=True,
                  crossover_lookback=3, min_crossover_gap=5, bars_calculate=20),
    "dem":   dict(open_orders_type=3, close_orders_type=0, level_open_orders=75,
                  level_close_orders=70, bars_calculate=20),
    "fbb":   dict(open_orders_type_1=1, open_orders_type_2=0, close_orders_type_1=0,
                  close_orders_type_2=0, level_open_orders_1=0,
                  level_open_orders_2=50, level_close_orders_1=40,
                  level_close_orders_2=40, bars_calculate=20, deviation=1.8),
    "mfi":   dict(open_orders_type=3, close_orders_type=0, level_open_orders=70,
                  level_close_orders=70, use_slope_filter=False,
                  use_divergence=False, use_hidden_divergence=False,
                  bars_calculate=12, slope_lookback=5, min_slope_strength=3,
                  divergence_bars=10),
    "ms":    dict(open_orders_type_1=8, open_orders_type_2=0, close_orders_type_1=0,
                  close_orders_type_2=0, level_open_orders_1=20,
                  level_open_orders_2=80, level_close_orders_1=50,
                  level_close_orders_2=65, use_confluence_filter=False,
                  use_macd_divergence=False, use_stoch_divergence=False,
                  use_histogram_divergence=False, fast_ema_period=3,
                  slow_ema_period=9, signal_period=2, k_period=5, d_period=3,
                  slowing_period=12),
}

print()
print(f"{'':10s} {'trades':>7s} {'sharpe':>7s} {'sortino':>8s} {'calmar':>7s} {'maxDD':>7s} {'equity':>9s}")
print("-" * 60)
for name, cls in STRATEGY_REGISTRY.items():
    try:
        s = cls(params=REAL_DEFAULT[name])
        sig = s.generate(df)
        pf, m = run_direction(df, sig.entries, sig.direction)
        print(f"{name:10s} {m['trades']:>7d} {m['sharpe']:>+7.2f} {m['sortino']:>+8.2f} "
              f"{m['calmar']:>+7.2f} {m['max_drawdown']:>7.2%} ${m['final_equity']:>8.0f}")
    except Exception as e:
        print(f"{name:10s} FAILED: {type(e).__name__}: {e}")