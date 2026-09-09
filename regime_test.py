"""Run backtest with regime-aware strategy selection.
Shows per-regime performance — which strategies win in which conditions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from data.cache import load as load_cache
from backtester.engine_full import run_full, GRID_NONE
from backtester.metrics_v2 import compute_all
from backtester.trade_journal import trades_to_dataframe
from core.regime import RegimeAwareStrategy, detect_regimes, regime_performance_summary
from strategies import STRATEGY_REGISTRY

print("=" * 70)
print("REGIME-AWARE BACKTEST — EURUSD H1 (1 Year)")
print("=" * 70)

# Load 1Y H1
df_full, _ = load_cache("EURUSD", "H1")
end_date = df_full.index[-1]
df = df_full[df_full.index >= end_date - pd.DateOffset(years=1)].copy()
print(f"Period: {df.index[0].date()} -> {df.index[-1].date()} ({len(df):,} bars)")

# Detect regimes
print("\nDetecting market regimes...")
regimes = detect_regimes(df)
regime_counts = regimes.value_counts()
print(f"\nRegime distribution:")
for regime, count in regime_counts.items():
    pct = count / len(df) * 100
    print(f"  {regime:18s} {count:>6d} bars ({pct:.1f}%)")

# Strategy map (user's example + variations)
configs = {
    "ALL strategies (no regime filter)": {
        "trending_up":   ["ac_ao", "adx", "dem", "fbb", "mfi", "ms"],
        "trending_down": ["ac_ao", "adx", "dem", "fbb", "mfi", "ms"],
        "ranging":       ["ac_ao", "adx", "dem", "fbb", "mfi", "ms"],
        "overextended":  ["ac_ao", "adx", "dem", "fbb", "mfi", "ms"],
        "volatile":      ["ac_ao", "adx", "dem", "fbb", "mfi", "ms"],
        "choppy":        ["ac_ao", "adx", "dem", "fbb", "mfi", "ms"],
    },
    "Conservative (only 1+4+5+6)": {
        "trending_up":   ["ac_ao", "fbb", "mfi", "ms"],
        "trending_down": ["ac_ao", "fbb", "mfi", "ms"],
        "ranging":       ["ac_ao", "fbb", "mfi", "ms"],
        "overextended":  ["ac_ao", "fbb", "mfi", "ms"],
        "volatile":      ["ac_ao", "fbb", "mfi", "ms"],
        "choppy":        ["ac_ao", "fbb", "mfi", "ms"],
    },
    "Regime-aware (smart)": {
        "trending_up":   ["ms", "adx", "ac_ao"],     # trend-following wins in trends
        "trending_down": ["ms", "adx", "ac_ao"],
        "ranging":       ["fbb", "dem"],             # mean-reversion wins in ranges
        "overextended":  ["mfi", "dem"],             # reversals at extremes
        "volatile":      ["fbb"],                     # only mean-reversion survives vol
        "choppy":        ["fbb", "dem"],
    },
    "Aggressive regime-aware": {
        "trending_up":   ["ac_ao", "adx", "ms", "fbb"],
        "trending_down": ["ac_ao", "adx", "ms", "fbb"],
        "ranging":       ["dem", "fbb", "mfi"],
        "overextended":  ["mfi", "dem", "ms"],
        "volatile":      [],
        "choppy":        ["fbb", "dem"],
    },
}

results = {}
for cfg_name, cfg in configs.items():
    print(f"\n{'=' * 70}\n{cfg_name}\n{'=' * 70}")
    strat = RegimeAwareStrategy(df, strategy_map=cfg)
    sig = strat.generate()
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    print(f"  Active entries: {entries.sum()}")
    result = run_full(df, {"regime_aware": (entries, direction)},
                       grid_mode=GRID_NONE, base_lot=0.1)
    m = result["metrics"]
    results[cfg_name] = m
    print(f"  Trades:   {m.get('n_trades', 0)}")
    print(f"  Win rate: {m.get('win_rate', 0):.1%}")
    print(f"  Max win:  ${m.get('largest_win', 0):.2f}")
    print(f"  Max loss: ${m.get('largest_loss', 0):.2f}")
    print(f"  Max DD:   {m.get('max_drawdown', 0):.2%}")
    print(f"  Net P&L:  ${m.get('net_pnl', 0):.2f}")
    print(f"  Sharpe:   {m.get('sharpe', 0):+.2f}")
    print(f"  Profit Factor: {m.get('profit_factor', 0):.2f}")
    if len(result["trades"]) > 0:
        print(f"\n  Per-regime performance:")
        regime_perf = regime_performance_summary(df, result["trades"])
        print(regime_perf.to_string(index=False))

# Comparison
print(f"\n{'=' * 70}")
print("COMPARISON")
print(f"{'=' * 70}")
print(f"{'Config':35s}  {'Trades':>7s}  {'WR':>5s}  {'Sharpe':>7s}  {'PF':>5s}  {'MaxDD':>7s}  {'NetP&L':>9s}")
for cfg_name, m in results.items():
    print(f"{cfg_name:35s}  {m.get('n_trades', 0):>7d}  {m.get('win_rate', 0):>4.0%}  "
          f"{m.get('sharpe', 0):>+7.2f}  {m.get('profit_factor', 0):>5.2f}  "
          f"{m.get('max_drawdown', 0):>6.2%}  ${m.get('net_pnl', 0):>+8.2f}")