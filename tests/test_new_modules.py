"""Smoke test for new modules: profile analyzer, ticker scanner, deep backtest, trade journal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

print("=" * 60)
print("Test 1: Strategy profile analyzer on real MQL5 file")
print("=" * 60)
from core.strategy_profile import analyze_strategy_file, format_profile_for_display
mq5_path = Path(r"D:\Trading\TRADING\youha created EA\Multi strat ea\multi strat newmeta.mq5")
profile = analyze_strategy_file(mq5_path)
print(f"Path: {profile['path']}")
print(f"Name: {profile['name']}")
print(f"Size: {profile['size_kb']} KB")
print(f"Parameters detected: {profile.get('parameter_count', 0)}")
print(f"Strictness estimate: {profile.get('strictness_estimate', '?')}")
print(f"\nFeatures:")
for k in ["has_grid", "has_recovery", "has_adaptive_sizing", "has_mtf",
          "has_divergence", "has_swaps", "has_break_even", "has_trailing_stop",
          "has_sl_tp", "has_quad_stoch", "has_confluence"]:
    mark = "[X]" if profile.get(k) else "[ ]"
    print(f"  {mark} {k.replace('_', ' ').title()}")
print(f"\nIndicators: {', '.join(profile['indicators_found'])}")
print(f"Risk features: {profile['risk_features']}")
print(f"\nSummary: {format_profile_for_display(profile)}")

print("\n" + "=" * 60)
print("Test 2: Trade journal export (CSV + HTML)")
print("=" * 60)
from backtester.trade_journal import export_to_csv, export_to_html
# Use real data
from data.cache import load as load_cache
df, _ = load_cache("EURUSD", "H1")
# Build synthetic trades from grid backtest
from strategies import STRATEGY_REGISTRY
from backtester.engine_grid import run_with_grid_recovery as run_grid
from backtester.grid_recovery import GridRecoveryManager, GRID_NONE
strat = STRATEGY_REGISTRY["fbb"](params={"open_orders_type_1": 1, "level_open_orders_1": 0,
                                          "level_open_orders_2": 50, "bars_calculate": 20,
                                          "deviation": 1.8})
sig = strat.generate(df)
entries = sig.entries.fillna(False).astype(bool)
direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
mgr = GridRecoveryManager(grid_mode=GRID_NONE, base_lot=0.1)
sig_arr = (entries.values.astype(bool), direction.values.astype(int))
for i in range(len(df)):
    bar = df.iloc[i]
    mgr.on_bar_close("fbb", int(sig_arr[1][i]) if sig_arr[0][i] else 0,
                     float(bar["high"]), float(bar["low"]), float(bar["close"]),
                     i, 0.7, 0.3)
trades = mgr.to_trades_df()
print(f"Generated {len(trades)} trades")
csv_path = export_to_csv(trades, df.index, "output/journal.csv")
print(f"CSV saved to: {csv_path}")
html_path = export_to_html(trades, df.index, "output/journal.html",
                              metrics={"sharpe": 1.2, "win_rate": 0.55,
                                        "profit_factor": 1.8, "max_drawdown": -0.12})
print(f"HTML saved to: {html_path}")
print(f"HTML size: {Path(html_path).stat().st_size} bytes")

print("\n" + "=" * 60)
print("Test 3: Ticker scanner (single symbol — fast smoke test)")
print("=" * 60)
from analysis.ticker_scanner import scan_symbols
# Only 2 symbols to keep it fast
df_syms = scan_symbols("fbb", ["EURUSD"], timeframe="H1", start="2023-01-01")
if not df_syms.empty:
    print(df_syms[["rank", "symbol", "sharpe", "calmar", "win_rate", "composite_score"]].to_string(index=False))

print("\n" + "=" * 60)
print("Test 4: Deep backtest mode (synthetic ticks)")
print("=" * 60)
from backtester.engine_deep import deep_backtest
import tempfile
# Use a tiny synthetic df for speed
small_df = df.iloc[:2000]
small_entries = entries.iloc[:2000]
small_direction = direction.iloc[:2000]
result = deep_backtest(small_df, {"fbb": (small_entries, small_direction)},
                        ticks_per_bar=5, grid_mode=GRID_NONE, progress_every=0)
print(f"  tick_source: {result['tick_source']}")
print(f"  spread_source: {result['spread_source']}")
print(f"  n_bars: {result['n_bars']}")
print(f"  n_ticks_synthesized: {result['n_ticks_synthesized']}")
print(f"  avg_spread_pips: {result['avg_spread_pips']:.2f}")
print(f"  max_spread_pips: {result['max_spread_pips']:.2f}")
print(f"  sharpe: {result['metrics']['sharpe']:+.2f}")
print(f"  trades: {result['metrics']['n_trades']}")
print(f"  deep_mode: {result['deep_mode']}")

print("\n" + "=" * 60)
print("Test 5: Hybrid Grid Lab integration")
print("=" * 60)
from backtester.hybrid_grid import GridEngine, AdaptiveState, Order
print(f"  GridEngine: {GridEngine.__name__}")
print(f"  AdaptiveState: {AdaptiveState.__name__}")
print(f"  Can be used for advanced grid simulation with funding + session profit-taking.")

print("\n" + "=" * 60)
print("ALL NEW MODULE TESTS COMPLETE")
print("=" * 60)