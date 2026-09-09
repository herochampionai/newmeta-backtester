"""Monitor multi-strategy EA without grid. Run on real EURUSD H1.
Show: number of trades, win rate, max win, max loss, max drawdown, full journal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from data.cache import load as load_cache
from backtester.engine_full import run_full, GRID_NONE
from strategies import STRATEGY_REGISTRY
from backtester.trade_journal import trades_to_dataframe, export_to_csv, export_to_html

print("=" * 70)
print("MULTI-STRATEGY EA — TRADE MONITOR (No Grid, H1, Real EURUSD)")
print("=" * 70)

# Load real H1 data
df_full, meta = load_cache("EURUSD", "H1")
print(f"\nFull data: {len(df_full)} bars EURUSD H1 ({df_full.index[0].date()} -> {df_full.index[-1].date()})")
assert df_full.index[0].minute == 0, "Not H1 timeframe"

# Use last 1 year (most representative of current market)
end_date = df_full.index[-1]
start_date = end_date - pd.DateOffset(years=1)
df = df_full[df_full.index >= start_date].copy()
print(f"Period: 1 year ({df.index[0].date()} -> {df.index[-1].date()}, {len(df):,} bars)")

# Build all 6 strategy signals with realistic defaults
STRATEGY_DEFAULTS = {
    "ac_ao": dict(open_orders_type=1, close_orders_type=0, level_open_orders=80,
                   level_close_orders=70, use_acceleration_filter=False,
                   use_ao_synchronization=False),
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
                   bars_calculate=12, slope_lookback=5, min_slope_strength=3),
    "ms":    dict(open_orders_type_1=8, open_orders_type_2=0, close_orders_type_1=0,
                   close_orders_type_2=0, level_open_orders_1=20,
                   level_open_orders_2=80, level_close_orders_1=50,
                   level_close_orders_2=65, use_confluence_filter=False,
                   use_macd_divergence=False, use_stoch_divergence=False,
                   use_histogram_divergence=False, fast_ema_period=3,
                   slow_ema_period=9, signal_period=2, k_period=5,
                   d_period=3, slowing_period=12),
}

# Generate signals per strategy
print("\nGenerating signals...")
signals = {}
for name, cls in STRATEGY_REGISTRY.items():
    if name in ("universal", "mtf_stoch"):
        continue
    params = STRATEGY_DEFAULTS.get(name, {})
    sig = cls(params=params).generate(df)
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    signals[name] = (entries, direction)
    print(f"  {name:8s} entries={entries.sum():>5d}  long={(direction==1).sum():>4d}  short={(direction==-1).sum():>4d}")

# Run backtest WITHOUT grid (pure signals only)
print(f"\nRunning backtest with {len(signals)} strategies, NO grid, NO recovery, NO adaptive, NO swap...")
result = run_full(df, signals, grid_mode=GRID_NONE, base_lot=0.1,
                   init_cash=10000, commission_pips=0.7, slippage_pips=0.3)

# Get metrics
m = result["metrics"]
trades = result["trades"]
equity = result["equity"]

print("\n" + "=" * 70)
print("FINAL TEST RESULTS — Multi Strategy EA on EURUSD H1 (no grid)")
print("=" * 70)

print(f"\nTimeframe: H1 (verified: first bar at {df.index[0]})")
print(f"Period: {df.index[0].date()} to {df.index[-1].date()} ({len(df):,} bars)")
print(f"\n[THE BIG NUMBERS YOU ASKED FOR]")
print(f"  Number of trades:    {m.get('n_trades', 0):>6}")
print(f"  Win rate:            {m.get('win_rate', 0):>6.1%}")
print(f"  Max win:             ${m.get('largest_win', 0):>8.2f}")
print(f"  Max loss:            ${m.get('largest_loss', 0):>8.2f}")
print(f"  Max drawdown:        {m.get('max_drawdown', 0):>7.2%}")

print(f"\n[ADDITIONAL METRICS]")
print(f"  Net P&L:             ${m.get('net_pnl', 0):>8.2f}")
print(f"  Total return:        {m.get('total_return', 0):>7.2%}")
print(f"  Sharpe:              {m.get('sharpe', 0):>+7.3f}")
print(f"  Sortino:             {m.get('sortino', 0):>+7.3f}")
print(f"  Calmar:              {m.get('calmar', 0):>+7.3f}")
print(f"  Profit factor:       {m.get('profit_factor', 0):>7.2f}")
print(f"  Avg win / loss:      ${m.get('avg_win', 0):>6.2f} / ${m.get('avg_loss', 0):>6.2f}")
print(f"  Expectancy:          ${m.get('expectancy', 0):>6.2f}")
print(f"  Recovery factor:     {m.get('recovery_factor', 0):>7.2f}")
print(f"  Trades per year:     {result.get('annual_trades', 0):>7.1f}")
print(f"  Final equity:        ${m.get('final_equity', 0):>8.2f}")

# Per-strategy breakdown
print(f"\n[PER-STRATEGY BREAKDOWN]")
for strat, sm in result.get("per_strategy", {}).items():
    nt = sm.get("n_trades", 0)
    pnl = sm.get("net_pnl", 0)
    wr = sm.get("win_rate", 0)
    pf = sm.get("profit_factor", 0)
    print(f"  {strat:8s}  trades={nt:>4d}  net=${pnl:>+8.2f}  WR={wr:.1%}  PF={pf:.2f}")

# Trade journal export
print("\n" + "=" * 70)
print("EXPORTING TRADE JOURNAL")
print("=" * 70)
csv_path = export_to_csv(trades, df.index, "output/multi_ea_no_grid_trades.csv")
print(f"CSV:  {csv_path}")
html_path = export_to_html(trades, df.index, "output/multi_ea_no_grid_journal.html",
                             metrics=m, strategy_name="Multi-Strategy EA (No Grid)")
print(f"HTML: {html_path}")

# Streak info
streak = result.get("streak_stats", {})
if streak:
    print(f"\n[STREAK INFO]")
    print(f"  Longest win streak:  {streak.get('longest_win_streak', 0)}")
    print(f"  Longest loss streak: {streak.get('longest_loss_streak', 0)}")
    print(f"  Total wins:          {streak.get('total_wins', 0)}")
    print(f"  Total losses:        {streak.get('total_losses', 0)}")

print("\n" + "=" * 70)
print("ALL DONE. Open output/multi_ea_no_grid_journal.html for full trade log.")
print("=" * 70)