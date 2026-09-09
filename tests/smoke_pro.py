"""Smoke test: MTF Quad Stochastic + confluence + adaptive sizing + swaps."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

from data.cache import load as load_cache
from strategies import STRATEGY_REGISTRY
from strategies._confluence import confluence
from strategies.mtf_stoch import QuadStochStrategy, resample_htf
from backtester.adaptive import AdaptiveSizer, AdaptiveConfig
from backtester.swaps import compute_swap_series, apply_weekend_filter
from backtester.analytics import strategy_scoreboard, streak_stats, annual_trade_count

print("=== Loading real EURUSD H1 data ===")
df, meta = load_cache("EURUSD", "H1")
print(f"  {len(df)} bars, {df.index[0]} to {df.index[-1]}")

print("\n=== Quad Stochastic (MTF) on real data ===")
# Simulate MTF: use H1 as entry, resample to H4, D1 for HTF context
# (Note: in production you'd want actual 5-min data for sniper)
df_mtf = resample_htf(df, htf_rules=["4h", "1d", "1d", "1d"],
                       quad_params=[(5, 3, 3), (14, 3, 3), (40, 4, 6), (60, 6, 10)])
sig = QuadStochStrategy(params=dict(
    quad_params=[(5, 3, 3), (14, 3, 3), (40, 4, 6), (60, 6, 10)],
    oversold=20, overbought=80, divergence_lookback=30,
)).generate(df_mtf)
print(f"  Entries: {sig.entries.sum()}, Long: {(sig.direction == 1).sum()}, Short: {(sig.direction == -1).sum()}")

print("\n=== Confluence (6 strategies on real data) ===")
directions = {}
for name, cls in STRATEGY_REGISTRY.items():
    if name in ("mtf_stoch", "regime_aware", "universal"):
        continue
    if name == "ms":
        params = dict(open_orders_type_1=8, open_orders_type_2=0,
                      level_open_orders_1=20, level_open_orders_2=80,
                      use_confluence_filter=False)
    elif name == "fbb":
        params = dict(open_orders_type_1=1, open_orders_type_2=0,
                      level_open_orders_1=0, level_open_orders_2=50,
                      bars_calculate=20, deviation=1.8)
    elif name == "mfi":
        params = dict(open_orders_type=3, level_open_orders=70,
                      level_close_orders=70, bars_calculate=12)
    elif name == "dem":
        params = dict(open_orders_type=3, level_open_orders=75, bars_calculate=20)
    elif name == "adx":
        params = dict(open_orders_type=1, level_open_orders_1=55,
                      level_open_orders_2=15, use_di_crossover=True)
    else:
        params = dict(open_orders_type=1, level_open_orders=80)
    s = cls(params=params)
    sig_s = s.generate(df)
    directions[name] = sig_s.direction

comp, conf, n_agree = confluence(directions, min_agreement=2)
print(f"  Composite signals: long={(comp == 1).sum()}, short={(comp == -1).sum()}, none={(comp == 0).sum()}")
print(f"  Mean confidence when signal fires: {conf[comp != 0].mean():.2f}")
print(f"  Max strategies agreeing: {n_agree.max()}")

print("\n=== Adaptive position sizing (simulated) ===")
sizer = AdaptiveSizer(AdaptiveConfig(base_lot=0.1,
                                       cautious_after=3, hot_after=5,
                                       cautious_multiplier=0.5, hot_multiplier=1.2,
                                       pause_dd_pct=0.10))
# Simulate 20 trades: alternating wins and losses
simulated_pnls = [50, -30, -40, 60, -25, -50, -55, 70, -80, -90,
                   -100, 80, 90, 100, 110, -150, -180, 120, 130, 140]
print(f"  Trade # | Result | Consec | Lot | State")
for i, pnl in enumerate(simulated_pnls, 1):
    sizer.on_trade_close(pnl)
    st = sizer.get_state()
    print(f"  {i:>7d} | {'W' if pnl > 0 else 'L':>6s} | "
          f"W:{st['consecutive_wins']} L:{st['consecutive_losses']} | "
          f"{st['current_lot']:.3f} | {'PAUSED' if st['paused'] else 'active'}")

print("\n=== Swap awareness ===")
swap = compute_swap_series(df.iloc[:1000], long_swap_pips=-0.5, short_swap_pips=0.2,
                            contract_size=100_000)
print(f"  Total swap over 1000 H1 bars: ${swap.sum():.2f}")
print(f"  Wed swaps: {(swap != 0).sum()} bars affected")

print("\n=== Trade-frequency scoreboard ===")
# Mock metrics for each strategy (would come from real backtest)
mock = {
    "ac_ao": dict(sharpe=0.27, calmar=0.21, win_rate=0.55, profit_factor=1.4,
                   net_pnl=696, n_trades=70, max_drawdown=-0.10, expectancy=10),
    "adx":   dict(sharpe=0.0, calmar=0.0, win_rate=0.0, profit_factor=0,
                   net_pnl=0, n_trades=0, max_drawdown=0, expectancy=0),
    "dem":   dict(sharpe=0.31, calmar=0.26, win_rate=0.58, profit_factor=1.5,
                   net_pnl=799, n_trades=196, max_drawdown=-0.09, expectancy=4),
    "fbb":   dict(sharpe=-0.52, calmar=-0.22, win_rate=0.50, profit_factor=0.92,
                   net_pnl=-1250, n_trades=894, max_drawdown=-0.19, expectancy=-1.4),
    "mfi":   dict(sharpe=-1.35, calmar=-0.31, win_rate=0.40, profit_factor=0.78,
                   net_pnl=-3064, n_trades=643, max_drawdown=-0.36, expectancy=-4.8),
    "ms":    dict(sharpe=0.0, calmar=0.0, win_rate=0.0, profit_factor=0,
                   net_pnl=0, n_trades=0, max_drawdown=0, expectancy=0),
    "mtf_stoch": dict(sharpe=0.45, calmar=0.38, win_rate=0.62, profit_factor=1.8,
                       net_pnl=1200, n_trades=120, max_drawdown=-0.08, expectancy=10),
}
sb = strategy_scoreboard(mock, rank_by="sharpe")
print("\n  Strategy scoreboard (ranked by Sharpe):")
print(sb.to_string(index=False))
print("\n  Lowest 3 by Sharpe:")
print(sb.nsmallest(3, "sharpe").to_string(index=False))
print(f"\n  Trades per year (n_bars={len(df)}, periods_per_year=252*24=6048):")
for strat, n_tr in [("ac_ao", 70), ("dem", 196), ("fbb", 894), ("mfi", 643)]:
    tpy = annual_trade_count(n_tr, len(df), 252 * 24)
    print(f"    {strat:8s}: {tpy:.1f} trades/year")