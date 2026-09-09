"""Smoke test: live fetcher + metrics on real data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.live_fetcher import fetch_with_priority, load_settings
from data.mt5_export import resolve_terminal
from backtester.metrics_v2 import compute_all

print("=== Settings ===")
s = load_settings()
print(f"  mt5_terminal: {s.get('mt5_terminal')}")
print(f"  resolved:     {resolve_terminal()}")

print("\n=== Live fetcher (priority chain) ===")
df, info = fetch_with_priority("EURUSD", "H1", "2024-01-01", "2024-12-31")
print(f"source: {info['source']}, rows: {len(df)}")
print(f"range:  {df.index[0]} -> {df.index[-1]}")
for i, c in enumerate(info.get("chain", [])):
    print(f"  {i+1}. {c}")

print("\n=== Backtest FBB on real data ===")
from strategies import STRATEGY_REGISTRY
from backtester.engine import run_direction

strat = STRATEGY_REGISTRY["fbb"](params=dict(
    open_orders_type_1=1, open_orders_type_2=0, close_orders_type_1=0,
    close_orders_type_2=0, level_open_orders_1=0, level_open_orders_2=50,
    level_close_orders_1=40, level_close_orders_2=40,
    bars_calculate=20, deviation=1.8))
sig = strat.generate(df)
pf, summary = run_direction(df, sig.entries, sig.direction)
returns = pf.returns().dropna()
equity = (1 + returns).cumprod() * 10000

# Use the comprehensive metric set
try:
    trades_df = pf.trades.records_readable
except Exception:
    trades_df = None

m = compute_all(returns, trades_df, equity)
print(f"  {len(m)} metrics computed")
print(f"  Total return:    {m['total_return']:+.2%}")
print(f"  Sharpe:          {m['sharpe']:+.2f}")
print(f"  Sortino:         {m['sortino']:+.2f}")
print(f"  Calmar:          {m['calmar']:+.2f}")
print(f"  Max DD:          {m['max_drawdown']:.2%}")
print(f"  Stability (R²):  {m['stability']:+.2f}")
print(f"  Recovery factor: {m['recovery_factor']:+.2f}")
print(f"  N trades:        {m.get('n_trades', 'n/a')}")
print(f"  Win rate:        {m.get('win_rate', 0):.1%}")
print(f"  Avg win/loss:    \${m.get('avg_win', 0):.2f} / \${m.get('avg_loss', 0):.2f}")
print(f"  Profit factor:   {m.get('profit_factor', 0):.2f}")
print(f"  Expectancy:      \${m.get('expectancy', 0):.2f}")