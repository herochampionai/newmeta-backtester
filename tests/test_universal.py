"""End-to-end test: drop PineScript, drop MQ5, drop .py, drop .txt — all should work."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import tempfile

from core.loader import load_any_strategy
from data.cache import load as load_cache
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_LOSS_AND_PROFIT
import pandas as pd

# Load real data
df, meta = load_cache("EURUSD", "H1")
print(f"Data: {len(df)} bars EURUSD H1\n")

# === Test 1: PineScript ===
print("=" * 60)
print("Test 1: PineScript v5 (RSI strategy)")
PINE_SAMPLE = '''//@version=5
strategy("RSI Mean Reversion", overlay=false, default_qty_type=strategy.percent_of_equity, default_qty_value=10)
length = input.int(14, "RSI Length")
oversold = input.float(30.0, "Oversold")
overbought = input.float(70.0, "Overbought")
rsi_val = ta.rsi(close, length)
long_condition = ta.crossover(rsi_val, oversold)
short_condition = ta.crossunder(rsi_val, overbought)
if (long_condition)
    strategy.entry("Long", strategy.long)
if (short_condition)
    strategy.entry("Short", strategy.short)
'''

with tempfile.NamedTemporaryFile(mode="w", suffix=".pine", delete=False, encoding="utf-8") as f:
    f.write(PINE_SAMPLE)
    pine_path = f.name

cls, params, info = load_any_strategy(pine_path)
print(f"  type: {info.get('type')}, suggested: {info.get('parsed_indicators')}")
print(f"  n_entry_conditions: {info.get('n_entry_conditions')}")
print(f"  params: {list(params.keys())[:5]}")
if cls is not None:
    strat = cls(params=params) if not isinstance(cls, type) or hasattr(cls, 'params') else cls()
    sig = strat.generate(df)
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    result = run_full(df, {"pine": (entries, direction)},
                       grid_mode=GRID_LOSS_AND_PROFIT,
                       grid_take_profit=50.0, grid_stop_loss=200.0,
                       pips_between_orders=20, grid_lot_multiplier=1.5,
                       max_grid_layers=3)
    m = result["metrics"]
    print(f"  Trades: {m.get('n_trades', 0)}, "
          f"Sharpe: {m['sharpe']:+.2f}, "
          f"Total: ${m.get('net_pnl', 0):+.0f}, "
          f"WR: {m.get('win_rate', 0):.1%}")

# === Test 2: MQL5 ===
print("\n" + "=" * 60)
print("Test 2: MQL5 EA (existing multi strat newmeta.mq5)")
mq5_path = Path(r"D:\Trading\TRADING\youha created EA\Multi strat ea\multi strat newmeta.mq5")
cls, params, info = load_any_strategy(mq5_path)
print(f"  type: {info.get('type')}, suggested: {info.get('suggested_strategy')}")
print(f"  inputs detected: {info.get('input_count', 0)}")
print(f"  matched strategies: {info.get('matched_strategies')}")

# === Test 3: .py (using existing strategy file) ===
print("\n" + "=" * 60)
print("Test 3: Python strategy (universal_strategy.py)")
py_path = Path("core/universal_strategy.py")
cls, params, info = load_any_strategy(py_path)
print(f"  type: {info.get('type')}")
print(f"  info: {info}")

# === Test 4: .txt keyword config ===
print("\n" + "=" * 60)
print("Test 4: .txt keyword config")
txt_sample = "fbb: open_orders_type_1=1, level_open_orders_2=80\nmfi: level_open_orders=70"
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
    f.write(txt_sample)
    txt_path = f.name
cls, params, info = load_any_strategy(txt_path)
print(f"  type: {info.get('type')}, suggested: {info.get('suggested_strategy')}")
print(f"  matched: {info.get('matched_strategies')}")

print("\n" + "=" * 60)
print("All loader tests complete.")