import sys
import importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location("cache", "data/cache.py")
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)

matches = cache.list_cache(symbol='XAUUSD', timeframe='M1')
p = matches[-1]
df = pd.read_parquet(p)

from strategies.light9.light9_strategy import Light9Strategy
s = Light9Strategy(params={'profile': 'hybrid'})
sig = s.generate(df)

entries = sig.entries
if hasattr(entries, 'fillna'):
    entries = entries.fillna(False).astype(bool)
else:
    entries = pd.Series(entries, index=df.index).fillna(False).astype(bool)

exits = sig.exits
if hasattr(exits, 'fillna'):
    exits = exits.fillna(False).astype(bool)
else:
    exits = pd.Series(exits, index=df.index).fillna(False).astype(bool)

direction = sig.direction
if hasattr(direction, 'fillna'):
    direction = direction.fillna(0).astype(int)
else:
    direction = pd.Series(direction, index=df.index).fillna(0).astype(int)

print('Original exits:')
print(f'  Sum: {exits.sum()}')
print(f'  First 10 exit times: {df.index[exits][:10]}')
print(f'  Exit hour distribution: {df.index[exits].hour.value_counts().sort_index().head()}')

# Now call _generate_tp_sl_exits from backtester.engine_full
from backtester.engine_full import _generate_tp_sl_exits
signals = {'light9_hybrid': (entries, exits, direction)}
augmented = _generate_tp_sl_exits(df, signals, pip_size=0.01, tp_pips=50.0, sl_pips=30.0)
aug_entries, aug_exits, aug_direction = augmented['light9_hybrid']
print('\\nAfter _generate_tp_sl_exits:')
print(f'  Exits sum: {aug_exits.sum()}')
print(f'  First 10 exit times: {df.index[aug_exits][:10]}')
print(f'  Exit hour distribution: {df.index[aug_exits].hour.value_counts().sort_index().head()}')

# Check if they are the same object
print(f'\\nAre exits the same object? {exits is aug_exits}')
print(f'Are exits equal? {exits.equals(aug_exits)}')