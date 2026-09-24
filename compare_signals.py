import sys
import importlib.util
import pandas as pd
from strategies.light9.light9_strategy import Light9Strategy

spec = importlib.util.spec_from_file_location("cache", "data/cache.py")
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)

matches = cache.list_cache(symbol='XAUUSD', timeframe='M1')
p = matches[-1]
df = pd.read_parquet(p)

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

print('Full data:')
print(f'  Entries: {entries.sum()}')
print(f'  Exits: {exits.sum()}')
print(f'  Direction non-zero: {(direction != 0).sum()}')

# Check exit hours
exit_hours = df.index[exits].hour
print('  Exit hour distribution:')
print(exit_hours.value_counts().sort_index())

# Now load only first month
mask = df.index <= '2024-01-31'
df_sub = df[mask]
print('\\nFirst month:')
print(f'  Bars: {len(df_sub)}')
s_sub = Light9Strategy(params={'profile': 'hybrid'})
sig_sub = s_sub.generate(df_sub)
entries_sub = sig_sub.entries
if hasattr(entries_sub, 'fillna'):
    entries_sub = entries_sub.fillna(False).astype(bool)
else:
    entries_sub = pd.Series(entries_sub, index=df_sub.index).fillna(False).astype(bool)
exits_sub = sig_sub.exits
if hasattr(exits_sub, 'fillna'):
    exits_sub = exits_sub.fillna(False).astype(bool)
else:
    exits_sub = pd.Series(exits_sub, index=df_sub.index).fillna(False).astype(bool)
direction_sub = sig_sub.direction
if hasattr(direction_sub, 'fillna'):
    direction_sub = direction_sub.fillna(0).astype(int)
else:
    direction_sub = pd.Series(direction_sub, index=df_sub.index).fillna(0).astype(int)
print(f'  Entries: {entries_sub.sum()}')
print(f'  Exits: {exits_sub.sum()}')
print(f'  Direction non-zero: {(direction_sub != 0).sum()}')
exit_hours_sub = df_sub.index[exits_sub].hour
print('  Exit hour distribution:')
print(exit_hours_sub.value_counts().sort_index())