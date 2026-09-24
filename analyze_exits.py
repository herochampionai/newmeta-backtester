import sys
import importlib.util
import pandas as pd
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from strategies.light9.light9_strategy import Light9Strategy

# Load data.cache module without triggering data.__init__ (which imports MetaTrader5)
spec = importlib.util.spec_from_file_location("cache", "data/cache.py")
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)

# Load XAUUSD M1 data
matches = cache.list_cache(symbol='XAUUSD', timeframe='M1')
if not matches:
    raise FileNotFoundError("no cache for XAUUSD M1")
p = matches[-1]  # newest by name (sha sorts)
df = pd.read_parquet(p)

s = Light9Strategy(params={'profile': 'hybrid'})
sig = s.generate(df)
tp = s.get_trailing_params()

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

print('Signal statistics:')
print(f'  Entries: {entries.sum()}')
print(f'  Exits: {exits.sum()}')
print(f'  Direction non-zero: {(direction != 0).sum()}')

# Check exit hours
exit_hours = df.index[exits].hour
print('\\nExit hour distribution:')
print(exit_hours.value_counts().sort_index())

# Run the backtest
signals = {'light9_hybrid': (entries, exits, direction)}
result = run_full(df, signals, init_cash=10000, commission_pips=0.7, slippage_pips=0.3, grid_mode=GRID_NONE, params=tp, pip_size=0.01)
trades = result['trades']
print(f'\\nNumber of trades: {len(trades)}')

if not trades.empty:
    # Convert timestamps to datetime if they are not already
    if 'Entry Timestamp' in trades.columns:
        trades['Entry Timestamp'] = pd.to_datetime(trades['Entry Timestamp'])
    if 'Exit Timestamp' in trades.columns:
        trades['Exit Timestamp'] = pd.to_datetime(trades['Exit Timestamp'])
    
    # Compute duration in hours
    trades['duration_hours'] = (trades['Exit Timestamp'] - trades['Entry Timestamp']).dt.total_seconds() / 3600
    
    print('\\nFirst 10 trades:')
    print(trades[['Entry Timestamp', 'Exit Timestamp', 'duration_hours', 'PnL']].head(10))
    
    print('\\nDuration statistics (hours):')
    print(trades['duration_hours'].describe())
    
    print('\\nExit hour distribution from trades:')
    exit_hours_from_trades = trades['Exit Timestamp'].dt.hour
    print(exit_hours_from_trades.value_counts().sort_index())
    
    # Check if exits are only at session boundaries (0, 7, 13) and within first 5 minutes
    # We'll check the minute of exit
    exits_minute = trades['Exit Timestamp'].dt.minute
    print('\\nExit minute distribution:')
    print(exits_minute.value_counts().sort_index().head(10))
    
    # Check if exit hour is in [0,7,13] and minute < 5
    valid_exit = ((trades['Exit Timestamp'].dt.hour.isin([0,7,13])) & (trades['Exit Timestamp'].dt.minute < 5))
    print(f'\\nNumber of exits at session boundaries (hour in [0,7,13] and minute < 5): {valid_exit.sum()} out of {len(trades)}')
    if not valid_exit.all():
        print('Some exits are not at session boundaries:')
        print(trades[~valid_exit][['Exit Timestamp', 'PnL']].head())
else:
    print('No trades')