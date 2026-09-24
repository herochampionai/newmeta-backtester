import sys
import importlib.util
import pandas as pd

# Load data.cache module without triggering data.__init__ (which imports MetaTrader5)
spec = importlib.util.spec_from_file_location("cache", "data/cache.py")
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)

# Load XAUUSD M1 data
matches = cache.list_cache(symbol='XAUUSD', timeframe='M1')
p = matches[-1]
df = pd.read_parquet(p)

from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from strategies.light9.light9_strategy import Light9Strategy

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

signals = {'light9_hybrid': (entries, exits, direction)}
result = run_full(df, signals, init_cash=10000, commission_pips=0.7, slippage_pips=0.3, grid_mode=GRID_NONE, params=tp, pip_size=0.01)
metrics = result['metrics']
trades = result['trades']
equity = result['equity']
net_pnl = equity.iloc[-1] - 10000
print(f'Trades={len(trades)} NetPnL={net_pnl:.2f} Sharpe={metrics.get("sharpe",0):.3f} WinRate={metrics.get("win_rate",0):.1%}')
print(f'Metrics keys: {list(metrics.keys())}')
print(f'Trade columns: {list(trades.columns)}')
if not trades.empty:
    pnl_col = None
    for col in trades.columns:
        if col.lower() == 'pnl':
            pnl_col = col
            break
    if pnl_col:
        pnl_series = trades[pnl_col]
        wins = (pnl_series > 0).sum()
        losses = (pnl_series < 0).sum()
        zeros = (pnl_series == 0).sum()
        print(f'Wins: {wins}, Losses: {losses}, Zeros: {zeros}')
        print(f'Total trades: {len(pnl_series)}')
        print(f'Win rate (wins/total): {wins/len(pnl_series):.2%}')
        print(f'Win rate (wins/(wins+losses)): {wins/(wins+losses):.2%}')
        print(f'PnL sum: {pnl_series.sum():.2f}')