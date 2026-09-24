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

from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
signals = {'light9_hybrid': (entries, exits, direction)}
result = run_full(df, signals, init_cash=10000, commission_pips=0.7, slippage_pips=0.3, grid_mode=GRID_NONE, params=s.get_trailing_params(), pip_size=0.01)
trades = result['trades']

if not trades.empty:
    # Ensure we have PnL column
    pnl_col = 'PnL' if 'PnL' in trades.columns else None
    if pnl_col is None:
        for col in trades.columns:
            if 'pnl' in col.lower():
                pnl_col = col
                break
    if pnl_col is not None:
        pnl = trades[pnl_col]
        print(f'Total PnL: {pnl.sum():.2f}')
        print(f'Number of trades: {len(pnl)}')
        wins = pnl > 0
        losses = pnl < 0
        zeros = pnl == 0
        print(f'Wins: {wins.sum()}, Losses: {losses.sum()}, Zeros: {zeros.sum()}')
        print(f'Win rate: {wins.sum()/len(pnl):.2%}')
        print(f'Average win: {pnl[wins].mean():.2f}')
        print(f'Average loss: {pnl[losses].mean():.2f}')
        print(f'Largest win: {pnl[wins].max():.2f}')
        print(f'Largest loss: {pnl[losses].min():.2f}')
        print(f'Profit factor: {pnl[wins].sum() / abs(pnl[losses].sum()):.2f}')
        print()
        print('Losing trades:')
        losing_trades = trades[losses]
        print(losing_trades[['Entry Timestamp', 'Exit Timestamp', pnl_col]].to_string())
    else:
        print('No PnL column found')
else:
    print('No trades')