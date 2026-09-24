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
metrics = result['metrics']
trades = result['trades']
equity = result['equity']

print('=== BACKTEST RESULTS ===')
print(f'Initial capital: $10,000')
print(f'Final equity: ${equity.iloc[-1]:.2f}')
print(f'Net P&L: ${equity.iloc[-1] - 10000:.2f}')
print(f'Total return: {(equity.iloc[-1]/10000 - 1)*100:.2f}%')
print()
print('=== TRADE STATISTICS ===')
print(f'Number of trades: {len(trades)}')
if 'PnL' in trades.columns:
    pnl = trades['PnL']
    print(f'Total P&L: ${pnl.sum():.2f}')
    print(f'Average P&L per trade: ${pnl.mean():.2f}')
    print(f'Median P&L per trade: ${pnl.median():.2f}')
    print(f'Standard deviation: ${pnl.std():.2f}')
    print(f'Best trade: ${pnl.max():.2f}')
    print(f'Worst trade: ${pnl.min():.2f}')
    wins = pnl > 0
    losses = pnl < 0
    print(f'Winning trades: {wins.sum()}')
    print(f'Losing trades: {losses.sum()}')
    if wins.sum() + losses.sum() > 0:
        win_rate = wins.sum() / (wins.sum() + losses.sum())
        print(f'Win rate (win/loss): {win_rate:.2%}')
    print(f'Win rate (win/total): {wins.sum()/len(pnl):.2%}')
    print()
    print('=== METRICS FROM ENGINE ===')
    print(f'Sharpe ratio: {metrics.get("sharpe", 0):.3f}')
    print(f'Win rate: {metrics.get("win_rate", 0):.1%}')
    print(f'Max drawdown: {metrics.get("max_drawdown", 0):.2%}')
    print()
    print('=== LOSS ANALYSIS ===')
    if losses.sum() > 0:
        losing_pnl = pnl[losses]
        print(f'Number of losing trades: {losses.sum()}')
        print(f'Average loss: ${losing_pnl.mean():.2f}')
        print(f'Median loss: ${losing_pnl.median():.2f}')
        print(f'Largest loss: ${losing_pnl.min():.2f}')
        print(f'Smallest loss: ${losing_pnl[losing_pnl < 0].max():.2f}')
        print()
        print('Losing trades detail:')
        losing_trades = trades[losses][['Entry Timestamp', 'Exit Timestamp', 'PnL']].copy()
        losing_trades['Duration'] = (losing_trades['Exit Timestamp'] - losing_trades['Entry Timestamp']).dt.total_seconds() / 3600
        print(losing_trades.to_string(index=False))
    else:
        print('No losing trades')
else:
    print('No PnL column in trades')