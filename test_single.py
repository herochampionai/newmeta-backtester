import sys
sys.path.insert(0, '.')
from data.cache import load
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from strategies.light9.light9_strategy import Light9Strategy
import pandas as pd

df, info = load('XAUUSD', 'M1')

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
if not trades.empty and 'pnl' in trades.columns and len(trades) > 0:
    print(f'Avg PnL: {trades["pnl"].mean():.2f}')
    print(f'Wins: {(trades["pnl"] > 0).sum()}, Losses: {(trades["pnl"] < 0).sum()}')
