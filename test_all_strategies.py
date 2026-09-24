import sys
sys.path.insert(0, '.')
from data.cache import load
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from strategies import STRATEGY_REGISTRY
import pandas as pd
import numpy as np

df, info = load('XAUUSD', 'M1')
print(f'Loaded {len(df)} rows')
print()

for name in ['mfi', 'bb_rsi', 'ac_ao', 'adx', 'dem', 'fbb', 'stoch533_mtf']:
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        continue
    try:
        s = cls()
        sig = s.generate(df)
        entries = sig.entries
        if hasattr(entries, 'fillna'):
            entries = entries.fillna(False).astype(bool)
        else:
            entries = pd.Series(entries, index=df.index).fillna(False).astype(bool)

        direction = sig.direction
        if hasattr(direction, 'fillna'):
            direction = direction.fillna(0).astype(int)
        else:
            direction = pd.Series(direction, index=df.index).fillna(0).astype(int)

        if hasattr(sig, 'exits') and sig.exits is not None:
            exits = sig.exits
            if hasattr(exits, 'fillna'):
                exits = exits.fillna(False).astype(bool)
            else:
                exits = pd.Series(exits, index=df.index).fillna(False).astype(bool)
            signals = {name: (entries, exits, direction)}
        else:
            signals = {name: (entries, direction)}

        result = run_full(df, signals, init_cash=10000, commission_pips=0.7, slippage_pips=0.3, grid_mode=GRID_NONE)
        metrics = result['metrics']
        trades = result['trades']
        equity = result['equity']
        net_pnl = equity.iloc[-1] - 10000
        n_trades = len(trades)
        sharpe = metrics.get('sharpe', 0)
        win_rate = metrics.get('win_rate', 0)
        max_dd = metrics.get('max_drawdown', 0)
        print(f'{name:15s}: Trades={n_trades:5d} NetPnL={net_pnl:8.2f} Sharpe={sharpe:.3f} WinRate={win_rate:.1%} MaxDD={max_dd:.2%}')
    except Exception as e:
        print(f'{name:15s}: ERROR - {e}')
