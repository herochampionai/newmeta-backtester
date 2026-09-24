import sys
sys.path.insert(0, '.')
from data.cache import load
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from strategies import STRATEGY_REGISTRY

df, info = load('XAUUSD', 'M1')
print(f'Loaded {len(df)} rows (XAUUSD M1)')

strategies_to_test = ['ac_ao', 'adx', 'dem', 'fbb', 'bb_rsi', 'mfi', 'light9']

results = {}
for name in strategies_to_test:
    if name not in STRATEGY_REGISTRY:
        print(f'{name}: NOT REGISTERED')
        continue
    cls = STRATEGY_REGISTRY[name]
    params = {'profile': 'original'} if name == 'light9' else {}
    try:
        s = cls(params=params)
        sig = s.generate(df)
        entries = sig.entries
        if hasattr(entries, 'fillna'):
            entries = entries.fillna(False).astype(bool)
        direction = sig.direction
        if hasattr(direction, 'fillna'):
            direction = direction.fillna(0).astype(int)
        else:
            direction = __import__('pandas').Series(direction, index=df.index).fillna(0).astype(int)
        if hasattr(sig, 'exits') and sig.exits is not None:
            exits = sig.exits
            if hasattr(exits, 'fillna'):
                exits = exits.fillna(False).astype(bool)
            else:
                exits = __import__('pandas').Series(exits, index=df.index).fillna(False).astype(bool)
            signals = {name: (entries, exits, direction)}
        else:
            signals = {name: (entries, direction)}
        result = run_full(df, signals, init_cash=10000, commission_pips=0.7, slippage_pips=0.3, grid_mode=GRID_NONE)
        metrics = result['metrics']
        equity = result['equity']
        net_pnl = equity.iloc[-1] - 10000
        n_trades = len(result['trades'])
        sharpe = metrics.get('sharpe', 0)
        max_dd = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        results[name] = {
            'trades': n_trades,
            'net_pnl': round(net_pnl, 2),
            'sharpe': round(sharpe, 2),
            'max_dd': round(max_dd, 4),
            'win_rate': round(win_rate, 4),
        }
        print(f'{name}: Trades={n_trades} NetPnL={net_pnl:.2f} Sharpe={sharpe:.2f} MaxDD={max_dd:.2%} WinRate={win_rate:.2%}')
    except Exception as e:
        print(f'{name}: ERROR - {e}')

# Sort by Sharpe
print('\n=== Sorted by Sharpe ===')
sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
for name, r in sorted_results:
    print(f'{name}: Sharpe={r["sharpe"]} NetPnL={r["net_pnl"]} Trades={r["trades"]}')
