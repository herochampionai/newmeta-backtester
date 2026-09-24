import sys
sys.path.insert(0, '.')
from data.cache import load
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from strategies.light9.light9_strategy import Light9Strategy

df, info = load('EURUSD', 'H1')
print(f'Loaded {len(df)} rows')

for profile_name in ['hunter', 'hybrid', 'heaven', 'original']:
    s = Light9Strategy(params={'profile': profile_name})
    sig = s.generate(df)
    signals = {"light9": (sig.entries.fillna(False).astype(bool), sig.exits.fillna(False).astype(bool), sig.direction.fillna(0).astype(int))}
    result = run_full(df, signals, init_cash=10000, commission_pips=0.7, slippage_pips=0.3, grid_mode=GRID_NONE)
    metrics = result['metrics']
    tp = s.get_trailing_params()
    trades = result['trades']
    equity = result['equity']
    net_pnl = equity.iloc[-1] - 10000
    print(f'{profile_name}: Trades={len(trades)} NetPnL={net_pnl:.2f} TP%={tp["tp_pct"]} SL%={tp["sl_pct"]} Trail={tp["trail_pips"]} MaxDD={metrics.get("max_drawdown", "N/A")} Sharpe={metrics.get("sharpe", "N/A")}')
