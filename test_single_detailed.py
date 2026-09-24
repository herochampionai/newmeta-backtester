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

signals = {'light9_hybrid': (entries, exits, direction)}
result = run_full(df, signals, init_cash=10000, commission_pips=0.7, slippage_pips=0.3, grid_mode=GRID_NONE, params=tp, pip_size=0.01)
metrics = result['metrics']
trades = result['trades']
equity = result['equity']
net_pnl = equity.iloc[-1] - 10000
print(f'Trades={len(trades)} NetPnL={net_pnl:.2f} Sharpe={metrics.get("sharpe",0):.3f} WinRate={metrics.get("win_rate",0):.1%}')
if not trades.empty:
    print('\\nTrade columns:', trades.columns.tolist())
    if 'pnl' in trades.columns:
        print('\\nFirst 10 trades PnL:')
        print(trades[['pnl']].head(10).to_string())
        print('\\nPnL stats:')
        print(f'  Mean: {trades["pnl"].mean():.2f}')
        print(f'  Median: {trades["pnl"].median():.2f}')
        print(f'  Std: {trades["pnl"].std():.2f}')
        print(f'  Min: {trades["pnl"].min():.2f}')
        print(f'  Max: {trades["pnl"].max():.2f}')
        print(f'  Zero PnL count: {(trades["pnl"] == 0).sum()}')
        print(f'  Positive PnL count: {(trades["pnl"] > 0).sum()}')
        print(f'  Negative PnL count: {(trades["pnl"] < 0).sum()}')
    else:
        print('\\nNo pnl column found')
        # Check for other PnL-like columns
        pnl_cols = [c for c in trades.columns if 'pnl' in c.lower() or 'profit' in c.lower()]
        if pnl_cols:
            print('Found PnL-like columns:', pnl_cols)
            for col in pnl_cols:
                print(f'  {col}: {trades[col].head()}')
else:
    print('No trades')