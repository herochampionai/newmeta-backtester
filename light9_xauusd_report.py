import sys
sys.path.insert(0, '.')
from data.cache import load
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from strategies.light9.light9_strategy import Light9Strategy
import pandas as pd
from strategies import STRATEGY_REGISTRY

df, info = load('XAUUSD', 'M1')
print(f'Loaded {len(df)} rows (XAUUSD M1)')
print(f'Date range: {df.index[0]} to {df.index[-1]}')
days = (df.index[-1] - df.index[0]).days
print(f'Days studied: {days}')
print()

for profile_name in ['hunter', 'hybrid', 'heaven', 'original']:
    s = Light9Strategy(params={'profile': profile_name})
    sig = s.generate(df)
    signals = {'light9': (sig.entries.fillna(False).astype(bool), pd.Series(sig.exits).fillna(False).astype(bool), sig.direction.fillna(0).astype(int))}
    result = run_full(df, signals, init_cash=10000, commission_pips=0.7, slippage_pips=0.3, grid_mode=GRID_NONE)

    trades = result['trades']
    metrics = result['metrics']
    equity = result['equity']
    n_trades = len(trades)

    if n_trades > 0:
        pnl_col = None
        for col in trades.columns:
            if 'pnl' in col.lower():
                pnl_col = col
                break
        if pnl_col is None:
            pnl_col = trades.columns[0]

        winning = trades[trades[pnl_col] > 0]
        losing = trades[trades[pnl_col] < 0]
        n_win = len(winning)
        n_los = len(losing)
        win_rate = n_win / n_trades if n_trades > 0 else 0

        avg_win = winning[pnl_col].mean() if len(winning) > 0 else 0
        avg_loss = losing[pnl_col].mean() if len(losing) > 0 else 0
        max_win = trades[pnl_col].max()
        max_loss = trades[pnl_col].min()

        gross_profit = winning[pnl_col].sum() if len(winning) > 0 else 0
        gross_loss = abs(losing[pnl_col].sum()) if len(losing) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        net_pnl = equity.iloc[-1] - 10000
        max_dd = metrics.get('max_drawdown', 0)
        sharpe = metrics.get('sharpe', 0)
        avg_per_trade = net_pnl / n_trades if n_trades > 0 else 0

        if 'exit_bar' in trades.columns and 'entry_bar' in trades.columns:
            durations = trades['exit_bar'] - trades['entry_bar']
            avg_duration_bars = durations.mean()
        else:
            avg_duration_bars = 'N/A'
    else:
        n_win, n_los, win_rate = 0, 0, 0
        avg_win, avg_loss, max_win, max_loss = 0, 0, 0, 0
        profit_factor, net_pnl, sharpe, max_dd, avg_per_trade = 0, 0, 0, 0, 0
        avg_duration_bars = 'N/A'

    print(f'=== {profile_name.upper()} ===')
    print(f'  Trades: {n_trades}')
    print(f'  Winning Trades: {n_win}')
    print(f'  Losing Trades: {n_los}')
    print(f'  Win Rate: {win_rate:.1%}')
    print(f'  Avg Per Trade: {avg_per_trade:.2f}')
    print(f'  Avg Win: {avg_win:.2f}')
    print(f'  Avg Loss: {avg_loss:.2f}')
    print(f'  Max Win: {max_win:.2f}')
    print(f'  Max Loss: {max_loss:.2f}')
    print(f'  Net PnL: {net_pnl:.2f}')
    print(f'  Profit Factor: {profit_factor:.2f}')
    print(f'  Sharpe: {sharpe:.2f}')
    print(f'  Max Drawdown: {max_dd:.2%}')
    print(f'  Avg Duration (bars): {avg_duration_bars}')
    print()

# Faceoff
print('=== FACEOFF: Light9 vs existing strategies (XAUUSD M1) ===')
for name in ['ac_ao', 'adx', 'dem', 'fbb', 'bb_rsi', 'mfi', 'light9']:
    if name not in STRATEGY_REGISTRY:
        continue
    cls = STRATEGY_REGISTRY[name]
    params = {'profile': 'hybrid'} if name == 'light9' else {}
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
        print(f'  {name}: Trades={n_trades} NetPnL={net_pnl:.2f} Sharpe={sharpe:.2f} WinRate={win_rate:.1%}')
    except Exception as e:
        print(f'  {name}: ERROR - {e}')
