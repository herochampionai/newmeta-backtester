"""Final regime engine search — best of both PnL and Sharpe objectives."""
import warnings; warnings.filterwarnings('ignore')
import json
import sys

sys.path.insert(0, '.')
from datetime import datetime, timezone

import MetaTrader5 as mt5
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import run_strategy
from strategies.top5_research import RegimeSwitchingEngineStrategy

eur = pd.read_csv('output/mt5_EURUSD_H1_2022_2026.csv', index_col='time', parse_dates=True)
if eur.index.tz is None:
    eur.index = eur.index.tz_localize('UTC')
eur = eur[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
eur_t = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
eur_v = eur[(eur.index >= pd.Timestamp('2022-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2024-09-17', tz='UTC'))]

# Extended 3Y data
mt5.initialize(path='D:\\MT5_Bybit\\terminal64.exe')
rates = mt5.copy_rates_range('EURUSD', mt5.TIMEFRAME_H1,
                              datetime(2021, 1, 1, tzinfo=timezone.utc),
                              datetime(2024, 1, 1, tzinfo=timezone.utc))
eur_3y = pd.DataFrame(rates)
eur_3y['time'] = pd.to_datetime(eur_3y['time'], unit='s', utc=True)
eur_3y.set_index('time', inplace=True)
eur_3y = eur_3y[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
mt5.shutdown()


def obj_combined(trial, df_t, df_v):
    """Maximize combined objective: min PnL + Sharpe + log(trades)."""
    params = {
        'adx_period': trial.suggest_int('adx_period', 8, 30),
        'atr_period': trial.suggest_int('atr_period', 8, 30),
        'bb_period': trial.suggest_int('bb_period', 10, 50),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.5),
        'trend_thresh': trial.suggest_float('trend_thresh', 18.0, 40.0),
        'range_thresh': trial.suggest_float('range_thresh', 8.0, 25.0),
        'expansion_mult': trial.suggest_float('expansion_mult', 1.1, 3.0),
        'atr_avg_period': trial.suggest_int('atr_avg_period', 10, 150),
        'sma_period': trial.suggest_int('sma_period', 50, 300),
        'cooldown': trial.suggest_int('cooldown', 1, 50),
    }
    m_t = run_strategy(df_t, RegimeSwitchingEngineStrategy, params, [], 'forex')
    m_v = run_strategy(df_v, RegimeSwitchingEngineStrategy, params, [], 'forex')
    if 'error' in m_t or 'error' in m_v: return -1e9
    if m_t['n_trades'] < 30 or m_v['n_trades'] < 30: return -1e9
    min_pnl = min(m_t['net_pnl'], m_v['net_pnl'])
    avg_sharpe = (m_t['sharpe'] + m_v['sharpe']) / 2
    import math
    log_trades = math.log(m_t['n_trades'] + m_v['n_trades'])
    return min_pnl + 50 * avg_sharpe + 100 * log_trades


# Run 1000 trials with combined objective
print('=' * 100)
print('Optuna Regime Engine — 1000 trials with COMBINED objective')
print('=' * 100)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=99))
study.optimize(lambda t: obj_combined(t, eur_t, eur_v), n_trials=1000, show_progress_bar=False)
best = study.best_params
m_t = run_strategy(eur_t, RegimeSwitchingEngineStrategy, best, [], 'forex')
m_v = run_strategy(eur_v, RegimeSwitchingEngineStrategy, best, [], 'forex')

# Verify on 3Y
half = len(eur_3y) // 2
m_3y1 = run_strategy(eur_3y.iloc[:half], RegimeSwitchingEngineStrategy, best, [], 'forex')
m_3y2 = run_strategy(eur_3y.iloc[half:], RegimeSwitchingEngineStrategy, best, [], 'forex')

print('\n=== Best Regime Engine v3 (Combined objective) ===')
print(f'Tuning (2024-26): ${m_t["net_pnl"]:+,.0f}/Sh {m_t["sharpe"]:+.2f}/{m_t["n_trades"]}t')
print(f'Val (2022-24):    ${m_v["net_pnl"]:+,.0f}/Sh {m_v["sharpe"]:+.2f}/{m_v["n_trades"]}t')
print(f'3Y-H1 (2021-22):  ${m_3y1["net_pnl"]:+,.0f}/Sh {m_3y1["sharpe"]:+.2f}/{m_3y1["n_trades"]}t')
print(f'3Y-H2 (2022-24):  ${m_3y2["net_pnl"]:+,.0f}/Sh {m_3y2["sharpe"]:+.2f}/{m_3y2["n_trades"]}t')
print(f'3Y Total:         ${m_3y1["net_pnl"]+m_3y2["net_pnl"]:+,.0f}')
robust = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
print(f'ROBUST (2Y): {robust}')
print(f'Params: {best}')

with open('output/regime_engine_v3.json', 'w') as f:
    json.dump({
        'name': 'Regime Engine v3 (Combined objective: PnL+Sharpe+trades)',
        'params': best,
        'tuning_pnl': m_t['net_pnl'], 'tuning_sharpe': m_t['sharpe'], 'tuning_trades': m_t['n_trades'],
        'val_pnl': m_v['net_pnl'], 'val_sharpe': m_v['sharpe'], 'val_trades': m_v['n_trades'],
        '3y_total_pnl': m_3y1['net_pnl']+m_3y2['net_pnl'],
        'robust_2y': robust,
    }, f, indent=2, default=str)
print('\nSaved → output/regime_engine_v3.json')


# Final summary
print()
print('=' * 100)
print('SUMMARY — Regime Engine versions')
print('=' * 100)
print(f'{"Version":20s} | {"T PnL":>8s} | {"V PnL":>7s} | {"3Y PnL":>8s} | {"T Trades":>8s} | {"V Trades":>8s} | ROBUST')
print('-' * 100)
v1 = {'T': 572, 'V': 472, '3Y': 1038, 'TT': 21, 'VT': 21, 'Robust': True}
v2 = {'T': 809, 'V': 757, '3Y': 660, 'TT': 132, 'VT': 142, 'Robust': True}
v3 = {'T': int(m_t['net_pnl']), 'V': int(m_v['net_pnl']), '3Y': int(m_3y1['net_pnl']+m_3y2['net_pnl']),
       'TT': m_t['n_trades'], 'VT': m_v['n_trades'], 'Robust': robust}
for label, v in [('v1 (orig)', v1), ('v2 (PnL opt)', v2), ('v3 (combined)', v3)]:
    print(f'{label:20s} | {v["T"]:>8} | {v["V"]:>7} | {v["3Y"]:>8} | {v["TT"]:>8} | {v["VT"]:>8} | {v["Robust"]}')
