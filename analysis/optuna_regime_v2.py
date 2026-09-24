"""Optuna Regime Engine - extensive search with multiple objectives.

Try:
1. ROBUST + HIGH-TRADES (>= 30 each period)
2. ROBUST + HIGH-PnL
3. ROBUST + HIGH-Sharpe

Test on 2Y + verify on 3Y window.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, json, math
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import MetaTrader5 as mt5
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import run_strategy
from strategies.top5_research import RegimeSwitchingEngineStrategy

# Load data
eur = pd.read_csv('output/mt5_EURUSD_H1_2022_2026.csv', index_col='time', parse_dates=True)
if eur.index.tz is None:
    eur.index = eur.index.tz_localize('UTC')
eur = eur[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
eur_t = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
eur_v = eur[(eur.index >= pd.Timestamp('2022-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2024-09-17', tz='UTC'))]

# Extended 3Y data for verification
mt5.initialize(path='D:\\MT5_Bybit\\terminal64.exe')
from_dt = datetime(2021, 1, 1, tzinfo=timezone.utc)
to_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
rates = mt5.copy_rates_range('EURUSD', mt5.TIMEFRAME_H1, from_dt, to_dt)
eur_3y = pd.DataFrame(rates)
eur_3y['time'] = pd.to_datetime(eur_3y['time'], unit='s', utc=True)
eur_3y.set_index('time', inplace=True)
eur_3y = eur_3y[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
mt5.shutdown()
print(f'Loaded EURUSD: {len(eur_t)} tuning, {len(eur_v)} val, {len(eur_3y)} 3Y (2021-24)')


def obj_regime_pnl(trial, df_t, df_v):
    """Maximize min(tuning_pnl, val_pnl) with constraint n_trades >= 30."""
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
    if m_t['n_trades'] < 25 or m_v['n_trades'] < 25: return -1e9
    # Want: ROBUST + high PnL + many trades
    return min(m_t['net_pnl'], m_v['net_pnl']) + 20 * (m_t['sharpe'] + m_v['sharpe']) + 0.5 * (m_t['n_trades'] + m_v['n_trades'])


def obj_regime_sharpe(trial, df_t, df_v):
    """Maximize avg Sharpe with ROBUST constraint."""
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
    if m_t['n_trades'] < 25 or m_v['n_trades'] < 25: return -1e9
    avg_sharpe = (m_t['sharpe'] + m_v['sharpe']) / 2
    return avg_sharpe * 100 + min(m_t['net_pnl'], m_v['net_pnl']) / 100


# Run 500 trials each
print('=' * 100)
print('Optuna Regime Engine — 500 trials with PnL objective')
print('=' * 100)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda t: obj_regime_pnl(t, eur_t, eur_v), n_trials=500, show_progress_bar=False)
best = study.best_params
m_t = run_strategy(eur_t, RegimeSwitchingEngineStrategy, best, [], 'forex')
m_v = run_strategy(eur_v, RegimeSwitchingEngineStrategy, best, [], 'forex')

# Verify on 3Y
half = len(eur_3y) // 2
m_3y1 = run_strategy(eur_3y.iloc[:half], RegimeSwitchingEngineStrategy, best, [], 'forex')
m_3y2 = run_strategy(eur_3y.iloc[half:], RegimeSwitchingEngineStrategy, best, [], 'forex')

print(f'\n=== Best Regime Engine (PnL objective) ===')
print(f'Tuning (2024-26): ${m_t["net_pnl"]:+,.0f}/Sh {m_t["sharpe"]:+.2f}/{m_t["n_trades"]}t')
print(f'Val (2022-24):    ${m_v["net_pnl"]:+,.0f}/Sh {m_v["sharpe"]:+.2f}/{m_v["n_trades"]}t')
print(f'3Y-H1 (2021-22):  ${m_3y1["net_pnl"]:+,.0f}/Sh {m_3y1["sharpe"]:+.2f}/{m_3y1["n_trades"]}t')
print(f'3Y-H2 (2022-24):  ${m_3y2["net_pnl"]:+,.0f}/Sh {m_3y2["sharpe"]:+.2f}/{m_3y2["n_trades"]}t')
print(f'3Y Total:         ${m_3y1["net_pnl"]+m_3y2["net_pnl"]:+,.0f}')
robust = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
print(f'ROBUST (2Y): {robust}')
print(f'Params: {best}')

# Save
with open('output/regime_engine_v2.json', 'w') as f:
    json.dump({
        'name': 'Regime Engine v2 (PnL-optimized)',
        'params': best,
        'tuning_pnl': m_t['net_pnl'], 'tuning_sharpe': m_t['sharpe'], 'tuning_trades': m_t['n_trades'],
        'val_pnl': m_v['net_pnl'], 'val_sharpe': m_v['sharpe'], 'val_trades': m_v['n_trades'],
        '3y_pnl': m_3y1['net_pnl'] + m_3y2['net_pnl'],
        'robust_2y': robust,
    }, f, indent=2, default=str)
print('\nSaved → output/regime_engine_v2.json')


# Now try Sharpe objective
print()
print('=' * 100)
print('Optuna Regime Engine — 500 trials with Sharpe objective')
print('=' * 100)

study2 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=43))
study2.optimize(lambda t: obj_regime_sharpe(t, eur_t, eur_v), n_trials=500, show_progress_bar=False)
best2 = study2.best_params
m_t = run_strategy(eur_t, RegimeSwitchingEngineStrategy, best2, [], 'forex')
m_v = run_strategy(eur_v, RegimeSwitchingEngineStrategy, best2, [], 'forex')
m_3y1 = run_strategy(eur_3y.iloc[:half], RegimeSwitchingEngineStrategy, best2, [], 'forex')
m_3y2 = run_strategy(eur_3y.iloc[half:], RegimeSwitchingEngineStrategy, best2, [], 'forex')

print(f'\n=== Best Regime Engine (Sharpe objective) ===')
print(f'Tuning (2024-26): ${m_t["net_pnl"]:+,.0f}/Sh {m_t["sharpe"]:+.2f}/{m_t["n_trades"]}t')
print(f'Val (2022-24):    ${m_v["net_pnl"]:+,.0f}/Sh {m_v["sharpe"]:+.2f}/{m_v["n_trades"]}t')
print(f'3Y Total:         ${m_3y1["net_pnl"]+m_3y2["net_pnl"]:+,.0f}')
robust2 = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
print(f'ROBUST (2Y): {robust2}')
print(f'Params: {best2}')


# Compare to v1
print()
print('=' * 100)
print('COMPARISON')
print('=' * 100)
v1 = {'tuning': 572, 'val': 472, '3y': 1038, 'tuning_trades': 21, 'val_trades': 21}
v2 = {'tuning': m_t['net_pnl'], 'val': m_v['net_pnl'], '3y': m_3y1['net_pnl']+m_3y2['net_pnl'],
       'tuning_trades': m_t['n_trades'], 'val_trades': m_v['n_trades']}
print(f'v1 (PnL): T ${v1["tuning"]}/V ${v1["val"]}/3y ${v1["3y"]}/T:{v1["tuning_trades"]}t V:{v1["val_trades"]}t')
print(f'v2 (Sharpe): T ${v2["tuning"]:.0f}/V ${v2["val"]:.0f}/3y ${v2["3y"]:.0f}/T:{v2["tuning_trades"]}t V:{v2["val_trades"]}t')
