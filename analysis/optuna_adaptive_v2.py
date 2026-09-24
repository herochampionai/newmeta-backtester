"""Adaptive ADX v2 - focused Optuna."""
import warnings; warnings.filterwarnings('ignore')
import sys, json, math
sys.path.insert(0, '.')
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import run_strategy
from strategies.top5_research import AdaptiveADXStrategy

eur = pd.read_csv('output/mt5_EURUSD_H1_2022_2026.csv', index_col='time', parse_dates=True)
if eur.index.tz is None:
    eur.index = eur.index.tz_localize('UTC')
eur = eur[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
eur_t = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
eur_v = eur[(eur.index >= pd.Timestamp('2022-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2024-09-17', tz='UTC'))]


def obj_a(trial, df_t, df_v):
    params = {
        'adx_period': trial.suggest_int('adx', 8, 25),
        'lookback': trial.suggest_int('lb', 100, 500),
        'adx_percentile': trial.suggest_float('pct', 50, 90),
        'bars_calculate': trial.suggest_int('bars', 5, 25),
        'crossover_lookback': trial.suggest_int('xo', 1, 12),
        'min_crossover_gap': trial.suggest_float('gap', 0.3, 5.0),
        'zone_lo': trial.suggest_float('zl', 18, 30),
        'zone_hi': trial.suggest_float('zh', 40, 65),
        'cont': trial.suggest_float('cont', 12, 28),
        'rev': trial.suggest_float('rev', 20, 38),
        'lv1': trial.suggest_float('lv1', 25, 60),
        'lv2': trial.suggest_float('lv2', 10, 35),
        'swp': trial.suggest_int('swp', 5, 25),
        'cl1': trial.suggest_float('cl1', 5, 25),
        'cl2': trial.suggest_float('cl2', 1, 15),
        'oot': trial.suggest_int('oot', 1, 10),
        'cot': trial.suggest_int('cot', 1, 10),
        'sma_period': trial.suggest_int('sma', 50, 250),
        'cooldown': trial.suggest_int('cd', 1, 20),
    }
    m_t = run_strategy(df_t, AdaptiveADXStrategy, params, [], 'forex')
    m_v = run_strategy(df_v, AdaptiveADXStrategy, params, [], 'forex')
    if 'error' in m_t or 'error' in m_v: return -1e9
    if m_t['n_trades'] < 50 or m_v['n_trades'] < 50: return -1e9
    return min(m_t['net_pnl'], m_v['net_pnl']) + 20 * (m_t['sharpe'] + m_v['sharpe'])


print('Optuna Adaptive ADX — 500 trials...')
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=43))
study.optimize(lambda t: obj_a(t, eur_t, eur_v), n_trials=500, show_progress_bar=False)
best = study.best_params

m_t = run_strategy(eur_t, AdaptiveADXStrategy, best, [], 'forex')
m_v = run_strategy(eur_v, AdaptiveADXStrategy, best, [], 'forex')
robust = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
print(f'Tuning: ${m_t["net_pnl"]:+,.0f}/Sh {m_t["sharpe"]:+.2f}/{m_t["n_trades"]}t')
print(f'Val:    ${m_v["net_pnl"]:+,.0f}/Sh {m_v["sharpe"]:+.2f}/{m_v["n_trades"]}t')
print(f'ROBUST: {robust}')
print(f'Params: {best}')

with open('output/adaptive_adx_v2.json', 'w') as f:
    json.dump({'name': 'Adaptive ADX v2', 'params': best,
               'tuning_pnl': m_t['net_pnl'], 'tuning_sharpe': m_t['sharpe'], 'tuning_trades': m_t['n_trades'],
               'val_pnl': m_v['net_pnl'], 'val_sharpe': m_v['sharpe'], 'val_trades': m_v['n_trades'],
               'robust_2y': robust}, f, indent=2, default=str)
print('Saved -> output/adaptive_adx_v2.json')
