"""Optuna-tune top 3 strategies — maximize min(tuning_pnl, val_pnl)."""
import warnings; warnings.filterwarnings('ignore')
import sys, json, math
sys.path.insert(0, '.')
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import run_strategy
from strategies.top5_research import (
    RegimeSwitchingEngineStrategy, AdaptiveADXStrategy, VolatilityBreakoutStrategy
)

eur = pd.read_csv('output/mt5_EURUSD_H1_2022_2026.csv', index_col='time', parse_dates=True)
if eur.index.tz is None:
    eur.index = eur.index.tz_localize('UTC')
eur = eur[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
eur_t = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
eur_v = eur[(eur.index >= pd.Timestamp('2022-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2024-09-17', tz='UTC'))]


def obj_regime(trial, df_t, df_v):
    params = {
        'adx_period': trial.suggest_int('adx_period', 8, 25),
        'atr_period': trial.suggest_int('atr_period', 8, 25),
        'bb_period': trial.suggest_int('bb_period', 15, 40),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0),
        'trend_thresh': trial.suggest_float('trend_thresh', 20.0, 35.0),
        'range_thresh': trial.suggest_float('range_thresh', 12.0, 22.0),
        'expansion_mult': trial.suggest_float('expansion_mult', 1.2, 2.5),
        'atr_avg_period': trial.suggest_int('atr_avg_period', 20, 100),
        'sma_period': trial.suggest_int('sma_period', 50, 250),
        'cooldown': trial.suggest_int('cooldown', 3, 30),
    }
    m_t = run_strategy(df_t, RegimeSwitchingEngineStrategy, params, [], 'forex')
    m_v = run_strategy(df_v, RegimeSwitchingEngineStrategy, params, [], 'forex')
    if 'error' in m_t or 'error' in m_v: return -1e9
    if m_t['n_trades'] < 20 or m_v['n_trades'] < 20: return -1e9
    return min(m_t['net_pnl'], m_v['net_pnl']) + 20 * (m_t['sharpe'] + m_v['sharpe'])


def obj_adaptive(trial, df_t, df_v):
    params = {
        'adx_period': trial.suggest_int('adx_period', 8, 25),
        'lookback': trial.suggest_int('lookback', 100, 500),
        'adx_percentile': trial.suggest_float('adx_percentile', 50.0, 90.0),
        'bars_calculate': trial.suggest_int('bars', 5, 25),
        'crossover_lookback': trial.suggest_int('xover', 1, 12),
        'min_crossover_gap': trial.suggest_float('gap', 0.3, 5.0),
        'zone_lo': trial.suggest_float('zl', 18.0, 30.0),
        'zone_hi': trial.suggest_float('zh', 40.0, 65.0),
        'cont': trial.suggest_float('cont', 12.0, 28.0),
        'rev': trial.suggest_float('rev', 20.0, 38.0),
        'lv1': trial.suggest_float('lv1', 25.0, 60.0),
        'lv2': trial.suggest_float('lv2', 10.0, 35.0),
        'swp': trial.suggest_int('swp', 5, 25),
        'cl1': trial.suggest_float('cl1', 5.0, 25.0),
        'cl2': trial.suggest_float('cl2', 1.0, 15.0),
        'oot': trial.suggest_int('oot', 1, 10),
        'cot': trial.suggest_int('cot', 1, 10),
        'sma_period': trial.suggest_int('sma_p', 50, 250),
        'cooldown': trial.suggest_int('cd', 1, 20),
    }
    m_t = run_strategy(df_t, AdaptiveADXStrategy, params, [], 'forex')
    m_v = run_strategy(df_v, AdaptiveADXStrategy, params, [], 'forex')
    if 'error' in m_t or 'error' in m_v: return -1e9
    if m_t['n_trades'] < 20 or m_v['n_trades'] < 20: return -1e9
    return min(m_t['net_pnl'], m_v['net_pnl']) + 20 * (m_t['sharpe'] + m_v['sharpe'])


def obj_vb(trial, df_t, df_v):
    params = {
        'adx_period': trial.suggest_int('adx_period', 8, 25),
        'atr_period': trial.suggest_int('atr_period', 8, 25),
        'atr_avg_period': trial.suggest_int('atr_avg_period', 10, 60),
        'donchian': trial.suggest_int('donchian', 10, 50),
        'adx_thresh': trial.suggest_float('adx_thresh', 18.0, 35.0),
        'sma_period': trial.suggest_int('sma_p', 50, 250),
        'cooldown': trial.suggest_int('cd', 1, 15),
    }
    m_t = run_strategy(df_t, VolatilityBreakoutStrategy, params, [], 'forex')
    m_v = run_strategy(df_v, VolatilityBreakoutStrategy, params, [], 'forex')
    if 'error' in m_t or 'error' in m_v: return -1e9
    if m_t['n_trades'] < 20 or m_v['n_trades'] < 20: return -1e9
    return min(m_t['net_pnl'], m_v['net_pnl']) + 20 * (m_t['sharpe'] + m_v['sharpe'])


print('=' * 100)
print('Optuna: ROBUST search for top 3 (min of tuning/val PnL)')
print('=' * 100)

results = []

# Regime Engine
print('\n=== #1 Regime Engine ===')
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda t: obj_regime(t, eur_t, eur_v), n_trials=100, show_progress_bar=False)
best = study.best_params
m_t = run_strategy(eur_t, RegimeSwitchingEngineStrategy, best, [], 'forex')
m_v = run_strategy(eur_v, RegimeSwitchingEngineStrategy, best, [], 'forex')
robust = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
print(f'Tuning:     ${m_t["net_pnl"]:+,.0f} / Sh {m_t["sharpe"]:+.2f} / {m_t["n_trades"]}t')
print(f'Validation: ${m_v["net_pnl"]:+,.0f} / Sh {m_v["sharpe"]:+.2f} / {m_v["n_trades"]}t')
print(f'ROBUST: {robust} | Best: {best}')
results.append({'name': '#1 Regime Engine', 'params': best, 'm_t': m_t, 'm_v': m_v, 'robust': robust})

# Adaptive ADX
print('\n=== #3 Adaptive ADX ===')
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda t: obj_adaptive(t, eur_t, eur_v), n_trials=100, show_progress_bar=False)
best = study.best_params
m_t = run_strategy(eur_t, AdaptiveADXStrategy, best, [], 'forex')
m_v = run_strategy(eur_v, AdaptiveADXStrategy, best, [], 'forex')
robust = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
print(f'Tuning:     ${m_t["net_pnl"]:+,.0f} / Sh {m_t["sharpe"]:+.2f} / {m_t["n_trades"]}t')
print(f'Validation: ${m_v["net_pnl"]:+,.0f} / Sh {m_v["sharpe"]:+.2f} / {m_v["n_trades"]}t')
print(f'ROBUST: {robust} | Best: {best}')
results.append({'name': '#3 Adaptive ADX', 'params': best, 'm_t': m_t, 'm_v': m_v, 'robust': robust})

# Vol Breakout
print('\n=== #2 Volatility Breakout ===')
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda t: obj_vb(t, eur_t, eur_v), n_trials=100, show_progress_bar=False)
best = study.best_params
m_t = run_strategy(eur_t, VolatilityBreakoutStrategy, best, [], 'forex')
m_v = run_strategy(eur_v, VolatilityBreakoutStrategy, best, [], 'forex')
robust = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
print(f'Tuning:     ${m_t["net_pnl"]:+,.0f} / Sh {m_t["sharpe"]:+.2f} / {m_t["n_trades"]}t')
print(f'Validation: ${m_v["net_pnl"]:+,.0f} / Sh {m_v["sharpe"]:+.2f} / {m_v["n_trades"]}t')
print(f'ROBUST: {robust} | Best: {best}')
results.append({'name': '#2 Volatility Breakout', 'params': best, 'm_t': m_t, 'm_v': m_v, 'robust': robust})


# Save results
with open('output/top3_research_results.json', 'w') as f:
    json.dump([{'name': r['name'], 'params': r['params'], 'm_t': r['m_t'], 'm_v': r['m_v'], 'robust': r['robust']} for r in results], f, indent=2, default=str)

print()
print('=' * 100)
print('SUMMARY')
print('=' * 100)
for r in results:
    if r['robust']:
        print(f'  ROBUST: {r["name"]} — T ${r["m_t"]["net_pnl"]:,.0f} / V ${r["m_v"]["net_pnl"]:,.0f}')
    else:
        print(f'  NOT ROBUST: {r["name"]} — T ${r["m_t"]["net_pnl"]:,.0f} / V ${r["m_v"]["net_pnl"]:,.0f}')
