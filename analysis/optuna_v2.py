"""Optuna v2: Regime Engine + Adaptive ADX with focus on HIGH PnL + ROBUST."""
import warnings; warnings.filterwarnings('ignore')
import sys, json, math
sys.path.insert(0, '.')
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import run_strategy
from strategies.top5_research import RegimeSwitchingEngineStrategy, AdaptiveADXStrategy

eur = pd.read_csv('output/mt5_EURUSD_H1_2022_2026.csv', index_col='time', parse_dates=True)
if eur.index.tz is None:
    eur.index = eur.index.tz_localize('UTC')
eur = eur[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
eur_t = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
eur_v = eur[(eur.index >= pd.Timestamp('2022-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2024-09-17', tz='UTC'))]


# Try multiple objectives to find best ROBUST variants

def obj_regime_min_pnl(trial, df_t, df_v):
    """Maximize min PnL with min trades 50+ each period."""
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
        'cooldown': trial.suggest_int('cooldown', 1, 80),
    }
    m_t = run_strategy(df_t, RegimeSwitchingEngineStrategy, params, [], 'forex')
    m_v = run_strategy(df_v, RegimeSwitchingEngineStrategy, params, [], 'forex')
    if 'error' in m_t or 'error' in m_v: return -1e9
    if m_t['n_trades'] < 50 or m_v['n_trades'] < 50: return -1e9
    return min(m_t['net_pnl'], m_v['net_pnl'])


def obj_adaptive_min_pnl(trial, df_t, df_v):
    """Maximize min PnL for Adaptive ADX with min trades 50+."""
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
    if m_t['n_trades'] < 50 or m_v['n_trades'] < 50: return -1e9
    return min(m_t['net_pnl'], m_v['net_pnl'])


# Run Optuna for Regime Engine v2
print('=' * 80)
print('Optuna Regime Engine v2 — min PnL objective (1500 trials)')
print('=' * 80)

study_r = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_r.optimize(lambda t: obj_regime_min_pnl(t, eur_t, eur_v), n_trials=1500, show_progress_bar=False)
best_r = study_r.best_params

m_t = run_strategy(eur_t, RegimeSwitchingEngineStrategy, best_r, [], 'forex')
m_v = run_strategy(eur_v, RegimeSwitchingEngineStrategy, best_r, [], 'forex')
robust_r = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0

print(f'\n=== Regime Engine v2 ===')
print(f'Tuning: ${m_t["net_pnl"]:+,.0f}/Sh {m_t["sharpe"]:+.2f}/{m_t["n_trades"]}t')
print(f'Val:    ${m_v["net_pnl"]:+,.0f}/Sh {m_v["sharpe"]:+.2f}/{m_v["n_trades"]}t')
print(f'ROBUST: {robust_r}')
print(f'Params: {best_r}')


# Run Optuna for Adaptive ADX v2
print()
print('=' * 80)
print('Optuna Adaptive ADX v2 — min PnL objective (1500 trials)')
print('=' * 80)

study_a = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=43))
study_a.optimize(lambda t: obj_adaptive_min_pnl(t, eur_t, eur_v), n_trials=1500, show_progress_bar=False)
best_a = study_a.best_params

m_t = run_strategy(eur_t, AdaptiveADXStrategy, best_a, [], 'forex')
m_v = run_strategy(eur_v, AdaptiveADXStrategy, best_a, [], 'forex')
robust_a = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0

print(f'\n=== Adaptive ADX v2 ===')
print(f'Tuning: ${m_t["net_pnl"]:+,.0f}/Sh {m_t["sharpe"]:+.2f}/{m_t["n_trades"]}t')
print(f'Val:    ${m_v["net_pnl"]:+,.0f}/Sh {m_v["sharpe"]:+.2f}/{m_v["n_trades"]}t')
print(f'ROBUST: {robust_a}')
print(f'Params: {best_a}')


# Save v2 results
with open('output/regime_engine_v2.json', 'w') as f:
    json.dump({'name': 'Regime Engine v2', 'params': best_r,
               'tuning_pnl': m_t['net_pnl'] if 'm_t' in dir() else None,
               'val_pnl': m_v['net_pnl'] if 'm_v' in dir() else None,
               'robust': robust_r}, f, indent=2, default=str)
with open('output/adaptive_adx_v2.json', 'w') as f:
    json.dump({'name': 'Adaptive ADX v2', 'params': best_a,
               'tuning_pnl': m_t['net_pnl'] if 'm_t' in dir() else None,
               'val_pnl': m_v['net_pnl'] if 'm_v' in dir() else None,
               'robust': robust_a}, f, indent=2, default=str)
print('\nSaved → output/regime_engine_v2.json, output/adaptive_adx_v2.json')
