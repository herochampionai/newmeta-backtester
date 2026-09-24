"""Optuna-tune the 6 edges with ROBUST objective."""
import warnings; warnings.filterwarnings('ignore')
import sys, json, math
sys.path.insert(0, '.')
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import run_strategy
from strategies.edges_2026 import (
    LiquiditySweepStrategy, VolatilityRegimeStrategy, SessionBasedStrategy,
    MarketStructureStrategy, SynthOrderFlowStrategy, EnsembleStrategy
)

eur = pd.read_csv('output/mt5_EURUSD_H1_2022_2026.csv', index_col='time', parse_dates=True)
if eur.index.tz is None:
    eur.index = eur.index.tz_localize('UTC')
eur = eur[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
eur_t = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
eur_v = eur[(eur.index >= pd.Timestamp('2022-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2024-09-17', tz='UTC'))]

nas = pd.read_csv('output/NDX_NAS100_H1_2024_2026.csv', index_col='Datetime')
nas.index = pd.to_datetime(nas.index, utc=True)
nas.columns = [c.lower().replace(' ', '_') for c in nas.columns]
nas = nas[['open', 'high', 'low', 'close', 'volume']]
nas_t = nas[(nas.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (nas.index < pd.Timestamp('2026-09-17', tz='UTC'))]


def obj(strategy_cls, param_fn, df_t, df_v, profile='forex'):
    def inner(trial):
        params = param_fn(trial)
        m_t = run_strategy(df_t, strategy_cls, params, [], profile)
        m_v = run_strategy(df_v, strategy_cls, params, [], profile)
        if 'error' in m_t or 'error' in m_v: return -1e9
        if m_t['n_trades'] < 20 or m_v['n_trades'] < 20: return -1e9
        return min(m_t['net_pnl'], m_v['net_pnl']) + 20 * (m_t['sharpe'] + m_v['sharpe'])
    return inner


def sweep_p(trial):
    return {
        'lookback': trial.suggest_int('lookback', 10, 50),
        'atr_period': trial.suggest_int('atr', 10, 25),
        'sl_atr': trial.suggest_float('sl_atr', 0.5, 3.0),
        'tp_rr': trial.suggest_float('tp_rr', 1.0, 3.0),
        'atr_avg_period': trial.suggest_int('atr_avg', 10, 50),
        'vol_avg_period': trial.suggest_int('vol_avg', 10, 50),
        'sma_period': trial.suggest_int('sma_p', 50, 300),
        'cooldown': trial.suggest_int('cd', 3, 30),
    }

def vr_p(trial):
    return {
        'atr_period': trial.suggest_int('atr', 10, 25),
        'bb_period': trial.suggest_int('bb_p', 10, 50),
        'bb_mult': trial.suggest_float('bb_m', 1.5, 3.0),
        'rv_period': trial.suggest_int('rv', 5, 30),
        'quiet_pct': trial.suggest_float('quiet', 10.0, 35.0),
        'explosive_pct': trial.suggest_float('exp', 65.0, 90.0),
        'lookback_pct': trial.suggest_int('lbp', 50, 400),
        'ma_period': trial.suggest_int('ma', 10, 100),
        'breakout_period': trial.suggest_int('bo', 10, 50),
        'cooldown': trial.suggest_int('cd', 3, 30),
    }

def sess_p(trial):
    return {
        'adx_period': trial.suggest_int('adx', 10, 30),
        'bb_period': trial.suggest_int('bb', 10, 30),
        'bb_mult': trial.suggest_float('bb_m', 1.5, 3.0),
        'atr_period': trial.suggest_int('atr', 10, 25),
        'cooldown': trial.suggest_int('cd', 3, 30),
    }

def struct_p(trial):
    return {
        'swing_n': trial.suggest_int('sn', 2, 8),
        'disp_atr_mult': trial.suggest_float('dam', 0.5, 2.5),
        'atr_period': trial.suggest_int('atr', 10, 25),
        'retest_window': trial.suggest_int('rw', 2, 10),
        'cooldown': trial.suggest_int('cd', 3, 30),
    }

def sof_p(trial):
    return {
        'atr_period': trial.suggest_int('atr', 10, 25),
        'range_atr_mult': trial.suggest_float('ram', 0.5, 3.0),
        'vol_atr_mult': trial.suggest_float('vam', 0.5, 3.0),
        'close_pct_top': trial.suggest_float('cpt', 0.6, 0.95),
        'close_pct_bot': trial.suggest_float('cpb', 0.05, 0.4),
        'atr_avg_period': trial.suggest_int('atr_avg', 10, 50),
        'vol_avg_period': trial.suggest_int('vol_avg', 10, 50),
        'sma_period': trial.suggest_int('sma', 50, 300),
        'cooldown': trial.suggest_int('cd', 3, 30),
    }

def ens_p(trial):
    return {
        'score_threshold': trial.suggest_float('thr', 60, 95),
        'cooldown': trial.suggest_int('cd', 5, 40),
    }


# Run Optuna for each strategy
print('=' * 80)
print('Optuna ROBUST search — 6 edges on EURUSD H1 (2Y)')
print('=' * 80)

results = []
strategies = [
    ('#1 Liq Sweep',     LiquiditySweepStrategy, sweep_p, 'forex'),
    ('#2 Vol Regime',    VolatilityRegimeStrategy, vr_p, 'forex'),
    ('#3 Session',       SessionBasedStrategy, sess_p, 'forex'),
    ('#4 Market Struct', MarketStructureStrategy, struct_p, 'forex'),
    ('#5 Ensemble',      EnsembleStrategy, ens_p, 'forex'),
    ('#6 Synth OF',      SynthOrderFlowStrategy, sof_p, 'forex'),
]

for name, cls, p_fn, profile in strategies:
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj(cls, p_fn, eur_t, eur_v, profile), n_trials=80, show_progress_bar=False)
    best = study.best_params
    m_t = run_strategy(eur_t, cls, best, [], profile)
    m_v = run_strategy(eur_v, cls, best, [], profile)
    robust = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
    print(f'{name:25s} T ${m_t["net_pnl"]:+7,.0f}/Sh {m_t["sharpe"]:+.2f}/{m_t["n_trades"]:3d}t | V ${m_v["net_pnl"]:+7,.0f}/Sh {m_v["sharpe"]:+.2f}/{m_v["n_trades"]:3d}t | ROBUST={robust}')
    results.append({'name': name, 'params': best, 'm_t': m_t, 'm_v': m_v, 'robust': robust})


print()
print('=' * 80)
print('Best ROBUST — test on NAS100')
print('=' * 80)

for r in results:
    if r['robust']:
        # Get the strategy class
        cls_map = {
            '#1 Liq Sweep': LiquiditySweepStrategy,
            '#2 Vol Regime': VolatilityRegimeStrategy,
            '#3 Session': SessionBasedStrategy,
            '#4 Market Struct': MarketStructureStrategy,
            '#5 Ensemble': EnsembleStrategy,
            '#6 Synth OF': SynthOrderFlowStrategy,
        }
        cls = cls_map[r['name']]
        m_nas = run_strategy(nas_t, cls, r['params'], [], 'nas100')
        if 'error' not in m_nas:
            print(f'{r["name"]:25s} NAS: ${m_nas["net_pnl"]:+,.0f}/Sh {m_nas["sharpe"]:+.2f}/{m_nas["n_trades"]:3d}t')
        else:
            print(f'{r["name"]:25s} NAS: ERROR')


# Save
with open('output/edges_2026_results.json', 'w') as f:
    json.dump([{'name': r['name'], 'params': r['params'], 'm_t': r['m_t'], 'm_v': r['m_v'], 'robust': r['robust']} for r in results], f, indent=2, default=str)
print('\nSaved → output/edges_2026_results.json')
