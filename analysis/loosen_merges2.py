"""More aggressive loosening: short cooldowns, wider windows, multiple primaries."""
import warnings; warnings.filterwarnings('ignore')
import sys, json, math
sys.path.insert(0, '.')
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import fetch_h1, run_strategy
from strategies._smart_merge import VoteStrategy, PrimaryWithFilterStrategy
from strategies.screener_and_linda import (
    ScreenerS5Strategy, ScreenerS6Strategy,
    ScreenerS7Strategy, ScreenerS8Strategy
)
from strategies.adx import ADX_Strategy
from strategies._combined import CombinedStrategy

eur = pd.read_csv('output/mt5_EURUSD_H1_2022_2026.csv', index_col='time', parse_dates=True)
if eur.index.tz is None:
    eur.index = eur.index.tz_localize('UTC')
eur = eur[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
eur_t = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
eur_v = eur[(eur.index >= pd.Timestamp('2022-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2024-09-17', tz='UTC'))]

print('=' * 100)
print('APPROACH 1: Vote with VERY loose cooldowns (cd=0)')
print('=' * 100)

# Vote-1-of-2 (OR-gate) with cooldowns=0 to maximize signals
adx_p = {
    'bars_calculate': 10, 'crossover_lookback': 7, 'min_crossover_gap': 0.83,
    'adx_zone_low': 22.5, 'adx_zone_high': 56.0,
    'continuation_level': 21.7, 'reversal_edge': 26.3,
    'level_open_orders_1': 51.6, 'level_open_orders_2': 21.3,
    'sweep_lookback': 16, 'level_close_orders_1': 16.8, 'level_close_orders_2': 3.59,
    'use_di_cross': True, 'open_orders_type': 1, 'close_orders_type': 6,
}

def obj_vote_loose(trial, df):
    sc6_p = {
        'fast': trial.suggest_int('fast', 3, 30),
        'slow': trial.suggest_int('slow', 20, 120),
        'sma_period': trial.suggest_int('sma_p', 50, 400),
        'cooldown': trial.suggest_int('cd', 0, 5),
    }
    params = {
        'n_strategies': 2, 'vote_threshold': 1,
        'cooldown': trial.suggest_int('outer_cd', 0, 3),
        '_strategy1': ADX_Strategy, '_strategy1_params': adx_p,
        '_strategy2': ScreenerS6Strategy, '_strategy2_params': sc6_p,
    }
    m = run_strategy(df, VoteStrategy, params, [], 'forex')
    if 'error' in m or m['n_trades'] < 20:
        return -1e9
    return m['net_pnl'] + 30 * m['sharpe'] + 200 * m['win_rate']

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda t: obj_vote_loose(t, eur_t), n_trials=80, show_progress_bar=False)
best = study.best_params
sc6_p = {'fast': best['fast'], 'slow': best['slow'], 'sma_period': best['sma_p'], 'cooldown': best['cd']}
params = {
    'n_strategies': 2, 'vote_threshold': 1, 'cooldown': best['outer_cd'],
    '_strategy1': ADX_Strategy, '_strategy1_params': adx_p,
    '_strategy2': ScreenerS6Strategy, '_strategy2_params': sc6_p,
}
m_t = run_strategy(eur_t, VoteStrategy, params, [], 'forex')
m_v = run_strategy(eur_v, VoteStrategy, params, [], 'forex')
robust = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
total = m_t['n_trades'] + m_v['n_trades']
print(f'  T ${m_t["net_pnl"]:+,.0f} Sh {m_t["sharpe"]:+.2f} WR {m_t["win_rate"]*100:.1f}% {m_t["n_trades"]}t')
print(f'  V ${m_v["net_pnl"]:+,.0f} Sh {m_v["sharpe"]:+.2f} WR {m_v["win_rate"]*100:.1f}% {m_v["n_trades"]}t')
print(f'  ROBUST: {robust} | Total trades: {total}')
print(f'  Params: sc6={sc6_p}, vote cd={best["outer_cd"]}')


print()
print('=' * 100)
print('APPROACH 2: Vote-1-of-3 (adx + sc_s6 + sc_s5) — more sources = more trades')
print('=' * 100)

def obj_vote_3(trial, df):
    sc6_p = {
        'fast': trial.suggest_int('s6_fast', 3, 30),
        'slow': trial.suggest_int('s6_slow', 20, 120),
        'sma_period': trial.suggest_int('s6_sma_p', 50, 400),
        'cooldown': trial.suggest_int('s6_cd', 0, 5),
    }
    sc5_p = {
        'k_period': trial.suggest_int('s5_k', 5, 20),
        'd_period': trial.suggest_int('s5_d', 2, 10),
        'slowing': trial.suggest_int('s5_slowing', 1, 10),
        'ob_level': trial.suggest_float('s5_ob', 70.0, 85.0),
        'os_level': trial.suggest_float('s5_os', 15.0, 30.0),
        'sma_filter': trial.suggest_categorical('s5_sma', [True, False]),
        'sma_period': trial.suggest_int('s5_sma_p', 50, 300),
        'cooldown': trial.suggest_int('s5_cd', 0, 5),
    }
    params = {
        'n_strategies': 3, 'vote_threshold': 1,
        'cooldown': trial.suggest_int('outer_cd', 0, 3),
        '_strategy1': ADX_Strategy, '_strategy1_params': adx_p,
        '_strategy2': ScreenerS6Strategy, '_strategy2_params': sc6_p,
        '_strategy3': ScreenerS5Strategy, '_strategy3_params': sc5_p,
    }
    m = run_strategy(df, VoteStrategy, params, [], 'forex')
    if 'error' in m or m['n_trades'] < 30:
        return -1e9
    return m['net_pnl'] + 30 * m['sharpe'] + 200 * m['win_rate']

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda t: obj_vote_3(t, eur_t), n_trials=100, show_progress_bar=False)
best = study.best_params
sc6_p = {'fast': best['s6_fast'], 'slow': best['s6_slow'], 'sma_period': best['s6_sma_p'], 'cooldown': best['s6_cd']}
sc5_p = {'k_period': best['s5_k'], 'd_period': best['s5_d'], 'slowing': best['s5_slowing'],
         'ob_level': best['s5_ob'], 'os_level': best['s5_os'], 'sma_filter': best['s5_sma'],
         'sma_period': best['s5_sma_p'], 'cooldown': best['s5_cd']}
params = {
    'n_strategies': 3, 'vote_threshold': 1, 'cooldown': best['outer_cd'],
    '_strategy1': ADX_Strategy, '_strategy1_params': adx_p,
    '_strategy2': ScreenerS6Strategy, '_strategy2_params': sc6_p,
    '_strategy3': ScreenerS5Strategy, '_strategy3_params': sc5_p,
}
m_t = run_strategy(eur_t, VoteStrategy, params, [], 'forex')
m_v = run_strategy(eur_v, VoteStrategy, params, [], 'forex')
robust = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
total = m_t['n_trades'] + m_v['n_trades']
print(f'  T ${m_t["net_pnl"]:+,.0f} Sh {m_t["sharpe"]:+.2f} WR {m_t["win_rate"]*100:.1f}% {m_t["n_trades"]}t')
print(f'  V ${m_v["net_pnl"]:+,.0f} Sh {m_v["sharpe"]:+.2f} WR {m_v["win_rate"]*100:.1f}% {m_v["n_trades"]}t')
print(f'  ROBUST: {robust} | Total trades: {total}')
print(f'  Params: sc6={sc6_p}, sc5={sc5_p}, cd={best["outer_cd"]}')


print()
print('=' * 100)
print('APPROACH 3: Primary+Filter with cw=1 (almost immediate confirmation)')
print('=' * 100)

def obj_primary_cw1(trial, df):
    sc6_p = {
        'fast': trial.suggest_int('s6_fast', 3, 30),
        'slow': trial.suggest_int('s6_slow', 20, 120),
        'sma_period': trial.suggest_int('s6_sma_p', 50, 400),
        'cooldown': trial.suggest_int('s6_cd', 0, 5),
    }
    adx_p_local = {
        'bars_calculate': trial.suggest_int('a_bars', 8, 20),
        'crossover_lookback': trial.suggest_int('a_xover', 3, 10),
        'min_crossover_gap': trial.suggest_float('a_gap', 0.5, 4.0),
        'adx_zone_low': trial.suggest_float('a_zl', 18.0, 28.0),
        'adx_zone_high': trial.suggest_float('a_zh', 45.0, 60.0),
        'continuation_level': trial.suggest_float('a_cont', 15.0, 25.0),
        'reversal_edge': trial.suggest_float('a_rev', 22.0, 35.0),
        'level_open_orders_1': trial.suggest_float('a_lv1', 30.0, 55.0),
        'level_open_orders_2': trial.suggest_float('a_lv2', 15.0, 30.0),
        'sweep_lookback': trial.suggest_int('a_swp', 5, 20),
        'level_close_orders_1': trial.suggest_float('a_cl1', 5.0, 20.0),
        'level_close_orders_2': trial.suggest_float('a_cl2', 2.0, 10.0),
        'use_di_cross': True, 'open_orders_type': 1, 'close_orders_type': 1,
    }
    params = {
        '_primary': ScreenerS6Strategy, '_primary_params': sc6_p,
        '_filter': ADX_Strategy, '_filter_params': adx_p_local,
        'confirm_window': trial.suggest_int('cw', 1, 3),  # short window
        'cooldown': trial.suggest_int('cd', 0, 5),
    }
    m = run_strategy(df, PrimaryWithFilterStrategy, params, [], 'forex')
    if 'error' in m or m['n_trades'] < 30:
        return -1e9
    return m['net_pnl'] + 30 * m['sharpe'] + 200 * m['win_rate']

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda t: obj_primary_cw1(t, eur_t), n_trials=80, show_progress_bar=False)
best = study.best_params
sc6_p = {'fast': best['s6_fast'], 'slow': best['s6_slow'], 'sma_period': best['s6_sma_p'], 'cooldown': best['s6_cd']}
adx_p_local = {k[2:]: v for k, v in best.items() if k.startswith('a_')}
adx_p_local['use_di_cross'] = True
adx_p_local['open_orders_type'] = 1
adx_p_local['close_orders_type'] = 1
params = {
    '_primary': ScreenerS6Strategy, '_primary_params': sc6_p,
    '_filter': ADX_Strategy, '_filter_params': adx_p_local,
    'confirm_window': best['cw'],
    'cooldown': best['cd'],
}
m_t = run_strategy(eur_t, PrimaryWithFilterStrategy, params, [], 'forex')
m_v = run_strategy(eur_v, PrimaryWithFilterStrategy, params, [], 'forex')
robust = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
total = m_t['n_trades'] + m_v['n_trades']
print(f'  T ${m_t["net_pnl"]:+,.0f} Sh {m_t["sharpe"]:+.2f} WR {m_t["win_rate"]*100:.1f}% {m_t["n_trades"]}t')
print(f'  V ${m_v["net_pnl"]:+,.0f} Sh {m_v["sharpe"]:+.2f} WR {m_v["win_rate"]*100:.1f}% {m_v["n_trades"]}t')
print(f'  ROBUST: {robust} | Total trades: {total}')
