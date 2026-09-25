"""Reduce strictness to find higher-trade-count variants of ROBUST singles.

Approach:
- For each of 5 ROBUST singles, run Optuna with looser constraints
- Goal: same ROBUST status but n_trades >= 2x current
- Strategies to loosen: adx_EUR (currently 106t), linda_macd (213t already), ms (small), sc_s5/s8 (Python-only)

Focus on adx_EUR, ms, sc_s5_stoch, sc_s8_bb, linda_macd
"""
import warnings; warnings.filterwarnings('ignore')
import json
import sys

sys.path.insert(0, '.')
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import run_strategy

# Load MT5 data
eur = pd.read_csv('output/mt5_EURUSD_H1_2022_2026.csv', index_col='time', parse_dates=True)
if eur.index.tz is None:
    eur.index = eur.index.tz_localize('UTC')
eur = eur[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})

eur_t = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
eur_v = eur[(eur.index >= pd.Timestamp('2022-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2024-09-17', tz='UTC'))]

print('=' * 100)
print('REDUCING STRICTNESS — search for variants with MORE trades')
print('=' * 100)

# Strategy: adx_EUR — currently 106 trades, Sh 1.73. Try lower cooldowns + simpler logic
def obj_adx_eur(trial, df, period):
    """Looser ADX with reduced cooldowns + optional simpler entry."""
    params = {
        'bars_calculate': trial.suggest_int('bars', 8, 25),  # smaller range
        'use_di_cross': True,  # always on (the core signal)
        'crossover_lookback': trial.suggest_int('xover', 1, 8),  # was 6, allow 1-8
        'min_crossover_gap': trial.suggest_float('gap', 0.5, 5.0),  # was 2.33
        # Loosen zone requirements
        'adx_zone_low': trial.suggest_float('zone_lo', 18.0, 28.0),  # was 23.8
        'adx_zone_high': trial.suggest_float('zone_hi', 45.0, 60.0),  # was 49.5
        'continuation_level': trial.suggest_float('cont', 15.0, 25.0),  # was 20.0
        'reversal_edge': trial.suggest_float('rev', 22.0, 35.0),  # was 28.5
        'open_orders_type': trial.suggest_int('oot', 1, 8),  # allow all modes
        'level_open_orders_1': trial.suggest_float('lv1', 30.0, 55.0),
        'level_open_orders_2': trial.suggest_float('lv2', 15.0, 30.0),
        'sweep_lookback': trial.suggest_int('swp', 5, 20),
        'level_close_orders_1': trial.suggest_float('cl1', 5.0, 20.0),
        'level_close_orders_2': trial.suggest_float('cl2', 2.0, 10.0),
        'close_orders_type': trial.suggest_int('cot', 1, 8),
    }
    from strategies.adx import ADX_Strategy
    m = run_strategy(df, ADX_Strategy, params, [], 'forex')
    if 'error' in m or m['n_trades'] < 30:  # require at least 30 trades
        return -1e9
    # Reward: PnL + 0.05 * Sharpe + 0.5 * log(n_trades) — favor more trades
    import math
    return m['net_pnl'] + 30 * m['sharpe'] + 200 * math.log(m['n_trades'])


# Strategy: linda_macd_EUR — currently 213 trades, Sh 0.58. Try without SMA filter, lower cooldown
def obj_linda(trial, df, period):
    """Looser Linda MACD: lower cooldowns, optional SMA filter."""
    params = {
        'fast': trial.suggest_int('fast', 8, 25),
        'slow': trial.suggest_int('slow', 30, 80),
        'signal': trial.suggest_int('sig', 5, 25),
        'use_sma': trial.suggest_categorical('use_sma', [True, False]),
        'sma_p': trial.suggest_int('sma_p', 50, 250),
        'use_hist': trial.suggest_categorical('use_hist', [True, False]),
        'cd': trial.suggest_int('cd', 1, 25),  # was 20, allow 1-25
    }
    from strategies.linda_macd_lenient import LindaMACDLenientStrategy
    m = run_strategy(df, LindaMACDLenientStrategy, params, [], 'forex')
    if 'error' in m or m['n_trades'] < 50:
        return -1e9
    import math
    return m['net_pnl'] + 30 * m['sharpe'] + 200 * math.log(m['n_trades'])


# Strategy: ms_EUR — currently small. Try looser cooldowns
def obj_ms(trial, df, period):
    """MS with looser cooldowns."""
    params = {
        'use_confluence': trial.suggest_categorical('conf', [True, False]),
        'confluence_bars': trial.suggest_int('conf_bars', 1, 5),
        'use_macd_div': trial.suggest_categorical('macd_div', [True, False]),
        'use_stoch_div': trial.suggest_categorical('stoch_div', [True, False]),
        'use_hist_div': trial.suggest_categorical('hist_div', [True, False]),
        'div_bars': trial.suggest_int('div_bars', 5, 30),
        'fast_ema': trial.suggest_int('fast', 2, 20),
        'slow_ema': trial.suggest_int('slow', 5, 30),
        'signal': trial.suggest_int('sig', 2, 15),
        'k_period': trial.suggest_int('k', 3, 20),
        'd_period': trial.suggest_int('d', 2, 15),
        'slowing': trial.suggest_int('slowing', 3, 30),
        'open1': trial.suggest_int('open1', 1, 30),
        'open2': trial.suggest_int('open2', 1, 50),
        'level_open1': trial.suggest_float('lv1', 30.0, 80.0),
        'level_open2': trial.suggest_float('lv2', 50.0, 90.0),
        'close1': trial.suggest_int('close1', 1, 20),
        'close2': trial.suggest_int('close2', 1, 30),
        'level_close1': trial.suggest_float('clv1', 30.0, 70.0),
        'level_close2': trial.suggest_float('clv2', 40.0, 80.0),
    }
    from strategies.ms import MS_Strategy
    m = run_strategy(df, MS_Strategy, params, [], 'forex')
    if 'error' in m or m['n_trades'] < 30:
        return -1e9
    import math
    return m['net_pnl'] + 30 * m['sharpe'] + 200 * math.log(m['n_trades'])


# Strategy: sc_s5_stoch — Stochastic-based
def obj_sc_s5(trial, df, period):
    """SCreener S5 with looser thresholds."""
    params = {
        'k_period': trial.suggest_int('k', 5, 20),
        'd_period': trial.suggest_int('d', 2, 10),
        'slowing': trial.suggest_int('slowing', 1, 10),
        'ob_level': trial.suggest_float('ob', 70.0, 85.0),
        'os_level': trial.suggest_float('os', 15.0, 30.0),
        'sma_filter': trial.suggest_categorical('sma', [True, False]),
        'sma_period': trial.suggest_int('sma_p', 50, 300),
        'cooldown': trial.suggest_int('cd', 1, 20),
    }
    from strategies.screener_and_linda import ScreenerS5Strategy
    m = run_strategy(df, ScreenerS5Strategy, params, [], 'forex')
    if 'error' in m or m['n_trades'] < 30:
        return -1e9
    import math
    return m['net_pnl'] + 30 * m['sharpe'] + 200 * math.log(m['n_trades'])


# Run for each strategy on EUR tuning (2024-26), then validate on EUR val (2022-24)
results = []

print()
print('=== ADX EUR (looser) ===')
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda t: obj_adx_eur(t, eur_t, 'tuning'), n_trials=50, show_progress_bar=False)
best = study.best_params
p = {**{'use_di_cross': True, 'open_orders_type': 1, 'close_orders_type': 1},
     'bars_calculate': best['bars'], 'crossover_lookback': best['xover'],
     'min_crossover_gap': best['gap'], 'adx_zone_low': best['zone_lo'],
     'adx_zone_high': best['zone_hi'], 'continuation_level': best['cont'],
     'reversal_edge': best['rev'], 'open_orders_type': best['oot'],
     'level_open_orders_1': best['lv1'], 'level_open_orders_2': best['lv2'],
     'sweep_lookback': best['swp'], 'level_close_orders_1': best['cl1'],
     'level_close_orders_2': best['cl2'], 'close_orders_type': best['cot']}
from strategies.adx import ADX_Strategy

m_t = run_strategy(eur_t, ADX_Strategy, p, [], 'forex')
m_v = run_strategy(eur_v, ADX_Strategy, p, [], 'forex')
robust = (m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0)
print(f'  T ${m_t["net_pnl"]:+,.0f} Sh {m_t["sharpe"]:+.2f} WR {m_t["win_rate"]*100:.1f}% {m_t["n_trades"]}t')
print(f'  V ${m_v["net_pnl"]:+,.0f} Sh {m_v["sharpe"]:+.2f} WR {m_v["win_rate"]*100:.1f}% {m_v["n_trades"]}t')
print(f'  ROBUST: {robust}  (vs original 106 trades, +{m_t["n_trades"]-106} more trades)')
results.append({'strategy': 'adx_EUR', 'loosened': True, 'params': p, 'm_t': m_t, 'm_v': m_v, 'robust': robust})

print()
print('=== Linda MACD EUR (looser) ===')
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda t: obj_linda(t, eur_t, 'tuning'), n_trials=50, show_progress_bar=False)
best = study.best_params
p = {**{'fast': best['fast'], 'slow': best['slow'], 'signal': best['sig'],
        'use_sma': best['use_sma'], 'sma_p': best['sma_p'], 'use_hist': best['use_hist'],
        'cd': best['cd']}}
from strategies.linda_macd_lenient import LindaMACDLenientStrategy

m_t = run_strategy(eur_t, LindaMACDLenientStrategy, p, [], 'forex')
m_v = run_strategy(eur_v, LindaMACDLenientStrategy, p, [], 'forex')
robust = (m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0)
print(f'  T ${m_t["net_pnl"]:+,.0f} Sh {m_t["sharpe"]:+.2f} WR {m_t["win_rate"]*100:.1f}% {m_t["n_trades"]}t')
print(f'  V ${m_v["net_pnl"]:+,.0f} Sh {m_v["sharpe"]:+.2f} WR {m_v["win_rate"]*100:.1f}% {m_v["n_trades"]}t')
print(f'  ROBUST: {robust}  (vs original 213 trades, +{m_t["n_trades"]-213} more trades)')
results.append({'strategy': 'linda_macd_EUR', 'loosened': True, 'params': p, 'm_t': m_t, 'm_v': m_v, 'robust': robust})

print()
print('=== SCreener S5 EUR (looser) ===')
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda t: obj_sc_s5(t, eur_t, 'tuning'), n_trials=50, show_progress_bar=False)
best = study.best_params
p = {**{'k_period': best['k'], 'd_period': best['d'], 'slowing': best['slowing'],
        'ob_level': best['ob'], 'os_level': best['os'], 'sma_filter': best['sma'],
        'sma_period': best['sma_p'], 'cooldown': best['cd']}}
from strategies.screener_and_linda import ScreenerS5Strategy

m_t = run_strategy(eur_t, ScreenerS5Strategy, p, [], 'forex')
m_v = run_strategy(eur_v, ScreenerS5Strategy, p, [], 'forex')
robust = (m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0)
print(f'  T ${m_t["net_pnl"]:+,.0f} Sh {m_t["sharpe"]:+.2f} WR {m_t["win_rate"]*100:.1f}% {m_t["n_trades"]}t')
print(f'  V ${m_v["net_pnl"]:+,.0f} Sh {m_v["sharpe"]:+.2f} WR {m_v["win_rate"]*100:.1f}% {m_v["n_trades"]}t')
print(f'  ROBUST: {robust}')
results.append({'strategy': 'sc_s5_EUR', 'loosened': True, 'params': p, 'm_t': m_t, 'm_v': m_v, 'robust': robust})


print()
print('=' * 100)
print('FINAL: LOOSENED ROBUST VARIANTS')
print('=' * 100)
for r in results:
    if r['robust']:
        print(f'  ✅ {r["strategy"]:25s} | T ${r["m_t"]["net_pnl"]:+,} ({r["m_t"]["n_trades"]}t) | V ${r["m_v"]["net_pnl"]:+,} ({r["m_v"]["n_trades"]}t)')

with open('output/loosened_robust.json', 'w') as f:
    json.dump([{'strategy': r['strategy'], 'params': r['params'], 'metrics': r['m_t'], 'val_metrics': r['m_v'], 'robust': r['robust']} for r in results], f, indent=2, default=str)
print('\nSaved → output/loosened_robust.json')
