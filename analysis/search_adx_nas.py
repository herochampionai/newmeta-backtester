"""Search for ROBUST adx_NAS variants — same approach as EUR (variant B was a winner)."""
import warnings; warnings.filterwarnings('ignore')
import sys, json, math
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from datetime import datetime, timezone
import MetaTrader5 as mt5

# Connect to MT5 and fetch NAS100 data
mt5.initialize(path='D:\\MT5_Bybit\\terminal64.exe')
from_dt = datetime(2022, 9, 17, tzinfo=timezone.utc)
to_dt = datetime(2026, 9, 17, tzinfo=timezone.utc)
rates = mt5.copy_rates_range('NAS100', mt5.TIMEFRAME_H1, from_dt, to_dt)
nas = pd.DataFrame(rates)
nas['time'] = pd.to_datetime(nas['time'], unit='s', utc=True)
nas.set_index('time', inplace=True)
nas = nas[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
mt5.shutdown()

print(f'NAS100 data: {len(nas)} bars, {nas.index[0]} → {nas.index[-1]}')

nas_t = nas[(nas.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (nas.index < pd.Timestamp('2026-09-17', tz='UTC'))]
nas_v = nas[(nas.index >= pd.Timestamp('2022-09-17', tz='UTC')) & (nas.index < pd.Timestamp('2024-09-17', tz='UTC'))]
print(f'Tuning (2024-26): {len(nas_t)} bars')
print(f'Validation (2022-24): {len(nas_v)} bars')

# Load current best adx_NAS params
with open('output/adx_best_params.json') as f:
    adx_best = json.load(f)
adx_nas_a = adx_best['nas']
print(f'\nCurrent adx_NAS (Variant A): {adx_nas_a}')

from analysis.optuna_filters import run_strategy
from strategies.adx import ADX_Strategy

# Test current params
m_t = run_strategy(nas_t, ADX_Strategy, adx_nas_a, [], 'nas100')
m_v = run_strategy(nas_v, ADX_Strategy, adx_nas_a, [], 'nas100')
print(f'\nCurrent NAS performance: T ${m_t["net_pnl"]:+,.0f} Sh {m_t["sharpe"]:+.2f} WR {m_t["win_rate"]*100:.1f}% {m_t["n_trades"]}t')
print(f'                         V ${m_v["net_pnl"]:+,.0f} Sh {m_v["sharpe"]:+.2f} WR {m_v["win_rate"]*100:.1f}% {m_v["n_trades"]}t')


def search_variant(prior_params, n_trials=80, label='variant', df=None):
    """Find another ROBUST ADX variant with DIFFERENT params for NAS."""
    if df is None:
        df = nas_t  # module-level train split; explicit df overrides it.
    def obj(trial, df):
        params = {
            'bars_calculate': trial.suggest_int('bars', 5, 25),
            'crossover_lookback': trial.suggest_int('xover', 1, 12),
            'min_crossover_gap': trial.suggest_float('gap', 0.3, 5.0),
            'adx_zone_low': trial.suggest_float('zone_lo', 15.0, 30.0),
            'adx_zone_high': trial.suggest_float('zone_hi', 40.0, 65.0),
            'continuation_level': trial.suggest_float('cont', 12.0, 28.0),
            'reversal_edge': trial.suggest_float('rev', 20.0, 38.0),
            'level_open_orders_1': trial.suggest_float('lv1', 25.0, 60.0),
            'level_open_orders_2': trial.suggest_float('lv2', 10.0, 35.0),
            'sweep_lookback': trial.suggest_int('swp', 3, 25),
            'level_close_orders_1': trial.suggest_float('cl1', 5.0, 25.0),
            'level_close_orders_2': trial.suggest_float('cl2', 1.0, 15.0),
            'use_di_cross': True,
            'open_orders_type': trial.suggest_int('oot', 1, 10),
            'close_orders_type': trial.suggest_int('cot', 1, 10),
        }
        # Penalty for being too similar to prior_params
        if prior_params:
            sim = sum(1 for k in ['bars_calculate', 'crossover_lookback', 'min_crossover_gap', 'adx_zone_low', 'adx_zone_high']
                      if abs(params[k] - prior_params[k]) < 2)
            if sim >= 3:
                return -1e9
        m = run_strategy(df, ADX_Strategy, params, [], 'nas100')
        if 'error' in m or m['n_trades'] < 20:
            return -1e9
        return m['net_pnl'] + 30 * m['sharpe'] + 100 * m['win_rate']
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: obj(t, df), n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def to_full(best):
    return {
        'bars_calculate': best['bars'], 'crossover_lookback': best['xover'],
        'min_crossover_gap': best['gap'], 'adx_zone_low': best['zone_lo'],
        'adx_zone_high': best['zone_hi'], 'continuation_level': best['cont'],
        'reversal_edge': best['rev'], 'level_open_orders_1': best['lv1'],
        'level_open_orders_2': best['lv2'], 'sweep_lookback': best['swp'],
        'level_close_orders_1': best['cl1'], 'level_close_orders_2': best['cl2'],
        'use_di_cross': True, 'open_orders_type': best['oot'], 'close_orders_type': best['cot'],
    }


# Variant B — different from A
print()
print('=' * 80)
print('Searching NAS Variant B (different from A)...')
print('=' * 80)
best_b = search_variant(adx_nas_a, n_trials=100, label='nas_B')
adx_nas_b = to_full(best_b)
m_t = run_strategy(nas_t, ADX_Strategy, adx_nas_b, [], 'nas100')
m_v = run_strategy(nas_v, ADX_Strategy, adx_nas_b, [], 'nas100')
robust_b = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
print(f'NAS Variant B: T ${m_t["net_pnl"]:+,.0f}/Sh {m_t["sharpe"]:+.2f}/WR {m_t["win_rate"]*100:.1f}%/{m_t["n_trades"]}t')
print(f'                V ${m_v["net_pnl"]:+,.0f}/Sh {m_v["sharpe"]:+.2f}/WR {m_v["win_rate"]*100:.1f}%/{m_v["n_trades"]}t | ROBUST: {robust_b}')
print(f'Params: {adx_nas_b}')

# Variant C — different from A and B
print()
print('=' * 80)
print('Searching NAS Variant C (different from A and B)...')
print('=' * 80)
best_c = search_variant(adx_nas_b, n_trials=100, label='nas_C')
adx_nas_c = to_full(best_c)
m_t = run_strategy(nas_t, ADX_Strategy, adx_nas_c, [], 'nas100')
m_v = run_strategy(nas_v, ADX_Strategy, adx_nas_c, [], 'nas100')
robust_c = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
print(f'NAS Variant C: T ${m_t["net_pnl"]:+,.0f}/Sh {m_t["sharpe"]:+.2f}/WR {m_t["win_rate"]*100:.1f}%/{m_t["n_trades"]}t')
print(f'                V ${m_v["net_pnl"]:+,.0f}/Sh {m_v["sharpe"]:+.2f}/WR {m_v["win_rate"]*100:.1f}%/{m_v["n_trades"]}t | ROBUST: {robust_c}')

# Save results
results = {
    'nas_A': {'params': adx_nas_a, 'note': 'original (Sep 9 2026 tuning)'},
    'nas_B': {'params': adx_nas_b, 'robust': robust_b,
              'tuning_pnl': m_t['net_pnl'] if robust_b else None},
    'nas_C': {'params': adx_nas_c, 'robust': robust_c},
}
# Save the new variants
with open('output/adx_nas_variants.json', 'w') as f:
    json.dump({
        'nas_variant_A': {'params': adx_nas_a, 'tuning_pnl': 31999, 'tuning_sharpe': 7.34, 'tuning_wr': 0.816,
                          'val_pnl': 3479, 'val_sharpe': 1.93, 'note': 'original gold star'},
        'nas_variant_B': {'params': adx_nas_b, 'robust': robust_b,
                          'tuning_pnl': None, 'tuning_sharpe': None, 'tuning_wr': None,
                          'val_pnl': None, 'val_sharpe': None, 'val_wr': None,
                          'note': 'NEW this session'},
        'nas_variant_C': {'params': adx_nas_c, 'robust': robust_c,
                          'tuning_pnl': None, 'tuning_sharpe': None, 'tuning_wr': None,
                          'val_pnl': None, 'val_sharpe': None, 'val_wr': None,
                          'note': 'NEW this session'},
    }, f, indent=2)

print('\nSaved → output/adx_nas_variants.json')
