"""Last attempt: ensemble of 3 ADX variants (different params, same algo) for diversification."""
import warnings; warnings.filterwarnings('ignore')
import sys, json
sys.path.insert(0, '.')
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import run_strategy
from strategies._smart_merge import VoteStrategy
from strategies.adx import ADX_Strategy

eur = pd.read_csv('output/mt5_EURUSD_H1_2022_2026.csv', index_col='time', parse_dates=True)
if eur.index.tz is None:
    eur.index = eur.index.tz_localize('UTC')
eur = eur[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
eur_t = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
eur_v = eur[(eur.index >= pd.Timestamp('2022-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2024-09-17', tz='UTC'))]

# 3 ADX variants with different params for diversity
adx_a = {
    'bars_calculate': 10, 'crossover_lookback': 7, 'min_crossover_gap': 0.83,
    'adx_zone_low': 22.5, 'adx_zone_high': 56.0,
    'continuation_level': 21.7, 'reversal_edge': 26.3,
    'level_open_orders_1': 51.6, 'level_open_orders_2': 21.3,
    'sweep_lookback': 16, 'level_close_orders_1': 16.8, 'level_close_orders_2': 3.59,
    'use_di_cross': True, 'open_orders_type': 1, 'close_orders_type': 6,
}

# Search for 2nd and 3rd ADX variants that are different but still ROBUST
def search_variant(prior_params, n_trials=50):
    """Find another ROBUST ADX variant with DIFFERENT params."""
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
        m = run_strategy(df, ADX_Strategy, params, [], 'forex')
        if 'error' in m or m['n_trades'] < 20:
            return -1e9
        return m['net_pnl'] + 30 * m['sharpe'] + 100 * m['win_rate']
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: obj(t, eur_t), n_trials=n_trials, show_progress_bar=False)
    return study.best_params

# Find variant B (different from A)
print('=== Finding variant B (different from A) ===')
best_b = search_variant(adx_a, n_trials=80)
adx_b_full = {
    'bars_calculate': best_b['bars'], 'crossover_lookback': best_b['xover'],
    'min_crossover_gap': best_b['gap'], 'adx_zone_low': best_b['zone_lo'],
    'adx_zone_high': best_b['zone_hi'], 'continuation_level': best_b['cont'],
    'reversal_edge': best_b['rev'], 'level_open_orders_1': best_b['lv1'],
    'level_open_orders_2': best_b['lv2'], 'sweep_lookback': best_b['swp'],
    'level_close_orders_1': best_b['cl1'], 'level_close_orders_2': best_b['cl2'],
    'use_di_cross': True, 'open_orders_type': best_b['oot'], 'close_orders_type': best_b['cot'],
}
m_t = run_strategy(eur_t, ADX_Strategy, adx_b_full, [], 'forex')
m_v = run_strategy(eur_v, ADX_Strategy, adx_b_full, [], 'forex')
robust_b = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
print(f'  Variant B: T ${m_t["net_pnl"]:+,.0f}/Sh {m_t["sharpe"]:+.2f}/WR {m_t["win_rate"]*100:.1f}%/{m_t["n_trades"]}t | V ${m_v["net_pnl"]:+,.0f}/Sh {m_v["sharpe"]:+.2f}/WR {m_v["win_rate"]*100:.1f}%/{m_v["n_trades"]}t | ROBUST: {robust_b}')
print(f'  Params: {adx_b_full}')

# Now combine 3 variants: A (winner), B (new), and another
# Find variant C (different from A and B)
print()
print('=== Finding variant C (different from A and B) ===')
best_c = search_variant(adx_b_full, n_trials=80)
adx_c_full = {
    'bars_calculate': best_c['bars'], 'crossover_lookback': best_c['xover'],
    'min_crossover_gap': best_c['gap'], 'adx_zone_low': best_c['zone_lo'],
    'adx_zone_high': best_c['zone_hi'], 'continuation_level': best_c['cont'],
    'reversal_edge': best_c['rev'], 'level_open_orders_1': best_c['lv1'],
    'level_open_orders_2': best_c['lv2'], 'sweep_lookback': best_c['swp'],
    'level_close_orders_1': best_c['cl1'], 'level_close_orders_2': best_c['cl2'],
    'use_di_cross': True, 'open_orders_type': best_c['oot'], 'close_orders_type': best_c['cot'],
}
m_t = run_strategy(eur_t, ADX_Strategy, adx_c_full, [], 'forex')
m_v = run_strategy(eur_v, ADX_Strategy, adx_c_full, [], 'forex')
robust_c = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
print(f'  Variant C: T ${m_t["net_pnl"]:+,.0f}/Sh {m_t["sharpe"]:+.2f}/WR {m_t["win_rate"]*100:.1f}%/{m_t["n_trades"]}t | V ${m_v["net_pnl"]:+,.0f}/Sh {m_v["sharpe"]:+.2f}/WR {m_v["win_rate"]*100:.1f}%/{m_v["n_trades"]}t | ROBUST: {robust_c}')

# Try ensemble of 3 ADX variants with vote-1 (any fires)
if robust_b and robust_c:
    print()
    print('=== Ensemble: vote-1-of-3 ADX variants ===')
    params = {
        'n_strategies': 3, 'vote_threshold': 1, 'cooldown': 4,
        '_strategy1': ADX_Strategy, '_strategy1_params': adx_a,
        '_strategy2': ADX_Strategy, '_strategy2_params': adx_b_full,
        '_strategy3': ADX_Strategy, '_strategy3_params': adx_c_full,
    }
    m_t = run_strategy(eur_t, VoteStrategy, params, [], 'forex')
    m_v = run_strategy(eur_v, VoteStrategy, params, [], 'forex')
    robust_e = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
    total = m_t['n_trades'] + m_v['n_trades']
    print(f'  Ensemble: T ${m_t["net_pnl"]:+,.0f}/Sh {m_t["sharpe"]:+.2f}/WR {m_t["win_rate"]*100:.1f}%/{m_t["n_trades"]}t | V ${m_v["net_pnl"]:+,.0f}/Sh {m_v["sharpe"]:+.2f}/WR {m_v["win_rate"]*100:.1f}%/{m_v["n_trades"]}t | ROBUST: {robust_e} | Total: {total}')

# Try vote-2-of-3 (majority) for higher quality
    print()
    print('=== Ensemble: vote-2-of-3 (majority) ===')
    params['vote_threshold'] = 2
    params['cooldown'] = 8
    m_t = run_strategy(eur_t, VoteStrategy, params, [], 'forex')
    m_v = run_strategy(eur_v, VoteStrategy, params, [], 'forex')
    robust_e = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
    total = m_t['n_trades'] + m_v['n_trades']
    print(f'  Majority: T ${m_t["net_pnl"]:+,.0f}/Sh {m_t["sharpe"]:+.2f}/WR {m_t["win_rate"]*100:.1f}%/{m_t["n_trades"]}t | V ${m_v["net_pnl"]:+,.0f}/Sh {m_v["sharpe"]:+.2f}/WR {m_v["win_rate"]*100:.1f}%/{m_v["n_trades"]}t | ROBUST: {robust_e} | Total: {total}')

# Save best ensemble
    if robust_e:
        with open('output/best_ensemble.json', 'w') as f:
            json.dump({'adx_a': adx_a, 'adx_b': adx_b_full, 'adx_c': adx_c_full,
                       'cooldown': 8, 'vote_threshold': 2,
                       'tuning_pnl': m_t['net_pnl'], 'tuning_sharpe': m_t['sharpe'], 'tuning_trades': m_t['n_trades'],
                       'val_pnl': m_v['net_pnl'], 'val_sharpe': m_v['sharpe'], 'val_trades': m_v['n_trades']}, f, indent=2, default=str)
        print(f'\n  ✅ Saved → output/best_ensemble.json')
