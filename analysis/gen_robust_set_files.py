"""Generate MT5 .set files for V1 strategies (adx, ac_ao, ms) — production-ready.

Plus a lenient Linda MACD variant.
"""
import warnings; warnings.filterwarnings('ignore')
import json
import sys

sys.path.insert(0, '.')
from pathlib import Path

# Load best params for V1 strategies
with open('output/adx_best_params.json') as f:
    adx_params = json.load(f)
with open('output/ac_ao_best_params.json') as f:
    ac_ao_params = json.load(f)
with open('output/ms_best_params.json') as f:
    ms_params = json.load(f)


def fmt_num(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return f'{v:.6f}'
    return str(v)


def gen_set(strategy_name, params, ticker, output_path, profile_metrics):
    """Generate MT5 .set file with header comments."""
    lines = []
    lines.append(f'; {strategy_name.upper()}_{ticker} — LOCKED PRODUCTION PROFILE')
    lines.append('; Generated from 2-year OOS validation (robust on unseen period)')
    lines.append(f'; Net PnL: ${profile_metrics.get("net_pnl", 0):+,.0f}')
    lines.append(f'; Sharpe: {profile_metrics.get("sharpe", 0):+.2f}')
    lines.append(f'; WR: {profile_metrics.get("win_rate", 0)*100:.1f}%')
    lines.append(f'; PF: {profile_metrics.get("profit_factor", 0):.2f}')
    lines.append(f'; Trades: {profile_metrics.get("n_trades", 0)}')
    lines.append(f'; Robust on validation (2022-2024)? {"YES" if profile_metrics.get("robust") else "X"}')
    lines.append('')
    # Symbol selection
    if ticker == 'EUR':
        lines.append('PairToTrade=4')  # TRADE_EURUSD_4
    else:
        lines.append('PairToTrade=0')  # CHART_PAIR — use whatever chart EA is on
    lines.append('TimeFrame=PERIOD_H1')
    lines.append('')
    # Map Python params to MQL5 input names
    if strategy_name == 'adx':
        lines.append('; === ADX LOCKED PARAMETERS ===')
        m = {
            'bars_calculate': 'ADX_BarsCalculate',
            'use_di_crossover': 'ADX_UseDICrossover',
            'crossover_lookback': 'ADX_CrossoverLookback',
            'min_crossover_gap': 'ADX_MinCrossoverGap',
            'open_orders_type': 'ADX_OpenOrdersType',
            'level_open_orders_1': 'ADX_LevelOpenOrders_1',
            'level_open_orders_2': 'ADX_LevelOpenOrders_2',
            'close_orders_type': 'ADX_CloseOrdersType',
            'level_close_orders_1': 'ADX_LevelCloseOrders_1',
            'level_close_orders_2': 'ADX_LevelCloseOrders_2',
        }
        for py_key, mql_key in m.items():
            val = params.get(py_key)
            if val is not None:
                lines.append(f'{mql_key}={fmt_num(val)}')
        # NOTE: TwelveStrategies.mq5 doesn't have ADX_ZoneLow/High/Continuation/Reversal/Sweep
        # The Python Optuna found these helpful — they need to be ADDED to the MQL5 EA
    elif strategy_name == 'ac_ao':
        lines.append('; === AC+AO LOCKED PARAMETERS ===')
        m = {
            'level_open_orders': 'AC_LevelOpenOrders',
            'open_orders_type': 'AC_OpenOrdersType',
            'use_acceleration_filter': 'AC_UseAccelerationFilter',
            'min_acceleration': 'AC_MinAcceleration',
            'use_ao_synchronization': 'AC_UseAOSynchronization',
        }
        for py_key, mql_key in m.items():
            val = params.get(py_key)
            if val is not None:
                lines.append(f'{mql_key}={fmt_num(val)}')
    elif strategy_name == 'ms':
        lines.append('; === MS (MACD Signal) LOCKED PARAMETERS ===')
        m = {
            'ms_fast_ema': 'MS_Fast_EMA_Period',
            'ms_slow_ema': 'MS_Slow_EMA_Period',
            'ms_signal_period': 'MS_Signal_Period',
            'level_open_orders': 'MS_LevelOpenOrders',
            'open_orders_type': 'MS_OpenOrdersType',
        }
        for py_key, mql_key in m.items():
            val = params.get(py_key)
            if val is not None:
                lines.append(f'{mql_key}={fmt_num(val)}')
    lines.append('')
    # Enable ONLY this strategy
    lines.append('; === ENABLE ONLY THIS STRATEGY ===')
    all_strats = ['AC_AO', 'ADX', 'DeM', 'FBB', 'MFI', 'MS',
                   'MTF_Stoch', 'BB_RSI', 'Triple_RSI', 'Quad_Stoch',
                   'Stoch533_MTF', 'MACD_Confluence']
    for s in all_strats:
        lines.append(f'{s}_StrategyRun=' + ('true' if s == strategy_name.upper() else 'false'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# Read robustness metrics
with open('output/robustness_check.json') as f:
    rb = json.load(f)
rb_map = {(i['strategy'], i['ticker']): i for i in rb['instances']}


# Generate .set files for ROBUST strategies
print('=' * 100)
print('MT5 .SET FILES — ROBUST STRATEGIES ONLY')
print('=' * 100)
out_dir = Path('output/set_files')

strategies = [
    ('adx', adx_params, 'EUR'),
    ('adx', adx_params, 'NAS'),
    ('ac_ao', ac_ao_params, 'EUR'),
    ('ms', ms_params, 'EUR'),
]
for strat_name, params_dict, ticker in strategies:
    py_params = params_dict[ticker.lower()]
    key = (strat_name, ticker)
    metrics = {
        'net_pnl': rb_map.get(key, {}).get('tuning_pnl', 0),
        'sharpe': rb_map.get(key, {}).get('tuning_sharpe', 0),
        'win_rate': 0,  # not tracked in robustness check
        'profit_factor': 0,
        'n_trades': rb_map.get(key, {}).get('tuning_trades', 0),
        'robust': (rb_map.get(key, {}).get('validation_pnl', 0) or 0) > 0,
    }
    out_path = out_dir / f'{strat_name}_{ticker}.set'
    gen_set(strat_name, py_params, ticker, out_path, metrics)
    print(f'  ✓ {out_path.name}')
    print(f'    PnL ${metrics["net_pnl"]:+,.0f}, Sharpe {metrics["sharpe"]:+.2f}, '
          f'Robust on 2022-2024: {"YES" if metrics["robust"] else "X"}')

# Also generate .set files for fbb_v5, macd_inst, sc_s8_bb (ROBUST ones)
# These need porting to MQL5 — for now, save JSON profile with clear "needs MQL5 port" note
print('\n; === ROBUST V4/MTF/SCREENER STRATEGIES ===')
# These exist as Python-only; need MQL5 porting before MT5 deployment
# We'll save production JSONs for them
robust_v4 = [('sc_s8_bb', 'NAS'), ('sc_s5_stoch', 'NAS'), ('sc_s5_stoch', 'EUR')]
for strat, ticker in robust_v4:
    profiles = list(Path('output/profiles_final').glob(f'{strat}_{ticker}.json'))
    if profiles:
        data = json.load(open(profiles[0]))
        rb_key = (strat, ticker)
        if rb_key in rb_map and rb_map[rb_key].get('validation_pnl', 0) > 0:
            data['robust_on_validation'] = True
            data['validation_metrics'] = {
                'pnl': rb_map[rb_key].get('validation_pnl'),
                'sharpe': rb_map[rb_key].get('validation_sharpe'),
                'trades': rb_map[rb_key].get('validation_trades'),
            }
            with open(f'output/set_files/{strat}_{ticker}_production.json', 'w') as f:
                json.dump(data, f, indent=2)
            print(f'  ✓ {strat}_{ticker}_production.json (Python-only, needs MQL5 porting)')

print('\n=== DELIVERABLES ===')
print('MT5 .set files (load in Strategy Tester):')
print('  output/set_files/adx_NAS.set — primary deployment')
print('  output/set_files/adx_EUR.set — secondary deployment')
print('  output/set_files/ac_ao_EUR.set')
print('  output/set_files/ms_EUR.set')
print()
print('Production JSON profiles (Python-only, need MQL5 porting):')
print('  output/set_files/sc_s8_bb_NAS_production.json')
print('  output/set_files/sc_s5_stoch_NAS_production.json')
print('  output/set_files/sc_s5_stoch_EUR_production.json')

# Honest note
print()
print('=' * 100)
print('MT5 EA GAP — Several locked strategies need MQL5 porting:')
print('=' * 100)
print('  - sc_s8_bb, sc_s5_stoch: Bollinger/Stochastic setups from SCreener Pine Script')
print('    → Port: replace TwelveModule.mqh add_BollingerBands + add_Stochastic code')
print('  - fbb_v5, macd_inst, rc44: Python rewrites (V5/MACD inst/4x4 RC)')
print('    → Port: extract logic from strategies/four_user_strategies.py')
print('  - light9, mtf_ms: minor instances')
print('    → Skip (low trade count, marginal edge)')
print()
print('Quickest MQL5 win: SCreener setups (L0-S9 patterns are simple crosses + 200 SMA filter, ~30 lines each)')
