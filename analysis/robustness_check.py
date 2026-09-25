"""Robustness check: validate all 20 locked instances on PRIOR OOS (2022-09 to 2024-09).

This tests if the strategies are robust on a DIFFERENT market period.
If a strategy only works on 2024-2026 but fails on 2022-2024, it's over-fit.

Period 1 (tuning): 2024-09 → 2026-09
Period 2 (validation): 2022-09 → 2024-09  ← THIS ONE

If both periods profit, the strategy is ROBUST.
If only Period 1 profits, it's likely OVERFIT.
"""
import warnings; warnings.filterwarnings('ignore')
import json
import sys

sys.path.insert(0, '.')
import importlib
from pathlib import Path

import pandas as pd

VALIDATION_START = pd.Timestamp('2022-09-17', tz='UTC')
VALIDATION_END = pd.Timestamp('2024-09-17', tz='UTC')


def resolve_class(sn):
    if sn in ('fbb_v5', 'dem_v5'):
        cls_name = 'FBBV5Strategy' if sn == 'fbb_v5' else 'DeMV5Strategy'
        return getattr(importlib.import_module('strategies.dem_fbb_v5'), cls_name)
    if sn == 'macd_confluence_v4':
        return getattr(importlib.import_module('strategies.trend_follow_v4_more'), 'MACDConfluenceV4')
    if sn == 'bb_rsi_v4':
        return getattr(importlib.import_module('strategies.trend_follow_v4_more'), 'BBRsiV4')
    four_map = {
        'bachelier_wave': ('strategies.four_user_strategies', 'BachelierWaveStrategy'),
        'macd_inst': ('strategies.four_user_strategies', 'MACDInstitutionalStrategy'),
        'rc44': ('strategies.four_user_strategies', 'RC44Strategy'),
    }
    if sn in four_map:
        mod, cls = four_map[sn]
        return getattr(importlib.import_module(mod), cls)
    screener_map = {
        'sc_s5_stoch': ('strategies.screener_and_linda', 'ScreenerS5Strategy'),
        'sc_s6_macross': ('strategies.screener_and_linda', 'ScreenerS6Strategy'),
        'sc_s7_macd': ('strategies.screener_and_linda', 'ScreenerS7Strategy'),
        'sc_s8_bb': ('strategies.screener_and_linda', 'ScreenerS8Strategy'),
    }
    if sn in screener_map:
        mod, cls = screener_map[sn]
        return getattr(importlib.import_module(mod), cls)
    from strategies import MULTI_STRAT_EA_REGISTRY
    if sn in MULTI_STRAT_EA_REGISTRY:
        return MULTI_STRAT_EA_REGISTRY[sn]
    return None


def looks_like_params(v):
    if not isinstance(v, dict): return False
    if len(v) < 2: return False
    return sum(1 for k, val in v.items() if isinstance(val, (int, float, bool, str))) >= 2


# Load all profiles from final consolidation
all_params = {}
profiles_dir = Path('output/profiles_final')
if profiles_dir.exists():
    for f in profiles_dir.glob('*.json'):
        try:
            data = json.load(open(f))
            key = f"{data['strategy']}_{data['ticker']}"
            all_params[key] = data.get('params', {})
        except Exception:
            pass

print('=' * 110)
print('ROBUSTNESS CHECK — same params on DIFFERENT OOS period')
print('Tuning period:  2024-09 → 2026-09 (the period Optuna was tuned on)')
print('Validation:    2022-09 → 2024-09 (NEW period, unseen by Optuna)')
print('=' * 110)

from analysis.optuna_filters import fetch_h1, run_strategy

eur = fetch_h1('D:/MT5_EuroPrinter/terminal64.exe', 'EURUSD')
nas = fetch_h1('D:/MT5_Bybit/terminal64.exe', 'NAS100')
eur_val = eur[(eur.index >= VALIDATION_START) & (eur.index < VALIDATION_END)]
nas_val = nas[(nas.index >= VALIDATION_START) & (nas.index < VALIDATION_END)]
print(f'EUR validation: {len(eur_val)} bars')
print(f'NAS validation: {len(nas_val)} bars')

# Run each locked instance on the prior period
results = []
for key, params in sorted(all_params.items()):
    if '_' not in key: continue
    parts = key.rsplit('_', 1)
    if len(parts) != 2: continue
    strat, ticker = parts
    if not looks_like_params(params): continue
    cls = resolve_class(strat)
    if cls is None: continue
    df = eur_val if ticker.upper() == 'EUR' else nas_val
    profile = 'forex' if ticker.upper() == 'EUR' else 'nas100'
    try:
        r = run_strategy(df, cls, params, [], profile)
        if 'error' not in r:
            r['strategy'] = strat
            r['ticker'] = ticker.upper()
            r['params'] = params
            results.append(r)
    except Exception:
        pass

# Show comparison: Tuning period vs Validation period
print('\n' + '=' * 110)
print('COMPARISON: tuning period (2024-09 → 2026-09) vs validation period (2022-09 → 2024-09)')
print('=' * 110)
print(f"{'Strategy':<28} {'Asset':<5} {'─ Tuning ─':>28} {'─ Validation ─':>28} {'Robust?'}")
print(f"{'':<28} {'':<5} {'PnL':>10} {'Sharpe':>7} {'WR':>5} {'PF':>5}    {'PnL':>10} {'Sharpe':>7} {'WR':>5} {'PF':>5}")
print('-' * 130)

# Need to re-run tuning period for direct comparison
tuning_results = []
for key, params in sorted(all_params.items()):
    if '_' not in key: continue
    parts = key.rsplit('_', 1)
    if len(parts) != 2: continue
    strat, ticker = parts
    if not looks_like_params(params): continue
    cls = resolve_class(strat)
    if cls is None: continue
    df_t = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))] if ticker.upper() == 'EUR' else nas[(nas.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (nas.index < pd.Timestamp('2026-09-17', tz='UTC'))]
    profile = 'forex' if ticker.upper() == 'EUR' else 'nas100'
    try:
        r = run_strategy(df_t, cls, params, [], profile)
        if 'error' not in r:
            r['strategy'] = strat
            r['ticker'] = ticker.upper()
            tuning_results.append(r)
    except Exception:
        pass

# Map by key
tuning_map = {(r['strategy'], r['ticker']): r for r in tuning_results}
val_map = {(r['strategy'], r['ticker']): r for r in results}

# Sort by combined PnL
all_keys = set(tuning_map.keys()) | set(val_map.keys())
for key in sorted(all_keys, key=lambda k: -(tuning_map.get(k, {}).get('sharpe', 0) + val_map.get(k, {}).get('sharpe', 0))):
    s, t = key
    t_r = tuning_map.get(key, {})
    v_r = val_map.get(key, {})
    t_pnl = t_r.get('net_pnl', 0)
    t_sh = t_r.get('sharpe', 0)
    t_wr = t_r.get('win_rate', 0) * 100
    t_pf = t_r.get('profit_factor', 0)
    v_pnl = v_r.get('net_pnl', 0)
    v_sh = v_r.get('sharpe', 0)
    v_wr = v_r.get('win_rate', 0) * 100
    v_pf = v_r.get('profit_factor', 0)
    robust = 'YES' if (v_pnl > 0 and v_sh > 0) else ('OK' if v_pnl > 0 else 'NO')
    print(f"{s:<28} {t:<5} ${t_pnl:>+9,.0f} {t_sh:>+6.2f} {t_wr:>4.0f}% {t_pf:>4.2f}    ${v_pnl:>+9,.0f} {v_sh:>+6.2f} {v_wr:>4.0f}% {v_pf:>4.2f}  {robust}")

# Summary
print('\n' + '=' * 110)
print('ROBUSTNESS SUMMARY')
print('=' * 110)
both_profit = sum(1 for k in all_keys if tuning_map.get(k, {}).get('net_pnl', 0) > 0 and val_map.get(k, {}).get('net_pnl', 0) > 0)
both_robust = sum(1 for k in all_keys if tuning_map.get(k, {}).get('sharpe', 0) > 0 and val_map.get(k, {}).get('sharpe', 0) > 0)
tuning_only = sum(1 for k in all_keys if tuning_map.get(k, {}).get('net_pnl', 0) > 0 and val_map.get(k, {}).get('net_pnl', 0) <= 0)
val_only = sum(1 for k in all_keys if tuning_map.get(k, {}).get('net_pnl', 0) <= 0 and val_map.get(k, {}).get('net_pnl', 0) > 0)
losing = sum(1 for k in all_keys if tuning_map.get(k, {}).get('net_pnl', 0) <= 0 and val_map.get(k, {}).get('net_pnl', 0) <= 0)
print(f'  ✓ ROBUST (profit on BOTH periods):  {both_robust}/{len(all_keys)}')
print(f'  ◐ Profitable on both (PnL>0):       {both_profit}/{len(all_keys)}')
print(f'  ✗ Tuning-only profit:                {tuning_only}/{len(all_keys)}  (likely OVERFIT)')
print(f'  ✗ Validation-only profit:            {val_only}/{len(all_keys)}')
print(f'  ✗ Losing on both:                    {losing}/{len(all_keys)}')

# Aggregate PnL comparison
total_tuning = sum(r.get('net_pnl', 0) for r in tuning_results)
total_val = sum(r.get('net_pnl', 0) for r in results)
print('\n  Combined PnL:')
print(f'    Tuning period (2024-09 → 2026-09):    ${total_tuning:+,.0f}')
print(f'    Validation period (2022-09 → 2024-09): ${total_val:+,.0f}')
print(f'    Total combined:                       ${total_tuning + total_val:+,.0f}')

# Save
out_dir = Path('output')
with open(out_dir / 'robustness_check.json', 'w') as f:
    json.dump({
        'tuning_period': {'start': '2024-09-17', 'end': '2026-09-17', 'total_pnl': total_tuning},
        'validation_period': {'start': '2022-09-17', 'end': '2024-09-17', 'total_pnl': total_val},
        'instances': [
            (
                {**{'strategy': k[0], 'ticker': k[1],
                  'tuning_pnl': tuning_map[k].get('net_pnl', 0),
                  'tuning_sharpe': tuning_map[k].get('sharpe', 0),
                  'tuning_trades': tuning_map[k].get('n_trades', 0),
                  'validation_pnl': val_map[k].get('net_pnl', 0) if k in val_map else None,
                  'validation_sharpe': val_map[k].get('sharpe', 0) if k in val_map else None,
                  'validation_trades': val_map[k].get('n_trades', 0) if k in val_map else None}}
                if k in tuning_map else
                {'strategy': k[0], 'ticker': k[1],
                 'tuning_pnl': None, 'validation_pnl': val_map[k].get('net_pnl', 0),
                 'validation_sharpe': val_map[k].get('sharpe', 0),
                 'validation_trades': val_map[k].get('n_trades', 0)}
            )
            for k in all_keys
        ]
    }, f, indent=2, default=str)
print('\n  Saved → output/robustness_check.json')
