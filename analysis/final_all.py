"""FINAL consolidated report — all locked strategies across all sessions."""
import warnings; warnings.filterwarnings('ignore')
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
import pandas as pd
import importlib

OOS_2Y_START = pd.Timestamp('2024-09-17', tz='UTC')
OOS_2Y_END = pd.Timestamp('2026-09-17', tz='UTC')


def resolve_class(sn):
    """Find the strategy class for a given name."""
    # Check if it's a V5 (dem_fbb_v5)
    if sn in ('fbb_v5', 'dem_v5'):
        cls_name = 'FBBV5Strategy' if sn == 'fbb_v5' else 'DeMV5Strategy'
        return getattr(importlib.import_module('strategies.dem_fbb_v5'), cls_name)
    # V4
    if sn == 'macd_confluence_v4':
        return getattr(importlib.import_module('strategies.trend_follow_v4_more'), 'MACDConfluenceV4')
    if sn == 'bb_rsi_v4':
        return getattr(importlib.import_module('strategies.trend_follow_v4_more'), 'BBRsiV4')
    # four_user_strategies
    four_map = {
        'bachelier_wave': ('strategies.four_user_strategies', 'BachelierWaveStrategy'),
        'macd_inst': ('strategies.four_user_strategies', 'MACDInstitutionalStrategy'),
        'ma8_ribbon': ('strategies.four_user_strategies', 'MA8RibbonStrategy'),
        'rc44': ('strategies.four_user_strategies', 'RC44Strategy'),
    }
    if sn in four_map:
        mod, cls = four_map[sn]
        return getattr(importlib.import_module(mod), cls)
    # screener + linda
    screener_map = {
        'sc_l0_stoch': ('strategies.screener_and_linda', 'ScreenerL0Strategy'),
        'sc_s5_stoch': ('strategies.screener_and_linda', 'ScreenerS5Strategy'),
        'sc_l1_macross': ('strategies.screener_and_linda', 'ScreenerL1Strategy'),
        'sc_s6_macross': ('strategies.screener_and_linda', 'ScreenerS6Strategy'),
        'sc_l2_macd': ('strategies.screener_and_linda', 'ScreenerL2Strategy'),
        'sc_s7_macd': ('strategies.screener_and_linda', 'ScreenerS7Strategy'),
        'sc_l3_bb': ('strategies.screener_and_linda', 'ScreenerL3Strategy'),
        'sc_s8_bb': ('strategies.screener_and_linda', 'ScreenerS8Strategy'),
        'sc_l4_supertrend': ('strategies.screener_and_linda', 'ScreenerL4Strategy'),
        'sc_s9_supertrend': ('strategies.screener_and_linda', 'ScreenerS9Strategy'),
        'linda_macd': ('strategies.screener_and_linda', 'LindaMACDStrategy'),
    }
    if sn in screener_map:
        mod, cls = screener_map[sn]
        return getattr(importlib.import_module(mod), cls)
    # Original V1 strategies
    from strategies import MULTI_STRAT_EA_REGISTRY
    if sn in MULTI_STRAT_EA_REGISTRY:
        return MULTI_STRAT_EA_REGISTRY[sn]
    return None


def looks_like_params(v):
    if not isinstance(v, dict): return False
    if len(v) < 2: return False
    return sum(1 for k, val in v.items() if isinstance(val, (int, float, bool, str))) >= 2


# Load all winning params from all _best*.json files
all_params = {}
for f in Path('output').glob('*_best*.json'):
    name = f.stem.replace('_best_params', '').replace('_best', '')
    if f.name in ('v4_best_params.json', 'v4_more_best_params.json'): continue
    if 'genetic' in f.name: continue
    try:
        data = json.load(open(f))
        if isinstance(data, dict):
            for ticker in ('eur', 'nas'):
                if ticker in data and isinstance(data[ticker], dict):
                    if 'params' in data[ticker]:
                        all_params[name + '_' + ticker.upper()] = data[ticker]['params']
                    elif looks_like_params(data[ticker]):
                        all_params[name + '_' + ticker.upper()] = data[ticker]
            for k, v in data.items():
                if isinstance(v, dict) and 'params' in v and looks_like_params(v['params']):
                    all_params[k] = v['params']
    except Exception:
        pass
# V4 files
for f in ['v4_best_params.json', 'v4_more_best_params.json']:
    p = Path('output/' + f)
    if not p.exists(): continue
    try:
        data = json.load(open(p))
        for strat_name, ticker_data in data.items():
            if not isinstance(ticker_data, dict): continue
            for ticker in ('eur', 'nas'):
                if ticker in ticker_data and isinstance(ticker_data[ticker], dict):
                    if 'params' in ticker_data[ticker]:
                        all_params[strat_name + '_' + ticker.upper()] = ticker_data[ticker]['params']
    except Exception:
        pass


print('=' * 100)
print('FINAL CONSOLIDATION — ALL strategies across ALL sessions, 2Y OOS revalidation')
print('=' * 100)

from analysis.optuna_filters import fetch_h1, run_strategy

eur = fetch_h1('D:/MT5_EuroPrinter/terminal64.exe', 'EURUSD')
nas = fetch_h1('D:/MT5_Bybit/terminal64.exe', 'NAS100')
eur_oos = eur[(eur.index >= OOS_2Y_START) & (eur.index < OOS_2Y_END)]
nas_oos = nas[(nas.index >= OOS_2Y_START) & (nas.index < OOS_2Y_END)]

# Dedupe and revalidate
seen = set()
results = []
errors = []
for key in sorted(all_params.keys()):
    if '_' not in key: continue
    parts = key.rsplit('_', 1)
    if len(parts) != 2: continue
    strat, ticker = parts
    if (strat, ticker) in seen: continue
    seen.add((strat, ticker))
    params = all_params.get(key)
    if not params or not looks_like_params(params): continue
    cls = resolve_class(strat)
    if cls is None: continue
    df = eur_oos if ticker.upper() == 'EUR' else nas_oos
    profile = 'forex' if ticker.upper() == 'EUR' else 'nas100'
    try:
        r = run_strategy(df, cls, params, [], profile)
        if 'error' not in r:
            r['strategy'] = strat
            r['ticker'] = ticker.upper()
            r['params'] = params
            results.append(r)
        else:
            errors.append((key, r['error']))
    except Exception as e:
        errors.append((key, str(e)[:50]))

results.sort(key=lambda r: -r['sharpe'])

print(f'\nLoaded {len(results)} instances, {len(errors)} errors\n')
print(f"{'Strategy':<28} {'Asset':<5} {'PnL':>10} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'Tr':>4} {'Status':<10}")
print('-' * 90)
for r in results:
    st = 'ROBUST' if (r['net_pnl'] > 0 and r['sharpe'] > 0) else 'X'
    print(f"{r['strategy']:<28} {r['ticker']:<5} ${r['net_pnl']:>+9,.0f} {r['sharpe']:>+6.2f} "
          f"{r['win_rate']*100:>4.0f}% {r['profit_factor']:>4.2f} {r['n_trades']:>4} {st:<10}")

# LOCKED
locked = [r for r in results if r['net_pnl'] > 0 and r['sharpe'] > 0]
locked.sort(key=lambda r: -r['sharpe'])

print('\n' + '=' * 100)
print(f"FINAL LOCKED — {len(locked)} profitable instances, {len(set(r['strategy'] for r in locked))} unique strategies")
print('=' * 100)
total_pnl = sum(r['net_pnl'] for r in locked)
total_trades = sum(r['n_trades'] for r in locked)
print(f'\n{"Strategy":<28} {"Asset":<5} {"PnL":>10} {"Sharpe":>7} {"WR":>5} {"PF":>5} {"Tr":>4} {"TPY":>5}')
print('-' * 90)
for r in locked:
    tpy = r['n_trades'] / 2.0
    print(f"{r['strategy']:<28} {r['ticker']:<5} ${r['net_pnl']:>+9,.0f} {r['sharpe']:>+6.2f} "
          f"{r['win_rate']*100:>4.0f}% {r['profit_factor']:>4.2f} {r['n_trades']:>4} {tpy:>5.1f}")

unique = set(r['strategy'] for r in locked)
print(f'\n  Total instances: {len(locked)}')
print(f'  Unique strategies: {len(unique)}')
print(f'  Combined PnL (2Y OOS): ${total_pnl:+,.0f}')
print(f'  Combined trades: {total_trades}')
if locked:
    avg_sh = sum(r['sharpe'] for r in locked) / len(locked)
    avg_wr = sum(r['win_rate']*100 for r in locked) / len(locked)
    avg_pf = sum(r['profit_factor'] for r in locked) / len(locked)
    print(f'  Avg Sharpe: {avg_sh:.2f}')
    print(f'  Avg WR: {avg_wr:.1f}%')
    print(f'  Avg PF: {avg_pf:.2f}')

# Show "great results" subset (Sharpe > 1.0)
great = [r for r in locked if r['sharpe'] > 1.0]
print(f'\n  >>> STRATEGIES WITH GREAT RESULTS (Sharpe > 1.0): {len(great)} <<<')
for r in great:
    print(f'      {r["strategy"]}_{r["ticker"]:<10} Sharpe {r["sharpe"]:+.2f} | PnL ${r["net_pnl"]:+,.0f} | WR {r["win_rate"]*100:.1f}% | PF {r["profit_factor"]:.2f}')

# Save final
out_dir = Path('output/profiles_final')
out_dir.mkdir(parents=True, exist_ok=True)
for r in locked:
    profile = {
        'strategy': r['strategy'],
        'ticker': r['ticker'],
        'timeframe': 'H1',
        'oos_window': {'start': OOS_2Y_START.strftime('%Y-%m-%d'),
                       'end': OOS_2Y_END.strftime('%Y-%m-%d'), 'months': 24},
        'metrics': {k: v for k, v in r.items() if k not in ('strategy', 'ticker', 'params')},
        'params': r['params'],
        'deployment': {
            'single_chart': True,
            'multi_instance_supported': True,
            'source': 'mt5_ea' if r['strategy'] in ['adx', 'ac_ao', 'ms', 'mfi', 'dem', 'fbb'] else 'python_port',
            'mt5_ea_name': 'TwelveStrategies.mq5' if r['strategy'] in ['adx', 'ac_ao', 'ms', 'mfi', 'dem', 'fbb'] else None,
        }
    }
    out_path = out_dir / (r['strategy'] + '_' + r['ticker'] + '.json')
    with open(out_path, 'w') as f:
        json.dump(profile, f, indent=2)
print(f'\nSaved {len(locked)} final profiles to {out_dir}')
