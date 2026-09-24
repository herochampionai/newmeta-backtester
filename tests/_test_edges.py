"""Test the 6 new edges on EURUSD H1 (MT5 data) + NAS100 H1 (yfinance data)."""
import warnings; warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '.')
import pandas as pd
from analysis.optuna_filters import run_strategy
from strategies.edges_2026 import (
    LiquiditySweepStrategy, VolatilityRegimeStrategy, SessionBasedStrategy,
    MarketStructureStrategy, SynthOrderFlowStrategy, EnsembleStrategy
)

# Load data
eur = pd.read_csv('output/mt5_EURUSD_H1_2022_2026.csv', index_col='time', parse_dates=True)
if eur.index.tz is None:
    eur.index = eur.index.tz_localize('UTC')
eur = eur[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
eur_t = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
eur_v = eur[(eur.index >= pd.Timestamp('2022-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2024-09-17', tz='UTC'))]

# NAS data from yfinance
nas = pd.read_csv('output/NDX_NAS100_H1_2024_2026.csv', index_col='Datetime')
nas.index = pd.to_datetime(nas.index, utc=True)
nas.columns = [c.lower().replace(' ', '_') for c in nas.columns]
# Rename to standard
col_map = {'adj_close': 'adj_close'}
nas = nas.rename(columns=col_map)
# Get the standard columns
nas = nas[['open', 'high', 'low', 'close', 'volume']]
nas_t = nas[(nas.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (nas.index < pd.Timestamp('2026-09-17', tz='UTC'))]

print(f'EURUSD: {len(eur_t)} tuning, {len(eur_v)} val')
print(f'NAS100: {len(nas_t)} bars')


def default_params():
    return {}


# Test each strategy on EURUSD first
print()
print('=' * 110)
print('6 NEW EDGES — EURUSD H1 default params')
print('=' * 110)
print(f'{"Edge":25s} | {"Tuning":30s} | {"Validation":30s} | ROBUST')
print('-' * 110)

for name, cls in [
    ('#1 Liq Sweep',     LiquiditySweepStrategy),
    ('#2 Vol Regime',    VolatilityRegimeStrategy),
    ('#3 Session',       SessionBasedStrategy),
    ('#4 Market Struct', MarketStructureStrategy),
    ('#5 Ensemble',      EnsembleStrategy),
    ('#6 Synth OF',      SynthOrderFlowStrategy),
]:
    m_t = run_strategy(eur_t, cls, default_params(), [], 'forex')
    m_v = run_strategy(eur_v, cls, default_params(), [], 'forex')
    if 'error' in m_t or 'error' in m_v:
        print(f'{name:25s} | ERROR: T={m_t.get("error", "n/a")}')
        continue
    robust = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
    t_str = f'${m_t["net_pnl"]:+7,.0f}/Sh {m_t["sharpe"]:+.2f}/{m_t["n_trades"]}t'
    v_str = f'${m_v["net_pnl"]:+7,.0f}/Sh {m_v["sharpe"]:+.2f}/{m_v["n_trades"]}t'
    flag = 'YES' if robust else 'X'
    print(f'{name:25s} | {t_str:30s} | {v_str:30s} | {flag}')


# Test on NAS100 too
print()
print('=' * 110)
print('6 NEW EDGES — NAS100 H1 default params')
print('=' * 110)
print(f'{"Edge":25s} | NAS100 2Y: PnL / Sh / Trades | ROBUST (vs original $32k)')
print('-' * 110)

for name, cls in [
    ('#1 Liq Sweep',     LiquiditySweepStrategy),
    ('#2 Vol Regime',    VolatilityRegimeStrategy),
    ('#3 Session',       SessionBasedStrategy),
    ('#4 Market Struct', MarketStructureStrategy),
    ('#5 Ensemble',      EnsembleStrategy),
    ('#6 Synth OF',      SynthOrderFlowStrategy),
]:
    m_t = run_strategy(nas_t, cls, default_params(), [], 'nas100')
    if 'error' in m_t:
        print(f'{name:25s} | ERROR: T={m_t.get("error", "n/a")}')
        continue
    robust = m_t['net_pnl'] > 0
    t_str = f'${m_t["net_pnl"]:+7,.0f}/Sh {m_t["sharpe"]:+.2f}/{m_t["n_trades"]}t'
    flag = 'YES' if robust else 'X'
    print(f'{name:25s} | {t_str:42s} | {flag}')
