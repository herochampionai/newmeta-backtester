"""Test top 5 research strategies on EURUSD H1."""
import warnings; warnings.filterwarnings('ignore')
import sys, json
sys.path.insert(0, '.')
import pandas as pd
from analysis.optuna_filters import run_strategy
from strategies.top5_research import (
    RegimeSwitchingEngineStrategy, VolatilityBreakoutStrategy,
    AdaptiveADXStrategy, CompressionExpansionStrategy, TurtleDonchianStrategy
)

eur = pd.read_csv('output/mt5_EURUSD_H1_2022_2026.csv', index_col='time', parse_dates=True)
if eur.index.tz is None:
    eur.index = eur.index.tz_localize('UTC')
eur = eur[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
eur_t = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
eur_v = eur[(eur.index >= pd.Timestamp('2022-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2024-09-17', tz='UTC'))]

print('=' * 110)
print('TOP 5 RESEARCH STRATEGIES — EURUSD H1 default params')
print('=' * 110)
print(f'{"Strategy":25s} | {"Tuning":30s} | {"Validation":30s} | ROBUST')
print('-' * 110)

strategies = [
    ('#1 Regime Engine',     RegimeSwitchingEngineStrategy, {
        'adx_period': 14, 'atr_period': 14, 'bb_period': 20, 'bb_mult': 2.0,
        'trend_thresh': 25, 'range_thresh': 18, 'expansion_mult': 1.5,
        'atr_avg_period': 50, 'sma_period': 200, 'cooldown': 10,
    }),
    ('#2 Volatility Breakout', VolatilityBreakoutStrategy, {
        'adx_period': 14, 'atr_period': 14, 'atr_avg_period': 20,
        'donchian': 20, 'adx_thresh': 25, 'sma_period': 200, 'cooldown': 5,
    }),
    ('#3 Adaptive ADX',      AdaptiveADXStrategy, {
        'adx_period': 14, 'lookback': 200, 'adx_percentile': 70,
        'bars_calculate': 13, 'crossover_lookback': 6, 'min_crossover_gap': 1.0,
        'sma_period': 200, 'cooldown': 4,
    }),
    ('#4 Compression Exp',   CompressionExpansionStrategy, {
        'bb_period': 20, 'bb_mult': 2.0, 'compression_period': 30,
        'compression_percentile': 20, 'adx_period': 14, 'adx_max': 25,
        'sma_period': 200, 'cooldown': 5,
    }),
    ('#7 Turtle Donchian',   TurtleDonchianStrategy, {
        'entry_period': 20, 'exit_period': 10, 'atr_period': 14,
        'atr_min': 0.5, 'sma_period': 200, 'cooldown': 5,
    }),
]

for name, cls, params in strategies:
    m_t = run_strategy(eur_t, cls, params, [], 'forex')
    m_v = run_strategy(eur_v, cls, params, [], 'forex')
    if 'error' in m_t or 'error' in m_v:
        print(f'{name:25s} | ERROR: T={m_t.get("error", "n/a")}')
        continue
    robust = m_t['net_pnl'] > 0 and m_v['net_pnl'] > 0
    t_str = f'{m_t["net_pnl"]:+7,.0f}/Sh {m_t["sharpe"]:+.2f}/{m_t["n_trades"]}t'
    v_str = f'{m_v["net_pnl"]:+7,.0f}/Sh {m_v["sharpe"]:+.2f}/{m_v["n_trades"]}t'
    flag = 'YES' if robust else 'X'
    print(f'{name:25s} | {t_str:30s} | {v_str:30s} | {flag}')
