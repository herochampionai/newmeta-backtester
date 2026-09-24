"""Verify regime_engine and adaptive_adx on a longer 3-year window for confidence."""
import warnings; warnings.filterwarnings('ignore')
import sys, json
sys.path.insert(0, '.')
import pandas as pd
from analysis.optuna_filters import run_strategy
from strategies.top5_research import RegimeSwitchingEngineStrategy, AdaptiveADXStrategy
from datetime import datetime, timezone
import MetaTrader5 as mt5

# Connect to MT5 and fetch 4-year data for more robust validation
mt5.initialize(path='D:\\MT5_Bybit\\terminal64.exe')
from_dt = datetime(2021, 1, 1, tzinfo=timezone.utc)
to_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)  # extended to 2021-2024
rates = mt5.copy_rates_range('EURUSD', mt5.TIMEFRAME_H1, from_dt, to_dt)
df_3y = pd.DataFrame(rates)
df_3y['time'] = pd.to_datetime(df_3y['time'], unit='s', utc=True)
df_3y.set_index('time', inplace=True)
df_3y = df_3y[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
mt5.shutdown()
print(f'EURUSD 3y data (2021-2024): {len(df_3y)} bars')

# Load best params from previous session
with open('output/top3_research_results.json') as f:
    results = json.load(f)

for r in results:
    name = r['name']
    params = r['params']
    cls = RegimeSwitchingEngineStrategy if 'Regime' in name else AdaptiveADXStrategy
    print(f'\n=== {name} ===')
    print(f'Params: {params}')

    # Test on 3-year window (2021-2024) split in 2 halves
    half = len(df_3y) // 2
    df_3y_1 = df_3y.iloc[:half]
    df_3y_2 = df_3y.iloc[half:]

    m_1 = run_strategy(df_3y_1, cls, params, [], 'forex')
    m_2 = run_strategy(df_3y_2, cls, params, [], 'forex')

    if 'error' not in m_1 and 'error' not in m_2:
        print(f'2021-H1:  ${m_1["net_pnl"]:+,.0f}/Sh {m_1["sharpe"]:+.2f}/{m_1["n_trades"]}t')
        print(f'2022-H2:  ${m_2["net_pnl"]:+,.0f}/Sh {m_2["sharpe"]:+.2f}/{m_2["n_trades"]}t')
        total_pnl = m_1['net_pnl'] + m_2['net_pnl']
        print(f'Combined 3y PnL: ${total_pnl:+,.0f}')
        # Compare to our existing robust ADX variants
        # ADX EUR_B was T $7,369 / Sh 5.30 / 293 trades on the same 2Y window
        print(f'(For comparison: adx_EUR_B T $7,369 / 293t over 2Y 2024-26)')
