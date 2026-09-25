"""Cross-validation: compare MT5-fetched data vs previously-fetched data,
then re-run backtests on MT5 data to verify Python results match.
"""
import warnings; warnings.filterwarnings('ignore')
import json
import sys

sys.path.insert(0, '.')
import os
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from analysis.optuna_filters import run_strategy
from strategies.linda_macd_lenient import LindaMACDLenientStrategy


def init_mt5():
    paths = ["D:\\MT5_Bybit\\terminal64.exe", "D:\\MT5_EuroPrinter\\terminal64.exe"]
    for p in paths:
        if mt5.initialize(path=p):
            return True
    return False


def fetch_mt5_data(symbol, from_dt, to_dt):
    """Fetch bars via MT5 API."""
    tf_map = {'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1}
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, from_dt, to_dt)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'tick_volume']].rename(columns={'tick_volume': 'volume'})


def compare_dataframes(df1, df2, label=""):
    """Compare two DataFrames bar-by-bar."""
    if len(df1) != len(df2):
        print(f"  {label}: bar count differs ({len(df1)} vs {len(df2)})")
    # Align by index
    common = df1.index.intersection(df2.index)
    if len(common) == 0:
        print(f"  {label}: no common bars")
        return None
    d1 = df1.loc[common]
    d2 = df2.loc[common]
    for col in ['open', 'high', 'low', 'close']:
        diff = (d1[col] - d2[col]).abs()
        n_diff = (diff > 1e-5).sum()
        max_diff = diff.max()
        if n_diff > 0:
            print(f"  {label} {col}: {n_diff}/{len(common)} bars differ, max diff {max_diff:.6f}")
        else:
            print(f"  {label} {col}: 100% match ({len(common)} bars)")
    return len(common)


def main():
    # Step 1: Connect and fetch fresh MT5 data
    if not init_mt5():
        print("MT5 init failed")
        return

    from_dt = datetime(2024, 9, 17, tzinfo=timezone.utc)
    to_dt = datetime(2026, 9, 17, tzinfo=timezone.utc)

    print("=" * 80)
    print("STEP 1: Fetch fresh MT5 data + compare to previous Python fetch")
    print("=" * 80)

    # Fresh MT5 data
    mt5_eur = fetch_mt5_data('EURUSD', from_dt, to_dt)
    print(f"\nFresh MT5 EURUSD data: {len(mt5_eur)} bars")
    print(f"  Range: {mt5_eur.index[0]} → {mt5_eur.index[-1]}")

    # Previous fetch from terminal (saved earlier as output/mt5_EURUSD_H1_2024_2026.csv)
    prev_csv = 'output/mt5_EURUSD_H1_2024_2026.csv'
    if os.path.exists(prev_csv):
        prev_eur = pd.read_csv(prev_csv, index_col='time', parse_dates=True)
        # Ensure UTC index
        if prev_eur.index.tz is None:
            prev_eur.index = prev_eur.index.tz_localize('UTC')
        print(f"Previous MT5 EURUSD data: {len(prev_eur)} bars")
        compare_dataframes(mt5_eur, prev_eur, "EURUSD")

    # Step 2: Fetch from D:/MT5_EuroPrinter/terminal64.exe (the OLD method)
    print("\n>>> Fetching EURUSD via D:/MT5_EuroPrinter path")
    mt5.shutdown()
    if mt5.initialize(path="D:\\MT5_EuroPrinter\\terminal64.exe"):
        old_eur = fetch_mt5_data('EURUSD', from_dt, to_dt)
        if old_eur is not None:
            print(f"Old MT5 fetch: {len(old_eur)} bars")
            compare_dataframes(mt5_eur, old_eur, "EURUSD old-vs-new")
        mt5.shutdown()

    # Step 3: Run the locked Linda MACD EUR strategy on fresh MT5 data
    print()
    print("=" * 80)
    print("STEP 2: Re-run Linda MACD EUR on fresh MT5 data (locked params)")
    print("=" * 80)

    with open('output/linda_macd_lenient_best.json') as f:
        linda_best = json.load(f)
    eur_params = linda_best['EUR']['params']
    print(f"\nLocked EUR params: {eur_params}")

    # Run on MT5 data
    m_mt5 = run_strategy(mt5_eur, LindaMACDLenientStrategy, eur_params, [], 'forex')
    if 'error' not in m_mt5:
        print("\nMT5 data result:")
        print(f"  PnL: ${m_mt5['net_pnl']:+,.0f}")
        print(f"  Sharpe: {m_mt5['sharpe']:+.2f}")
        print(f"  WR: {m_mt5['win_rate']*100:.1f}%")
        print(f"  PF: {m_mt5['profit_factor']:.2f}")
        print(f"  Trades: {m_mt5['n_trades']}")

    # Compare with locked profile
    profile_path = 'output/profiles_final/linda_macd_lenient_EUR.json'
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            locked = json.load(f)
        print("\nLocked profile result (from previous backtest):")
        for k in ['net_pnl', 'sharpe', 'win_rate', 'profit_factor', 'n_trades']:
            print(f"  {k}: {locked.get(k, 'N/A')}")

    # Step 4: Test ADX strategy on MT5 data (the gold star)
    print()
    print("=" * 80)
    print("STEP 3: Run ADX EUR on fresh MT5 data")
    print("=" * 80)

    # Load ADX best params
    adx_path = 'output/adx_best_params.json'
    if os.path.exists(adx_path):
        with open(adx_path) as f:
            adx_best = json.load(f)
        # Check structure - might be {asset: {params}} or {asset: params}
        if 'EUR' in adx_best:
            if isinstance(adx_best['EUR'], dict) and 'params' in adx_best['EUR']:
                adx_eur_params = adx_best['EUR']['params']
            else:
                adx_eur_params = adx_best['EUR']
            print(f"\nADX EUR params: {adx_eur_params}")

            # Find the ADX strategy class
            from strategies.adx import ADX_Strategy
            m_adx = run_strategy(mt5_eur, ADX_Strategy, adx_eur_params, [], 'forex')
            if 'error' not in m_adx:
                print("\nADX EUR on MT5 data:")
                print(f"  PnL: ${m_adx['net_pnl']:+,.0f}")
                print(f"  Sharpe: {m_adx['sharpe']:+.2f}")
                print(f"  WR: {m_adx['win_rate']*100:.1f}%")
                print(f"  PF: {m_adx['profit_factor']:.2f}")
                print(f"  Trades: {m_adx['n_trades']}")


if __name__ == "__main__":
    main()
