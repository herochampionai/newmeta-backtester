"""Cross-validate Python backtester against MT5 data via MetaTrader5 Python API.

Strategy:
1. Connect to running MT5 terminal (Tickmill-Live)
2. Fetch EURUSD + NAS100 H1 for 2024-09-17 → 2026-09-17 (the 2Y OOS window)
3. Save the data so we can verify tick-by-tick what MT5 sees
4. Also fetch ticks if available for model-4 comparison

This is the cleanest cross-validation since both Python and MT5 tester use
the same data source from the same broker.
"""
import warnings; warnings.filterwarnings('ignore')
import json
import sys

sys.path.insert(0, '.')
from datetime import datetime, timezone

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print('MetaTrader5 package not installed')
    sys.exit(1)


def init_mt5():
    """Initialize MT5 connection. Path is the running terminal's executable."""
    # Try a few paths - any one that works
    paths = [
        "D:\\MT5_Bybit\\terminal64.exe",
        "D:\\MT5_EuroPrinter\\terminal64.exe",
    ]
    for p in paths:
        if mt5.initialize(path=p):
            info = mt5.terminal_info()
            print(f"Connected: {info.name if info else 'OK'}")
            acc = mt5.account_info()
            if acc:
                print(f"  Account: {acc.login} on {acc.server}")
                print(f"  Leverage: {acc.leverage}")
            return True
    print("Failed to initialize any MT5 terminal")
    return False


def fetch_ohlc(symbol, timeframe_str, from_date, to_date):
    """Fetch OHLC bars from MT5."""
    tf_map = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }
    tf = tf_map[timeframe_str]
    rates = mt5.copy_rates_range(symbol, tf, from_date, to_date)
    if rates is None or len(rates) == 0:
        print(f"  No data for {symbol} {timeframe_str}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    return df


def fetch_ticks(symbol, from_date, to_date):
    """Fetch tick data from MT5 (limited - usually caps at recent dates)."""
    ticks = mt5.copy_ticks_range(symbol, from_date, to_date, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return None
    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    return df


def main():
    if not init_mt5():
        return

    print()
    print("=" * 80)
    print("CROSS-VALIDATION: Python backtester vs MT5 data")
    print("=" * 80)

    # 2Y OOS window: 2024-09-17 → 2026-09-17
    from_dt = datetime(2024, 9, 17, tzinfo=timezone.utc)
    to_dt = datetime(2026, 9, 17, tzinfo=timezone.utc)

    symbols = [
        ('EURUSD', 'D:/MT5_EuroPrinter/terminal64.exe'),  # EURUSD on any terminal
        ('NAS100', 'D:/MT5_Bybit/terminal64.exe'),        # NAS100 on Bybit
    ]

    results = {}
    for symbol, _ in symbols:
        print(f"\n>>> {symbol} H1 2024-09-17 → 2026-09-17")
        df = fetch_ohlc(symbol, 'H1', from_dt, to_dt)
        if df is None:
            continue
        print(f"  Bars: {len(df)}")
        print(f"  Date range: {df.index[0]} → {df.index[-1]}")
        print(f"  Price range: {df['low'].min():.5f} - {df['high'].max():.5f}")
        # Volume stats
        print(f"  Avg tick_volume: {df['tick_volume'].mean():.0f}")
        print(f"  Total bars (expected ~12400 for 2Y H1): delta = {len(df) - 12400}")

        # Save for later use
        out = f"output/mt5_{symbol}_H1_2024_2026.csv"
        df.to_csv(out)
        print(f"  Saved: {out}")
        results[symbol] = {
            'bars': len(df),
            'first': str(df.index[0]),
            'last': str(df.index[-1]),
            'low': float(df['low'].min()),
            'high': float(df['high'].max()),
        }

    # Also try to fetch some tick data for EURUSD (small range)
    print("\n>>> EURUSD ticks sample (last 7 days)")
    ticks_from = datetime(2026, 9, 11, tzinfo=timezone.utc)
    ticks_to = datetime(2026, 9, 18, tzinfo=timezone.utc)
    ticks = fetch_ticks('EURUSD', ticks_from, ticks_to)
    if ticks is not None:
        print(f"  Tick count: {len(ticks)}")
        print(f"  Bid range: {ticks['bid'].min():.5f} - {ticks['bid'].max():.5f}")
    else:
        print("  No tick data (MT5 limits tick history)")

    mt5.shutdown()

    # Save results summary
    with open('output/mt5_cross_validation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSummary saved → output/mt5_cross_validation.json")


if __name__ == "__main__":
    main()
