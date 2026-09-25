"""Connect to running MT5 and run the strategy tester programmatically."""
import warnings; warnings.filterwarnings('ignore')
import sys

sys.path.insert(0, '.')
import os

import MetaTrader5 as mt5
import pandas as pd

# Find running MT5 terminal
print("=" * 80)
print("MT5 TERMINAL CONNECTION")
print("=" * 80)

# Try multiple paths - the running one
paths_to_try = [
    "D:\\MT5_EuroPrinter\\terminal64.exe",
    "D:\\MT5_Bybit\\terminal64.exe",
    "D:\\work D\\mt5-mcp\\MT5_EuroPrinter\\terminal64.exe",
]

connected = False
for path in paths_to_try:
    if not os.path.exists(path):
        continue
    print(f"Trying: {path}")
    if mt5.initialize(path=path):
        info = mt5.terminal_info()
        if info and info.connected:
            acc = mt5.account_info()
            print(f"  CONNECTED to {path}")
            if acc:
                print(f"  Account: {acc.login} on {acc.server}")
                print(f"  Leverage: {acc.leverage}")
            connected = True
            break
        else:
            print("  Initialized but NOT connected")
            mt5.shutdown()

if not connected:
    print("ERROR: No running MT5 found")
    sys.exit(1)

# Check what's available
print()
print("=" * 80)
print("CHECKING SYMBOLS")
print("=" * 80)

# Check EURUSD
sym = mt5.symbol_info("EURUSD")
if sym:
    print(f"EURUSD: {sym.path}, digits={sym.digits}, spread={sym.spread}")
else:
    print("EURUSD not found")

# Check available ticks for EURUSD
from_dt = pd.Timestamp('2024-09-17', tz='UTC').to_pydatetime()
to_dt = pd.Timestamp('2026-09-17', tz='UTC').to_pydatetime()
rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_H1, from_dt, to_dt)
if rates is not None and len(rates) > 0:
    df = pd.DataFrame(rates)
    print(f"EURUSD H1 bars available: {len(df)} ({df['time'].min()} to {df['time'].max()})")

mt5.shutdown()
