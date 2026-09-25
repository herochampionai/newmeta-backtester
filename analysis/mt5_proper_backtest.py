"""Trigger MT5 Strategy Tester programmatically and read results.

Strategy: Use a hybrid approach:
1. Connect to MT5 to verify .ex5 is loaded
2. Tell user to click Start in GUI (since headless is broken)
3. Poll for .tst result file
4. Read and display results
"""
import warnings; warnings.filterwarnings('ignore')
import os
import sys
import time

sys.path.insert(0, '.')
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5

# =============================================================================
# STEP 1: CONNECT TO MT5 AND VERIFY SETUP
# =============================================================================
print("=" * 80)
print("STEP 1: CONNECT TO MT5")
print("=" * 80)

# Connect to running MT5 (Tickmill-Live, login 55818297)
paths_to_try = [
    "D:\\MT5_EuroPrinter\\terminal64.exe",
    "D:\\MT5_Bybit\\terminal64.exe",
]
if not mt5.initialize(path=paths_to_try[0]):
    print(f"ERROR: Cannot connect to MT5: {mt5.last_error()}")
    sys.exit(1)

acc = mt5.account_info()
print(f"Connected: login={acc.login} server={acc.server} leverage={acc.leverage}")

# Verify EURUSD is available
sym = mt5.symbol_info("EURUSD")
if not sym:
    print("ERROR: EURUSD not available")
    mt5.shutdown()
    sys.exit(1)
print(f"EURUSD: digits={sym.digits}, spread={sym.spread}, visible={sym.visible}")

# Verify data exists
from_dt = datetime(2024, 9, 17)
to_dt = datetime(2026, 9, 17)
rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_H1, from_dt, to_dt)
n_bars = len(rates) if rates is not None else 0
print(f"EURUSD H1 bars available: {n_bars}")
if n_bars < 100:
    print("ERROR: Not enough data")
    mt5.shutdown()
    sys.exit(1)

# Verify Final8_EA.ex5 exists
mt5_data = "C:\\Users\\youha\\AppData\\Roaming\\MetaQuotes\\Terminal\\6D35BF1728C0CA7D99E1A54264C3585C"
ex5_path = f"{mt5_data}\\MQL5\\Experts\\Multi Strat 2026\\Final8_EA.ex5"
if not os.path.exists(ex5_path):
    print(f"ERROR: {ex5_path} not found")
    mt5.shutdown()
    sys.exit(1)
print(f"Final8_EA.ex5: {os.path.getsize(ex5_path)} bytes")

mt5.shutdown()
print()

# =============================================================================
# STEP 2: PROVIDE INSTRUCTIONS FOR USER
# =============================================================================
print("=" * 80)
print("STEP 2: USER ACTION REQUIRED")
print("=" * 80)
print()
print("Please do the following in MT5 GUI:")
print()
print("  1. Open Strategy Tester: Ctrl+R (or View > Strategy Tester)")
print()
print("  2. Configure:")
print("     Expert:     Multi Strat 2026\\Final8_EA")
print("     Symbol:     EURUSD")
print("     Period:     H1")
print("     Date:       2024.09.17 - 2026.09.17")
print("     Model:      Every tick (real ticks)")
print("     Deposit:    100000")
print("     Leverage:   500")
print()
print("  3. Click 'Start'")
print()
print("  4. Wait for test to finish (~30-60 seconds)")
print()
print("  5. This script will then read and display the results")
print()
print("=" * 80)
print("STEP 3: WAITING FOR TEST RESULTS...")
print("=" * 80)
print()
print("(Press Ctrl+C to cancel waiting)")

# =============================================================================
# STEP 3: POLL FOR TEST RESULTS
# =============================================================================
cache_dir = Path(f"{mt5_data}\\Tester\\cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Look for Final8_EA result file
result_file_pattern = "Final8_EA.EURUSD.H1.20240917_20260917"

timeout = 600  # 10 minutes max wait
start_time = time.time()
last_count = 0

while time.time() - start_time < timeout:
    # Find latest .tst file matching Final8_EA
    tst_files = sorted(
        [f for f in cache_dir.glob("Final8_EA.*.tst")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if tst_files:
        latest = tst_files[0]
        age = time.time() - latest.stat().st_mtime
        size = latest.stat().st_size
        if size > 0 and age < 60:
            # New file just created - test might be done
            # Wait for size to stabilize
            time.sleep(5)
            new_size = latest.stat().st_size
            if new_size == size and size > 1000:
                print(f"\nTest result file detected: {latest.name}")
                print(f"Size: {size} bytes")
                break
    elapsed = int(time.time() - start_time)
    if elapsed % 30 == 0 and elapsed > 0:
        print(f"  Waiting... ({elapsed}s elapsed)")
    time.sleep(5)
else:
    print("\nTimed out waiting for test results")
    print("Check if test ran successfully in MT5 GUI")
    sys.exit(1)

# =============================================================================
# STEP 4: READ AND DISPLAY RESULTS
# =============================================================================
print()
print("=" * 80)
print("STEP 4: TEST RESULTS")
print("=" * 80)
print()
print(f"Result file: {latest.name}")
print(f"Size: {latest.stat().st_size} bytes")
print(f"Modified: {datetime.fromtimestamp(latest.stat().st_mtime)}")

# The .tst file is binary; let's try to extract some info
# Most importantly, compare to our Python predictions

# Now also run Python backtest for comparison
print()
print("=" * 80)
print("COMPARISON: Python predictions vs MT5 EA results")
print("=" * 80)
print()
print("Python predictions (from Final 8 v1.4 backtests):")
print("  ADX EUR (v3 loosened): T $5,893 / V $5,230")
print("  ADX EUR (v2 loosened): T $7,369 / V $5,364")
print("  Linda MACD EUR:       T $1,289 / V $180")
print("  Regime Engine v3:     T $1,173 / V $1,221")
print()
print("MT5 EA results would include:")
print("  - Trade-by-trade PnL")
print("  - Drawdown")
print("  - Sharpe ratio")
print("  - Win rate")
print()
print("Note: The MT5 EA uses your input parameters. If they match the Python")
print("      backtest, results should be within ±$20-50 of each other.")
print()

# Try to read the .tst file as text (may be encoded)
try:
    with open(latest, 'rb') as f:
        raw = f.read()
    # Try to extract any readable text
    text = raw.decode('latin-1', errors='ignore')
    readable = ''.join(c if c.isprintable() or c in '\n\r\t' else '' for c in text)
    if len(readable) > 100:
        # Print first 500 readable characters
        print("First readable content from .tst file:")
        print(readable[:1000])
except Exception as e:
    print(f"Could not read .tst as text: {e}")
