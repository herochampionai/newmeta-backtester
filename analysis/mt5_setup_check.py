"""Proper MT5 backtest: connect to running terminal and verify setup."""
import warnings; warnings.filterwarnings('ignore')
import os, sys
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# Connect to running MT5
mt5_path = r'D:\MT5_EuroPrinter\terminal64.exe'
if not mt5.initialize(path=mt5_path):
    print(f'Connect failed: {mt5.last_error()}')
    sys.exit(1)
print('=' * 80)
print('CONNECTED TO MT5')
print('=' * 80)

acc = mt5.account_info()
print(f'Account: {acc.login}')
print(f'Server: {acc.server}')
print(f'Leverage: {acc.leverage}')
print(f'Equity: ${acc.equity:.2f}')

# Verify EURUSD data
print()
print('=' * 80)
print('DATA CHECK')
print('=' * 80)

rates = mt5.copy_rates_range('EURUSD', mt5.TIMEFRAME_H1,
                              datetime(2024, 9, 17), datetime(2026, 9, 17))
df = pd.DataFrame(rates)
print(f'EURUSD H1 bars: {len(df)}')
print(f'Date range: {pd.to_datetime(df["time"].min(), unit="s")} -> {pd.to_datetime(df["time"].max(), unit="s")}')

# Verify .ex5 is in correct location
ex5_path = r'C:\Users\youha\AppData\Roaming\MetaQuotes\Terminal\6D35BF1728C0CA7D99E1A54264C3585C\MQL5\Experts\Multi Strat 2026\Final8_EA.ex5'
print()
print('=' * 80)
print('EA CHECK')
print('=' * 80)
if os.path.exists(ex5_path):
    print(f'EXISTS: {ex5_path}')
    print(f'Size: {os.path.getsize(ex5_path)} bytes')
else:
    print(f'MISSING: {ex5_path}')

mt5.shutdown()

print()
print('=' * 80)
print('NEXT STEPS')
print('=' * 80)
print()
print('MT5 is connected. To run the Strategy Tester:')
print()
print('1. In MT5 GUI: Press Ctrl+R (or View > Strategy Tester)')
print('2. Configure:')
print('   Expert: Multi Strat 2026\\Final8_EA')
print('   Symbol: EURUSD')
print('   Period: H1')
print('   Date: 2024.09.17 - 2026.09.17')
print('   Model: Every tick')
print('3. Click Start')
print('4. When done, check Experts tab for results')
print()
print('After test runs, .tst file will appear in:')
print('C:\\Users\\youha\\AppData\\Roaming\\MetaQuotes\\Terminal\\6D35BF1728C0CA7D99E1A54264C3585C\\Tester\\cache\\')
print()
print('I can read that file and show you the numbers.')
