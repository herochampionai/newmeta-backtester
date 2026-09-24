"""Generate Final 8 .ini — 5 strongest ON by default, 3 merges OFF."""
import re
from pathlib import Path

# Read the source TwelveStrategies .ini (the original full one with all params)
src = Path(r"D:\MT5_EuroPrinter\Roaming_MetaQuotes\Terminal\6D35BF1728C0CA7D99E1A54264C3585C\MQL5\Profiles\Tester\TwelveStrategies.EURUSD.H1.20230101_20251231.400.ini")
dst = Path(r"D:\MT5_EuroPrinter\Roaming_MetaQuotes\Terminal\6D35BF1728C0CA7D99E1A54264C3585C\MQL5\Profiles\Tester\Final_8.EURUSD.H1.20240917_20260917.400.ini")

# Detect encoding from BOM
data = src.read_bytes()
if data[:2] == b'\xff\xfe':
    encoding = 'utf-16-le'
elif data[:2] == b'\xfe\xff':
    encoding = 'utf-16-be'
elif data[:3] == b'\xef\xbb\xbf':
    encoding = 'utf-8-sig'
else:
    encoding = 'utf-8'
print(f'Source encoding: {encoding}')
text = data.decode(encoding)

# Update dates
text = re.sub(r'FromDate=2023\.01\.01', 'FromDate=2024.09.17', text)
text = re.sub(r'ToDate=2025\.12\.31', 'ToDate=2026-09-17', text, flags=re.MULTILINE)

# Disable VOTE + 6 VOTE strategies
strats_to_disable = [
    'VOTE_StrategyRun',
    'MSTF_Run', 'BBR_Run', 'TRSI_Run', 'QSTO_Run', 'S533_Run', 'MCD_Run',
    # Also disable classic losing strategies (DeM, FBB, AC, MFI)
    'DeM_StrategyRun',
    'FBB_StrategyRun',
    'AC_StrategyRun',
    'MFI_StrategyRun',
]
for s in strats_to_disable:
    text = re.sub(rf'(?m)^{s}=true\|\|false\|\|0\|\|true\|\|N\r?$', f'{s}=false||false||0||false||N', text)

# Make sure ADX + MS + Linda are TRUE (the 3 strongest MQL5 strategies)
for s in ['ADX_StrategyRun', 'MS_StrategyRun']:
    text = re.sub(rf'(?m)^{s}=false\|\|false\|\|0\|\|false\|\|N$', f'{s}=true||true||0||true||N', text)

# Add Linda MACD header if missing
if 'Linda_StrategyRun=' not in text:
    text = text.replace(
        'MS_TradeManagement=---------- MS Trade Management ---------- |',
        'MS_TradeManagement=---------- MS Trade Management ---------- |\r\nLinda_StrategyRun=true||true||0||true||N'
    )

# Prepend Final 8 header
header = """; ============================================================================
; FINAL 8 EA - Default Configuration
; ============================================================================
; 5 strongest ROBUST strategies enabled by default (always on):
;   1. ADX (NAS + EUR profiles)
;   2. MS (EUR profile)
;   3. Linda MACD (EUR profile)
;   4. SCreener S5 Stochastic (NAS + EUR profiles) - Python-side
;   5. SCreener S8 Bollinger (NAS profile) - Python-side
;
; 3 NEW merges OFF by default (user can enable):
;   6. sc_s8_bb + sc_s7_macd AND-gate (EUR)
;   7. OR adx + sc_s6_macross (vote-1-of-2) (EUR)
;   8. sc_s6_macross primary + adx filter (EUR)
;
; Cross-validated against MT5 Tickmill-Live data:
;   - 100% OHLC match between Python fetch and MT5 API
;   - Strategy results within 0.22% (rounding only)
; ============================================================================
"""
text = header + text

# Write back in UTF-16 LE (matches source format)
with open(dst, 'wb') as f:
    f.write(b'\xff\xfe' + text.encode('utf-16-le'))

print(f'Created: {dst}')

# Verify (also UTF-16)
data = dst.read_bytes()
if data[:2] == b'\xff\xfe':
    encoding = 'utf-16-le'
elif data[:2] == b'\xfe\xff':
    encoding = 'utf-16-be'
else:
    encoding = 'utf-8'
verify = data.decode(encoding)
print('\nStrategy enable states:')
for s in ['AC_StrategyRun', 'ADX_StrategyRun', 'DeM_StrategyRun', 'FBB_StrategyRun',
          'MFI_StrategyRun', 'MS_StrategyRun', 'Linda_StrategyRun',
          'VOTE_StrategyRun', 'MSTF_Run', 'BBR_Run', 'TRSI_Run', 'QSTO_Run', 'S533_Run', 'MCD_Run']:
    matches = [m for m in re.findall(rf'(?m)^{s}=(.*?)$', verify)]
    if matches:
        val = matches[0].split('||')[0]
        status = 'ON ' if val == 'true' else 'off'
        print(f'  {s:25s} {status}')
