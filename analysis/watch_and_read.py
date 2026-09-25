"""The proper way: have user run tester in GUI, then poll for results.

This script:
1. Waits for Final8_EA .tst file to appear in MT5 cache
2. Reads the .tst file structure (MT5 proprietary binary format)
3. Extracts: header info, trade list, equity curve
4. Reports back to user

Best practice for MT5 Strategy Tester output:
- .tst file (binary): complete test results
- Journal log: human-readable test progress
"""
import sys
import time
from datetime import datetime
from pathlib import Path

cache_dir = Path(r'C:\Users\youha\AppData\Roaming\MetaQuotes\Terminal\6D35BF1728C0CA7D99E1A54264C3585C\Tester\cache')
journal_dir = Path(r'C:\Users\youha\AppData\Roaming\MetaQuotes\Terminal\6D35BF1728C0CA7D99E1A54264C3585C\Tester\logs')

print('=' * 80)
print('WAITING FOR FINAL8_EA TESTER RESULTS')
print('=' * 80)
print()
print('Watching for new .tst files in:', cache_dir)
print()
print('HOW TO RUN THE TEST (do this in MT5 GUI):')
print()
print('  1. Open MT5 (it should already be open)')
print('  2. Press Ctrl+R to open Strategy Tester')
print('  3. In the Tester panel:')
print('     - Expert: Multi Strat 2026\\Final8_EA')
print('     - Symbol: EURUSD')
print('     - Period: H1')
print('     - Date range: 2024.09.17 - 2026.09.17')
print('     - Modeling: Every tick based on real ticks')
print('     - Deposit: 100000')
print('     - Leverage: 500')
print('  4. Click START button (bottom)')
print('  5. Wait ~30-60 seconds for test to complete')
print()
print('This script will detect the results file automatically.')
print()
print('=' * 80)

# Track existing files
existing_files = set()
if cache_dir.exists():
    existing_files = {f.name for f in cache_dir.glob('*.tst')}

timeout = 1800  # 30 minutes
start = time.time()
found_file = None

while time.time() - start < timeout:
    try:
        current_files = {f.name for f in cache_dir.glob('Final8_EA.*.tst')}
        new_files = current_files - existing_files

        if new_files:
            for fname in new_files:
                fpath = cache_dir / fname
                print(f'\nNew file: {fname}')
                # Wait for size to stabilize (test still writing)
                prev_size = -1
                for _ in range(10):  # wait up to 50 sec for stable size
                    time.sleep(5)
                    cur_size = fpath.stat().st_size
                    if cur_size == prev_size:
                        break
                    prev_size = cur_size
                print(f'Size stable: {prev_size} bytes')
                found_file = fpath
                break

        if found_file:
            break

        elapsed = int(time.time() - start)
        if elapsed > 0 and elapsed % 60 == 0:
            print(f'  Watching... ({elapsed}s elapsed) - no Final8_EA .tst yet')
            # Also check today's journal
            today_log = journal_dir / f'{datetime.now().strftime("%Y%m%d")}.log'
            if today_log.exists():
                # Print last 5 lines
                try:
                    with open(today_log, 'r') as f:
                        lines = f.readlines()
                    for line in lines[-5:]:
                        print(f'  [journal] {line.rstrip()}')
                except:
                    pass

    except Exception as e:
        print(f'  Error: {e}')

    time.sleep(5)

if not found_file:
    print('\nTimed out. Please run the test manually in MT5 GUI.')
    sys.exit(1)

# Parse the .tst file
print()
print('=' * 80)
print('TEST RESULTS')
print('=' * 80)
print()
print(f'File: {found_file.name}')
print(f'Size: {found_file.stat().st_size} bytes')
print(f'Modified: {datetime.fromtimestamp(found_file.stat().st_mtime)}')
print()

# .tst file is a proprietary binary format
# Header is Unicode text, body has trade records
# Extract what we can
try:
    with open(found_file, 'rb') as f:
        raw = f.read()

    # Decode header (Unicode UTF-16 LE)
    text = raw.decode('utf-16-le', errors='ignore')

    # Print first readable text
    printable = ''.join(c if (c.isprintable() or c in '\n\r\t') else ' ' for c in text[:5000])
    print('File header (first 5000 bytes decoded):')
    print('-' * 80)
    for chunk in printable.split('  '):
        if len(chunk) > 3:
            print(chunk)
    print('-' * 80)

    # Look for embedded numbers (PnL, etc.)
    # Common patterns in .tst: 4-byte floats, 8-byte doubles
    print()
    print('Looking for embedded PnL/equity numbers...')

    # Try to find numeric sequences
    import re
    # Look for sequences of digits with possible decimal/sign
    numbers = re.findall(r'-?\d+\.\d{2,6}', text[:50000])
    if numbers:
        print(f'Found {len(numbers)} numbers in first 50KB:')
        # Look for large positive numbers (PnL)
        big_nums = [float(n) for n in numbers if abs(float(n)) > 100]
        big_nums.sort(reverse=True)
        print(f'  Top 20 numbers: {big_nums[:20]}')

    # Save raw for later analysis
    raw_path = f'output/mt5_tester_raw/{found_file.name}'
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, 'wb') as f:
        f.write(raw)
    print(f'\nRaw .tst saved to: {raw_path}')

except Exception as e:
    print(f'Parse error: {e}')
    import traceback
    traceback.print_exc()

print()
print('=' * 80)
print('RECOMMENDED: VIEW RESULTS IN MT5 GUI')
print('=' * 80)
print('The Strategy Tester in MT5 GUI shows:')
print('  - Total Net Profit')
print('  - Profit Factor')
print('  - Sharpe Ratio')
print('  - Recovery Factor')
print('  - Total Trades')
print('  - Win Rate')
print('  - Max Drawdown')
print('  - Equity Curve graph')
print()
print('These are visible in the MT5 Strategy Tester BACKTEST tab.')
