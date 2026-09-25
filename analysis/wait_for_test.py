"""Wait for MT5 tester results and read them."""
import warnings; warnings.filterwarnings('ignore')
import time
from datetime import datetime
from pathlib import Path

cache_dir = Path(r'C:\Users\youha\AppData\Roaming\MetaQuotes\Terminal\6D35BF1728C0CA7D99E1A54264C3585C\Tester\cache')
tester_logs = Path(r'C:\Users\youha\AppData\Roaming\MetaQuotes\Terminal\6D35BF1728C0CA7D99E1A54264C3585C\Tester\logs')

print('=' * 80)
print('WAITING FOR FINAL8_EA TEST RESULTS')
print('=' * 80)
print()
print('Watching: ' + str(cache_dir))
print('Press Ctrl+C to stop')
print()

last_files = set()
timeout = 900  # 15 min
start = time.time()

while time.time() - start < timeout:
    try:
        current_files = {f.name for f in cache_dir.glob('Final8_EA.*.tst')}
        new_files = current_files - last_files
        last_files = current_files

        if new_files:
            print(f'  New file detected: {new_files}')
            for fname in new_files:
                fpath = cache_dir / fname
                # Wait for file to finish writing (size stable for 5s)
                size1 = fpath.stat().st_size
                time.sleep(5)
                size2 = fpath.stat().st_size
                if size1 == size2 and size2 > 1000:
                    print(f'  File stable: {fname} ({size1} bytes)')
                    print()
                    print('=' * 80)
                    print('TEST COMPLETED — READING RESULTS')
                    print('=' * 80)
                    print()

                    # Read the .tst file
                    try:
                        with open(fpath, 'rb') as f:
                            raw = f.read()
                        # Convert to text (Unicode UTF-16 LE)
                        text = raw.decode('utf-16-le', errors='ignore')
                        # Find readable strings
                        printable = ''.join(c if (c.isprintable() or c in '\n\r\t') else '\x00' for c in text)
                        # Split by null bytes
                        chunks = [c for c in printable.split('\x00') if len(c) > 5]
                        print('Strings found in .tst file:')
                        for chunk in chunks[:30]:
                            print(f'  {chunk[:100]}')
                    except Exception as e:
                        print(f'  Read error: {e}')

                    # Also check for trade log files
                    print()
                    print('=' * 80)
                    print('CHECKING MT5 JOURNAL')
                    print('=' * 80)
                    today = datetime.now().strftime('%Y%m%d')
                    log_file = tester_logs / f'{today}.log'
                    if log_file.exists():
                        print(f'Reading: {log_file}')
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                        # Filter for Final8_EA related lines
                        final8_lines = [l for l in lines if 'Final8_EA' in l or 'EURUSD' in l or 'trade' in l.lower()]
                        for line in final8_lines[-30:]:
                            print('  ' + line.rstrip())
                    break

        elapsed = int(time.time() - start)
        if elapsed % 30 == 0 and elapsed > 0:
            print(f'  Waiting... ({elapsed}s elapsed)')

    except Exception as e:
        print(f'  Error: {e}')

    time.sleep(3)

print()
print('Done waiting')
