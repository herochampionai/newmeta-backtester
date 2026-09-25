"""Regression test: all strategies must instantiate and generate signals
without error after the BaseStrategy _aliases system change."""
import sys
sys.path.insert(0, '.')

import importlib
import traceback
from pathlib import Path
import pandas as pd

# Load real data (freshest EURUSD H1 cache; hash suffixes change on refetch).
from pathlib import Path as _Path
_candidates = sorted(_Path('data/cache').glob('EURUSD_H1*.parquet'),
                     key=lambda p: p.stat().st_mtime, reverse=True)
assert _candidates, 'No EURUSD_H1 cache in data/cache — run the Yahoo fetch first'
df = pd.read_parquet(_candidates[0])

# Discover all strategy files (skip base/init/indicators)
strategies_dir = Path('strategies')
strategy_files = sorted([
    f.stem for f in strategies_dir.glob('*.py')
    if f.stem not in ('__init__', 'indicators', '_base', 'adx')
    and not f.stem.startswith('_')
])

print(f'Found {len(strategy_files)} non-base strategy files')
print('=' * 75)

results = []
for stem in strategy_files:
    try:
        mod = importlib.import_module(f'strategies.{stem}')
        # Find all classes that look like strategies
        strategy_classes = []
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and attr_name != 'BaseStrategy':
                try:
                    from strategies._base import BaseStrategy
                    if issubclass(attr, BaseStrategy):
                        strategy_classes.append(attr)
                except Exception:
                    pass
        if not strategy_classes:
            results.append((stem, 'skipped', 'no BaseStrategy subclass'))
            continue
        for cls in strategy_classes:
            try:
                # Try with no params
                inst = cls()
                sig = inst.generate(df)
                n_e = int(sig.entries.sum()) if hasattr(sig, 'entries') else 0
                results.append((f'{stem}.{cls.__name__}', 'PASS', f'entries={n_e}'))
            except Exception as e:
                # Config-gated base classes (e.g. MTFStrategy requires ltf_strategy_cls)
                # are by design, not regressions — mark as skipped.
                if 'ltf_strategy_cls' in str(e):
                    results.append((f'{stem}.{cls.__name__}', 'skipped',
                                    'config-gated base class (needs ltf_strategy_cls)'))
                else:
                    results.append((f'{stem}.{cls.__name__}', 'FAIL', f'{type(e).__name__}: {str(e)[:60]}'))
    except Exception as e:
        results.append((stem, 'IMPORT_FAIL', f'{type(e).__name__}: {str(e)[:60]}'))

# Summary
passed = sum(1 for _, s, _ in results if s == 'PASS')
failed = sum(1 for _, s, _ in results if s.startswith('FAIL') or s == 'IMPORT_FAIL')
skipped = sum(1 for _, s, _ in results if s == 'skipped')

print(f'\n=== Summary: {passed} PASS, {failed} FAIL, {skipped} skipped ===\n')

if failed > 0:
    print('FAILURES:')
    for name, status, msg in results:
        if status.startswith('FAIL') or status == 'IMPORT_FAIL':
            print(f'  [{status:12s}] {name}: {msg}')
else:
    print('All strategies work after alias system change.')

# Also verify _resolve_params behavior on a strategy without aliases
print('\n=== _resolve_params regression test ===')
from strategies.adx import ADX_Strategy
s = ADX_Strategy(name='adx', params={})
resolved = s._resolve_params()
print(f'  empty params resolved: {resolved} (expected: {{}})')

s = ADX_Strategy(name='adx', params={'unknown_key': 42})
resolved = s._resolve_params()
print(f'  unknown param resolved: {resolved} (expected: {{"unknown_key": 42}})')

s = ADX_Strategy(name='adx', params={'adx_period': 14})
resolved = s._resolve_params()
print(f'  alias applied: bars_calculate in resolved = {"bars_calculate" in resolved}')
