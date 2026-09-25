"""CI/local fixture fetch — Yahoo H1 bars into data/cache (exit 0 = ok).

Fresh clones have an empty data/cache (gitignored), which breaks every
data-dependent suite. This script pulls 1y H1 for the symbols the suites
need and writes sha-named parquet + meta via data.cache.write.
Suites glob the cache (never hardcode hashes), so any refetch works.

Usage: python tests/fetch_ci_fixture.py
"""
import sys
sys.path.insert(0, '.')

import hashlib
import os
import pandas as pd
from data.live_fetcher import fetch_with_priority
from data.cache import write

# FIXTURE_SYMBOLS="EURUSD,ETHUSDT" in CI (Yahoo-reliable only).
SYMBOLS = [s.strip() for s in os.environ.get("FIXTURE_SYMBOLS", "EURUSD,ETHUSDT,BTCUSDT").split(",") if s.strip()]
failures = []

for sym in SYMBOLS:
    try:
        df, info = fetch_with_priority(sym, 'H1', start='2024-01-01', end=None)
        src = info.get('source', '?')
        if src == 'synthetic' or (df is not None and len(df) < 500):
            # Synthetic or stub-sized data is useless as a fixture.
            print(f"  [SKIP] {sym}: source={src} rows={len(df) if df is not None else 0}")
            failures.append(sym)
            continue
        sha = hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:16]
        meta = {'symbol': sym, 'timeframe': 'H1', 'sha': sha,
                'source': src, 'rows': len(df)}
        p = write(df, meta)
        print(f"  [OK] {sym}: {len(df)} bars source={src} -> {p.name}")
    except Exception as e:
        print(f"  [FAIL] {sym}: {type(e).__name__}: {str(e)[:100]}")
        failures.append(sym)

if failures:
    print(f"fixture incomplete, missing: {failures}")
    sys.exit(1)
print("FIXTURE READY")
