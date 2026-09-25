"""Signal-tuple contract test — locks the 3-tuple direction fix (3aa290d).

Bug class: run_pure reads a 2-tuple as (entries, DIRECTION). Every caller
once passed (entries, exits), silently feeding exits as direction — every
direction-aware strategy backtested as open-once-hold-forever (n=1) and
ADX numbers were exits-as-direction artifacts. This test fails if any
caller regresses to the 2-tuple form or drops direction.

Run: python tests/test_signal_contract.py  (exit 0 = pass)
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from run_pipeline import _sig_tuple, _make_strategy
from backtester.engine_full import run_full

from pathlib import Path as _Path
_candidates = sorted(_Path('data/cache').glob('ETHUSDT_H1*.parquet'),
                     key=lambda p: p.stat().st_mtime, reverse=True)
assert _candidates, 'No ETHUSDT_H1 cache in data/cache — run the Yahoo fetch first'
DF = pd.read_parquet(_candidates[0])

# Direction-aware strategies: emit entries with mixed +1/-1 direction,
# (near-)zero exits. Under the old bug each produced exactly 1 trade.
CASES = [
    ('bb_rsi', {}),
    ('macd_confluence', {}),
    ('triple_rsi', {}),
    ('mfi', {}),
]

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


print("=" * 70)
print("Signal-tuple contract: 3-tuple (entries, exits, direction)")
print("=" * 70)

# 1. _sig_tuple returns a 3-tuple whose third leg is the true direction.
sg0 = _make_strategy('bb_rsi', {}).generate(DF)
tup = _sig_tuple(sg0)
check("tuple_len_3", len(tup) == 3, f"len={len(tup)}")
check("direction_matches",
      (pd.Series(tup[2]).values == pd.Series(sg0.direction).fillna(0).values.astype(int)).all(),
      "third leg == signal direction")
check("direction_mixed",
      set(pd.Series(tup[2]).unique()) >= {1, -1},
      f"values={sorted(pd.Series(tup[2]).unique())}")

# 2. Correct tuple -> real multi-trade, two-sided backtests.
for name, params in CASES:
    s = _make_strategy(name, params)
    sg = s.generate(DF)
    r = run_full(DF, {name: _sig_tuple(sg)}, init_cash=10000,
                 symbol='ETHUSDT', strict_data=False)
    trades = r.get('trades', pd.DataFrame())
    n = len(trades)
    dcol = next((c for c in ('Direction', 'direction', 'side') if c in trades.columns), None)
    dirs = set(trades[dcol].unique()) if dcol and n else set()
    check(f"{name}_multi_trade", n > 10, f"trades={n} (old bug gave exactly 1)")
    check(f"{name}_two_sided", len(dirs) >= 2, f"sides={sorted(map(str, dirs))}")

# 3. Old wrong form must NOT reproduce correct results (locks the convention:
#    if anyone "simplifies" back to (entries, exits), counts collapse to 1).
s = _make_strategy('bb_rsi', {})
sg = s.generate(DF)
wrong = run_full(DF, {'bb_rsi': (sg.entries.values.astype(int),
                                 sg.exits.values.astype(int))},
                 init_cash=10000, symbol='ETHUSDT', strict_data=False)
n_wrong = len(wrong.get('trades', pd.DataFrame()))
check("wrong_tuple_collapses", n_wrong <= 1,
      f"2-tuple (entries, exits) gives {n_wrong} trade(s) — proves direction was dropped")

print("=" * 70)
if failures:
    print(f"CONTRACT BROKEN: {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("CONTRACT HOLDS: all checks passed")
