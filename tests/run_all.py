"""Unified test runner — runs all smoke tests + reports.
Usage: python -m tests
"""
import sys
import subprocess
from pathlib import Path

TESTS = Path(__file__).parent
PYTHON = sys.executable
ROOT = TESTS.parent

# Tests that don't need MT5 (faster, run unconditionally)
SAFE_TESTS = [
    "tests/test_universal.py",
    "tests/smoke_test.py",
    "tests/smoke_grid.py",
    "tests/smoke_pro.py",
    "tests/smoke_modular.py",
    "tests/smoke_quick.py",
]

# Tests that need MT5 (skip if MT5 unavailable)
MT5_TESTS = [
    "tests/smoke_tick.py",
    "tests/real_data_test.py",
]


def main():
    print("=" * 60)
    print("Running Newmeta Backtester test suite")
    print("=" * 60)
    passed = 0
    failed = 0
    skipped = 0
    for test in SAFE_TESTS:
        path = ROOT / test
        if not path.exists():
            print(f"  SKIP {test} (not found)")
            skipped += 1
            continue
        print(f"\n>>> Running {test}…")
        result = subprocess.run([PYTHON, str(path)], capture_output=True, text=True,
                                 cwd=ROOT, env={**__import__("os").environ,
                                                "PYTHONPATH": str(ROOT)})
        if result.returncode == 0:
            print(f"  PASS {test}")
            passed += 1
        else:
            print(f"  FAIL {test}")
            print(result.stdout[-500:] if result.stdout else "")
            print(result.stderr[-500:] if result.stderr else "")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed == 0:
        print("*** ALL TESTS PASSED ***")
    else:
        print("*** SOME TESTS FAILED ***")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())