"""Unified test runner — every suite, one exit code.

Usage:
    python -m tests            # everything
    python -m tests --quick    # skip slow suites (e2e, regression, contract)

Suites needing data/cache: run `python tests/fetch_ci_fixture.py` first
(CI does this automatically). MT5-terminal suites are excluded — they
need a live terminal and are covered by tests/smoke_tick.py manually.
"""
import os
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).parent
PYTHON = sys.executable
ROOT = TESTS.parent
ENV = {**os.environ, "PYTHONPATH": str(ROOT)}
PER_SUITE_TIMEOUT = int(os.environ.get("SUITE_TIMEOUT_SEC", "1500"))

# Script suites: exit 0 = pass. Order: fast synthetic first, data last.
SCRIPT_SUITES = [
    ("smoke", "tests/smoke_test.py"),
    ("smoke-quick", "tests/smoke_quick.py"),
    ("smoke-modular", "tests/smoke_modular.py"),
    ("smoke-pro", "tests/smoke_pro.py"),
    ("smoke-grid", "tests/smoke_grid.py"),
    ("smoke-tick-engine", "tests/smoke_tick_engine.py"),
    ("universal", "tests/test_universal.py"),
    ("tax-audit", "tests/test_tax_audit.py"),
    ("mcp-tools", "tests/test_mcp_tools.py"),
    ("strategies-regression", "tests/test_strategies_regression.py"),
    ("signal-contract", "tests/test_signal_contract.py"),
    ("e2e-pipeline", "tests/e2e_pipeline.py"),
]
SLOW = {"strategies-regression", "signal-contract", "e2e-pipeline"}

# Module self-tests: `python -m <mod>`, exit 0 = pass.
MODULE_SUITES = [
    "analysis.permutation_test",
    "analysis.review_gate",
    "analysis.auto_iterate",
    "analysis.universe_screener",
]


def _run(name, cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                           env=ENV, timeout=PER_SUITE_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"  FAIL {name} (timeout {PER_SUITE_TIMEOUT}s)")
        return False
    if r.returncode == 0:
        tail = (r.stdout or "").strip().splitlines()[-1:] if r.stdout else []
        print(f"  PASS {name}" + (f" — {tail[0][:100]}" if tail else ""))
        return True
    print(f"  FAIL {name} (exit {r.returncode})")
    print("  --- stdout tail ---")
    print("\n".join((r.stdout or "").splitlines()[-8:]))
    print("  --- stderr tail ---")
    print("\n".join((r.stderr or "").splitlines()[-8:]))
    return False


def main():
    quick = "--quick" in sys.argv
    print("=" * 60)
    print(f"Running Newmeta Backtester suite ({'quick' if quick else 'full'})")
    print("=" * 60)
    passed, failed = 0, 0
    for name, rel in SCRIPT_SUITES:
        if quick and name in SLOW:
            print(f"  SKIP {name} (--quick)")
            continue
        path = ROOT / rel
        if not path.exists():
            print(f"  SKIP {name} (not found)")
            continue
        print(f"\n>>> {name}: {rel}")
        if _run(name, [PYTHON, str(path)]):
            passed += 1
        else:
            failed += 1
    for mod in MODULE_SUITES:
        print(f"\n>>> module: {mod}")
        if _run(mod, [PYTHON, "-m", mod]):
            passed += 1
        else:
            failed += 1
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("*** ALL TESTS PASSED ***" if failed == 0 else "*** SOME TESTS FAILED ***")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
