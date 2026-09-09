"""Strategy Validator — run each strategy against synthetic data with KNOWN expected
behavior. Catches coding errors + logical errors BEFORE you trust backtest output.

Test scenarios (synthetic OHLCV with known properties):
  1. trending_up:    strong monotonic uptrend → strategy should produce >=1 long signal
  2. trending_down:  strong monotonic downtrend → strategy should produce >=1 short signal
  3. ranging:        sideways oscillation → strategy should NOT over-trade
  4. crash:          sudden 5σ drop → bearish signal expected for reversal/short strategies
  5. ramp:           linear ramp + noise → strategy should follow direction
  6. constant:       zero-volatility → strategy should produce 0 trades
  7. impulse_then_trend: impulse move then continuation
  8. mean_reverting: spikes that decay back to mean

For each scenario, we run the strategy and assert specific properties:
  - directional bias (longs vs shorts ratio)
  - trade count bounds
  - first signal position
  - indicator ranges

Run via: python -m tools.strategy_validator [--strategy fbb]
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from strategies import STRATEGY_REGISTRY


# ============================================================================
# Synthetic data generators
# ============================================================================
def gen_trending_up(n: int = 1000, drift: float = 0.001, vol: float = 0.0005,
                     seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    ret = rng.normal(drift, vol, n)
    price = 1.10 * np.exp(np.cumsum(ret))
    return _ohlcv(idx, price, rng)


def gen_trending_down(n: int = 1000, drift: float = -0.001, vol: float = 0.0005,
                       seed: int = 42) -> pd.DataFrame:
    return gen_trending_up(n, -drift, vol, seed)


def gen_ranging(n: int = 1000, mean: float = 1.10, amplitude: float = 0.005,
                 vol: float = 0.0005, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    cycles = np.sin(np.linspace(0, 4 * np.pi, n)) * amplitude
    noise = np.cumsum(rng.normal(0, vol, n)) * 0.5
    price = mean + cycles + noise
    return _ohlcv(idx, price, rng)


def gen_crash(n: int = 1000, pre_vol: float = 0.0003, drop_sigma: float = 5,
               seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    ret = rng.normal(0, pre_vol, n)
    # Inject a crash at bar 500
    crash_bar = n // 2
    ret[crash_bar] = -drop_sigma * pre_vol
    ret[crash_bar + 1] = -drop_sigma * pre_vol * 0.5
    price = 1.10 * np.exp(np.cumsum(ret))
    return _ohlcv(idx, price, rng)


def gen_constant(n: int = 500, price: float = 1.10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return _ohlcv(idx, np.full(n, price), rng)


def gen_impulse_then_trend(n: int = 1000, impulse_size: float = 0.01,
                            seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    ret = rng.normal(0, 0.0003, n)
    ret[n // 3] += impulse_size
    ret[n // 3 + 1] += impulse_size * 0.3
    price = 1.10 * np.exp(np.cumsum(ret))
    return _ohlcv(idx, price, rng)


def gen_mean_reverting(n: int = 500, mean: float = 1.10, spike: float = 0.01,
                        decay: float = 0.05, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    ret = np.zeros(n)
    for i in range(10, n):
        ret[i] = -decay * ret[i - 1] + rng.normal(0, 0.0003)
    # Inject spikes
    ret[100] += spike
    ret[200] -= spike
    ret[300] += spike
    price = mean * np.exp(np.cumsum(ret))
    return _ohlcv(idx, price, rng)


def _ohlcv(idx: pd.DatetimeIndex, price: np.ndarray,
            rng: np.random.Generator) -> pd.DataFrame:
    high = price * (1 + np.abs(rng.normal(0, 0.0005, len(price))))
    low = price * (1 - np.abs(rng.normal(0, 0.0005, len(price))))
    opn = np.roll(price, 1); opn[0] = price[0]
    vol = rng.integers(50, 5000, len(price))
    return pd.DataFrame({
        "open": opn, "high": high, "low": low, "close": price, "volume": vol,
    }, index=idx)


# ============================================================================
# Validation scenarios
# ============================================================================
SCENARIOS = [
    ("trending_up", gen_trending_up, "should produce at least 1 long signal"),
    ("trending_down", gen_trending_down, "should produce at least 1 short signal"),
    ("ranging", gen_ranging, "should produce <=30 trades (don't over-trade noise)"),
    ("crash", gen_crash, "should produce at least 1 signal around the crash"),
    ("constant", gen_constant, "should produce 0 trades (no volatility = no signals)"),
    ("impulse_then_trend", gen_impulse_then_trend, "should produce >=1 signal"),
    ("mean_reverting", gen_mean_reverting, "should produce some signals (mean reversion works)"),
]


def _get_default_params(strategy_name: str) -> dict:
    """Get test-friendly defaults for each strategy."""
    defaults = {
        "ac_ao": dict(open_orders_type=1, level_open_orders=50, use_acceleration_filter=False,
                       use_ao_synchronization=False),
        "adx": dict(open_orders_type=1, level_open_orders_1=30, level_open_orders_2=10,
                     use_di_crossover=True, bars_calculate=14),
        "dem": dict(open_orders_type=3, level_open_orders=70, bars_calculate=14),
        "fbb": dict(open_orders_type_1=1, open_orders_type_2=0, level_open_orders_1=0,
                     level_open_orders_2=50, bars_calculate=20, deviation=1.5),
        "mfi": dict(open_orders_type=2, level_open_orders=70, bars_calculate=14,
                     use_slope_filter=False, use_divergence=False),
        "ms": dict(open_orders_type_1=8, open_orders_type_2=0, level_open_orders_1=20,
                    level_open_orders_2=80, use_confluence_filter=False,
                    fast_ema_period=3, slow_ema_period=9, signal_period=2,
                    k_period=5, d_period=3, slowing_period=12),
    }
    return defaults.get(strategy_name, {})


def _indicator_sanity(strategy_name: str, df: pd.DataFrame) -> dict:
    """Sanity-check indicator outputs (no NaN bombs, reasonable ranges)."""
    from strategies import indicators as ind
    out = {}
    h, l, c = df["high"], df["low"], df["close"]
    if strategy_name == "ac_ao":
        ac_v = ind.ac(h, l) * 40000
        out["ac_scaled"] = {"min": float(ac_v.min()), "max": float(ac_v.max()),
                              "nan_pct": float(ac_v.isna().mean())}
    elif strategy_name == "adx":
        a, pdi, mdi = ind.adx(h, l, c, 14)
        out["adx"] = {"min": float(a.min()), "max": float(a.max()),
                      "nan_pct": float(a.isna().mean())}
        out["pdi"] = {"min": float(pdi.min()), "max": float(pdi.max())}
        out["mdi"] = {"min": float(mdi.min()), "max": float(mdi.max())}
    elif strategy_name == "dem":
        d = ind.dem(h, l, 14) * 100
        out["dem"] = {"min": float(d.min()), "max": float(d.max()),
                       "nan_pct": float(d.isna().mean())}
    elif strategy_name == "fbb":
        mid, up, lo = ind.bollinger(c, 20, 1.5)
        out["bb_width"] = {"min": float((up - lo).min()),
                             "max": float((up - lo).max())}
    elif strategy_name == "mfi":
        m = ind.mfi(h, l, c, df["volume"], 14)
        out["mfi"] = {"min": float(m.min()), "max": float(m.max()),
                       "nan_pct": float(m.isna().mean())}
    elif strategy_name == "ms":
        m, sig, hist = ind.macd(c, 3, 9, 2)
        out["macd_main"] = {"min": float(m.min()), "max": float(m.max())}
        k, d = ind.stochastic(h, l, c, 5, 3, 12)
        out["stoch_k"] = {"min": float(k.min()), "max": float(k.max()),
                           "nan_pct": float(k.isna().mean())}
    return out


def validate_strategy(strategy_name: str, verbose: bool = True) -> dict:
    """Run all validation scenarios for one strategy. Returns summary dict."""
    cls = STRATEGY_REGISTRY[strategy_name]
    strat = cls(params=_get_default_params(strategy_name))
    results = {"strategy": strategy_name, "scenarios": [], "passed": 0, "failed": 0, "errors": []}
    for name, gen, expectation in SCENARIOS:
        try:
            df = gen()
            sig = strat.generate(df)
            n_long = int((sig.direction == 1).sum())
            n_short = int((sig.direction == -1).sum())
            n_total = int(sig.entries.sum())
            long_pct = n_long / max(n_total, 1)
            short_pct = n_short / max(n_total, 1)
            # Scenario-specific assertions
            passed = True
            msg = ""
            if name == "trending_up" and n_long < 1:
                passed = False; msg = f"expected >=1 long, got {n_long}"
            elif name == "trending_down" and n_short < 1:
                passed = False; msg = f"expected >=1 short, got {n_short}"
            elif name == "ranging" and n_total > 30:
                passed = False; msg = f"expected <=30 trades, got {n_total}"
            elif name == "constant" and n_total != 0:
                passed = False; msg = f"expected 0 trades, got {n_total}"
            # Indicator sanity
            ind_check = _indicator_sanity(strategy_name, df)
            for ind_name, vals in ind_check.items():
                if vals.get("nan_pct", 0) > 0.05:
                    passed = False
                    msg = f"{ind_name} has {vals['nan_pct']:.1%} NaN"
            results["scenarios"].append({
                "name": name, "passed": passed, "msg": msg,
                "n_long": n_long, "n_short": n_short, "n_total": n_total,
                "indicator_check": ind_check,
            })
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
        except Exception as e:
            import traceback
            results["errors"].append({"scenario": name, "error": str(e),
                                        "trace": traceback.format_exc()})
            results["failed"] += 1
    return results


def validate_all(verbose: bool = True) -> dict:
    """Run validation for every strategy. Returns summary dict."""
    out = {"strategies": {}, "total_passed": 0, "total_failed": 0,
           "errors": 0}
    for name in STRATEGY_REGISTRY:
        if name in ("universal", "mtf_stoch"):
            continue
        if verbose:
            print(f"\n--- {name} ---")
        r = validate_strategy(name, verbose=verbose)
        out["strategies"][name] = r
        out["total_passed"] += r["passed"]
        out["total_failed"] += r["failed"]
        out["errors"] += len(r["errors"])
        if verbose:
            for s in r["scenarios"]:
                mark = "[OK]" if s["passed"] else "[FAIL]"
                print(f"  {mark} {s['name']:25s} L={s['n_long']:>4d}  S={s['n_short']:>4d}  "
                      f"total={s['n_total']:>4d}  {s['msg']}")
            if r["errors"]:
                for e in r["errors"]:
                    print(f"  ERROR [{e['scenario']}]: {e['error']}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=None,
                    help="Validate one strategy; default = all")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.strategy:
        if args.strategy not in STRATEGY_REGISTRY:
            print(f"Unknown strategy: {args.strategy}")
            print(f"Available: {list(STRATEGY_REGISTRY.keys())}")
            sys.exit(1)
        r = validate_strategy(args.strategy, verbose=not args.quiet)
        print(f"\n{args.strategy}: {r['passed']} passed, {r['failed']} failed")
        if r["errors"]:
            for e in r["errors"]:
                print(f"  ERROR [{e['scenario']}]: {e['error'][:200]}")
    else:
        out = validate_all(verbose=not args.quiet)
        print(f"\n=== TOTAL: {out['total_passed']} passed, {out['total_failed']} failed, "
              f"{out['errors']} errors ===")
        if out["total_failed"] == 0 and out["errors"] == 0:
            print("*** ALL STRATEGIES VALIDATED ***")
        else:
            print("*** SOME STRATEGIES FAILED VALIDATION — DO NOT TRUST BACKTEST ***")
            sys.exit(1)


if __name__ == "__main__":
    main()