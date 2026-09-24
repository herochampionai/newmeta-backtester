"""PBO — Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado, Zhu).

Answers: if I pick the best-looking backtest, what is the probability it
underperforms the median out-of-sample? PBO near 0 = selection is real;
near 0.5 = you are flipping coins; above 0.5 = selection is adversarial.

Method (combinatorially symmetric cross-validation, sampled):
  1. Stack N trial equity curves -> (T x N) log-return matrix.
  2. Partition rows into S contiguous blocks (default 16).
  3. Sample even IS/OOS block splits (full enumeration is C(16,8)/2 = 6435;
     200 sampled splits is standard practice and documented in the report).
  4. Per split: rank trials by IS Sharpe; take the IS-best trial's OOS rank r.
  5. PBO = fraction of splits where r falls below the OOS median.

Honest simplifications vs the paper: sampled (not exhaustive) splits, no
parametric logit-distribution fitting — the reported number is the raw
frequency, which is the quantity you act on.

Usage:
    from analysis.pbo import probability_of_overfitting
    rep = probability_of_overfitting(df, run_equity, param_grid)
    # run_equity(params) -> pd.Series of equity values
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import itertools
import json
import time


@dataclass
class PBOReport:
    """Overfitting probability with audit trail."""
    pbo: float
    n_trials: int
    n_splits: int
    n_partitions: int
    expected_oos_rank: float
    median_oos_rank: float
    sharpe_freq: float = 1.0  # fraction of splits with a defined IS-best
    elapsed_sec: float = 0.0

    def verdict(self) -> str:
        if self.pbo < 0.2:
            return "LOW"
        if self.pbo < 0.4:
            return "ELEVATED"
        return "HIGH"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict()
        for k in ("pbo", "expected_oos_rank", "median_oos_rank", "sharpe_freq"):
            d[k] = round(d[k], 3)
        return d

    def save(self, out_path: str | Path) -> str:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return str(out_path)


def _sharpe(x, periods_per_year: float) -> float:
    import numpy as np
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    sd = x.std(ddof=1)
    if sd == 0:
        return 0.0
    import math
    return float(x.mean() / sd * math.sqrt(periods_per_year))


def probability_of_overfitting(df, run_equity, param_grid: list[dict],
                               n_partitions: int = 16, n_splits: int = 200,
                               periods_per_year: float = 8760.0,
                               seed: int = 7, verbose: bool = True) -> PBOReport:
    """Estimate PBO over N trial configs.

    Args:
        df: price data (only its length/index used for alignment context).
        run_equity: callable(params) -> pd.Series of equity values.
        param_grid: candidate configs (N >= 3).
        n_partitions: contiguous time blocks (even S required).
        n_splits: sampled even IS/OOS splits (exhaustive if <= available).
        periods_per_year: Sharpe annualization (8760 = hourly).
    """
    import numpy as np
    import pandas as pd
    t0 = time.time()
    if n_partitions % 2:
        raise ValueError("n_partitions must be even")
    if len(param_grid) < 3:
        raise ValueError("need >= 3 trial configs")

    curves = []
    for p in param_grid:
        eq = run_equity(p)
        eq = pd.Series(eq).dropna().astype(float).reset_index(drop=True)
        curves.append(eq.values)
    T = min(len(c) for c in curves)
    if T < n_partitions * 4:
        raise ValueError(f"only {T} equity points for {n_partitions} partitions")
    rets = np.array([np.diff(np.log(np.maximum(c[:T], 1e-12))) for c in curves]).T
    n_trials = len(param_grid)

    # Partition row indices into S contiguous blocks.
    edges = np.linspace(0, len(rets), n_partitions + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_partitions)]

    # Enumerate even splits; sample when combinatorially explosive.
    rng = np.random.default_rng(seed)
    half = n_partitions // 2
    all_combos = list(itertools.combinations(range(n_partitions), half))
    # Symmetry: (A|B) == (B|A) — keep canonical half to avoid double counting.
    seen, combos = set(), []
    for c in all_combos:
        key = tuple(sorted(c))
        comp = tuple(sorted(set(range(n_partitions)) - set(c)))
        canon = min(key, comp)
        if canon not in seen:
            seen.add(canon)
            combos.append((set(key), set(comp)))
    if len(combos) > n_splits:
        combos = [combos[i] for i in rng.choice(len(combos), n_splits, replace=False)]

    below_median = 0
    ranks = []
    for is_blocks, oos_blocks in combos:
        is_idx = np.concatenate([blocks[b] for b in sorted(is_blocks)])
        oos_idx = np.concatenate([blocks[b] for b in sorted(oos_blocks)])
        is_sr = [_sharpe(rets[is_idx, j], periods_per_year) for j in range(n_trials)]
        oos_sr = [_sharpe(rets[oos_idx, j], periods_per_year) for j in range(n_trials)]
        best_is = int(np.argmax(is_sr))
        # OOS rank of the IS-best (1 = best). Ties -> average-ish via sort order.
        order = sorted(range(n_trials), key=lambda j: oos_sr[j], reverse=True)
        rank = order.index(best_is) + 1
        ranks.append(rank)
        if rank > n_trials / 2:
            below_median += 1

    import numpy as _np
    pbo = below_median / len(combos)
    if verbose:
        print(f"  PBO: {pbo:.3f} [{PBOReport(pbo, 0, 0, 0, 0.0, 0.0).verdict()}] "
              f"over {len(combos)} splits x {n_trials} trials, "
              f"median OOS rank of IS-best: {float(_np.median(ranks)):.1f}")
    return PBOReport(pbo=round(pbo, 4), n_trials=n_trials, n_splits=len(combos),
                     n_partitions=n_partitions,
                     expected_oos_rank=round(float(_np.mean(ranks)), 2),
                     median_oos_rank=round(float(_np.median(ranks)), 2),
                     elapsed_sec=round(time.time() - t0, 1))


# ---------- Self-test ----------
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # Case 1: one trial dominates everywhere -> PBO ~ 0.
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(0.001, 0.01, 2000))
    curves = {"a": 10000 + np.cumsum(rng.normal(0.5, 1.0, 2000)),
              "b": 10000 + np.cumsum(rng.normal(0.0, 1.0, 2000)),
              "c": 10000 + np.cumsum(rng.normal(-0.2, 1.0, 2000)),
              "d": 10000 + np.cumsum(rng.normal(0.1, 1.0, 2000))}
    df = pd.DataFrame({"close": [1.0] * 2000})
    rep = probability_of_overfitting(
        df, lambda p: pd.Series(curves[p["k"]]),
        [{"k": k} for k in ("a", "b", "c", "d")],
        n_partitions=8, n_splits=50, verbose=True)
    assert rep.pbo < 0.3, rep.pbo
    print("dominant-trial PBO OK:", rep.pbo, rep.verdict())

    # Case 2: pure noise trials -> PBO ~ 0.5 (coin flip).
    noisy = {f"t{i}": 10000 + np.cumsum(rng.normal(0, 1.0, 2000)) for i in range(6)}
    rep2 = probability_of_overfitting(
        df, lambda p: pd.Series(noisy[p["k"]]),
        [{"k": k} for k in noisy], n_partitions=8, n_splits=50, verbose=True)
    assert 0.2 < rep2.pbo < 0.8, rep2.pbo
    print("noise-trial PBO OK:", rep2.pbo, rep2.verdict())
    rep.save("output/reports/_selftest_pbo.json")
    print("SELF-TEST PASS")
