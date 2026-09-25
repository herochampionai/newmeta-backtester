"""Bootstrap Sharpe significance — is this Sharpe luck? (White-lite)

Resample trades WITH REPLACEMENT N times, recompute Sharpe each time, and
ask: what fraction of resamples fall at or below zero? That fraction is the
p-value against H0 (true Sharpe <= 0). No normality assumption (unlike
t-tests), no multiple-testing correction needed for the single question.

Why resampling-with-replacement and not order shuffling: Sharpe from trade
PnLs is order-invariant (shuffling changes neither mean nor sd), so a pure
permutation test would return p ~= 1 for EVERY input, skilled or not. The
bootstrap instead measures sampling variability — the right null.

This is the cheap, robust sibling of PBO: PBO asks "will my selection
procedure disappoint"; bootstrap asks "is THIS result distinguishable
from noise". Report both.

Usage:
    from analysis.permutation_test import permutation_pvalue
    rep = permutation_pvalue(trade_pnls, n_permutations=5000, seed=7)
    # rep: {p_value, observed_sharpe, verdict, ...}
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PermutationReport:
    """Bootstrap significance outcome (resampling with replacement)."""
    p_value: float
    observed_sharpe: float
    mean_shuffled_sharpe: float  # mean bootstrap Sharpe (sanity: ~= observed)
    n_permutations: int
    n_trades: int
    elapsed_sec: float = 0.0

    def verdict(self, alpha: float = 0.05) -> str:
        if self.p_value < 0.01:
            return "STRONG"
        if self.p_value < alpha:
            return "WEAK"
        return "NO_EVIDENCE"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict()
        for k in ("p_value", "observed_sharpe", "mean_shuffled_sharpe"):
            d[k] = round(d[k], 4)
        return d

    def save(self, out_path: str | Path) -> str:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return str(out_path)


def _sharpe_from_pnls(pnls, periods_per_year: float = 1.0) -> float:
    """Sharpe of a trade-PnL sequence (per-trade units by default)."""
    import numpy as np
    x = np.asarray(pnls, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    sd = x.std(ddof=1)
    if sd == 0:
        return 0.0
    import math
    return float(x.mean() / sd * math.sqrt(periods_per_year))


def permutation_pvalue(trade_pnls, n_permutations: int = 5000,
                       periods_per_year: float = 1.0, seed: int = 7,
                       verbose: bool = True) -> PermutationReport:
    """Bootstrap H0 (true Sharpe <= 0): fraction of resampled Sharpes <= 0.

    Resampling is WITH replacement (plain order shuffling cannot work here:
    mean/sd Sharpe is order-invariant, so every shuffle would tie). Result
    fields keep legacy names (p_value, n_permutations) for API stability.
    """
    import numpy as np
    t0 = time.time()
    x = np.asarray(list(trade_pnls), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        raise ValueError(f"need >= 10 trades, have {len(x)}")
    obs = _sharpe_from_pnls(x, periods_per_year)
    rng = np.random.default_rng(seed)
    beats = 0
    boot_sum = 0.0
    for _ in range(n_permutations):
        sample = rng.choice(x, size=len(x), replace=True)
        s = _sharpe_from_pnls(sample, periods_per_year)
        boot_sum += s
        if s <= 0:
            beats += 1
    # Add-one smoothing so p is never exactly 0 (honest with finite draws).
    p = (beats + 1) / (n_permutations + 1)
    rep = PermutationReport(p_value=round(p, 4), observed_sharpe=round(obs, 4),
                            mean_shuffled_sharpe=round(boot_sum / n_permutations, 4),
                            n_permutations=n_permutations,
                            n_trades=len(x), elapsed_sec=round(time.time() - t0, 1))
    if verbose:
        print(f"  bootstrap: p={p:.4f} [{rep.verdict()}] "
              f"obs Sharpe={obs:.3f} over {n_permutations} resamples")
    return rep


# ---------- Self-test ----------
if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)

    # Skilled: consistent edge -> tiny p.
    skilled = list(rng.normal(5, 20, 100))
    r1 = permutation_pvalue(skilled, n_permutations=1000, verbose=True)
    assert r1.p_value < 0.05, r1.p_value
    assert r1.verdict() in ("STRONG", "WEAK")

    # Noise: zero-mean flips -> large p.
    noise = list(rng.normal(0, 20, 100))
    r2 = permutation_pvalue(noise, n_permutations=1000, verbose=True)
    assert r2.p_value > 0.05, r2.p_value
    assert r2.verdict() == "NO_EVIDENCE"

    # Mean bootstrap Sharpe tracks the observed (sanity on the new field).
    assert abs(r1.mean_shuffled_sharpe - r1.observed_sharpe) < 0.15, r1.mean_shuffled_sharpe
    assert abs(r2.mean_shuffled_sharpe) < 0.15, r2.mean_shuffled_sharpe
    print("SELF-TEST PASS")
