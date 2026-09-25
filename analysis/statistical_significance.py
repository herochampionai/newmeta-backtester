"""Statistical significance & multiple testing correction for optimization trust.

Features:
- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014):
  Adjusts Sharpe for the number of trials tested (anti-overfit).
- Probabilistic Sharpe Ratio (Marcos Lopez de Prado):
  P(SR > benchmark SR) with confidence.
- Multiple testing correction:
  * Bonferroni (conservative)
  * Benjamini-Hochberg FDR (less conservative, controls false discovery rate)
- Integration with composite criterion: deflated criterion scores.

References:
- Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio".
- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning".
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------- Probabilistic Sharpe Ratio ----------
def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float = 0.0,
    n_trials: int = 1,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> dict:
    """P(SR > benchmark) — probability that true SR exceeds benchmark.

    PSR = Φ( (SR_hat - SR*) / σ(SR_hat) )
    where σ(SR_hat) = sqrt(1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / sqrt(T-1)
    For multiple trials, use Deflated Sharpe (which extends PSR).
    """
    try:
        if n_trials < 2 or not np.isfinite(observed_sr):
            return {"psr": None, "note": "need n_trials >= 2"}

        # Standard error of Sharpe ratio
        sr = float(observed_sr)
        sr_std = math.sqrt(
            (1 + 0.5 * sr**2 - skewness * sr + (kurtosis - 3) / 4 * sr**2)
            / max(n_trials - 1, 1)
        )
        if sr_std == 0:
            return {"psr": None, "note": "zero std"}

        z = (sr - benchmark_sr) / sr_std
        psr = float(scipy_stats.norm.cdf(z))

        return {
            "psr": round(psr, 4),
            "z_score": round(z, 4),
            "sr_std": round(sr_std, 4),
            "benchmark_sr": benchmark_sr,
            "verdict": (
                "STRONG" if psr > 0.95 else
                "GOOD" if psr > 0.85 else
                "WEAK" if psr > 0.5 else
                "POOR"
            )
        }
    except Exception as e:
        return {"psr": None, "error": str(e)[:100]}


# ---------- Deflated Sharpe Ratio ----------
def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sr: float = 0.0
) -> dict:
    """Deflated Sharpe Ratio: P(max SR across n_trials > benchmark SR).

    Adjusts for multiple testing. If you tested 100 param combinations and the
    best has SR=2.0, this asks: what's the probability the best of 100 random
    SRs exceeds benchmark?

    Reference: Bailey & Lopez de Prado (2014), Eq. 3-4.
    """
    try:
        if n_trials < 2 or not np.isfinite(observed_sr):
            return {"dsr": None, "note": "need n_trials >= 2"}

        sr = float(observed_sr)
        # Expected maximum of N i.i.d. standard normals
        e_max = _expected_max_normal(n_trials)

        # Standard error of observed SR
        sr_std = math.sqrt(
            (1 + 0.5 * sr**2 - skewness * sr + (kurtosis - 3) / 4 * sr**2)
            / max(n_trials - 1, 1)
        )

        # Deflated Sharpe: P(true SR > benchmark | best of n_trials)
        # DSR = Φ( (SR_hat - SR*) * sqrt(T-1) / σ_SR )
        # adjusted by e_max for multiple testing
        if sr_std == 0:
            return {"dsr": None, "note": "zero std"}

        # z-score adjusted for multiple testing (subtract expected max)
        z = ((sr - benchmark_sr) * math.sqrt(max(n_trials - 1, 1)) / sr_std) - e_max
        dsr = float(scipy_stats.norm.cdf(z))

        return {
            "dsr": round(dsr, 4),
            "z_adjusted": round(z, 4),
            "expected_max_sr": round(e_max, 4),
            "n_trials": n_trials,
            "observed_sr": sr,
            "benchmark_sr": benchmark_sr,
            "verdict": (
                "STRONG" if dsr > 0.95 else
                "GOOD" if dsr > 0.85 else
                "WEAK" if dsr > 0.5 else
                "OVERFIT" if dsr < 0.1 else
                "CHECK"
            )
        }
    except Exception as e:
        return {"dsr": None, "error": str(e)[:100]}


def _expected_max_normal(n: int) -> float:
    """Expected maximum of N i.i.d. standard normals. E[max(Z_1,...,Z_N)].

    Approximation from Bailey & Lopez de Prado (2014):
    E[max] ≈ (1-γ)Φ^(-1)(1-1/N) + γΦ^(-1)(1-1/(Ne))
    where γ ≈ 0.5772 (Euler-Mascheroni).
    """
    if n <= 1:
        return 0.0
    gamma = 0.5772156649  # Euler-Mascheroni
    inv_cdf1 = scipy_stats.norm.ppf(1 - 1 / n)
    inv_cdf2 = scipy_stats.norm.ppf(1 - 1 / (n * math.e))
    return (1 - gamma) * inv_cdf1 + gamma * inv_cdf2


# ---------- Multiple Testing Correction ----------
def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> dict:
    """Bonferroni: reject if p < alpha / n_tests. Conservative but simple."""
    n = len(p_values)
    if n == 0:
        return {"rejected": [], "adjusted_alpha": alpha, "n_tests": 0}
    adj_alpha = alpha / n
    rejected = [i for i, p in enumerate(p_values) if p < adj_alpha]
    return {
        "method": "bonferroni",
        "n_tests": n,
        "original_alpha": alpha,
        "adjusted_alpha": round(adj_alpha, 6),
        "rejected_indices": rejected,
        "n_rejected": len(rejected)
    }


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR correction. Less conservative than Bonferroni.

    Procedure:
    1. Sort p-values
    2. For each rank k, threshold = k/n * alpha
    3. Reject all p_i <= max threshold
    """
    n = len(p_values)
    if n == 0:
        return {"rejected": [], "n_tests": 0}

    # Sort with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    sorted_p = [p for _, p in indexed]

    # Find largest k where p_(k) <= k/n * alpha
    rejected_sorted = []
    for k in range(1, n + 1):
        threshold = k / n * alpha
        if sorted_p[k - 1] <= threshold:
            rejected_sorted.append(k)

    # All p-values up to max k are rejected
    max_k = max(rejected_sorted) if rejected_sorted else 0
    rejected_set = {indexed[i][0] for i in range(max_k)}

    return {
        "method": "benjamini_hochberg",
        "n_tests": n,
        "alpha": alpha,
        "rejected_indices": sorted(rejected_set),
        "n_rejected": len(rejected_set),
        "thresholds": [round(k / n * alpha, 6) for k in range(1, n + 1)]
    }


# ---------- Integration with optimization trials ----------
def deflated_criterion_score(
    trials_df: pd.DataFrame,
    metric: str = "score",
    benchmark_sr: float = 0.0,
    min_trials: int = 20
) -> dict:
    """Compute DSR-aware score for the best trial in an Optuna study.

    Returns a dict with:
    - dsr: Deflated Sharpe probability
    - best_score: the raw best score
    - deflated_score: best_score * DSR (penalized for overfitting risk)
    - verdict: STRONG/GOOD/WEAK/OVERFIT/CHECK
    """
    try:
        if trials_df is None or len(trials_df) < min_trials:
            return {"dsr": None, "note": f"need {min_trials}+ trials"}

        scores = trials_df[metric].values
        best = float(np.nanmax(scores))
        # Estimate skewness/kurtosis of trial scores
        clean = scores[np.isfinite(scores)]
        if len(clean) < 5:
            return {"dsr": None, "note": "insufficient clean trials"}
        skew = float(scipy_stats.skew(clean))
        kurt = float(scipy_stats.kurtosis(clean, fisher=False))  # excess=False → regular

        dsr_result = deflated_sharpe_ratio(
            observed_sr=best,
            n_trials=len(clean),
            skewness=skew,
            kurtosis=kurt,
            benchmark_sr=benchmark_sr
        )
        dsr = dsr_result.get("dsr")
        if dsr is None:
            return dsr_result

        deflated = best * dsr
        return {
            "dsr": dsr,
            "best_score": round(best, 4),
            "deflated_score": round(deflated, 4),
            "n_trials": len(clean),
            "skewness": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "verdict": dsr_result["verdict"]
        }
    except Exception as e:
        return {"dsr": None, "error": str(e)[:100]}


# ---------- Criterion functions for Optuna ----------
def criterion_deflated_sharpe(metrics: dict, n_trials: int = 1, skewness: float = 0.0,
                              kurtosis: float = 3.0) -> float:
    """Optimizable criterion: Deflated Sharpe Ratio.

    Note: This requires n_trials/skewness/kurtosis to be set per-study.
    Use deflated_criterion_score() in the post-optimization analysis.
    """
    sr = float(metrics.get("sharpe", 0) or 0)
    dsr_result = deflated_sharpe_ratio(sr, n_trials, skewness, kurtosis)
    dsr = dsr_result.get("dsr")
    if dsr is None or not np.isfinite(dsr):
        return sr  # fall back to raw Sharpe
    return dsr


def criterion_psr(metrics: dict, n_trials: int = 1, benchmark_sr: float = 0.0,
                  skewness: float = 0.0, kurtosis: float = 3.0) -> float:
    """Probabilistic Sharpe Ratio (no multiple testing correction)."""
    sr = float(metrics.get("sharpe", 0) or 0)
    psr_result = probabilistic_sharpe_ratio(sr, benchmark_sr, n_trials, skewness, kurtosis)
    psr = psr_result.get("psr")
    if psr is None or not np.isfinite(psr):
        return sr
    return psr


def criterion_deflated_complex(metrics: dict) -> float:
    """Combine MQL5 Complex Score with Deflated Sharpe correction.

    Use when n_trials is known (set via context).
    For post-hoc analysis, use deflated_criterion_score() instead.
    """
    from analysis.composite_criterion import criterion_mql5_complex
    base = criterion_mql5_complex(metrics)
    sr = float(metrics.get("sharpe", 0) or 0)
    # Simple penalty: if SR < 0.5, cap the score
    if sr < 0.5:
        return min(base, 50.0)
    return base


# ---------- Self-test ----------
if __name__ == "__main__":
    # PSR test
    psr = probabilistic_sharpe_ratio(2.0, 0.0, 252)
    print(f"PSR(SR=2.0, T=252): {psr}")

    # DSR test (with multiple testing penalty)
    dsr10 = deflated_sharpe_ratio(2.0, 10)  # 10 trials
    print(f"DSR(SR=2.0, N=10): {dsr10}")
    dsr100 = deflated_sharpe_ratio(2.0, 100)  # 100 trials
    print(f"DSR(SR=2.0, N=100): {dsr100}")
    dsr1000 = deflated_sharpe_ratio(2.0, 1000)  # 1000 trials
    print(f"DSR(SR=2.0, N=1000): {dsr1000}")

    # Multiple testing correction
    pvals = [0.001, 0.01, 0.04, 0.03, 0.5]
    print(f"\nBonferroni: {bonferroni_correction(pvals)}")
    print(f"BH: {benjamini_hochberg(pvals)}")

    # Deflated criterion from trials
    import pandas as pd
    np.random.seed(42)
    trials = pd.DataFrame({"score": np.random.randn(100).cumsum() + 5})
    print(f"\nDeflated criterion (100 trials): {deflated_criterion_score(trials)}")
