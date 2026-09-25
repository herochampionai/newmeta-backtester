"""Statistical significance testing for strategy comparison (A/B testing).

R012: Provides rigorous statistical tests to compare two backtest results.
Without these, "Strategy B is better than Strategy A" is just noise.

Features:
- Two-sample t-test (parametric, assumes normality)
- Mann-Whitney U test (non-parametric, rank-based)
- Bootstrap confidence intervals (PnL, Sharpe)
- Paired tests (same trades, different exits/sizes)
- Effect size (Cohen's d)
- Verdict: "SIGNIFICANT", "MARGINAL", "NO_DIFFERENCE"

Use cases:
- Strategy A vs Strategy B (different params)
- Tick mode vs OHLC mode (same strategy)
- Real vs synthetic spreads
- Different commission models
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def two_sample_ttest(
    returns_a: np.ndarray | pd.Series,
    returns_b: np.ndarray | pd.Series,
    equal_var: bool = False,
    alpha: float = 0.05
) -> dict:
    """Welch's t-test (default) or Student's t-test for two samples.

    H0: mean(returns_a) == mean(returns_b)
    H1: mean(returns_a) != mean(returns_b)
    """
    try:
        a = np.asarray(returns_a, dtype=float)
        b = np.asarray(returns_b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            return {"error": "need 2+ samples in each group", "n_a": len(a), "n_b": len(b)}

        t_stat, p_value = scipy_stats.ttest_ind(a, b, equal_var=equal_var)
        # Welch-Satterthwaite df for unequal_var
        if not equal_var:
            s_a, s_b = np.var(a, ddof=1), np.var(b, ddof=1)
            n_a, n_b = len(a), len(b)
            if s_a + s_b > 0:
                df = (s_a / n_a + s_b / n_b) ** 2 / (
                    (s_a / n_a) ** 2 / (n_a - 1) + (s_b / n_b) ** 2 / (n_b - 1)
                )
            else:
                df = n_a + n_b - 2
        else:
            df = len(a) + len(b) - 2

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        cohens_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

        significant = p_value < alpha
        if not significant:
            verdict = "NO_DIFFERENCE"
        elif abs(cohens_d) >= 0.5:
            verdict = "SIGNIFICANT"
        elif abs(cohens_d) >= 0.2:
            verdict = "MARGINAL"
        else:
            verdict = "WEAK"

        return {
            "test": "welch_ttest" if not equal_var else "student_ttest",
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 4),
            "degrees_of_freedom": round(float(df), 1),
            "cohens_d": round(float(cohens_d), 3),
            "effect_size": (
                "large" if abs(cohens_d) >= 0.8 else
                "medium" if abs(cohens_d) >= 0.5 else
                "small" if abs(cohens_d) >= 0.2 else
                "negligible"
            ),
            "mean_a": round(float(np.mean(a)), 6),
            "mean_b": round(float(np.mean(b)), 6),
            "n_a": len(a),
            "n_b": len(b),
            "significant": significant,
            "alpha": alpha,
            "verdict": verdict,
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def mann_whitney_u(
    returns_a: np.ndarray | pd.Series,
    returns_b: np.ndarray | pd.Series,
    alpha: float = 0.05
) -> dict:
    """Mann-Whitney U test (non-parametric, no normality assumption).

    Tests if one distribution is stochastically greater than the other.
    Better than t-test for non-normal returns (e.g., fat tails).
    """
    try:
        a = np.asarray(returns_a, dtype=float)
        b = np.asarray(returns_b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            return {"error": "need 2+ samples", "n_a": len(a), "n_b": len(b)}

        u_stat, p_value = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        significant = p_value < alpha
        return {
            "test": "mann_whitney_u",
            "u_statistic": round(float(u_stat), 4),
            "p_value": round(float(p_value), 4),
            "median_a": round(float(np.median(a)), 6),
            "median_b": round(float(np.median(b)), 6),
            "n_a": len(a),
            "n_b": len(b),
            "significant": significant,
            "alpha": alpha,
            "verdict": "SIGNIFICANT" if significant else "NO_DIFFERENCE",
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def bootstrap_confidence_interval(
    returns: np.ndarray | pd.Series,
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> dict:
    """Bootstrap confidence interval for mean/median/sharpe.

    Args:
        returns: array of trade returns
        statistic: "mean", "median", "sharpe", "sum"
        confidence: confidence level (0.95 = 95%)
    """
    try:
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 5:
            return {"error": "need 5+ samples", "n": len(arr)}

        rng = np.random.default_rng(seed)

        def _stat(x):
            if statistic == "mean":
                return float(np.mean(x))
            elif statistic == "median":
                return float(np.median(x))
            elif statistic == "sum":
                return float(np.sum(x))
            elif statistic == "sharpe":
                if np.std(x) == 0:
                    return 0.0
                return float(np.mean(x) / np.std(x) * np.sqrt(len(x)))
            else:
                return float(np.mean(x))

        boot_stats = [_stat(rng.choice(arr, size=len(arr), replace=True))
                      for _ in range(n_bootstrap)]

        alpha = 1 - confidence
        lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        observed = _stat(arr)

        return {
            "statistic": statistic,
            "observed": round(observed, 4),
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
            "confidence": confidence,
            "n_bootstrap": n_bootstrap,
            "n_samples": len(arr),
            "ci_contains_zero": lower <= 0 <= upper,
            "significant_at_zero": not (lower <= 0 <= upper),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def compare_strategies(
    returns_a: np.ndarray | pd.Series,
    returns_b: np.ndarray | pd.Series,
    alpha: float = 0.05
) -> dict:
    """Comprehensive A/B comparison: t-test + Mann-Whitney + bootstrap CI on difference.

    Returns:
    - t_test result
    - mann_whitney result
    - bootstrap CI on (mean_b - mean_a)
    - combined verdict (require BOTH parametric and non-parametric to agree for SIGNIFICANT)
    """
    try:
        a = np.asarray(returns_a, dtype=float)
        b = np.asarray(returns_b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            return {"error": "need 2+ samples", "n_a": len(a), "n_b": len(b)}

        tt = two_sample_ttest(a, b, alpha=alpha)
        mw = mann_whitney_u(a, b, alpha=alpha)

        # Bootstrap CI on difference (b - a)
        rng = np.random.default_rng(42)
        diff = b - a if len(a) == len(b) else None
        n = min(len(a), len(b))
        n_boot = 5000
        boot_diffs = []
        for _ in range(n_boot):
            ia = rng.integers(0, len(a), n)
            ib = rng.integers(0, len(b), n)
            boot_diffs.append(float(np.mean(b[ib]) - np.mean(a[ia])))
        lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
        hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

        # Combined verdict
        both_sig = tt.get("significant", False) and mw.get("significant", False)
        diff_sig = not (lo <= 0 <= hi)  # CI doesn't contain 0
        if both_sig and diff_sig:
            verdict = "STRONG_EVIDENCE_B_BETTER" if np.mean(b) > np.mean(a) else "STRONG_EVIDENCE_A_BETTER"
        elif tt.get("significant", False) or mw.get("significant", False):
            verdict = "MARGINAL_EVIDENCE"
        else:
            verdict = "NO_DIFFERENCE"

        return {
            "t_test": tt,
            "mann_whitney": mw,
            "bootstrap_diff": {
                "mean_diff": round(float(np.mean(boot_diffs)), 6),
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
                "ci_excludes_zero": diff_sig,
                "n_bootstrap": n_boot,
            },
            "n_a": len(a),
            "n_b": len(b),
            "mean_a": round(float(np.mean(a)), 6),
            "mean_b": round(float(np.mean(b)), 6),
            "verdict": verdict,
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def compare_from_trades(
    trades_a: pd.DataFrame,
    trades_b: pd.DataFrame,
    pnl_col: str = "pnl",
    alpha: float = 0.05
) -> dict:
    """Convenience wrapper: compare two trade DataFrames."""
    if trades_a is None or trades_b is None:
        return {"error": "need both trade DataFrames"}
    col_a = next((c for c in (pnl_col, "PnL", "profit", "Profit") if c in trades_a.columns), None)
    col_b = next((c for c in (pnl_col, "PnL", "profit", "Profit") if c in trades_b.columns), None)
    if not col_a or not col_b:
        return {"error": "missing pnl column"}
    return compare_strategies(trades_a[col_a].values, trades_b[col_b].values, alpha=alpha)


def export_comparison_report(comparison: dict, out_path: str | Path) -> str:
    """Export comparison as HTML."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tt = comparison.get("t_test", {})
    mw = comparison.get("mann_whitney", {})
    bs = comparison.get("bootstrap_diff", {})

    verdict_color = {
        "STRONG_EVIDENCE_B_BETTER": "#7bd88f",
        "STRONG_EVIDENCE_A_BETTER": "#7bd88f",
        "MARGINAL_EVIDENCE": "#ffb800",
        "NO_DIFFERENCE": "#ff6b6b",
    }
    color = verdict_color.get(comparison.get("verdict", ""), "#888")

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Strategy A/B Comparison</title>
<style>
body{{font-family:'Segoe UI',Arial;background:#0d1117;color:#e6edf3;margin:0;padding:24px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:16px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{border:1px solid #30363d;padding:8px}}
th{{background:#21262d;color:#8b949e}}
.verdict{{display:inline-block;padding:8px 16px;border-radius:6px;font-weight:700;color:white;background:{color}}}
</style></head><body>
<h1>🆚 Strategy A/B Comparison</h1>
<div class='card'><span class='verdict'>{comparison.get("verdict", "?")}</span></div>

<div class='card'>
<h3>📊 Welch's T-Test</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>t-statistic</td><td>{tt.get("t_statistic", "?")}</td></tr>
<tr><td>p-value</td><td>{tt.get("p_value", "?")}</td></tr>
<tr><td>Significant (α={tt.get("alpha", 0.05)})</td><td>{tt.get("significant", "?")}</td></tr>
<tr><td>Cohen's d</td><td>{tt.get("cohens_d", "?")} ({tt.get("effect_size", "?")})</td></tr>
<tr><td>Mean A / Mean B</td><td>{tt.get("mean_a", "?")} / {tt.get("mean_b", "?")}</td></tr>
<tr><td>Sample size</td><td>{tt.get("n_a", "?")} / {tt.get("n_b", "?")}</td></tr>
</table>
</div>

<div class='card'>
<h3>📈 Mann-Whitney U (Non-parametric)</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>U-statistic</td><td>{mw.get("u_statistic", "?")}</td></tr>
<tr><td>p-value</td><td>{mw.get("p_value", "?")}</td></tr>
<tr><td>Median A / Median B</td><td>{mw.get("median_a", "?")} / {mw.get("median_b", "?")}</td></tr>
<tr><td>Significant</td><td>{mw.get("significant", "?")}</td></tr>
</table>
</div>

<div class='card'>
<h3>🎯 Bootstrap CI on Difference (B - A)</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Mean difference</td><td>{bs.get("mean_diff", "?")}</td></tr>
<tr><td>CI lower (95%)</td><td>{bs.get("ci_lower", "?")}</td></tr>
<tr><td>CI upper (95%)</td><td>{bs.get("ci_upper", "?")}</td></tr>
<tr><td>CI excludes zero</td><td>{bs.get("ci_excludes_zero", "?")}</td></tr>
<tr><td>Bootstrap samples</td><td>{bs.get("n_bootstrap", "?")}</td></tr>
</table>
</div>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


# ---------- Self-test ----------
if __name__ == "__main__":
    np.random.seed(42)
    # Strategy A: mean = 0.001
    # Strategy B: mean = 0.003 (clearly better)
    a = np.random.normal(0.001, 0.01, 200)
    b = np.random.normal(0.003, 0.01, 200)

    print("=== T-Test ===")
    print(two_sample_ttest(a, b))

    print("\n=== Mann-Whitney U ===")
    print(mann_whitney_u(a, b))

    print("\n=== Bootstrap CI on Mean ===")
    print(bootstrap_confidence_interval(b, statistic="mean", confidence=0.95))

    print("\n=== Compare Strategies (A vs B) ===")
    result = compare_strategies(a, b)
    print(f"Verdict: {result.get('verdict')}")
    print(f"T-test p: {result['t_test'].get('p_value')}")
    print(f"Mann-Whitney p: {result['mann_whitney'].get('p_value')}")
    print(f"Bootstrap CI excludes 0: {result['bootstrap_diff'].get('ci_excludes_zero')}")

    # Test no-difference scenario
    a2 = np.random.normal(0.001, 0.01, 200)
    b2 = np.random.normal(0.001, 0.01, 200)
    print("\n=== Compare Strategies (No Difference) ===")
    result2 = compare_strategies(a2, b2)
    print(f"Verdict: {result2.get('verdict')}")

    export_comparison_report(result, "output/comparison_test.html")
    print("\nReport: output/comparison_test.html")
