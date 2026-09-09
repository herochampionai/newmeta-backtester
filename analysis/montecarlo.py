"""Monte Carlo robustness: bootstrap trade sequences to get confidence
intervals on metrics. Also: parameter perturbation, regime shuffle."""
from __future__ import annotations
import numpy as np
import pandas as pd
from backtester.metrics import metrics_from_returns


def bootstrap_returns(returns: pd.Series, n_sims: int = 5000, block_size: int = 24,
                      seed: int = 42) -> np.ndarray:
    """Block bootstrap (preserves intra-day vol clustering). Returns (n_sims × len)."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    out = np.empty((n_sims, n))
    for i in range(n_sims):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n - block_size)
            idx.extend(range(start, start + block_size))
        idx = idx[:n]
        out[i] = returns.values[idx]
    return out


def ci_metrics(returns: pd.Series, n_sims: int = 5000, block: int = 24,
               confidence: float = 0.95, periods_per_year: int = 252 * 24) -> pd.DataFrame:
    sims = bootstrap_returns(returns, n_sims=n_sims, block_size=block)
    rows = []
    for r in sims:
        s = pd.Series(r)
        m = metrics_from_returns(s, periods_per_year=periods_per_year)
        rows.append(m)
    df = pd.DataFrame(rows)
    lo = (1 - confidence) / 2
    hi = 1 - lo
    summary = df.describe(percentiles=[lo, hi]).T
    return summary.loc[["sharpe", "calmar", "max_drawdown", "cagr"]]


def permutation_test(returns_a: pd.Series, returns_b: pd.Series, n_sims: int = 5000,
                     seed: int = 42) -> float:
    """Probability that the Sharpe difference arose by chance."""
    rng = np.random.default_rng(seed)
    obs = abs(returns_a.mean() / returns_a.std() - returns_b.mean() / returns_b.std())
    combined = pd.concat([returns_a, returns_b]).values
    n_a = len(returns_a)
    count = 0
    for _ in range(n_sims):
        rng.shuffle(combined)
        a = combined[:n_a]
        b = combined[n_a:]
        s = abs(a.mean() / a.std() - b.mean() / b.std())
        if s >= obs:
            count += 1
    return count / n_sims