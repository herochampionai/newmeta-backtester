"""Portfolio allocation: Markowitz (Ledoit-Wolf shrunk), risk-parity, Kelly, equal.
Inputs: per-strategy daily returns DataFrame. Output: weight vector + portfolio stats."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize


def _to_daily(bar_returns: pd.DataFrame, bars_per_day: int = 24) -> pd.DataFrame:
    """Aggregate bar returns into daily returns (geometric compounding)."""
    return (1 + bar_returns).groupby(bar_returns.index.date).prod() - 1


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance estimator. Better than sample cov for n<p."""
    return LedoitWolf().fit(returns.values).covariance_


def markowitz(mu: np.ndarray, cov: np.ndarray, max_w: float = 0.4,
              max_dd_constraint: float | None = None) -> np.ndarray:
    """Long-only max-Sharpe with L2 regularization, weight caps, sum-to-1."""
    n = len(mu)
    eps = 1e-4 * np.eye(n)
    cov_r = cov + eps

    def neg_sharpe(w):
        p = w @ mu
        v = np.sqrt(w @ cov_r @ w)
        return -(p / v) if v > 0 else 1e9

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if max_dd_constraint is not None:
        # Add max-DD proxy via vol: w'*Sigma*w <= max_dd^2
        cons.append({"type": "ineq",
                     "fun": lambda w: max_dd_constraint**2 - w @ cov_r @ w})
    bounds = [(0, max_w)] * n
    w0 = np.ones(n) / n
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    w = res.x if res.success else w0
    return np.clip(w, 0, None) / max(np.sum(w), 1e-9)


def risk_parity(cov: np.ndarray) -> np.ndarray:
    """Equal risk contribution weights: w_i ∝ 1/sqrt(Σ_ii)."""
    inv_vol = 1.0 / np.sqrt(np.diag(cov) + 1e-12)
    w = inv_vol / inv_vol.sum()
    return w


def kelly(mu: np.ndarray, cov: np.ndarray, max_w: float = 0.25) -> np.ndarray:
    """Naive per-strategy Kelly clipped to max_w. f_i* = mu_i / sigma_i^2 (Gaussian)."""
    var = np.diag(cov) + 1e-12
    f = mu / var
    f = np.clip(f, 0, max_w)
    return f / max(f.sum(), 1e-9)


def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def allocate(daily_returns: pd.DataFrame, method: str = "markowitz",
             max_w: float = 0.4, max_dd: float | None = 0.15) -> np.ndarray:
    """daily_returns: strategy × day. Returns weight vector (sums to 1)."""
    mu = daily_returns.mean().values
    cov = ledoit_wolf_cov(daily_returns)
    n = daily_returns.shape[1]
    if method == "markowitz":
        return markowitz(mu, cov, max_w=max_w, max_dd_constraint=max_dd)
    if method == "risk_parity":
        return risk_parity(cov)
    if method == "kelly":
        return kelly(mu, cov, max_w=max_w)
    if method == "equal":
        return equal_weight(n)
    raise ValueError(f"unknown method {method}")