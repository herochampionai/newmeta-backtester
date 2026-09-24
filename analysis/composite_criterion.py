"""Composite criterion for Optuna optimization and parameter study.
Combines Sharpe + Sortino + Calmar + Profit Factor + Drawdown + WinRate into flexible objective functions.
User can select presets or tune custom weights in the UI.

R003: Includes Deflated Sharpe / PSR / multiple-testing-corrected criteria
to prevent accepting over-optimized results.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Any

# R003: import statistical significance criteria
from analysis.statistical_significance import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    criterion_deflated_sharpe as _dsr_fn,
    criterion_psr as _psr_fn,
    criterion_deflated_complex as _deflated_complex_fn,
)

# Reference normalization scales
SHARPE_REF = 2.0       # 2.0 Sharpe = 100% on this component
SORTINO_REF = 3.0      # 3.0 Sortino = 100%
CALMAR_REF = 3.0       # 3.0 Calmar = 100%
PF_REF = 2.5           # 2.5 PF = 100%
MDD_REF = 0.10         # -10% DD = 100% (we use -|MDD| / ref)
WINRATE_REF = 0.70     # 70% Win Rate = 100%


def composite_score(metrics: dict, weights: dict | None = None) -> float:
    """Combine metrics into a single 0-100 score.

    Default weights: 30% Sharpe + 25% Calmar + 20% ProfitFactor + 15% Drawdown + 10% Sortino.
    """
    if weights is None:
        weights = {"sharpe": 0.30, "calmar": 0.25, "pf": 0.20, "dd": 0.15, "sortino": 0.10}

    sharpe = float(metrics.get("sharpe", 0) or 0)
    calmar = float(metrics.get("calmar", 0) or 0)
    sortino = float(metrics.get("sortino", 0) or 0)
    pf = float(metrics.get("profit_factor", 0) or 0)
    max_dd = abs(float(metrics.get("max_drawdown", 0) or 0))
    win_rate = float(metrics.get("win_rate", 0) or 0)

    # Component normalization in [0, 1]
    sharpe_n = min(max(sharpe / SHARPE_REF, 0.0), 2.0)
    calmar_n = min(max(calmar / CALMAR_REF, 0.0), 2.0)
    sortino_n = min(max(sortino / SORTINO_REF, 0.0), 2.0)
    pf_n = min(max(pf / PF_REF, 0.0), 2.0)
    dd_n = min(max(1.0 - (max_dd / MDD_REF), 0.0), 1.0)
    win_n = min(max(win_rate / WINRATE_REF, 0.0), 1.5)

    w_sharpe = weights.get("sharpe", 0.0)
    w_calmar = weights.get("calmar", 0.0)
    w_sortino = weights.get("sortino", 0.0)
    w_pf = weights.get("pf", 0.0)
    w_dd = weights.get("dd", 0.0)
    w_win = weights.get("win_rate", 0.0)

    total_w = w_sharpe + w_calmar + w_sortino + w_pf + w_dd + w_win
    if total_w <= 0:
        total_w = 1.0

    score = (
        (w_sharpe * sharpe_n +
         w_calmar * calmar_n +
         w_sortino * sortino_n +
         w_pf * pf_n +
         w_dd * dd_n +
         w_win * win_n) / total_w
    ) * 100.0

    return float(score)


def criterion_sharpe_only(m: dict) -> float:
    v = float(m.get("sharpe", 0) or 0)
    return v if np.isfinite(v) else -10.0


def criterion_calmar_only(m: dict) -> float:
    v = float(m.get("calmar", 0) or 0)
    return v if np.isfinite(v) else -10.0


def criterion_sortino_only(m: dict) -> float:
    v = float(m.get("sortino", 0) or 0)
    return v if np.isfinite(v) else -10.0


def criterion_pf_only(m: dict) -> float:
    v = float(m.get("profit_factor", 0) or 0)
    return v if np.isfinite(v) else 0.0


def criterion_net_pnl_only(m: dict) -> float:
    v = float(m.get("net_pnl", 0) or 0)
    return v if np.isfinite(v) else -1e9


def criterion_win_rate_guarded(m: dict) -> float:
    """Win rate with trade count guard."""
    wr = float(m.get("win_rate", 0) or 0)
    n = int(m.get("n_trades", 0) or 0)
    if n < 30:
        return wr * (n / 30.0)
    return wr


# ---------- R003: Statistical significance criteria ----------
def criterion_deflated_sharpe_simple(m: dict) -> float:
    """Deflated Sharpe Ratio (single-trial approximation).

    For single-backtest scoring, uses n_trades as T and applies DSR
    with default skewness/kurtosis. For multi-trial analysis, use
    deflated_criterion_score() from statistical_significance.
    """
    sr = float(m.get("sharpe", 0) or 0)
    n_trades = int(m.get("n_trades", 0) or 0)
    if n_trades < 2:
        return sr  # not enough data
    # Treat each backtest as 1 trial; with 1 trial DSR == SR
    # For more rigor, the optimizer's n_trials should be passed via context
    return _dsr_fn(m, n_trials=max(1, n_trades // 30), skewness=0.0, kurtosis=3.0)


def criterion_psr_simple(m: dict) -> float:
    """Probabilistic Sharpe Ratio (P(SR > 0)).

    Simple version: uses n_trades as T, benchmark SR = 0.
    """
    sr = float(m.get("sharpe", 0) or 0)
    n_trades = int(m.get("n_trades", 0) or 0)
    if n_trades < 2:
        return sr
    return _psr_fn(m, n_trials=n_trades, benchmark_sr=0.0)


def criterion_deflated_complex(m: dict) -> float:
    """MQL5 Complex Score with SR guard (simple proxy for DSR-corrected score)."""
    return _deflated_complex_fn(m)


def criterion_pf_drawdown_realistic(m: dict) -> float:
    """Institutional score: high PF, low Drawdown, robust sample size."""
    pf = float(m.get("profit_factor", 0) or 0)
    max_dd = abs(float(m.get("max_drawdown", 0) or 0))
    n_trades = int(m.get("n_trades", 0) or 0)

    if not np.isfinite(pf):
        pf = 0.0
    if not np.isfinite(max_dd):
        max_dd = 1.0

    pf_score = min(pf / 2.5, 2.0) * 50.0
    dd_score = max(0.0, 1.0 - (max_dd / 0.20)) * 40.0
    trade_score = min(n_trades / 50.0, 1.0) * 10.0

    low_trade_penalty = 35.0 if n_trades < 25 else 0.0
    high_dd_penalty = 30.0 if max_dd > 0.25 else 0.0

    return float(pf_score + dd_score + trade_score - low_trade_penalty - high_dd_penalty)


def criterion_drawdown_only(m: dict) -> float:
    """Maximize = minimize drawdown. Returns negative DD so higher is better."""
    dd = abs(float(m.get("max_drawdown", 1.0) or 1.0))
    n = int(m.get("n_trades", 0) or 0)
    if not np.isfinite(dd):
        return -1e9
    guard = min(n / 30.0, 1.0)  # ignore lucky 3-trade low-DD flukes
    return float(-dd * 100.0 * guard)


def criterion_mql5_complex(m: dict) -> float:
    """MQL5 'Complex Criterion max' mirror (0-100, staged like MT5).

    MT5 stages: Deals -> Expected Payoff -> Recovery -> Sharpe -> DD.
    Formula undisclosed; this reproduces its behavior + color zones
    (red <20, dark-green >80):
      0-20  = deal-count gate (MT5 discards <10-deal passes first)
      20-100 = Expected(25) + Recovery(25) + Sharpe(20) + DD(10) blended
    Any losing / tiny-sample pass stays red, exactly like Tester.
    """
    n = int(m.get("n_trades", 0) or 0)
    net = float(m.get("net_pnl", 0) or 0)
    exp = float(m.get("expectancy", (net / n) if n else 0) or 0)
    rec = float(m.get("recovery_factor", 0) or 0)
    sharpe = float(m.get("sharpe", 0) or 0)
    dd = abs(float(m.get("max_drawdown", 1.0) or 1.0))
    pf = float(m.get("profit_factor", 0) or 0)
    if not all(np.isfinite(v) for v in (exp, rec, sharpe, dd, pf)):
        return 0.0
    # Stage 1 — deals gate (MT5 filters first by number of deals)
    if n < 10:
        return float(n * 2.0)  # 0-18 red zone
    if net <= 0 or exp <= 0:
        return float(min(19.0, 5.0 + n * 0.1))  # losing stays red
    deals_c = min(n / 200.0, 1.0) * 20.0            # 200+ deals = full
    exp_c = min(max(exp, 0.0) / max(net / max(n, 1) * 2.0, 1e-9), 1.0) * 0.0  # placeholder
    # Normalize expected payoff vs avg-risk: use PF as efficiency proxy (stable across symbols)
    exp_c = min(max(pf - 1.0, 0.0) / 1.5, 1.0) * 25.0
    rec_c = min(max(rec, 0.0) / 5.0, 1.0) * 25.0    # rec 5+ = excellent (MT5 scale)
    sh_c = min(max(sharpe, 0.0) / 2.0, 1.0) * 20.0  # sharpe 2+ = full
    dd_c = max(0.0, 1.0 - dd / 0.30) * 10.0         # 30%+ DD = zero
    return float(min(100.0, deals_c + exp_c + rec_c + sh_c + dd_c))


def criterion_mql5_balance(m: dict) -> float:
    return float(m.get("net_pnl", 0) or 0)


def criterion_mql5_expected(m: dict) -> float:
    n = int(m.get("n_trades", 0) or 0)
    v = float(m.get("expectancy", 0) or 0)
    if n < 10 or not np.isfinite(v):
        return -1e9
    return v


def criterion_mql5_recovery(m: dict) -> float:
    return float(m.get("recovery_factor", 0) or 0)


def criterion_full(m: dict) -> float:
    """Full-criterion study: PF + DD + WinRate + Sharpe + Calmar + sample guard.

    Single number for 'which ticker fits the strategy most'. Balanced for
    ranking symbols, not just params. Higher is better.
    """
    pf = float(m.get("profit_factor", 0) or 0)
    dd = abs(float(m.get("max_drawdown", 1.0) or 1.0))
    wr = float(m.get("win_rate", 0) or 0)
    sharpe = float(m.get("sharpe", 0) or 0)
    calmar = float(m.get("calmar", 0) or 0)
    n = int(m.get("n_trades", 0) or 0)
    net = float(m.get("net_pnl", 0) or 0)
    if not all(np.isfinite(v) for v in (pf, dd, wr, sharpe, calmar, net)):
        return -1e9
    if n < 20 or net <= 0:
        return -50.0 + min(pf, 1.0) * 10.0
    pf_c = min(pf / 2.0, 2.0) * 30.0
    dd_c = max(0.0, 1.0 - dd / 0.25) * 30.0
    wr_c = min(max(wr / 0.6, 0.0), 1.2) * 15.0
    sh_c = min(max(sharpe / 2.0, 0.0), 2.0) * 15.0
    ca_c = min(max(calmar / 3.0, 0.0), 2.0) * 10.0
    return float(pf_c + dd_c + wr_c + sh_c + ca_c)


def criterion_composite(weights: dict) -> Callable[[dict], float]:
    """Returns a callable that scores via composite_score with given weights."""
    def _score(m: dict) -> float:
        return composite_score(m, weights)
    return _score


CRITERION_PRESETS = {
    "MQL5 Complex Criterion max (0-100)": criterion_mql5_complex,
    "MQL5 Balance max": criterion_mql5_balance,
    "MQL5 Profit Factor max": criterion_pf_only,
    "MQL5 Expected Payoff max": criterion_mql5_expected,
    "MQL5 Recovery Factor max": criterion_mql5_recovery,
    "MQL5 Sharpe Ratio max": criterion_sharpe_only,
    "MQL5 Drawdown min": criterion_drawdown_only,
    "Full Criterion (All-in-One)": criterion_full,
    "PF/DD Realistic (Institutional)": criterion_pf_drawdown_realistic,
    "Composite (Balanced)": criterion_composite({"sharpe": 0.35, "calmar": 0.25, "pf": 0.20, "dd": 0.10, "sortino": 0.10}),
    "Composite (Conservative Low-DD)": criterion_composite({"sharpe": 0.20, "calmar": 0.35, "pf": 0.15, "dd": 0.30}),
    "Composite (Aggressive Max Growth)": criterion_composite({"sharpe": 0.40, "calmar": 0.10, "pf": 0.40, "dd": 0.05, "sortino": 0.05}),
    "Sharpe Ratio": criterion_sharpe_only,
    "Calmar Ratio": criterion_calmar_only,
    "Sortino Ratio": criterion_sortino_only,
    "Profit Factor": criterion_pf_only,
    "Drawdown (Minimize)": criterion_drawdown_only,
    "Net P&L ($)": criterion_net_pnl_only,
    "Win Rate (%) (Guarded)": criterion_win_rate_guarded,
    # R003: Deflated Sharpe / PSR / multiple testing correction
    "Deflated Sharpe Ratio (DSR)": criterion_deflated_sharpe_simple,
    "Probabilistic SR (PSR)": criterion_psr_simple,
    "Deflated Complex (MQL5 + SR guard)": criterion_deflated_complex,
}


def composite_to_dict(score: float, weights: dict) -> dict:
    return {
        "sharpe_weight": weights.get("sharpe", 0),
        "calmar_weight": weights.get("calmar", 0),
        "sortino_weight": weights.get("sortino", 0),
        "pf_weight": weights.get("pf", 0),
        "dd_weight": weights.get("dd", 0),
        "win_rate_weight": weights.get("win_rate", 0),
        "total_score": score,
    }
