"""Composite criterion for Optuna optimization.
Combines Sharpe + Calmar + ProfitFactor + max_drawdown into a single score.
User picks weights via sliders in the UI."""
from __future__ import annotations
import numpy as np


# Reference ranges for normalization (so each component contributes 0-1 typically)
SHARPE_REF = 2.0       # 2.0 Sharpe = 100% on this component
CALMAR_REF = 3.0      # 3.0 Calmar = 100%
PF_REF = 2.5          # 2.5 PF = 100%
MDD_REF = 0.10        # -10% DD = 100% (we use -|MDD| / ref)


def composite_score(metrics: dict, weights: dict | None = None) -> float:
    """Combine metrics into a single 0-100 score.

    Default weights: 40% Sharpe + 30% Calmar + 20% ProfitFactor + 10% (inverse Drawdown).
    """
    if weights is None:
        weights = {"sharpe": 0.4, "calmar": 0.3, "pf": 0.2, "dd": 0.1}

    sharpe = metrics.get("sharpe", 0)
    calmar = metrics.get("calmar", 0)
    pf = metrics.get("profit_factor", 0)
    max_dd = abs(metrics.get("max_drawdown", 0))

    # Each component in [0, 1] (clipped)
    sharpe_n = min(max(sharpe / SHARPE_REF, 0), 1)
    calmar_n = min(max(calmar / CALMAR_REF, 0), 1)
    pf_n = min(max(pf / PF_REF, 0), 1)
    dd_n = min(max(1 - (max_dd / MDD_REF), 0), 1)  # -10% DD = 0, 0% DD = 1

    score = (weights["sharpe"] * sharpe_n +
              weights["calmar"] * calmar_n +
              weights["pf"] * pf_n +
              weights["dd"] * dd_n) * 100
    return float(score)


# Convenience criterion functions for Optuna (higher is better)
def criterion_sharpe_only(m: dict) -> float:
    return m.get("sharpe", 0)


def criterion_calmar_only(m: dict) -> float:
    return m.get("calmar", 0)


def criterion_pf_only(m: dict) -> float:
    return m.get("profit_factor", 0)


def criterion_composite(weights: dict) -> callable:
    """Returns a callable that scores via composite_score with given weights."""
    def _score(m: dict) -> float:
        return composite_score(m, weights)
    return _score


# Built-in presets
CRITERION_PRESETS = {
    "Sharpe": criterion_sharpe_only,
    "Calmar": criterion_calmar_only,
    "Profit Factor": criterion_pf_only,
    "Composite (Balanced)": criterion_composite({"sharpe": 0.4, "calmar": 0.3,
                                                   "pf": 0.2, "dd": 0.1}),
    "Composite (Conservative)": criterion_composite({"sharpe": 0.2, "calmar": 0.4,
                                                       "pf": 0.2, "dd": 0.2}),
    "Composite (Aggressive)": criterion_composite({"sharpe": 0.5, "calmar": 0.2,
                                                     "pf": 0.3, "dd": 0.0}),
}


def composite_to_dict(score: float, weights: dict) -> dict:
    """Decompose a composite score back into its parts for display."""
    return {
        "sharpe_weight": weights.get("sharpe", 0),
        "calmar_weight": weights.get("calmar", 0),
        "pf_weight": weights.get("pf", 0),
        "dd_weight": weights.get("dd", 0),
        "total_score": score,
    }