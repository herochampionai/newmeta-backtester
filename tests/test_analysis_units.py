"""Pytest-collected unit tests for analysis modules (fast, no data needed).

The module `python -m analysis.X` self-tests also run via `python -m tests`;
this file gives CI a standard `pytest` entry point with isolated asserts.
"""
import sys
sys.path.insert(0, '.')

import numpy as np


def test_bootstrap_skilled():
    from analysis.permutation_test import permutation_pvalue
    rng = np.random.default_rng(0)
    rep = permutation_pvalue(list(rng.normal(5, 20, 100)),
                             n_permutations=500, verbose=False)
    assert rep.p_value < 0.05
    assert rep.verdict() in ("STRONG", "WEAK")
    assert abs(rep.mean_shuffled_sharpe - rep.observed_sharpe) < 0.2


def test_bootstrap_noise():
    from analysis.permutation_test import permutation_pvalue
    rng = np.random.default_rng(1)
    rep = permutation_pvalue(list(rng.normal(0, 20, 100)),
                             n_permutations=500, verbose=False)
    assert rep.p_value > 0.05
    assert rep.verdict() == "NO_EVIDENCE"


def test_gate_accepts_good():
    from analysis.review_gate import review
    g = review(backtest_metrics={"net_pnl": 297.0, "sharpe": 0.59,
                                 "max_drawdown": -0.04},
               wf_verdict="ACCEPT", wf_passed=6, wf_windows=8,
               psr_verdict="STRONG", dsr_verdict="STRONG",
               stress_verdict="ROBUST", trades=78)
    assert g["pass"]


def test_gate_rejects_bad():
    from analysis.review_gate import review
    g = review(backtest_metrics={"net_pnl": -266.0, "sharpe": -0.55,
                                 "max_drawdown": -0.09},
               wf_verdict="OVERFIT", wf_passed=1, wf_windows=8,
               psr_verdict="POOR", dsr_verdict="OVERFIT",
               stress_verdict="ROBUST", trades=82)
    assert not g["pass"] and len(g["reasons"]) >= 3


def test_gate_bootstrap_leg():
    from analysis.review_gate import review
    g = review(backtest_metrics={"net_pnl": 300.0, "sharpe": 0.6,
                                 "max_drawdown": -0.04},
               wf_verdict="ACCEPT", wf_passed=6, wf_windows=8,
               psr_verdict="POOR", dsr_verdict="WEAK", perm_verdict="STRONG",
               stress_verdict="ROBUST", trades=80)
    assert g["checks"]["significance"]["pass"]
    assert g["pass"]


def test_competition_study_complete():
    from analysis import competition_study as cs
    assert len(cs.matrix()) >= 3
    assert len(cs.our_leads()) >= 1
    assert len(cs.gaps()) >= 1
