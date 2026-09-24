import pandas as pd

from analysis.readiness import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WATCH,
    assess_backtest_readiness,
    assess_research_readiness,
    assess_walkforward_readiness,
)


def test_backtest_pass_watch_fail():
    passing = assess_backtest_readiness({
        "profit_factor": 1.75,
        "max_drawdown": -0.12,
        "trades": 64,
        "sharpe": 1.1,
        "calmar": 1.4,
    })
    assert passing["status"] == STATUS_PASS
    assert passing["score"] > 75

    watch = assess_backtest_readiness({
        "profit_factor": 1.22,
        "max_drawdown": -0.23,
        "n_trades": 24,
        "sharpe": 0.45,
        "calmar": 0.35,
    })
    assert watch["status"] == STATUS_WATCH
    assert any("drawdown" in reason for reason in watch["reasons"])

    failing = assess_backtest_readiness({
        "profit_factor": 0.94,
        "max_drawdown": -0.18,
        "trades": 80,
        "sharpe": 0.2,
        "calmar": 0.2,
    })
    assert failing["status"] == STATUS_FAIL
    assert any("profit factor" in reason for reason in failing["reasons"])


def test_walkforward_pass_and_fail_from_dataframe():
    passing_df = pd.DataFrame({
        "test_profit_factor": [1.4, 1.2, 1.8, 0.95],
        "test_sharpe": [0.9, 0.5, 1.2, -0.1],
        "train_sharpe": [1.1, 0.8, 1.4, 0.2],
        "test_calmar": [1.1, 0.7, 1.5, -0.05],
        "test_max_dd": [-0.10, -0.12, -0.08, -0.18],
    })
    verdict = assess_walkforward_readiness(passing_df)
    assert verdict["status"] == STATUS_PASS
    assert verdict["checks"]["oos_positive_ratio"] == 0.75

    failing_df = pd.DataFrame({
        "test_profit_factor": [0.8, 0.9, 0.7],
        "test_sharpe": [-0.4, -0.2, -0.8],
        "test_max_dd": [-0.12, -0.42, -0.20],
    })
    verdict = assess_walkforward_readiness(failing_df)
    assert verdict["status"] == STATUS_FAIL
    assert any("OOS" in reason or "drawdown" in reason for reason in verdict["reasons"])


def test_combined_research_verdict_uses_worst_section_status():
    result = assess_research_readiness(
        metrics={
            "profit_factor": 1.8,
            "max_drawdown": -0.10,
            "trades": 80,
            "sharpe": 1.0,
            "calmar": 1.2,
        },
        walkforward=pd.DataFrame({"test_sharpe": [-0.2], "test_max_dd": [-0.08]}),
    )
    assert result["status"] == STATUS_FAIL
    assert "backtest" in result["sections"]
    assert "walkforward" in result["sections"]
