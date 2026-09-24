from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import execution_stress


def test_run_execution_stress_applies_cost_multipliers(monkeypatch):
    calls = []

    def fake_run_full(df, signals_by_strategy, **kwargs):
        calls.append(kwargs)
        cost = (
            kwargs.get("spread_pips", 0)
            + kwargs.get("slippage_pips", 0)
            + kwargs.get("commission_pips", 0)
            + kwargs.get("commission_pct", 0)
        )
        return {
            "metrics": {
                "final_equity": 10_000 - cost,
                "profit_factor": 2.0,
                "max_drawdown": -0.05,
                "n_trades": 12,
            }
        }

    monkeypatch.setattr(execution_stress, "run_full", fake_run_full)

    df = pd.DataFrame({"close": [1.0, 1.1]})
    signals = {"demo": (pd.Series([False, True]), pd.Series([0, 1]))}
    table = execution_stress.run_execution_stress(
        df,
        signals,
        run_full_kwargs={
            "spread_pips": 1.0,
            "slippage_pips": 0.5,
            "commission_pips": 0.7,
            "commission_pct": 0.075,
            "init_cash": 10_000,
        },
        metric_keys=("final_equity", "profit_factor", "max_drawdown", "n_trades"),
    )

    assert list(table["scenario"]) == [
        "normal",
        "spread_x2",
        "slippage_x2",
        "spread_slippage_x3",
        "commission_x2",
    ]
    assert table.loc[table["scenario"] == "spread_x2", "spread_pips"].item() == 2.0
    assert table.loc[table["scenario"] == "slippage_x2", "slippage_pips"].item() == 1.0
    worst = table.loc[table["scenario"] == "spread_slippage_x3"].iloc[0]
    assert worst["spread_pips"] == 3.0
    assert worst["slippage_pips"] == 1.5
    commission = table.loc[table["scenario"] == "commission_x2"].iloc[0]
    assert commission["commission_pips"] == 1.4
    assert commission["commission_pct"] == 0.15
    assert calls[0]["init_cash"] == 10_000


def test_run_execution_stress_can_return_list(monkeypatch):
    def fake_run_full(df, signals_by_strategy, **kwargs):
        return {"metrics": {"final_equity": 9999.0}}

    monkeypatch.setattr(execution_stress, "run_full", fake_run_full)
    rows = execution_stress.run_execution_stress(
        pd.DataFrame({"close": [1.0]}),
        {"demo": (pd.Series([True]), pd.Series([1]))},
        metric_keys=("final_equity",),
        as_dataframe=False,
    )

    assert isinstance(rows, list)
    assert rows[0]["scenario"] == "normal"
    assert rows[0]["final_equity"] == 9999.0
