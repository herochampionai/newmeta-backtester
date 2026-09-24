"""Execution-cost stress testing helpers for Newmeta Research Lab.

The helpers in this module intentionally stay close to ``run_full``.  They run
one already-built signal set through a fixed list of execution-cost scenarios
and return compact metrics rows suitable for UI tables, CSV export, or tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

from backtester.engine_full import run_full


DEFAULT_METRIC_KEYS: tuple[str, ...] = (
    "final_equity",
    "total_return",
    "net_pnl",
    "n_trades",
    "win_rate",
    "profit_factor",
    "sharpe",
    "calmar",
    "max_drawdown",
)


@dataclass(frozen=True)
class ExecutionStressScenario:
    """One execution-cost stress scenario.

    Multipliers apply only to the matching execution-cost arguments. All other
    run assumptions remain unchanged.
    """

    name: str
    label: str
    spread_mult: float = 1.0
    slippage_mult: float = 1.0
    commission_mult: float = 1.0


DEFAULT_EXECUTION_STRESS_SCENARIOS: tuple[ExecutionStressScenario, ...] = (
    ExecutionStressScenario("normal", "Normal"),
    ExecutionStressScenario("spread_x2", "Spread x2", spread_mult=2.0),
    ExecutionStressScenario("slippage_x2", "Slippage x2", slippage_mult=2.0),
    ExecutionStressScenario(
        "spread_slippage_x3",
        "Spread + slippage x3",
        spread_mult=3.0,
        slippage_mult=3.0,
    ),
    ExecutionStressScenario("commission_x2", "Commission x2", commission_mult=2.0),
)


def _base_costs(kwargs: Mapping[str, object]) -> dict[str, float]:
    return {
        "commission_pips": float(kwargs.get("commission_pips", 0.7) or 0.0),
        "commission_pct": float(kwargs.get("commission_pct", 0.0) or 0.0),
        "slippage_pips": float(kwargs.get("slippage_pips", 0.3) or 0.0),
        "spread_pips": float(kwargs.get("spread_pips", 0.0) or 0.0),
    }


def _scenario_kwargs(base_kwargs: Mapping[str, object], scenario: ExecutionStressScenario) -> dict:
    costs = _base_costs(base_kwargs)
    stressed = dict(base_kwargs)
    stressed["spread_pips"] = costs["spread_pips"] * scenario.spread_mult
    stressed["slippage_pips"] = costs["slippage_pips"] * scenario.slippage_mult
    stressed["commission_pips"] = costs["commission_pips"] * scenario.commission_mult
    stressed["commission_pct"] = costs["commission_pct"] * scenario.commission_mult
    return stressed


def _metric_value(metrics: Mapping[str, object], key: str) -> float | int | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        if key == "n_trades":
            return int(value)
        return float(value)
    except (TypeError, ValueError):
        return None


def _result_row(
    scenario: ExecutionStressScenario,
    result: Mapping[str, object],
    run_kwargs: Mapping[str, object],
    metric_keys: Sequence[str],
) -> dict:
    metrics = result.get("metrics", {}) or {}
    row = {
        "scenario": scenario.name,
        "label": scenario.label,
        "spread_mult": float(scenario.spread_mult),
        "slippage_mult": float(scenario.slippage_mult),
        "commission_mult": float(scenario.commission_mult),
        "spread_pips": float(run_kwargs.get("spread_pips", 0.0) or 0.0),
        "slippage_pips": float(run_kwargs.get("slippage_pips", 0.0) or 0.0),
        "commission_pips": float(run_kwargs.get("commission_pips", 0.0) or 0.0),
        "commission_pct": float(run_kwargs.get("commission_pct", 0.0) or 0.0),
    }
    for key in metric_keys:
        row[key] = _metric_value(metrics, key)
    return row


def run_execution_stress(
    df: pd.DataFrame,
    signals_by_strategy: dict[str, tuple],
    run_full_kwargs: Mapping[str, object] | None = None,
    scenarios: Iterable[ExecutionStressScenario] | None = None,
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
    as_dataframe: bool = True,
) -> pd.DataFrame | list[dict]:
    """Run one signal set through execution-cost stress scenarios.

    Parameters mirror ``run_full``: pass ``df``, ``signals_by_strategy``, and any
    normal ``run_full`` assumptions in ``run_full_kwargs``. The helper changes
    only ``spread_pips``, ``slippage_pips``, ``commission_pips``, and
    ``commission_pct`` according to each scenario.
    """
    base_kwargs = dict(run_full_kwargs or {})
    scenario_list = tuple(scenarios or DEFAULT_EXECUTION_STRESS_SCENARIOS)
    rows: list[dict] = []

    for scenario in scenario_list:
        stressed_kwargs = _scenario_kwargs(base_kwargs, scenario)
        result = run_full(df, signals_by_strategy, **stressed_kwargs)
        rows.append(_result_row(scenario, result, stressed_kwargs, metric_keys))

    if not as_dataframe:
        return rows
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_METRIC_KEYS",
    "ExecutionStressScenario",
    "DEFAULT_EXECUTION_STRESS_SCENARIOS",
    "run_execution_stress",
]
