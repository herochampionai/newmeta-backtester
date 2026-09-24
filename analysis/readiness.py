"""Institutional-style readiness verdicts for Newmeta research outputs.

The functions in this module are deterministic and UI-agnostic. They classify
single backtest metrics and walk-forward out-of-sample summaries into:
PASS, WATCH, or FAIL.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable, Mapping

import pandas as pd


STATUS_PASS = "PASS"
STATUS_WATCH = "WATCH"
STATUS_FAIL = "FAIL"


@dataclass(frozen=True)
class ReadinessThresholds:
    """Default research thresholds for profit-factor/drawdown selection."""

    min_trades_fail: int = 10
    min_trades_watch: int = 30
    min_profit_factor_pass: float = 1.40
    min_profit_factor_watch: float = 1.10
    max_drawdown_pass: float = 0.20
    max_drawdown_watch: float = 0.30
    min_sharpe_pass: float = 0.80
    min_sharpe_watch: float = 0.20
    min_calmar_pass: float = 0.80
    min_calmar_watch: float = 0.20
    min_wf_windows_fail: int = 2
    min_wf_windows_pass: int = 3
    min_oos_positive_ratio_pass: float = 0.60
    min_oos_positive_ratio_watch: float = 0.45
    max_oos_drawdown_watch: float = 0.35
    max_sharpe_degradation_watch: float = 1.00


DEFAULT_THRESHOLDS = ReadinessThresholds()


@dataclass(frozen=True)
class ReadinessVerdict:
    status: str
    score: float
    reasons: list[str]
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_backtest_readiness(
    metrics: Mapping[str, Any],
    thresholds: ReadinessThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Classify one backtest metric dictionary into PASS/WATCH/FAIL.

    Expected metric names match the backtester: profit_factor, max_drawdown,
    trades or n_trades, sharpe, calmar. Missing Sharpe/Calmar are treated as
    neutral so older result dictionaries can still be classified.
    """
    pf = _number(metrics, "profit_factor", default=0.0)
    max_dd = _drawdown_abs(metrics)
    trades = int(_number_any(metrics, ("trades", "n_trades", "total_trades"), default=0.0))
    sharpe = _optional_number(metrics, "sharpe")
    calmar = _optional_number(metrics, "calmar")

    checks = {
        "profit_factor": pf,
        "max_drawdown_abs": max_dd,
        "trades": trades,
        "sharpe": sharpe,
        "calmar": calmar,
        "thresholds": asdict(thresholds),
    }

    fail_reasons: list[str] = []
    watch_reasons: list[str] = []

    if trades < thresholds.min_trades_fail:
        fail_reasons.append(f"too few trades for research validity ({trades} < {thresholds.min_trades_fail})")
    elif trades < thresholds.min_trades_watch:
        watch_reasons.append(f"limited trade sample ({trades} < {thresholds.min_trades_watch})")

    if pf < 1.0:
        fail_reasons.append(f"profit factor below breakeven ({pf:.2f} < 1.00)")
    elif pf < thresholds.min_profit_factor_watch:
        watch_reasons.append(f"profit factor is weak ({pf:.2f} < {thresholds.min_profit_factor_watch:.2f})")
    elif pf < thresholds.min_profit_factor_pass:
        watch_reasons.append(f"profit factor is not yet pass-grade ({pf:.2f} < {thresholds.min_profit_factor_pass:.2f})")

    if max_dd > thresholds.max_drawdown_watch:
        fail_reasons.append(f"drawdown is too high ({max_dd:.1%} > {thresholds.max_drawdown_watch:.1%})")
    elif max_dd > thresholds.max_drawdown_pass:
        watch_reasons.append(f"drawdown needs review ({max_dd:.1%} > {thresholds.max_drawdown_pass:.1%})")

    if sharpe is not None:
        if sharpe < 0:
            fail_reasons.append(f"Sharpe is negative ({sharpe:.2f})")
        elif sharpe < thresholds.min_sharpe_watch:
            watch_reasons.append(f"Sharpe is weak ({sharpe:.2f} < {thresholds.min_sharpe_watch:.2f})")
        elif sharpe < thresholds.min_sharpe_pass:
            watch_reasons.append(f"Sharpe is below pass-grade ({sharpe:.2f} < {thresholds.min_sharpe_pass:.2f})")

    if calmar is not None:
        if calmar < 0:
            fail_reasons.append(f"Calmar is negative ({calmar:.2f})")
        elif calmar < thresholds.min_calmar_watch:
            watch_reasons.append(f"Calmar is weak ({calmar:.2f} < {thresholds.min_calmar_watch:.2f})")
        elif calmar < thresholds.min_calmar_pass:
            watch_reasons.append(f"Calmar is below pass-grade ({calmar:.2f} < {thresholds.min_calmar_pass:.2f})")

    score = _backtest_score(pf, max_dd, trades, sharpe, calmar, thresholds)
    if fail_reasons:
        return ReadinessVerdict(STATUS_FAIL, score, fail_reasons, checks).to_dict()
    if watch_reasons:
        return ReadinessVerdict(STATUS_WATCH, score, watch_reasons, checks).to_dict()
    return ReadinessVerdict(STATUS_PASS, score, ["backtest metrics meet pass-grade thresholds"], checks).to_dict()


def assess_walkforward_readiness(
    walkforward: Any,
    thresholds: ReadinessThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Classify walk-forward OOS stability into PASS/WATCH/FAIL.

    Accepts a wf_summary DataFrame, a list of WFWindow objects, or a list of
    dictionaries. Uses test_profit_factor/test_sharpe/test_calmar/test_max_dd
    when available and degrades gracefully when only part of the OOS set exists.
    """
    df = _to_walkforward_frame(walkforward)
    windows = int(len(df))
    checks: dict[str, Any] = {"windows": windows, "thresholds": asdict(thresholds)}

    if windows == 0:
        return ReadinessVerdict(
            STATUS_FAIL,
            0.0,
            ["no walk-forward windows available"],
            checks,
        ).to_dict()

    test_pf = _series(df, "test_profit_factor")
    test_sharpe = _series(df, "test_sharpe")
    test_calmar = _series(df, "test_calmar")
    test_dd = _abs_series(df, "test_max_dd", "test_max_drawdown")
    train_sharpe = _series(df, "train_sharpe")

    positive = _positive_oos_series(test_pf, test_sharpe, test_calmar)
    positive_ratio = float(positive.mean()) if positive is not None and len(positive) else None
    avg_test_pf = _mean_or_none(test_pf)
    avg_test_sharpe = _mean_or_none(test_sharpe)
    avg_test_calmar = _mean_or_none(test_calmar)
    worst_test_dd = _max_or_none(test_dd)
    sharpe_degradation = _mean_or_none(train_sharpe - test_sharpe) if len(train_sharpe) and len(test_sharpe) else None

    checks.update({
        "avg_test_profit_factor": avg_test_pf,
        "avg_test_sharpe": avg_test_sharpe,
        "avg_test_calmar": avg_test_calmar,
        "worst_test_drawdown_abs": worst_test_dd,
        "oos_positive_ratio": positive_ratio,
        "train_to_test_sharpe_degradation": sharpe_degradation,
    })

    fail_reasons: list[str] = []
    watch_reasons: list[str] = []

    if windows < thresholds.min_wf_windows_fail:
        fail_reasons.append(f"too few walk-forward windows ({windows} < {thresholds.min_wf_windows_fail})")
    elif windows < thresholds.min_wf_windows_pass:
        watch_reasons.append(f"limited walk-forward sample ({windows} < {thresholds.min_wf_windows_pass})")

    if positive_ratio is None:
        watch_reasons.append("no OOS profitability/stability metric available")
    elif positive_ratio < thresholds.min_oos_positive_ratio_watch:
        fail_reasons.append(
            f"OOS positive-window ratio is too low ({positive_ratio:.0%} < {thresholds.min_oos_positive_ratio_watch:.0%})"
        )
    elif positive_ratio < thresholds.min_oos_positive_ratio_pass:
        watch_reasons.append(
            f"OOS positive-window ratio needs review ({positive_ratio:.0%} < {thresholds.min_oos_positive_ratio_pass:.0%})"
        )

    if avg_test_pf is not None and avg_test_pf < 1.0:
        fail_reasons.append(f"average OOS profit factor below breakeven ({avg_test_pf:.2f})")
    elif avg_test_pf is not None and avg_test_pf < thresholds.min_profit_factor_watch:
        watch_reasons.append(f"average OOS profit factor is weak ({avg_test_pf:.2f})")

    if avg_test_sharpe is not None:
        if avg_test_sharpe < 0:
            fail_reasons.append(f"average OOS Sharpe is negative ({avg_test_sharpe:.2f})")
        elif avg_test_sharpe < thresholds.min_sharpe_watch:
            watch_reasons.append(f"average OOS Sharpe is weak ({avg_test_sharpe:.2f})")

    if avg_test_calmar is not None and avg_test_calmar < 0:
        fail_reasons.append(f"average OOS Calmar is negative ({avg_test_calmar:.2f})")

    if worst_test_dd is not None and worst_test_dd > thresholds.max_oos_drawdown_watch:
        fail_reasons.append(f"worst OOS drawdown is too high ({worst_test_dd:.1%})")
    elif worst_test_dd is not None and worst_test_dd > thresholds.max_drawdown_pass:
        watch_reasons.append(f"worst OOS drawdown needs review ({worst_test_dd:.1%})")

    if sharpe_degradation is not None and sharpe_degradation > thresholds.max_sharpe_degradation_watch:
        watch_reasons.append(f"train-to-test Sharpe degradation is large ({sharpe_degradation:.2f})")

    score = _walkforward_score(positive_ratio, avg_test_pf, avg_test_sharpe, worst_test_dd, windows, thresholds)
    if fail_reasons:
        return ReadinessVerdict(STATUS_FAIL, score, fail_reasons, checks).to_dict()
    if watch_reasons:
        return ReadinessVerdict(STATUS_WATCH, score, watch_reasons, checks).to_dict()
    return ReadinessVerdict(STATUS_PASS, score, ["walk-forward OOS stability meets pass-grade thresholds"], checks).to_dict()


def assess_research_readiness(
    metrics: Mapping[str, Any] | None = None,
    walkforward: Any | None = None,
    thresholds: ReadinessThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Combined verdict for a backtest and optional walk-forward validation."""
    sections: dict[str, Any] = {}
    statuses: list[str] = []
    scores: list[float] = []

    if metrics is not None:
        sections["backtest"] = assess_backtest_readiness(metrics, thresholds)
        statuses.append(sections["backtest"]["status"])
        scores.append(float(sections["backtest"]["score"]))

    if walkforward is not None:
        sections["walkforward"] = assess_walkforward_readiness(walkforward, thresholds)
        statuses.append(sections["walkforward"]["status"])
        scores.append(float(sections["walkforward"]["score"]))

    if not sections:
        return ReadinessVerdict(STATUS_FAIL, 0.0, ["no metrics supplied"], {}).to_dict()

    status = STATUS_PASS
    if STATUS_FAIL in statuses:
        status = STATUS_FAIL
    elif STATUS_WATCH in statuses:
        status = STATUS_WATCH

    reasons = []
    for name, verdict in sections.items():
        reasons.extend(f"{name}: {reason}" for reason in verdict["reasons"])

    return {
        "status": status,
        "score": round(sum(scores) / len(scores), 2),
        "reasons": reasons,
        "sections": sections,
    }


def _backtest_score(
    pf: float,
    max_dd: float,
    trades: int,
    sharpe: float | None,
    calmar: float | None,
    t: ReadinessThresholds,
) -> float:
    pf_score = min(max(pf, 0.0) / t.min_profit_factor_pass, 1.4) * 35.0
    dd_score = max(0.0, 1.0 - (max_dd / t.max_drawdown_watch)) * 25.0
    trade_score = min(max(trades, 0) / t.min_trades_watch, 1.0) * 15.0
    sharpe_score = 12.5 if sharpe is None else min(max(sharpe, 0.0) / t.min_sharpe_pass, 1.2) * 12.5
    calmar_score = 12.5 if calmar is None else min(max(calmar, 0.0) / t.min_calmar_pass, 1.2) * 12.5
    return round(max(0.0, min(100.0, pf_score + dd_score + trade_score + sharpe_score + calmar_score)), 2)


def _walkforward_score(
    positive_ratio: float | None,
    avg_pf: float | None,
    avg_sharpe: float | None,
    worst_dd: float | None,
    windows: int,
    t: ReadinessThresholds,
) -> float:
    ratio_score = 25.0 if positive_ratio is None else min(max(positive_ratio, 0.0) / t.min_oos_positive_ratio_pass, 1.2) * 25.0
    pf_score = 20.0 if avg_pf is None else min(max(avg_pf, 0.0) / t.min_profit_factor_pass, 1.2) * 20.0
    sharpe_score = 20.0 if avg_sharpe is None else min(max(avg_sharpe, 0.0) / t.min_sharpe_pass, 1.2) * 20.0
    dd_score = 20.0 if worst_dd is None else max(0.0, 1.0 - (worst_dd / t.max_oos_drawdown_watch)) * 20.0
    window_score = min(max(windows, 0) / t.min_wf_windows_pass, 1.0) * 15.0
    return round(max(0.0, min(100.0, ratio_score + pf_score + sharpe_score + dd_score + window_score)), 2)


def _to_walkforward_frame(walkforward: Any) -> pd.DataFrame:
    if isinstance(walkforward, pd.DataFrame):
        return walkforward.copy()
    rows: list[dict[str, Any]] = []
    for item in _as_iterable(walkforward):
        if isinstance(item, Mapping):
            rows.append(dict(item))
            continue
        train_metrics = getattr(item, "train_metrics", {}) or {}
        test_metrics = getattr(item, "test_metrics", {}) or {}
        row = {
            "train_sharpe": train_metrics.get("sharpe"),
            "train_calmar": train_metrics.get("calmar"),
            "train_profit_factor": train_metrics.get("profit_factor"),
            "test_sharpe": test_metrics.get("sharpe"),
            "test_calmar": test_metrics.get("calmar"),
            "test_profit_factor": test_metrics.get("profit_factor"),
            "test_max_dd": test_metrics.get("max_drawdown"),
            "test_trades": test_metrics.get("trades", test_metrics.get("n_trades")),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _as_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def _number(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    return _finite_float(metrics.get(key), default)


def _number_any(metrics: Mapping[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        if key in metrics:
            return _finite_float(metrics.get(key), default)
    return default


def _optional_number(metrics: Mapping[str, Any], key: str) -> float | None:
    if key not in metrics or metrics.get(key) is None:
        return None
    return _finite_float(metrics.get(key), 0.0)


def _drawdown_abs(metrics: Mapping[str, Any]) -> float:
    value = _number_any(metrics, ("max_drawdown", "max_dd", "drawdown"), default=1.0)
    return abs(value)


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if isfinite(out) else default


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna().astype(float)


def _abs_series(df: pd.DataFrame, *columns: str) -> pd.Series:
    for column in columns:
        if column in df:
            return pd.to_numeric(df[column], errors="coerce").dropna().abs().astype(float)
    return pd.Series(dtype=float)


def _mean_or_none(series: pd.Series) -> float | None:
    return None if len(series) == 0 else float(series.mean())


def _max_or_none(series: pd.Series) -> float | None:
    return None if len(series) == 0 else float(series.max())


def _positive_oos_series(
    test_pf: pd.Series,
    test_sharpe: pd.Series,
    test_calmar: pd.Series,
) -> pd.Series | None:
    if len(test_pf):
        return test_pf > 1.0
    if len(test_sharpe):
        return test_sharpe > 0.0
    if len(test_calmar):
        return test_calmar > 0.0
    return None
