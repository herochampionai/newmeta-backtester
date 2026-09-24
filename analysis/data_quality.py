"""Data quality gates for Newmeta Research Lab backtests."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DataQualityThresholds:
    min_rows_fail: int = 200
    min_rows_watch: int = 1000
    max_missing_ratio_watch: float = 0.001
    max_missing_ratio_fail: float = 0.01
    max_gap_ratio_watch: float = 0.01
    max_gap_ratio_fail: float = 0.05


DEFAULT_DATA_QUALITY_THRESHOLDS = DataQualityThresholds()
REQUIRED_OHLC = ("open", "high", "low", "close")


def assess_data_quality(
    df: pd.DataFrame,
    timeframe: str | None = None,
    source: str | None = None,
    thresholds: DataQualityThresholds = DEFAULT_DATA_QUALITY_THRESHOLDS,
) -> dict[str, Any]:
    """Return PASS/WATCH/FAIL quality verdict for a price dataframe."""
    checks: dict[str, Any] = {
        "rows": int(len(df)) if df is not None else 0,
        "source": source,
        "timeframe": timeframe,
        "thresholds": asdict(thresholds),
    }
    fail: list[str] = []
    watch: list[str] = []

    if df is None or df.empty:
        return _verdict("FAIL", 0.0, ["no data loaded"], checks)

    missing_cols = [c for c in REQUIRED_OHLC if c not in df.columns]
    checks["missing_columns"] = missing_cols
    if missing_cols:
        fail.append("missing required OHLC columns: " + ", ".join(missing_cols))
        return _verdict("FAIL", 5.0, fail, checks)

    rows = len(df)
    if rows < thresholds.min_rows_fail:
        fail.append(f"too few bars for research ({rows} < {thresholds.min_rows_fail})")
    elif rows < thresholds.min_rows_watch:
        watch.append(f"limited bar sample ({rows} < {thresholds.min_rows_watch})")

    index = df.index
    duplicate_count = int(index.duplicated().sum()) if hasattr(index, "duplicated") else 0
    checks["duplicate_timestamps"] = duplicate_count
    if duplicate_count:
        fail.append(f"duplicate timestamps found ({duplicate_count})")

    monotonic = bool(index.is_monotonic_increasing) if hasattr(index, "is_monotonic_increasing") else True
    checks["monotonic_increasing"] = monotonic
    if not monotonic:
        fail.append("timestamps are not sorted ascending")

    ohlc = df.loc[:, REQUIRED_OHLC].apply(pd.to_numeric, errors="coerce")
    missing_cells = int(ohlc.isna().sum().sum())
    total_cells = max(int(ohlc.size), 1)
    missing_ratio = missing_cells / total_cells
    checks["missing_ohlc_cells"] = missing_cells
    checks["missing_ohlc_ratio"] = missing_ratio
    if missing_ratio > thresholds.max_missing_ratio_fail:
        fail.append(f"missing OHLC data is too high ({missing_ratio:.2%})")
    elif missing_ratio > thresholds.max_missing_ratio_watch:
        watch.append(f"missing OHLC data needs review ({missing_ratio:.2%})")

    invalid_high_low = int((ohlc["high"] < ohlc["low"]).sum())
    invalid_open_close = int(((ohlc["high"] < ohlc[["open", "close"]].max(axis=1)) | (ohlc["low"] > ohlc[["open", "close"]].min(axis=1))).sum())
    nonpositive = int((ohlc <= 0).any(axis=1).sum())
    checks.update({
        "invalid_high_low_bars": invalid_high_low,
        "invalid_open_close_bars": invalid_open_close,
        "nonpositive_price_bars": nonpositive,
    })
    if invalid_high_low or invalid_open_close:
        fail.append(f"invalid OHLC geometry ({invalid_high_low + invalid_open_close} bars)")
    if nonpositive:
        fail.append(f"nonpositive prices found ({nonpositive} bars)")

    gap_info = _gap_check(index, timeframe)
    checks.update(gap_info)
    gap_ratio = float(gap_info.get("gap_ratio", 0.0) or 0.0)
    if gap_ratio > thresholds.max_gap_ratio_fail:
        fail.append(f"large timestamp gaps ({gap_ratio:.2%})")
    elif gap_ratio > thresholds.max_gap_ratio_watch:
        watch.append(f"timestamp gaps need review ({gap_ratio:.2%})")

    score = _score(checks, thresholds)
    if fail:
        return _verdict("FAIL", score, fail, checks)
    if watch:
        return _verdict("WATCH", score, watch, checks)
    return _verdict("PASS", score, ["data is research-ready"], checks)


def _gap_check(index: pd.Index, timeframe: str | None) -> dict[str, Any]:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return {"expected_step_seconds": None, "large_gaps": 0, "gap_ratio": 0.0}
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    expected = _timeframe_seconds(timeframe) or float(diffs.median())
    if not expected or expected <= 0:
        return {"expected_step_seconds": None, "large_gaps": 0, "gap_ratio": 0.0}
    large_gaps = int((diffs > expected * 1.5).sum())
    return {
        "expected_step_seconds": float(expected),
        "large_gaps": large_gaps,
        "gap_ratio": large_gaps / max(len(diffs), 1),
    }


def _timeframe_seconds(timeframe: str | None) -> int | None:
    if not timeframe:
        return None
    tf = timeframe.upper().strip()
    unit = tf[-1]
    try:
        value = int(tf[:-1]) if len(tf) > 1 else 1
    except ValueError:
        return None
    return {"M": 60, "H": 3600, "D": 86400, "W": 604800}.get(unit, 0) * value or None


def _score(checks: dict[str, Any], thresholds: DataQualityThresholds) -> float:
    score = 100.0
    rows = int(checks.get("rows", 0) or 0)
    if rows < thresholds.min_rows_watch:
        score -= 18.0
    score -= min(float(checks.get("missing_ohlc_ratio", 0.0) or 0.0) / thresholds.max_missing_ratio_fail, 1.0) * 30.0
    score -= min(float(checks.get("gap_ratio", 0.0) or 0.0) / thresholds.max_gap_ratio_fail, 1.0) * 20.0
    score -= min(int(checks.get("duplicate_timestamps", 0) or 0), 10) * 2.0
    score -= min(int(checks.get("invalid_high_low_bars", 0) or 0) + int(checks.get("invalid_open_close_bars", 0) or 0), 10) * 4.0
    return round(max(0.0, min(100.0, score)), 2)


def _verdict(status: str, score: float, reasons: list[str], checks: dict[str, Any]) -> dict[str, Any]:
    return {"status": status, "score": round(float(score), 2), "reasons": reasons, "checks": checks}
