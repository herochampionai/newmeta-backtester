"""Data Quality Dashboard — comprehensive data health analysis for backtesting.

Features:
- Gap/outlier/stale detection with severity scoring
- Volume profile vs trading session (London/NY/Asian/Tokyo)
- Spread percentile bands (p1/p5/p50/p95/p99)
- Auto-grade on load (A/B/C/D/F with detailed rationale)
- HTML dashboard export for visual inspection
- JSON summary for CI/CD gates
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import pandas as pd

# Session definitions (UTC)
SESSIONS = {
    "tokyo": (0, 9),      # 00:00-09:00 UTC
    "london": (8, 17),    # 08:00-17:00 UTC
    "ny": (13, 22),       # 13:00-22:00 UTC
    "overlap_lon_ny": (13, 17),  # 13:00-17:00 UTC (London/NY overlap)
}

# Quality thresholds
GAP_MULTIPLIER = 5.0        # gap > 5x median interval
GAP_TOLERANCE_PCT = 0.02    # max 2% bars can be gaps
STALE_DAYS = 7              # tail older than 7 days = stale
MIN_BARS = 100              # minimum bars for any grade
OUTLIER_ZSCORE = 4.0        # price move > 4 sigma = outlier
SPREAD_PCTL_BANDS = [1, 5, 50, 95, 99]


@dataclass
class GapReport:
    total_gaps: int
    gap_pct: float
    max_gap_bars: int
    max_gap_hours: float
    gap_times: list[str]
    severity: str  # "none", "low", "medium", "high", "critical"
    expected_gaps: int = 0  # gaps spanning Sat/Sun (scheduled FX close)
    unexpected_gaps: int = 0  # gaps inside the trading week (real holes)


@dataclass
class OutlierReport:
    count: int
    pct: float
    max_zscore: float
    outlier_times: list[str]
    severity: str


@dataclass
class StaleReport:
    is_stale: bool
    days_since_last: float
    last_timestamp: str
    severity: str


@dataclass
class VolumeProfile:
    by_session: dict[str, dict]  # session -> {mean, median, std, pct_of_total}
    by_hour: dict[int, float]    # hour -> mean volume
    session_concentration: float  # Herfindahl index of volume across sessions
    severity: str


@dataclass
class SpreadProfile:
    percentiles: dict[int, float]  # pctl -> spread in pips
    mean: float
    median: float
    std: float
    max_spread: float
    max_spread_time: str
    severity: str


@dataclass
class DataQualityReport:
    symbol: str
    timeframe: str
    source: str
    bars: int
    date_range: tuple[str, str]
    grade: str  # A/B/C/D/F
    score: float  # 0-100
    gaps: GapReport
    outliers: OutlierReport
    stale: StaleReport
    volume: VolumeProfile
    spread: Optional[SpreadProfile]
    flags: list[str]
    loud_banner: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


def _session_of_hour(hour: int) -> str:
    for name, (start, end) in SESSIONS.items():
        if start <= hour < end:
            return name
    return "other"


def analyze_gaps(df: pd.DataFrame) -> GapReport:
    """Detect time gaps in the data."""
    if df is None or len(df) < 2:
        return GapReport(0, 0.0, 0, 0.0, [], "none")

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return GapReport(0, 0.0, 0, 0.0, [], "unknown")

    diffs = idx.to_series().diff().dropna()
    if len(diffs) == 0:
        return GapReport(0, 0.0, 0, 0.0, [], "none")

    med = diffs.median()
    med_seconds = med.total_seconds()
    if med_seconds <= 0:
        return GapReport(0, 0.0, 0, 0.0, [], "unknown")

    gap_threshold = med * GAP_MULTIPLIER
    gaps = diffs[diffs > gap_threshold]

    gap_times = [str(t) for t in gaps.index]
    max_gap = gaps.max() if len(gaps) > 0 else pd.Timedelta(0)
    max_gap_bars = int(max_gap / med) if med_seconds > 0 else 0
    gap_pct = len(gaps) / len(diffs) * 100

    # R021: session-aware gaps. A gap whose interval covers Saturday or
    # Sunday is the scheduled FX weekend close — expected, not a data hole.
    # Severity is scored on UNEXPECTED (in-week) gaps only.
    expected = 0
    for ts, width in gaps.items():
        start = ts - width
        # any calendar day in [start, ts] falling on Sat/Sun?
        days = pd.date_range(start=start.normalize(), end=ts.normalize(), freq="D")
        if any(d.weekday() >= 5 for d in days):
            expected += 1
    unexpected = len(gaps) - expected
    unexp_pct = unexpected / len(diffs) * 100

    if unexp_pct == 0:
        severity = "none"
    elif unexp_pct < 0.5:
        severity = "low"
    elif unexp_pct < 2.0:
        severity = "medium"
    elif unexp_pct < 5.0:
        severity = "high"
    else:
        severity = "critical"

    return GapReport(
        total_gaps=int(len(gaps)),
        gap_pct=round(gap_pct, 2),
        max_gap_bars=max_gap_bars,
        max_gap_hours=round(max_gap.total_seconds() / 3600, 2),
        gap_times=gap_times[:20],
        severity=severity,
        expected_gaps=expected,
        unexpected_gaps=unexpected,
    )


def analyze_outliers(df: pd.DataFrame, price_col: str = "close") -> OutlierReport:
    """Detect price outliers using z-score on returns."""
    if df is None or len(df) < 20 or price_col not in df.columns:
        return OutlierReport(0, 0.0, 0.0, [], "none")

    returns = df[price_col].pct_change().dropna()
    if len(returns) < 10:
        return OutlierReport(0, 0.0, 0.0, [], "none")

    mean_r = returns.mean()
    std_r = returns.std()
    if std_r == 0:
        return OutlierReport(0, 0.0, 0.0, [], "none")

    zscores = np.abs((returns - mean_r) / std_r)
    outliers = zscores[zscores > OUTLIER_ZSCORE]

    outlier_times = [str(t) for t in outliers.index]
    max_z = float(zscores.max()) if len(zscores) > 0 else 0.0
    pct = len(outliers) / len(returns) * 100

    if pct == 0:
        severity = "none"
    elif pct < 0.1:
        severity = "low"
    elif pct < 0.5:
        severity = "medium"
    elif pct < 1.0:
        severity = "high"
    else:
        severity = "critical"

    return OutlierReport(
        count=int(len(outliers)),
        pct=round(pct, 3),
        max_zscore=round(max_z, 2),
        outlier_times=outlier_times[:20],
        severity=severity
    )


def analyze_stale(df: pd.DataFrame) -> StaleReport:
    """Check if data tail is stale."""
    if df is None or len(df) == 0:
        return StaleReport(True, float('inf'), "", "critical")

    last_ts = df.index[-1]
    now = pd.Timestamp.now(tz="UTC")
    if last_ts.tz is None:
        last_ts = last_ts.tz_localize("UTC")
    days = (now - last_ts).total_seconds() / 86400

    is_stale = days > STALE_DAYS
    if days < 1:
        severity = "none"
    elif days < 3:
        severity = "low"
    elif days < 7:
        severity = "medium"
    elif days < 30:
        severity = "high"
    else:
        severity = "critical"

    return StaleReport(
        is_stale=is_stale,
        days_since_last=round(days, 1),
        last_timestamp=str(last_ts),
        severity=severity
    )


def _looks_like_fx_spot(symbol: str) -> bool:
    """Heuristic: FX spot pairs (EURUSD, GBPJPY, EUR/USD) have no centralized volume."""
    s = (symbol or "").upper().replace("/", "").replace("-", "").replace("_", "")
    if len(s) == 6 and s.isalpha():
        return True
    for ccy in ("USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "XAU", "XAG"):
        if s.startswith(ccy) or s.endswith(ccy):
            return True
    return False


def analyze_volume(df: pd.DataFrame, symbol: str = "") -> VolumeProfile:
    """Analyze volume profile by session and hour."""
    if df is None or len(df) == 0 or "volume" not in df.columns:
        return VolumeProfile({}, {}, 0.0, "unknown")

    vol = df["volume"].astype(float)
    if vol.sum() == 0:
        # FX spot has no centralized volume feed — flag it, don't tank the grade.
        # Equities/futures with zero volume IS suspicious (data error).
        if _looks_like_fx_spot(symbol):
            return VolumeProfile({}, {}, 0.0, "no_volume_data")
        return VolumeProfile({}, {}, 0.0, "zero_volume")

    # By session
    hours = df.index.hour
    sessions = hours.map(_session_of_hour)
    by_session = {}
    for sess in SESSIONS.keys():
        mask = sessions == sess
        if mask.any():
            v = vol[mask]
            by_session[sess] = {
                "mean": round(float(v.mean()), 2),
                "median": round(float(v.median()), 2),
                "std": round(float(v.std()), 2),
                "pct_of_total": round(float(v.sum() / vol.sum() * 100), 1)
            }

    # By hour
    by_hour = {}
    for h in range(24):
        mask = hours == h
        if mask.any():
            by_hour[h] = round(float(vol[mask].mean()), 2)

    # Herfindahl index (concentration)
    session_pcts = [v["pct_of_total"] / 100 for v in by_session.values()]
    hhi = sum(p * p for p in session_pcts)

    if hhi > 0.7:
        severity = "high"
    elif hhi > 0.5:
        severity = "medium"
    else:
        severity = "low"

    return VolumeProfile(
        by_session=by_session,
        by_hour=by_hour,
        session_concentration=round(hhi, 3),
        severity=severity
    )


def analyze_spread(df: pd.DataFrame, pip_size: float = 0.0001) -> Optional[SpreadProfile]:
    """Analyze spread distribution (requires bid/ask or spread column)."""
    if df is None or len(df) == 0:
        return None

    spread_col = None
    for c in ["spread", "spread_pips", "bid_ask_spread"]:
        if c in df.columns:
            spread_col = c
            break

    if spread_col is None:
        # Try to compute from bid/ask
        if "bid" in df.columns and "ask" in df.columns:
            spread_pips = (df["ask"] - df["bid"]) / pip_size
        else:
            return None
    else:
        spread_pips = df[spread_col].astype(float)
        if spread_col != "spread_pips":
            spread_pips = spread_pips / pip_size

    spread_pips = spread_pips.dropna()
    if len(spread_pips) == 0:
        return None

    pctls = {}
    for p in SPREAD_PCTL_BANDS:
        pctls[p] = round(float(np.percentile(spread_pips, p)), 1)

    max_idx = spread_pips.idxmax()
    max_time = str(max_idx) if max_idx is not None else ""

    # Severity based on p99/p50 ratio
    ratio = pctls[99] / pctls[50] if pctls[50] > 0 else 0
    if ratio > 10:
        severity = "critical"
    elif ratio > 5:
        severity = "high"
    elif ratio > 3:
        severity = "medium"
    else:
        severity = "low"

    return SpreadProfile(
        percentiles=pctls,
        mean=round(float(spread_pips.mean()), 1),
        median=round(float(spread_pips.median()), 1),
        std=round(float(spread_pips.std()), 1),
        max_spread=round(float(spread_pips.max()), 1),
        max_spread_time=max_time,
        severity=severity
    )


def compute_grade(report: DataQualityReport) -> tuple[str, float]:
    """Compute overall grade and score from sub-reports."""
    score = 100.0
    flags = []

    # Bars check
    if report.bars < MIN_BARS:
        score -= 50
        flags.append(f"insufficient_bars:{report.bars}")

    # Gaps
    if report.gaps.severity == "critical":
        score -= 30
        flags.append("critical_gaps")
    elif report.gaps.severity == "high":
        score -= 20
        flags.append("high_gaps")
    elif report.gaps.severity == "medium":
        score -= 10
        flags.append("medium_gaps")

    # Outliers
    if report.outliers.severity == "critical":
        score -= 20
        flags.append("critical_outliers")
    elif report.outliers.severity == "high":
        score -= 15
        flags.append("high_outliers")
    elif report.outliers.severity == "medium":
        score -= 5
        flags.append("medium_outliers")

    # Stale
    if report.stale.severity == "critical":
        score -= 50
        flags.append("critical_stale")
    elif report.stale.severity == "high":
        score -= 30
        flags.append("high_stale")
    elif report.stale.severity == "medium":
        score -= 15
        flags.append("medium_stale")

    # Volume
    if report.volume.severity == "zero_volume":
        score -= 25
        flags.append("zero_volume")
    elif report.volume.severity == "no_volume_data":
        score -= 5
        flags.append("no_volume_data_fx_spot")
    elif report.volume.severity == "high":
        score -= 10
        flags.append("volume_concentrated")

    # Spread
    if report.spread and report.spread.severity == "critical":
        score -= 15
        flags.append("extreme_spreads")
    elif report.spread and report.spread.severity == "high":
        score -= 10
        flags.append("high_spreads")

    # Source penalty
    src = report.source.lower()
    if "synthetic" in src:
        score -= 40
        flags.append("synthetic_source")
    elif "yahoo" in src:
        score -= 10
        flags.append("yahoo_source")
        # Survivorship honesty: Yahoo equities contain only survivors —
        # delisted failures are invisible, flattering every backtest.
        # FX spot / metals / crypto have no listed-company survivorship problem.
        s = (report.symbol or "").upper().replace("/", "")
        is_fx_like = ((len(s) == 6 and s.isalpha())
                      or s in ("XAUUSD", "XAGUSD")
                      or s.endswith("USDT") or s.endswith("BTC"))
        if not is_fx_like:
            score -= 5
            flags.append("no_survivorship_control")

    score = max(0, min(100, score))

    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    return grade, round(score, 1)


def analyze_data_quality(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    timeframe: str = "?",
    source: str = "?",
    pip_size: float = 0.0001
) -> DataQualityReport:
    """Main entry point: full data quality analysis."""
    if df is None or len(df) == 0:
        return DataQualityReport(
            symbol=symbol, timeframe=timeframe, source=source, bars=0,
            date_range=("", ""), grade="F", score=0.0,
            gaps=GapReport(0, 0, 0, 0, [], "critical"),
            outliers=OutlierReport(0, 0, 0, [], "critical"),
            stale=StaleReport(True, float('inf'), "", "critical"),
            volume=VolumeProfile({}, {}, 0, "critical"),
            spread=None,
            flags=["empty_dataframe"],
            loud_banner=f"[DATA F] {source} | 0 bars | EMPTY"
        )

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            pass

    # Run all analyses
    gaps = analyze_gaps(df)
    outliers = analyze_outliers(df)
    stale = analyze_stale(df)
    volume = analyze_volume(df, symbol)
    spread = analyze_spread(df, pip_size)

    bars = len(df)
    date_range = (str(df.index[0]), str(df.index[-1]))

    # Build preliminary report for grading
    prelim = DataQualityReport(
        symbol=symbol, timeframe=timeframe, source=source, bars=bars,
        date_range=date_range, grade="F", score=0.0,
        gaps=gaps, outliers=outliers, stale=stale,
        volume=volume, spread=spread, flags=[], loud_banner=""
    )

    grade, score = compute_grade(prelim)

    # Collect all flags
    all_flags = prelim.flags + gaps.gap_times[:5] + outliers.outlier_times[:5]
    if stale.is_stale:
        all_flags.append(f"stale_{stale.days_since_last}d")

    loud_banner = f"[DATA {grade}] {source} | {bars} bars | {date_range[0]} → {date_range[1]} | score={score}"
    if all_flags:
        loud_banner += f" | {'; '.join(all_flags[:5])}"

    return DataQualityReport(
        symbol=symbol, timeframe=timeframe, source=source, bars=bars,
        date_range=date_range, grade=grade, score=score,
        gaps=gaps, outliers=outliers, stale=stale,
        volume=volume, spread=spread, flags=all_flags, loud_banner=loud_banner
    )


def export_html_dashboard(report: DataQualityReport, out_path: str | Path) -> str:
    """Export interactive HTML dashboard."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grade_colors = {"A": "#00d47e", "B": "#7bd88f", "C": "#ffb800", "D": "#ff8800", "F": "#ff3366"}
    color = grade_colors.get(report.grade, "#888")

    # Volume by session table
    vol_rows = ""
    for sess, stats in report.volume.by_session.items():
        vol_rows += f"<tr><td>{sess}</td><td>{stats['mean']}</td><td>{stats['median']}</td><td>{stats['pct_of_total']}%</td></tr>"

    # Spread percentiles
    spread_rows = ""
    if report.spread:
        for p, v in report.spread.percentiles.items():
            spread_rows += f"<tr><td>P{p}</td><td>{v} pips</td></tr>"

    # Gap details
    gap_rows = ""
    for t in report.gaps.gap_times[:10]:
        gap_rows += f"<tr><td>{t}</td></tr>"

    # Outlier details
    out_rows = ""
    for t in report.outliers.outlier_times[:10]:
        out_rows += f"<tr><td>{t}</td></tr>"

    spread_html = ""
    if report.spread:
        spread_html = f"""<table><tr><th>Percentile</th><th>Spread (pips)</th></tr>{spread_rows}</table>
<p>Mean: {report.spread.mean} | Median: {report.spread.median} | Std: {report.spread.std} | Max: {report.spread.max_spread} at {report.spread.max_spread_time}</p>"""
    else:
        spread_html = "<p>No spread data available (need bid/ask or spread column)</p>"

    gap_html = ""
    if report.gaps.gap_times:
        gap_html = f"<h4>Gap Times (first 10)</h4><table><tr><th>Timestamp</th></tr>{gap_rows}</table>"
    else:
        gap_html = "<p>No gaps detected ✓</p>"

    outlier_html = ""
    if report.outliers.outlier_times:
        outlier_html = f"<h4>Outlier Times (first 10)</h4><table><tr><th>Timestamp</th></tr>{out_rows}</table>"
    else:
        outlier_html = "<p>No outliers detected ✓</p>"

    vol_rows_html = vol_rows or "<tr><td colspan=4>No volume data</td></tr>"
    vol_conc = "⚠️ High concentration" if report.volume.session_concentration > 0.7 else "✓ Well distributed"

    flags_html = "".join(f"<li>{f}</li>" for f in report.flags) or "<li>None</li>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Data Quality: {report.symbol} {report.timeframe}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:24px}}
h1{{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:8px}}
.grade-badge{{display:inline-block;padding:8px 16px;border-radius:6px;font-weight:700;font-size:24px;color:#fff;background:{color}}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{border:1px solid #30363d;padding:8px;text-align:left}}
th{{background:#21262d;color:#8b949e}}
tr:nth-child(even){{background:#161b22}}
.severity-low{{color:#7bd88f}}
.severity-medium{{color:#ffb800}}
.severity-high{{color:#ff8800}}
.severity-critical{{color:#ff3366}}
.banner{{background:#161b22;border-left:4px solid {color};padding:12px 16px;margin:16px 0;font-family:monospace}}
</style></head><body>
<h1>📊 Data Quality Dashboard — {report.symbol} {report.timeframe}</h1>
<div class='banner'>{report.loud_banner}</div>
<div class='card'><span class='grade-badge'>{report.grade}</span> Overall Score: <b>{report.score}/100</b></div>

<div class='grid'>
<div class='card'>
<h3>📈 Overview</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Symbol</td><td>{report.symbol}</td></tr>
<tr><td>Timeframe</td><td>{report.timeframe}</td></tr>
<tr><td>Source</td><td>{report.source}</td></tr>
<tr><td>Bars</td><td>{report.bars:,}</td></tr>
<tr><td>Date Range</td><td>{report.date_range[0]} → {report.date_range[1]}</td></tr>
<tr><td>Grade</td><td>{report.grade}</td></tr>
<tr><td>Score</td><td>{report.score}/100</td></tr>
</table>
</div>

<div class='card'>
<h3>⏱️ Gaps <span class='severity-{report.gaps.severity}'>{report.gaps.severity.upper()}</span></h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Gaps</td><td>{report.gaps.total_gaps}</td></tr>
<tr><td>Gap %</td><td>{report.gaps.gap_pct}%</td></tr>
<tr><td>Max Gap (bars)</td><td>{report.gaps.max_gap_bars}</td></tr>
<tr><td>Max Gap (hours)</td><td>{report.gaps.max_gap_hours}</td></tr>
</table>
{gap_html}
</div>

<div class='card'>
<h3>🎯 Outliers <span class='severity-{report.outliers.severity}'>{report.outliers.severity.upper()}</span></h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Count</td><td>{report.outliers.count}</td></tr>
<tr><td>% of Bars</td><td>{report.outliers.pct}%</td></tr>
<tr><td>Max Z-Score</td><td>{report.outliers.max_zscore}</td></tr>
</table>
{outlier_html}
</div>

<div class='card'>
<h3>🕐 Staleness <span class='severity-{report.stale.severity}'>{report.stale.severity.upper()}</span></h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Last Bar</td><td>{report.stale.last_timestamp}</td></tr>
<tr><td>Days Since</td><td>{report.stale.days_since_last}</td></tr>
<tr><td>Is Stale (>7d)</td><td>{'YES ⚠️' if report.stale.is_stale else 'No ✓'}</td></tr>
</table>
</div>

<div class='card'>
<h3>📊 Volume Profile <span class='severity-{report.volume.severity}'>{report.volume.severity.upper()}</span></h3>
<table>
<tr><th>Session</th><th>Mean Vol</th><th>Median Vol</th><th>% of Total</th></tr>
{vol_rows_html}
</table>
<p>Session Concentration (HHI): <b>{report.volume.session_concentration}</b> {vol_conc}</p>
</div>

<div class='card'>
<h3>📏 Spread Profile <span class='severity-{report.spread.severity if report.spread else "unknown"}'>{report.spread.severity.upper() if report.spread else "N/A"}</span></h3>
{spread_html}
</div>
</div>

<div class='card'>
<h3>🚩 Flags</h3>
<ul>{flags_html}</ul>
</div>

<p style='color:#8b949e;font-size:12px;margin-top:24px'>Generated: {pd.Timestamp.now(tz="UTC").isoformat()}</p>
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


def export_json_summary(report: DataQualityReport, out_path: str | Path) -> str:
    """Export JSON summary for CI/CD gates."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.to_json(), encoding="utf-8")
    return str(out_path)


def gate_check(report: DataQualityReport, min_grade: str = "C") -> dict:
    """CI/CD gate: pass/fail based on minimum grade."""
    grade_order = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
    passed = grade_order.get(report.grade, 0) >= grade_order.get(min_grade, 3)
    return {
        "passed": passed,
        "grade": report.grade,
        "score": report.score,
        "min_required": min_grade,
        "blocking_flags": [f for f in report.flags if any(kw in f for kw in ["critical", "empty", "zero_volume", "synthetic"])]
    }


# Convenience: auto-grade on DataFrame load
def auto_grade(df: pd.DataFrame, symbol: str = "", tf: str = "", source: str = "") -> DataQualityReport:
    """One-liner: analyze and return report. Use in pipelines."""
    return analyze_data_quality(df, symbol, tf, source)


if __name__ == "__main__":
    # Quick self-test
    import pandas as pd
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
    np.random.seed(42)
    px = 1.1 + np.cumsum(np.random.randn(500) * 0.001)
    df = pd.DataFrame({
        "open": px, "high": px + 0.001, "low": px - 0.001, "close": px,
        "volume": np.random.randint(100, 1000, 500),
        "spread": np.random.uniform(0.5, 3, 500)
    }, index=idx)

    # Inject some gaps
    df = df.iloc[::2]  # every other bar
    # Inject outlier
    df.iloc[10, df.columns.get_loc("close")] *= 1.05

    report = analyze_data_quality(df, "EURUSD", "H1", "test_source")
    print(report.loud_banner)
    print(f"Grade: {report.grade}, Score: {report.score}")
    print(f"Gaps: {report.gaps.total_gaps} ({report.gaps.severity})")
    print(f"Outliers: {report.outliers.count} ({report.outliers.severity})")
    print(f"Stale: {report.stale.is_stale} ({report.stale.severity})")
    print(f"Volume: {report.volume.severity}")
    print(f"Spread: {report.spread.severity if report.spread else 'N/A'}")

    # Export
    export_html_dashboard(report, "output/data_quality_test.html")
    export_json_summary(report, "output/data_quality_test.json")
    print("Exported to output/")