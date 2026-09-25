"""R009: Adversarial stress testing — survival scenarios beyond normal market conditions.

Tests strategies against:
- Historical crisis replay (2008 GFC, 2020 COVID, 2022 rates)
- Liquidity crunch (spread → ∞, volume → 0)
- Correlation breakdown (all assets → 1 in panic)
- Flash crash simulation (sudden -10% intraday move)
- Slippage spikes (10x normal)
- Spread widening (5x normal)

Use cases:
- Validate strategy survives tail risk
- Identify hidden leverage/correlations
- Set realistic max position sizes
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Historical crisis definitions (approximate dates and shock magnitudes)
CRISIS_SCENARIOS = {
    "2008_GFC": {
        "description": "Global Financial Crisis - Lehman collapse",
        "period": ("2008-09-01", "2009-03-01"),
        "shock_pct": -0.40,  # -40% peak-to-trough
        "volatility_mult": 3.5,
        "spread_mult": 5.0,
        "correlation_spike": 0.85,
    },
    "2020_COVID": {
        "description": "COVID-19 crash",
        "period": ("2020-02-20", "2020-04-01"),
        "shock_pct": -0.34,
        "volatility_mult": 4.0,
        "spread_mult": 8.0,
        "correlation_spike": 0.90,
    },
    "2022_RATES": {
        "description": "2022 rate hiking cycle (stocks + bonds down)",
        "period": ("2022-01-01", "2022-10-31"),
        "shock_pct": -0.25,
        "volatility_mult": 1.8,
        "spread_mult": 2.0,
        "correlation_spike": 0.70,
    },
    "Flash_Crash_2010": {
        "description": "May 6 2010 Flash Crash",
        "period": ("2010-05-06", "2010-05-07"),
        "shock_pct": -0.09,  # intraday
        "volatility_mult": 10.0,
        "spread_mult": 20.0,
        "correlation_spike": 0.95,
    },
    "Black_Monday_1987": {
        "description": "1987 Black Monday",
        "period": ("1987-10-19", "1987-10-20"),
        "shock_pct": -0.22,
        "volatility_mult": 8.0,
        "spread_mult": 15.0,
        "correlation_spike": 0.90,
    },
    "Dotcom_2000": {
        "description": "Dot-com bubble burst",
        "period": ("2000-03-10", "2002-10-09"),
        "shock_pct": -0.78,
        "volatility_mult": 2.0,
        "spread_mult": 1.5,
        "correlation_spike": 0.60,
    },
}


@dataclass
class StressResult:
    """Result of a single stress scenario."""
    scenario: str
    description: str
    base_metrics: dict
    stressed_metrics: dict
    survived: bool
    max_dd_change: float
    pnl_change_pct: float
    notes: str = ""


def replay_crisis(df: pd.DataFrame, scenario: str = "2020_COVID") -> pd.DataFrame:
    """Replay a historical crisis scenario by modifying price/volume/spread.

    Args:
        df: OHLC dataframe
        scenario: key from CRISIS_SCENARIOS

    Returns:
        New dataframe with crisis modifications applied to the scenario period
    """
    if scenario not in CRISIS_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Options: {list(CRISIS_SCENARIOS.keys())}")

    sc = CRISIS_SCENARIOS[scenario]
    start, end = pd.Timestamp(sc["period"][0], tz="UTC"), pd.Timestamp(sc["period"][1], tz="UTC")

    out = df.copy()
    # Find rows in crisis period (or use random subset if no date overlap)
    mask = (df.index >= start) & (df.index <= end)
    if mask.sum() < 5:
        # Use first 20% of data as proxy
        n = max(int(len(df) * 0.2), 20)
        mask = pd.Series(False, index=df.index)
        mask.iloc[:n] = True

    # Apply shock to prices (scale to target shock_pct)
    if mask.any():
        # Get current price level
        crisis_close = df.loc[mask, "close"]
        if len(crisis_close) > 1:
            current_level = float(crisis_close.iloc[0])
            target_level = current_level * (1 + sc["shock_pct"])
            # Gradual shock: linearly interpolate
            n_crisis = mask.sum()
            shock_path = np.linspace(current_level, target_level, n_crisis)
            adjustment = shock_path - crisis_close.values
            out.loc[mask, "close"] = crisis_close.values + adjustment
            out.loc[mask, "open"] = out.loc[mask, "open"] + adjustment
            # Widen high/low to reflect volatility spike
            hl_noise = np.abs(np.random.default_rng(42).normal(0, 1, n_crisis))
            out.loc[mask, "high"] = np.maximum(out.loc[mask, "high"] + adjustment, out.loc[mask, "close"] + hl_noise * sc["volatility_mult"] * 0.001)
            out.loc[mask, "low"] = np.minimum(out.loc[mask, "low"] + adjustment, out.loc[mask, "close"] - hl_noise * sc["volatility_mult"] * 0.001)

    return out


def simulate_liquidity_crunch(df: pd.DataFrame, spread_mult: float = 10.0,
                              volume_mult: float = 0.1) -> pd.DataFrame:
    """Simulate liquidity crunch: spread widens, volume drops."""
    out = df.copy()
    if "spread" in out.columns:
        out["spread"] = out["spread"] * spread_mult
    elif "bid" in out.columns and "ask" in out.columns:
        mid = (out["bid"] + out["ask"]) / 2
        old_spread = out["ask"] - out["bid"]
        new_spread = old_spread * spread_mult
        out["bid"] = mid - new_spread / 2
        out["ask"] = mid + new_spread / 2
    if "volume" in out.columns:
        out["volume"] = out["volume"] * volume_mult
    return out


def simulate_correlation_breakdown(df_returns: pd.DataFrame,
                                   target_correlation: float = 0.85) -> pd.DataFrame:
    """Force all asset returns to a high correlation (panic mode).

    Args:
        df_returns: DataFrame of returns (each column = one asset)
        target_correlation: target correlation between all pairs
    """
    if df_returns.shape[1] < 2:
        return df_returns

    # Generate common factor
    n = len(df_returns)
    common_factor = np.random.default_rng(42).normal(0, 1, n)
    idiosyncratic = df_returns.values

    # Mix to achieve target correlation
    # corr(asset, common_factor) = sqrt(target_correlation)
    weight = np.sqrt(target_correlation)
    new_returns = weight * common_factor[:, None] + (1 - weight) * idiosyncratic

    # Normalize to preserve variance per asset
    for j in range(new_returns.shape[1]):
        col_std = df_returns.iloc[:, j].std()
        if col_std > 0:
            new_returns[:, j] = new_returns[:, j] * (col_std / new_returns[:, j].std())

    return pd.DataFrame(new_returns, index=df_returns.index, columns=df_returns.columns)


def simulate_flash_crash(df: pd.DataFrame, magnitude: float = -0.10,
                         recovery_bars: int = 5) -> pd.DataFrame:
    """Simulate an intraday flash crash: sudden drop then partial recovery."""
    out = df.copy()
    if len(out) < recovery_bars + 2:
        return out

    # Pick a random bar to crash
    rng = np.random.default_rng(42)
    crash_bar = rng.integers(recovery_bars, len(out) - recovery_bars)

    # Apply sudden drop
    crash_factor = 1 + magnitude
    out.iloc[crash_bar, out.columns.get_loc("close")] *= crash_factor
    out.iloc[crash_bar, out.columns.get_loc("low")] *= crash_factor
    if "open" in out.columns:
        out.iloc[crash_bar, out.columns.get_loc("open")] *= crash_factor

    # Gradual recovery over next bars (50% recovery)
    recovery_factor = 1 + magnitude * 0.5  # partial recovery
    recovery_path = np.linspace(crash_factor, recovery_factor, recovery_bars)
    for k in range(1, recovery_bars + 1):
        if crash_bar + k < len(out):
            out.iloc[crash_bar + k, out.columns.get_loc("close")] *= recovery_path[k] / recovery_path[k - 1] if k > 0 else 1

    return out


def simulate_spread_widening(df: pd.DataFrame, mult: float = 5.0) -> pd.DataFrame:
    """Simulate spread widening (e.g., during off-hours or news events)."""
    return simulate_liquidity_crunch(df, spread_mult=mult, volume_mult=1.0)


def run_stress_test(
    df: pd.DataFrame,
    strategy_fn: Callable,
    base_metrics_fn: Callable | None = None,
    scenarios: list[str] | None = None,
    custom_shocks: list[dict] | None = None
) -> dict:
    """Run full stress test suite.

    Args:
        df: OHLC dataframe
        strategy_fn: function that runs backtest and returns metrics dict
        base_metrics_fn: optional pre-computed base metrics
        scenarios: list of CRISIS_SCENARIOS keys (default: all)
        custom_shocks: list of custom shock dicts with keys:
          {"name", "shock_pct", "volatility_mult", "spread_mult"}

    Returns:
        dict with:
        - base_metrics: baseline backtest
        - results: list of StressResult per scenario
        - survived_count: n scenarios where strategy was profitable or DD < 50%
        - failed_count: n scenarios where strategy lost >50%
        - verdict: ROBUST / MARGINAL / FRAGILE
    """
    scenarios = scenarios or list(CRISIS_SCENARIOS.keys())

    # Get base metrics
    if base_metrics_fn is not None:
        base_metrics = base_metrics_fn
    else:
        base_metrics = strategy_fn(df)
    base_pnl = base_metrics.get("net_pnl", 0)
    base_dd = abs(base_metrics.get("max_drawdown", 0))

    results = []
    for scenario in scenarios:
        if scenario not in CRISIS_SCENARIOS:
            continue
        try:
            stressed_df = replay_crisis(df, scenario)
            stressed_metrics = strategy_fn(stressed_df)
            sp = stressed_metrics.get("net_pnl", 0)
            sdd = abs(stressed_metrics.get("max_drawdown", 0))
            dd_change = sdd - base_dd
            pnl_change = ((sp - base_pnl) / abs(base_pnl) * 100) if base_pnl != 0 else 0

            # Survival: profitable OR max DD < 50%
            survived = sp > 0 or sdd < 0.50

            sc = CRISIS_SCENARIOS[scenario]
            results.append(StressResult(
                scenario=scenario,
                description=sc["description"],
                base_metrics={"net_pnl": base_pnl, "max_drawdown": base_dd},
                stressed_metrics={"net_pnl": sp, "max_drawdown": sdd},
                survived=survived,
                max_dd_change=round(dd_change, 4),
                pnl_change_pct=round(pnl_change, 1),
                notes=f"shock={sc['shock_pct']:.0%} vol_mult={sc['volatility_mult']}",
            ))
        except Exception as e:
            results.append(StressResult(
                scenario=scenario,
                description=CRISIS_SCENARIOS[scenario]["description"],
                base_metrics={}, stressed_metrics={},
                survived=False, max_dd_change=0, pnl_change_pct=0,
                notes=f"error: {str(e)[:80]}"
            ))

    survived_count = sum(1 for r in results if r.survived)
    failed_count = len(results) - survived_count
    survival_rate = survived_count / max(len(results), 1)

    if survival_rate >= 0.8:
        verdict = "ROBUST"
    elif survival_rate >= 0.5:
        verdict = "MARGINAL"
    else:
        verdict = "FRAGILE"

    return {
        "base_metrics": base_metrics,
        "results": [r.__dict__ for r in results],
        "survived_count": survived_count,
        "failed_count": failed_count,
        "survival_rate": round(survival_rate, 2),
        "verdict": verdict,
    }


def export_stress_report(result: dict, out_path: str | Path) -> str:
    """Export stress test report as HTML."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = result.get("base_metrics", {})
    results = result.get("results", [])
    verdict = result.get("verdict", "?")
    survived = result.get("survived_count", 0)
    failed = result.get("failed_count", 0)

    color = {"ROBUST": "#7bd88f", "MARGINAL": "#ffb800", "FRAGILE": "#ff6b6b"}.get(verdict, "#888")

    rows = ""
    for r in results:
        cls = "good" if r["survived"] else "bad"
        rows += f"<tr class='{cls}'><td>{r['scenario']}</td><td>{r['description']}</td>"
        rows += f"<td>${r['base_metrics'].get('net_pnl', 0):.0f}</td>"
        rows += f"<td>${r['stressed_metrics'].get('net_pnl', 0):.0f}</td>"
        rows += f"<td>{r['pnl_change_pct']:+.0f}%</td>"
        rows += f"<td>{r['max_dd_change']:+.3f}</td>"
        rows += f"<td>{'✓' if r['survived'] else '✗'}</td>"
        rows += f"<td>{r['notes']}</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Stress Test Report</title>
<style>
body{{font-family:'Segoe UI',Arial;background:#0d1117;color:#e6edf3;margin:0;padding:24px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:16px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{border:1px solid #30363d;padding:8px}}
th{{background:#21262d;color:#8b949e}}
tr.good{{background:#1a2a1a}}
tr.bad{{background:#3a1a1a}}
.verdict{{display:inline-block;padding:8px 16px;border-radius:6px;font-weight:700;color:white;background:{color}}}
</style></head><body>
<h1>🔥 Stress Test Report</h1>
<div class='card'>
<span class='verdict'>{verdict}</span>
Survived: <b>{survived}/{survived+failed}</b> ({result.get('survival_rate', 0):.0%})
</div>

<div class='card'>
<h3>Baseline</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Net PnL</td><td>${base.get('net_pnl', 0):.2f}</td></tr>
<tr><td>Max DD</td><td>{abs(base.get('max_drawdown', 0)):.2%}</td></tr>
<tr><td>Sharpe</td><td>{base.get('sharpe', 0):.2f}</td></tr>
</table>
</div>

<div class='card'>
<h3>Scenario Results</h3>
<table>
<tr><th>Scenario</th><th>Description</th><th>Base PnL</th><th>Stressed PnL</th><th>Change</th><th>DD Change</th><th>Survived</th><th>Notes</th></tr>
{rows}
</table>
</div>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


# ---------- Self-test ----------
if __name__ == "__main__":
    import pandas as pd

    from backtester.engine_full import run_full
    from strategies import STRATEGY_REGISTRY

    # Generate test data
    idx = pd.date_range("2015-01-01", periods=3000, freq="D", tz="UTC")
    np.random.seed(42)
    px = 1.1 + np.cumsum(np.random.randn(3000) * 0.01)
    df = pd.DataFrame({
        "open": px, "high": px + 0.01, "low": px - 0.01, "close": px,
        "volume": np.random.randint(1000, 10000, 3000),
    }, index=idx)

    def strategy_test(data):
        cls = STRATEGY_REGISTRY["adx"]
        strat = cls(params={"bars_calculate": 14})
        sig = strat.generate(data)
        entries = sig.entries.fillna(False).astype(bool)
        direction = pd.Series(sig.direction, index=data.index).fillna(0).astype(int)
        signals = {"adx": (entries, direction)}
        res = run_full(data, signals)
        return res.get("metrics", {})

    print("=== Stress Test ===")
    result = run_stress_test(df, strategy_test)
    print(f"Verdict: {result['verdict']}")
    print(f"Survived: {result['survived_count']}/{result['survived_count']+result['failed_count']}")
    print()
    for r in result["results"]:
        status = "✓" if r["survived"] else "✗"
        print(f"  {status} {r['scenario']:20s} PnL: ${r['stressed_metrics'].get('net_pnl', 0):>8.0f} ({r['pnl_change_pct']:+.0f}%) DD: {r['max_dd_change']:+.3f}")

    export_stress_report(result, "output/stress_test.html")
    print()
    print("Report: output/stress_test.html")
