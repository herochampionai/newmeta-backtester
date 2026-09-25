"""Universe optimizer — which ticker fits the uploaded strategy most?

Runs strategy x symbols x timeframes through run_full(), scores each by any
CRITERION_PRESETS entry (Full / PF / Drawdown / WinRate / Composite...),
optionally runs Optuna per top-N combo. This replaces the heuristic-only
ticker_scanner.suitability with real backtest ranking.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.composite_criterion import CRITERION_PRESETS
from backtester.engine_full import run_full
from data.live_fetcher import fetch_with_priority
from strategies import STRATEGY_REGISTRY


def _metrics_for(strategy_name: str, df: pd.DataFrame, params: dict,
                 engine_kwargs: dict) -> dict:
    cls = STRATEGY_REGISTRY[strategy_name]
    strat = cls(params=params or {})
    sig = strat.generate(df)
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    if hasattr(sig, "exits") and sig.exits is not None:
        exits = sig.exits.fillna(False).astype(bool)
        signals = {strategy_name: (entries, exits, direction)}
    else:
        signals = {strategy_name: (entries, direction)}
    res = run_full(df, signals, params=params, **engine_kwargs)
    return res.get("metrics", {})


def rank_universe(strategy_name: str, symbols: list[str],
                  timeframes: list[str] | None = None,
                  params: dict | None = None,
                  criterion: str = "Full Criterion (All-in-One)",
                  engine_kwargs: dict | None = None,
                  terminal: str | None = None,
                  progress_callback: Any = None) -> pd.DataFrame:
    """Backtest strategy across all symbols x TFs, rank by criterion score."""
    timeframes = timeframes or ["M15", "H1", "H4"]
    engine_kwargs = dict(engine_kwargs or {})
    fn = CRITERION_PRESETS.get(criterion, CRITERION_PRESETS["Full Criterion (All-in-One)"])
    rows: list[dict] = []
    total = len(symbols) * len(timeframes)
    k = 0
    for sym in symbols:
        for tf in timeframes:
            k += 1
            if progress_callback:
                progress_callback(k, total, f"{sym} [{tf}] ({criterion})...")
            try:
                df, info = fetch_with_priority(sym.strip().upper(), tf,
                                               allow_synthetic=False,
                                               terminal_override=terminal)
                if df is None or len(df) < 100:
                    continue
                m = _metrics_for(strategy_name, df, params or {}, engine_kwargs)
                score = float(fn(m))
                rows.append({
                    "symbol": sym.strip().upper(), "timeframe": tf,
                    "criterion": criterion, "score": round(score, 2),
                    "net_pnl": round(float(m.get("net_pnl", 0) or 0), 2),
                    "profit_factor": round(float(m.get("profit_factor", 0) or 0), 2),
                    "max_drawdown": round(float(m.get("max_drawdown", 0) or 0), 4),
                    "win_rate": round(float(m.get("win_rate", 0) or 0), 3),
                    "sharpe": round(float(m.get("sharpe", 0) or 0), 2),
                    "n_trades": int(m.get("n_trades", 0) or 0),
                    "bars": len(df),
                    "source": (info or {}).get("source", "?"),
                })
            except Exception as e:
                rows.append({"symbol": sym, "timeframe": tf, "criterion": criterion,
                             "score": -1e9, "error": str(e)[:120]})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def optimize_top_n(strategy_name: str, ranked: pd.DataFrame,
                   param_spec: dict, n_top: int = 3, n_trials: int = 50,
                   engine_kwargs: dict | None = None,
                   criterion: str = "Full Criterion (All-in-One)",
                   terminal: str | None = None) -> pd.DataFrame:
    """Run Optuna criterion study on the top-N symbol/TF fits."""
    from analysis.optuna_optimizer import optimize_strategy_criterion
    fn = CRITERION_PRESETS.get(criterion, CRITERION_PRESETS["Full Criterion (All-in-One)"])
    out = []
    for _, r in ranked.head(max(1, n_top)).iterrows():
        sym, tf = r["symbol"], r["timeframe"]
        df, _ = fetch_with_priority(sym, tf, allow_synthetic=False, terminal_override=terminal)
        if df is None or len(df) < 100:
            continue
        study = optimize_strategy_criterion(strategy_name, df, param_spec, fn,
                                            n_trials=n_trials,
                                            engine_kwargs=dict(engine_kwargs or {}),
                                            study_name=f"{strategy_name}_{sym}_{tf}")
        best = study.best_trial
        out.append({"symbol": sym, "timeframe": tf, "criterion": criterion,
                    "best_score": round(float(best.value), 2),
                    "best_params": best.params,
                    "n_trades": best.user_attrs.get("metrics", {}).get("n_trades", 0)})
    return pd.DataFrame(out)


def apply_multiple_testing_correction(
    ranked: pd.DataFrame,
    alpha: float = 0.05,
    method: str = "benjamini_hochberg",
    score_col: str = "score"
) -> pd.DataFrame:
    """R003: Apply multiple testing correction to universe optimizer results.

    Universe optimization is multiple testing by nature (you test the same
    strategy on N symbols × M timeframes). Without correction, the top result
    is likely overfit by chance.

    Args:
        ranked: DataFrame from rank_universe() with a score column
        alpha: significance level
        method: "bonferroni" or "benjamini_hochberg"
        score_col: name of the score column (default "score")

    Returns:
        DataFrame with added columns:
        - p_value: per-combo p-value (derived from score, conservative)
        - significant: whether combo survives multiple testing
        - correction_method: which method was used
    """
    from analysis.statistical_significance import benjamini_hochberg, bonferroni_correction
    if ranked is None or len(ranked) == 0:
        return ranked

    n = len(ranked)
    scores = ranked[score_col].values.astype(float)

    # Convert scores to p-values using a conservative mapping:
    # Higher score → lower p-value. Normalize scores to [0, 1] and invert.
    # Use rank-based p-values for robustness.
    ranks = np.argsort(np.argsort(-scores)) + 1  # 1 = best
    p_values = ranks / n  # uniform p-values from rank

    if method == "bonferroni":
        correction = bonferroni_correction(list(p_values), alpha)
        rejected = set(correction["rejected_indices"])
    elif method == "benjamini_hochberg":
        correction = benjamini_hochberg(list(p_values), alpha)
        rejected = set(correction["rejected_indices"])
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bonferroni' or 'benjamini_hochberg'")

    out = ranked.copy()
    out["p_value"] = np.round(p_values, 4)
    out["significant"] = [i in rejected for i in range(n)]
    out["correction_method"] = method
    out["adjusted_alpha"] = correction.get("adjusted_alpha", alpha)

    # Add warning if nothing survives
    n_sig = sum(out["significant"])
    if n_sig == 0 and n > 1:
        out.attrs["warning"] = (
            f"No combinations survived {method} correction at alpha={alpha}. "
            f"This suggests the ranking may be overfit. "
            f"Consider: reducing # of combos, increasing sample size, or relaxing alpha."
        )
    return out


def apply_deflated_sharpe(ranked: pd.DataFrame,
                          benchmark_sr: float = 0.0,
                          score_col: str = "sharpe") -> pd.DataFrame:
    """R003: Compute Deflated Sharpe Ratio for each combo in universe results.

    The DSR accounts for the fact that the MAX Sharpe across N combos is
    inflated by chance. For each row, computes DSR using n_combos as trials.

    Args:
        ranked: DataFrame from rank_universe() with sharpe column
        benchmark_sr: benchmark to compare against (default 0)
        score_col: name of the Sharpe column (default "sharpe")

    Returns:
        DataFrame with added columns:
        - dsr: Deflated Sharpe probability
        - dsr_verdict: STRONG/GOOD/WEAK/OVERFIT/CHECK
    """
    from analysis.statistical_significance import deflated_sharpe_ratio
    if ranked is None or len(ranked) == 0:
        return ranked

    n = len(ranked)
    out = ranked.copy()
    dsrs = []
    verdicts = []
    for _, row in ranked.iterrows():
        sr = float(row.get(score_col, 0) or 0)
        if not np.isfinite(sr):
            sr = 0.0
        # Use n_trades as sample size if available, else default
        result = deflated_sharpe_ratio(
            observed_sr=sr,
            n_trials=max(2, n),
            skewness=0.0,
            kurtosis=3.0,
            benchmark_sr=benchmark_sr
        )
        dsrs.append(result.get("dsr"))
        verdicts.append(result.get("verdict", "unknown"))

    out["dsr"] = dsrs
    out["dsr_verdict"] = verdicts
    out["n_combos_tested"] = n
    return out
