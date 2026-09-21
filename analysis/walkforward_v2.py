"""Walk-forward analysis v2 — enhanced with anchored mode, stability heatmap, auto-accept.

R004 enhancements over basic walk_forward:
- Anchored vs rolling toggle (anchored: IS grows from start; rolling: fixed window)
- Parameter stability heatmap (how much each param changes across windows)
- Auto-accept based on WFE threshold (default 0.5)
- Integration with run_full (newer engine) — supports tick_mode, swap, etc.
- Consolidated WF summary with aggregate metrics (mean WFE, WFE distribution)
- Per-window trade count tracking for low-data warning

Backward compat: walk_forward() and wf_summary() from analysis.walkforward still work.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import numpy as np
import pandas as pd

from backtester.pro_suite import wfe_gate


@dataclass
class WFWindowV2:
    """Enhanced walk-forward window with stability tracking."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: dict
    test_metrics: dict
    params: dict
    train_score: float = 0.0
    test_score: float = 0.0
    wfe: float = 0.0
    idle_day_count: int = 0
    n_train_trades: int = 0
    n_test_trades: int = 0
    passed: bool = False
    note: str = ""


@dataclass
class WFReport:
    """Aggregated walk-forward report."""
    windows: list[WFWindowV2]
    mode: str  # "rolling" or "anchored"
    aggregate: dict
    param_stability: dict  # param -> std/mean across windows
    passed_count: int
    failed_count: int
    auto_accept: bool  # Whether overall pass criterion was met
    auto_accept_threshold: float
    overall_verdict: str  # "ROBUST", "ACCEPT", "OVERFIT"

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for w in self.windows:
            rows.append({
                "window_id": w.window_id,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "train_score": round(w.train_score, 3),
                "test_score": round(w.test_score, 3),
                "wfe": round(w.wfe, 3),
                "n_train_trades": w.n_train_trades,
                "n_test_trades": w.n_test_trades,
                "idle_days": w.idle_day_count,
                "passed": w.passed,
                "note": w.note,
            })
        return pd.DataFrame(rows)

    def param_stability_dataframe(self) -> pd.DataFrame:
        """DataFrame showing param stability across windows."""
        if not self.param_stability:
            return pd.DataFrame()
        rows = []
        for param, stats in self.param_stability.items():
            rows.append({
                "param": param,
                "mean": round(stats["mean"], 4),
                "std": round(stats["std"], 4),
                "cv": round(stats["cv"], 3),  # coefficient of variation
                "min": round(stats["min"], 4),
                "max": round(stats["max"], 4),
                "stability": "STABLE" if stats["cv"] < 0.2 else ("DRIFT" if stats["cv"] < 0.5 else "UNSTABLE")
            })
        return pd.DataFrame(rows).sort_values("cv", ascending=False)


def _compute_window_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int,
    test_months: int,
    roll_months: int,
    anchored: bool
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate list of (train_start, train_end, test_start, test_end) tuples.

    Rolling: fixed train window size, advances by roll_months
    Anchored: train window grows from start, advances by roll_months
    """
    windows = []
    if anchored:
        # Train start fixed, train end advances
        cursor = start
        train_start = start
        while True:
            train_end = cursor + pd.DateOffset(months=train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)
            if test_end > end:
                break
            windows.append((train_start, train_end, test_start, test_end))
            cursor += pd.DateOffset(months=roll_months)
    else:
        # Rolling: fixed train window, advances by roll_months
        cursor = start
        while True:
            train_start = cursor
            train_end = cursor + pd.DateOffset(months=train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)
            if test_end > end:
                break
            windows.append((train_start, train_end, test_start, test_end))
            cursor += pd.DateOffset(months=roll_months)
    return windows


def _optimize_on_train(
    df_train: pd.DataFrame,
    strategy_name: str,
    param_spec: dict,
    score_fn: Callable,
    n_trials: int,
    engine_kwargs: dict,
    seed: int = 42,
    verbose: bool = False
) -> tuple[dict, float]:
    """Run Optuna on train window, return (best_params, best_score)."""
    import optuna
    from strategies import STRATEGY_REGISTRY
    from backtester.engine_full import run_full

    # Suppress Optuna logging unless verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    cls = STRATEGY_REGISTRY[strategy_name]
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )

    def obj(trial):
        params = _sample_params(trial, param_spec)
        try:
            strat = cls(params=params)
            sig = strat.generate(df_train)
            entries = sig.entries.fillna(False).astype(bool)
            direction = pd.Series(sig.direction, index=df_train.index).fillna(0).astype(int)
            if hasattr(sig, "exits") and sig.exits is not None:
                exits = sig.exits.fillna(False).astype(bool)
                signals = {strategy_name: (entries, exits, direction)}
            else:
                signals = {strategy_name: (entries, direction)}
            res = run_full(df_train, signals, params=params, **engine_kwargs)
            metrics = res.get("metrics", {})
            return score_fn(metrics)
        except Exception:
            return -1e9

    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


def _sample_params(trial, param_spec: dict) -> dict:
    """Sample params from spec (handles int/float/categorical/log)."""
    params = {}
    for name, spec in param_spec.items():
        if isinstance(spec, dict):
            kind = spec.get("type", "float")
            if kind == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif kind == "log":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            params[name] = trial.suggest_float(name, spec[0], spec[1])
        elif isinstance(spec, (list, tuple)) and len(spec) > 2:
            params[name] = trial.suggest_categorical(name, list(spec))
        else:
            params[name] = spec
    return params


def walk_forward_v2(
    df: pd.DataFrame,
    strategy_name: str,
    param_spec: dict,
    criterion_fn: Callable,
    train_months: int = 12,
    test_months: int = 3,
    roll_months: int = 3,
    n_trials: int = 50,
    anchored: bool = False,
    min_train_bars: int = 100,
    min_test_bars: int = 30,
    wfe_threshold: float = 0.5,
    engine_kwargs: dict | None = None,
    seed: int = 42,
    progress_callback: Optional[Callable] = None,
    verbose: bool = False
) -> WFReport:
    """Enhanced walk-forward with anchored/rolling toggle, stability, auto-accept.

    Args:
        df: OHLC dataframe
        strategy_name: name from STRATEGY_REGISTRY
        param_spec: dict of param_name -> (low, high) or {type, low, high, choices}
        criterion_fn: function that scores metrics dict → float
        train_months: training window length in months
        test_months: testing window length in months
        roll_months: how much to advance per step
        n_trials: Optuna trials per window
        anchored: True = expanding train window from start; False = rolling fixed window
        min_train_bars: skip window if train has fewer bars
        min_test_bars: skip window if test has fewer bars
        wfe_threshold: per-window WFE threshold (test/train ratio)
        engine_kwargs: passed to run_full

    Returns:
        WFReport with windows, aggregate stats, param stability, auto-accept verdict
    """
    engine_kwargs = dict(engine_kwargs or {})
    if df is None or len(df) < 100:
        return WFReport(
            windows=[], mode="rolling" if not anchored else "anchored",
            aggregate={}, param_stability={},
            passed_count=0, failed_count=0,
            auto_accept=False, auto_accept_threshold=wfe_threshold,
            overall_verdict="NO_DATA"
        )

    start = df.index[0]
    end = df.index[-1]
    window_dates = _compute_window_dates(
        start, end, train_months, test_months, roll_months, anchored
    )

    windows = []
    all_params = {}  # param -> list of values across windows

    total = len(window_dates)
    for i, (tr_start, tr_end, te_start, te_end) in enumerate(window_dates):
        if progress_callback:
            progress_callback(i + 1, total, f"WF window {i+1}/{total}")

        df_train = df[(df.index >= tr_start) & (df.index < tr_end)]
        df_test = df[(df.index >= te_start) & (df.index < te_end)]

        if len(df_train) < min_train_bars or len(df_test) < min_test_bars:
            continue

        # Optimize on train
        try:
            best_params, train_score = _optimize_on_train(
                df_train, strategy_name, param_spec, criterion_fn,
                n_trials, engine_kwargs, seed, verbose
            )
        except Exception as e:
            windows.append(WFWindowV2(
                window_id=i, train_start=tr_start, train_end=tr_end,
                test_start=te_start, test_end=te_end,
                train_metrics={}, test_metrics={}, params={},
                passed=False, note=f"train_optimize_failed: {str(e)[:80]}"
            ))
            continue

        # Score on test
        try:
            from strategies import STRATEGY_REGISTRY
            from backtester.engine_full import run_full
            cls = STRATEGY_REGISTRY[strategy_name]
            strat = cls(params=best_params)
            sig_tr = strat.generate(df_train)
            sig_te = strat.generate(df_test)

            entries_tr = sig_tr.entries.fillna(False).astype(bool)
            direction_tr = pd.Series(sig_tr.direction, index=df_train.index).fillna(0).astype(int)
            if hasattr(sig_tr, "exits") and sig_tr.exits is not None:
                exits_tr = sig_tr.exits.fillna(False).astype(bool)
                signals_tr = {strategy_name: (entries_tr, exits_tr, direction_tr)}
            else:
                signals_tr = {strategy_name: (entries_tr, direction_tr)}
            res_tr = run_full(df_train, signals_tr, params=best_params, **engine_kwargs)
            m_tr = res_tr.get("metrics", {})

            entries_te = sig_te.entries.fillna(False).astype(bool)
            direction_te = pd.Series(sig_te.direction, index=df_test.index).fillna(0).astype(int)
            if hasattr(sig_te, "exits") and sig_te.exits is not None:
                exits_te = sig_te.exits.fillna(False).astype(bool)
                signals_te = {strategy_name: (entries_te, exits_te, direction_te)}
            else:
                signals_te = {strategy_name: (entries_te, direction_te)}
            res_te = run_full(df_test, signals_te, params=best_params, **engine_kwargs)
            m_te = res_te.get("metrics", {})

            test_score = float(criterion_fn(m_te))

            # WFE calculation
            wfe = wfe_gate(m_tr, m_te, threshold=wfe_threshold)
            wfe_val = wfe.get("wfe", 0.0)
            passed = wfe.get("pass", False)
            verdict = wfe.get("verdict", "OVERFIT")

            # Idle day tracking
            test_entries = entries_te
            entry_dates = set(test_entries[test_entries].index.strftime("%Y-%m-%d"))
            all_dates = set(df_test.index.strftime("%Y-%m-%d"))
            idle_days = sorted(all_dates - entry_dates)

            windows.append(WFWindowV2(
                window_id=i,
                train_start=tr_start, train_end=tr_end,
                test_start=te_start, test_end=te_end,
                train_metrics=m_tr, test_metrics=m_te,
                params=best_params,
                train_score=train_score, test_score=test_score,
                wfe=wfe_val,
                idle_day_count=len(idle_days),
                n_train_trades=int(m_tr.get("n_trades", 0)),
                n_test_trades=int(m_te.get("n_trades", 0)),
                passed=passed,
                note=verdict
            ))

            # Track params for stability
            for k, v in best_params.items():
                all_params.setdefault(k, []).append(float(v) if isinstance(v, (int, float)) else 0.0)

        except Exception as e:
            windows.append(WFWindowV2(
                window_id=i, train_start=tr_start, train_end=tr_end,
                test_start=te_start, test_end=te_end,
                train_metrics={}, test_metrics={}, params={},
                passed=False, note=f"test_score_failed: {str(e)[:80]}"
            ))

    # Aggregate stats
    wfe_vals = [w.wfe for w in windows if w.wfe != 0]
    test_scores = [w.test_score for w in windows if w.test_score != 0]
    train_scores = [w.train_score for w in windows if w.train_score != 0]

    aggregate = {
        "n_windows": len(windows),
        "mean_wfe": round(float(np.mean(wfe_vals)), 3) if wfe_vals else 0.0,
        "median_wfe": round(float(np.median(wfe_vals)), 3) if wfe_vals else 0.0,
        "min_wfe": round(float(np.min(wfe_vals)), 3) if wfe_vals else 0.0,
        "max_wfe": round(float(np.max(wfe_vals)), 3) if wfe_vals else 0.0,
        "wfe_std": round(float(np.std(wfe_vals)), 3) if wfe_vals else 0.0,
        "mean_train_score": round(float(np.mean(train_scores)), 3) if train_scores else 0.0,
        "mean_test_score": round(float(np.mean(test_scores)), 3) if test_scores else 0.0,
        "score_degradation": round(
            1.0 - (np.mean(test_scores) / np.mean(train_scores))
            if train_scores and np.mean(train_scores) > 0 else 0.0, 3
        ),
    }

    # Param stability
    param_stability = {}
    for param, values in all_params.items():
        if len(values) < 2:
            continue
        arr = np.array(values)
        m = float(arr.mean())
        s = float(arr.std())
        param_stability[param] = {
            "mean": m,
            "std": s,
            "cv": s / m if abs(m) > 1e-9 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    passed_count = sum(1 for w in windows if w.passed)
    failed_count = len(windows) - passed_count

    # Auto-accept: 70% of windows pass WFE
    auto_accept = passed_count / max(len(windows), 1) >= 0.7
    if aggregate["mean_wfe"] >= 0.7:
        overall = "ROBUST"
    elif aggregate["mean_wfe"] >= wfe_threshold:
        overall = "ACCEPT"
    else:
        overall = "OVERFIT"

    return WFReport(
        windows=windows,
        mode="anchored" if anchored else "rolling",
        aggregate=aggregate,
        param_stability=param_stability,
        passed_count=passed_count,
        failed_count=failed_count,
        auto_accept=auto_accept,
        auto_accept_threshold=wfe_threshold,
        overall_verdict=overall,
    )


def wf_heatmap_data(report: WFReport, metric: str = "wfe") -> dict:
    """Generate data for a walk-forward WFE heatmap (per window).

    Returns dict suitable for plotly/matplotlib:
    - x: window_id or test date
    - y: WFE values
    - threshold line
    """
    windows = report.windows
    if not windows:
        return {"x": [], "y": [], "threshold": report.auto_accept_threshold}

    x = [w.test_start.strftime("%Y-%m-%d") for w in windows]
    y = [w.wfe for w in windows]
    colors = ["green" if w.passed else "red" for w in windows]

    return {
        "x": x,
        "y": y,
        "colors": colors,
        "threshold": report.auto_accept_threshold,
        "mean": report.aggregate.get("mean_wfe", 0),
    }


def wf_param_stability_heatmap_data(report: WFReport) -> dict:
    """Generate data for parameter stability heatmap.

    Returns dict with params as rows, windows as columns.
    """
    if not report.windows:
        return {"params": [], "windows": [], "matrix": []}

    param_names = list(report.param_stability.keys())
    window_ids = [w.window_id for w in report.windows]

    matrix = []
    for param in param_names:
        row = []
        for w in report.windows:
            val = w.params.get(param)
            if val is None or not isinstance(val, (int, float)):
                val = 0.0
            row.append(float(val))
        matrix.append(row)

    return {
        "params": param_names,
        "windows": window_ids,
        "matrix": matrix,
        "stability": report.param_stability_dataframe().to_dict("records"),
    }


if __name__ == "__main__":
    # Quick self-test
    import pandas as pd
    import numpy as np

    # Generate test data with a known regime
    idx = pd.date_range("2020-01-01", periods=3000, freq="h", tz="UTC")
    np.random.seed(42)
    px = 1.1 + np.cumsum(np.random.randn(3000) * 0.001)
    df = pd.DataFrame({
        "open": px, "high": px + 0.001, "low": px - 0.001, "close": px,
        "volume": np.random.randint(100, 1000, 3000)
    }, index=idx)

    # Test with simple Sharpe criterion
    from analysis.composite_criterion import criterion_sharpe_only

    param_spec = {"period": (5, 50)}
    report = walk_forward_v2(
        df, "sma_cross", param_spec, criterion_sharpe_only,
        train_months=2, test_months=1, roll_months=1,
        n_trials=10, anchored=False
    )

    print(f"Mode: {report.mode}")
    print(f"Windows: {len(report.windows)}")
    print(f"Aggregate: {report.aggregate}")
    print(f"Passed: {report.passed_count}, Failed: {report.failed_count}")
    print(f"Verdict: {report.overall_verdict}")
    print(f"Param stability:")
    print(report.param_stability_dataframe())