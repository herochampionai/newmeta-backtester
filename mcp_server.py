"""R022: MCP server — the backtester as agent tools.

Exposes the pipeline stages as Model Context Protocol tools over stdio, so
any MCP-capable agent (ours, Claude, MT5's new assistant) can drive research:
backtest -> quality gate -> walk-forward -> sensitivity -> stress -> full chain.

Run:
    python mcp_server.py            # stdio (for MCP clients)
    python -m mcp_server            # same

Tool speed guide (for agents): backtest/data_quality are seconds on cached
data; walk_forward/sensitivity/stress take ~30-90s; full_pipeline ~1-2 min.
Clients should set generous timeouts for the slow ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("newmeta-backtester")


def _clean(obj):
    """Make numpy/pandas scalars JSON-safe."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (pd.isna(obj)):
        return 0.0
    return obj


def _parse_params(params_json) -> dict:
    if params_json is None:
        return {}
    if isinstance(params_json, dict):
        return params_json
    return json.loads(params_json)


def _load(symbol: str, timeframe: str, source: str):
    import run_pipeline as rp
    df = rp.load_data(symbol, timeframe, source=source, duka_days=90)
    if df is None or len(df) == 0:
        raise RuntimeError(f"no data for {symbol} {timeframe} (source={source})")
    return df


def _exec(df, symbol: str, slippage_pips: float, spread_pips) -> dict:
    """Same honesty rule as run_pipeline: explicit spread wins, else bar
    mean spread_pips, else 0.5p. Zero-spread backtests are fantasy."""
    if spread_pips is None:
        spread_pips = (float(df["spread_pips"].mean())
                       if "spread_pips" in df.columns else 0.5)
    pip = 0.01 if "JPY" in symbol.upper() else 0.0001
    return {"commission_pips": 0.7, "slippage_pips": float(slippage_pips),
            "spread_pips": float(spread_pips), "pip_size": pip,
            "symbol": symbol}


def _backtest_df(df, strategy: str, params: dict, capital: float, symbol: str,
                 slippage_pips: float = 0.3, spread_pips=None):
    import run_pipeline as rp
    from backtester.engine_full import run_full
    strat = rp._make_strategy(strategy, params)
    sig = strat.generate(df)
    r = run_full(df, {strategy: rp._sig_tuple(sig)},
                 init_cash=capital, **_exec(df, symbol, slippage_pips, spread_pips),
                 strict_data=False)
    return r.get("metrics", {}), r.get("trades", pd.DataFrame())


@mcp.tool()
def pbo(symbol: str = "EURUSD", timeframe: str = "H1", strategy: str = "adx",
        grids_json: str = '{"bars_calculate": [10, 14, 20]}',
        capital: float = 10000.0, source: str = "auto",
        n_partitions: int = 16, n_splits: int = 200, seed: int = 7,
        max_configs: int = 12) -> dict:
    """Overfitting probability of the SELECTION procedure (Bailey CSCV).
    grids_json: {param: [values]}. LOW = selection is trustworthy.
    Minutes (one full backtest per config)."""
    import itertools
    import random as _rnd
    import run_pipeline as rp
    from backtester.engine_full import run_full
    from analysis.pbo import probability_of_overfitting
    grids = _parse_params(grids_json)
    keys = list(grids)
    prod = list(itertools.product(*[grids[k] for k in keys]))
    if len(prod) < 3:
        return {"error": f"need >=3 configs, grid gives {len(prod)}"}
    picks = [dict(zip(keys, c)) for c in
             (prod if len(prod) <= max_configs
              else _rnd.Random(seed).sample(prod, max_configs))]
    df = _load(symbol, timeframe, source)
    ex = _exec(df, symbol, 0.3, None)

    def _eq(p):
        strat = rp._make_strategy(strategy, p)
        sig = strat.generate(df)
        return run_full(df, {strategy: rp._sig_tuple(sig)},
                        init_cash=capital, strict_data=False, **ex).get("equity")
    rep = probability_of_overfitting(df, _eq, picks,
                                     n_partitions=n_partitions,
                                     n_splits=n_splits, seed=seed, verbose=False)
    return _clean(rep.to_dict())


@mcp.tool()
def hunt(symbols: str = "auto", strategies: str = 'adx:{"bars_calculate": 14}',
         timeframe: str = "H1", min_sharpe: float = 0.3,
         tune: int = 20, max_candidates: int = 5, seed: int = 7) -> dict:
    """Full edge hunt: screen -> tune -> embargoed WF -> gate. Returns ONLY
    walk-forward survivors (ACCEPT/ROBUST). Empty survivors = no trade.
    Minutes. Strategies use | separator (JSON contains commas)."""
    import hunt as hunt_mod
    import json as _json
    from pathlib import Path as _Path
    import glob as _glob
    import os as _os
    before = set(_glob.glob("output/reports/hunt_*.json"))
    rc = hunt_mod.main(["--symbols", symbols, "--strategies", strategies,
                        "--timeframe", timeframe, "--min-sharpe", str(min_sharpe),
                        "--tune", str(tune),
                        "--max-candidates", str(max_candidates),
                        "--seed", str(seed)])
    after = [p for p in _glob.glob("output/reports/hunt_*.json") if p not in before]
    if not after:
        return {"returncode": rc, "survivors": [], "note": "no report written"}
    latest = max(after, key=_os.path.getmtime)
    return _clean({"returncode": rc, **_json.loads(_Path(latest).read_text())})


@mcp.tool()
def backtest(symbol: str = "EURUSD", timeframe: str = "H1", strategy: str = "adx",
             params_json: str = "{}", capital: float = 10000.0,
             source: str = "auto", slippage_pips: float = 0.3,
             spread_pips: float | None = None) -> dict:
    """Run a backtest. Returns net PnL, Sharpe, max drawdown, trades, win rate. Fast (seconds on cached data)."""
    """Run a backtest. Returns net PnL, Sharpe, max drawdown, trades, win rate. Fast (seconds on cached data)."""
    params = _parse_params(params_json)
    df = _load(symbol, timeframe, source)
    metrics, trades = _backtest_df(df, strategy, params, capital, symbol,
                                   slippage_pips, spread_pips)
    return _clean({
        "symbol": symbol, "timeframe": timeframe, "strategy": strategy,
        "params": params, "bars": len(df),
        "spread_pips": _exec(df, symbol, slippage_pips, spread_pips)["spread_pips"],
        "net_pnl": round(float(metrics.get("net_pnl", 0) or 0), 2),
        "sharpe": round(float(metrics.get("sharpe", 0) or 0), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0) or 0) * 100, 2),
        "trades": len(trades),
        "win_rate": round(float(metrics.get("win_rate", 0) or 0), 4),
    })


@mcp.tool()
def data_quality(symbol: str = "EURUSD", timeframe: str = "H1",
                 source: str = "auto") -> dict:
    """Grade the dataset (A-F) with gap/outlier/stale/volume/spread flags. Fast."""
    from backtester.data_quality import analyze_data_quality, gate_check
    df = _load(symbol, timeframe, source)
    dq = analyze_data_quality(df, symbol, timeframe)
    gate = gate_check(dq, min_grade="C")
    return _clean({
        "symbol": symbol, "timeframe": timeframe, "bars": len(df),
        "grade": dq.grade, "score": dq.score,
        "gate_passed_C": bool(gate["passed"]),
        "gaps": {"total": dq.gaps.total_gaps, "pct": dq.gaps.gap_pct},
        "outliers": {"count": dq.outliers.count, "pct": dq.outliers.pct},
        "stale_days": dq.stale.days_since_last,
        "volume_severity": dq.volume.severity,
        "spread_severity": dq.spread.severity if dq.spread else None,
        "flags": dq.flags,
    })


@mcp.tool()
def walk_forward(symbol: str = "EURUSD", timeframe: str = "H1", strategy: str = "adx",
                 params_json: str = "{}", train_months: int = 4, test_months: int = 1,
                 trials: int = 5, capital: float = 10000.0,
                 source: str = "auto", slippage_pips: float = 0.3,
                 spread_pips: float | None = None) -> dict:
    """Anchored walk-forward validation. SLOW (~30-60s). Returns verdict
    (ACCEPT/OVERFIT), passed/failed windows, and best OOS params."""
    from analysis.walkforward_v2 import walk_forward_v2
    params = _parse_params(params_json)
    df = _load(symbol, timeframe, source)

    def criterion_fn(p):
        m, _ = _backtest_df(df, strategy, p, capital, symbol,
                            slippage_pips, spread_pips)
        return m.get("sharpe", 0)

    from analysis.auto_iterate import bounds_from_params
    spec = bounds_from_params(params) if params else {}
    if not spec:
        return {"error": "no numeric params to vary", "verdict": "SKIPPED"}
    wf = walk_forward_v2(df=df, strategy_name=strategy, param_spec=spec,
                         criterion_fn=criterion_fn, train_months=train_months,
                         test_months=test_months, roll_months=1, n_trials=trials,
                         anchored=True, min_train_bars=500, min_test_bars=200)
    wins = list(wf.windows or [])
    best = None
    best_score = float("-inf")
    for w in wins:
        if w.passed and w.test_score is not None and w.test_score > best_score:
            best, best_score = w, w.test_score
    if best is None and wins:
        scored = [w for w in wins if w.test_score is not None]
        if scored:
            best = max(scored, key=lambda w: w.test_score)
            best_score = best.test_score
    return _clean({
        "verdict": wf.overall_verdict, "windows": len(wins),
        "passed": wf.passed_count, "failed": wf.failed_count,
        "best_params": dict(best.params) if best and best.params else {},
        "best_oos_score": best_score if best_score != float("-inf") else 0.0,
    })


@mcp.tool()
def sensitivity(symbol: str = "EURUSD", timeframe: str = "H1", strategy: str = "adx",
                params_json: str = "{}", n_samples: int = 32,
                capital: float = 10000.0, source: str = "auto",
                slippage_pips: float = 0.3,
                spread_pips: float | None = None) -> dict:
    """Sobol parameter sensitivity (total/first-order indices). SLOW (~30s+)."""
    from analysis.parameter_sensitivity import analyze_parameter_sensitivity
    params = _parse_params(params_json)
    df = _load(symbol, timeframe, source)

    def evaluator(p):
        m, _ = _backtest_df(df, strategy, p, capital, symbol,
                            slippage_pips, spread_pips)
        return m.get("net_pnl", 0)

    specs = []
    for k, v in params.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v == 0:
            continue
        if isinstance(v, int):
            low, high = max(1, int(v * 0.5)), int(v * 1.5) + 1
            specs.append({"name": k, "low": low, "high": high, "type": "int"})
        else:
            specs.append({"name": k, "low": v * 0.5, "high": v * 1.5, "type": "float"})
    if not specs:
        return {"error": "no numeric params", "ranking": []}
    sens = analyze_parameter_sensitivity(param_specs=specs, evaluator=evaluator,
                                         n_samples=n_samples)
    return _clean({
        "ranking": sens.get("importance_ranking", []),
        "sobol_S": sens.get("sobol", {}).get("S", []),
    })


@mcp.tool()
def stress(symbol: str = "EURUSD", timeframe: str = "H1", strategy: str = "adx",
           params_json: str = "{}", capital: float = 10000.0,
           source: str = "auto", slippage_pips: float = 0.3,
           spread_pips: float | None = None) -> dict:
    """Adversarial stress across 6 crisis scenarios. SLOW (~30s).
    Returns ROBUST/MARGINAL/FRAGILE verdict + worst scenario."""
    from analysis.adversarial_stress import CRISIS_SCENARIOS, run_stress_test
    params = _parse_params(params_json)
    df = _load(symbol, timeframe, source)

    def strat_fn(p):
        def fn(d):
            m, _ = _backtest_df(d, strategy, p, capital, symbol,
                                slippage_pips, spread_pips)
            return m
        return fn

    res = run_stress_test(df=df, strategy_fn=strat_fn(params))
    return _clean({
        "verdict": res.get("verdict"), "worst_scenario": res.get("worst_scenario"),
        "scenarios": list(CRISIS_SCENARIOS.keys()),
    })


@mcp.tool()
def significance(symbol: str = "EURUSD", timeframe: str = "H1", strategy: str = "adx",
                 params_json: str = "{}", capital: float = 10000.0,
                 n_trials: int = 5, source: str = "auto",
                 slippage_pips: float = 0.3,
                 spread_pips: float | None = None) -> dict:
    """Statistical significance: A/B t-test + Mann-Whitney + PSR + Deflated
    Sharpe (multiple-testing corrected). Fast."""
    from analysis.stat_tests import compare_strategies
    from analysis.statistical_significance import deflated_sharpe_ratio, probabilistic_sharpe_ratio
    params = _parse_params(params_json)
    df = _load(symbol, timeframe, source)
    metrics, trades = _backtest_df(df, strategy, params, capital, symbol,
                                   slippage_pips, spread_pips)
    pnl_col = next((c for c in ("pnl", "PnL", "profit", "Profit", "net_pnl")
                    if c in trades.columns), None)
    if pnl_col is None or len(trades) < 10:
        return {"error": f"need >=10 trades with PnL (have {len(trades)})"}
    rets = trades[pnl_col].values.astype(float)
    v = compare_strategies(rets, rets * 0.5)
    obs_sr = float(metrics.get("sharpe", 0) or 0)
    s = pd.Series(rets)
    skew = float(s.skew()) if len(s) > 2 else 0.0
    kurt = float(s.kurtosis() + 3.0) if len(s) > 3 else 3.0
    psr = probabilistic_sharpe_ratio(obs_sr, n_trials=n_trials, skewness=skew, kurtosis=kurt)
    dsr = deflated_sharpe_ratio(obs_sr, n_trials=n_trials, skewness=skew, kurtosis=kurt)
    return _clean({
        "ab_verdict": v.get("verdict"),
        "t_p": v.get("t_test", {}).get("p_value"),
        "cohens_d": v.get("t_test", {}).get("cohens_d"),
        "psr": psr.get("psr"), "psr_verdict": psr.get("verdict"),
        "dsr": dsr.get("dsr"), "dsr_verdict": dsr.get("verdict"),
        "observed_sharpe": obs_sr, "n_trades": len(trades),
    })


@mcp.tool()
def full_pipeline(symbol: str = "EURUSD", timeframe: str = "H1", strategy: str = "adx",
                  params_json: str = "{}", capital: float = 10000.0,
                  source: str = "auto", auto_iterate: int = 0,
                  tick_mode: str = "off", slippage_pips: float = 0.3,
                  spread_pips: float | None = None, leverage: float = 30.0) -> dict:
    """Run the entire 8-stage chain. VERY SLOW (1-3 min). Returns the
    consolidated report (same artifact as run_pipeline.py)."""
    import argparse

    import run_pipeline as rp
    args = argparse.Namespace(
        symbol=symbol, timeframe=timeframe, strategy=strategy,
        params=params_json if isinstance(params_json, str) else json.dumps(params_json),
        capital=capital, source=source, duka_days=90, min_grade="C", force=True,
        wf_train=4, wf_test=1, wf_anchored=True, auto_iterate=auto_iterate,
        skip_stress=False, skip_sensitivity=False, skip_significance=False,
        tick_mode=tick_mode, slippage_pips=slippage_pips, spread_pips=spread_pips,
        commission_pips=0.7, leverage=leverage,
        list_data=False)
    code = rp.run_pipeline(args)
    reports = sorted(Path("output/reports").glob("pipeline_*.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    # Exclude iterate audits; want the main report. Pick newest non-iterate file
    # if the newest is an iterate audit... simplest: newest pipeline_ file that
    # does not end with _iterate.json.
    main = [p for p in reports if not p.name.endswith("_iterate.json")]
    latest = main[0] if main else (reports[0] if reports else None)
    summary = {}
    if latest:
        summary = json.loads(latest.read_text())
    return _clean({"exit_code": code, "report_file": str(latest) if latest else None,
                   "report": summary})


@mcp.tool()
def screen_universe(strategies_json: str = '[{"name": "adx", "params": {}}]',
                    symbols_json: str | None = None, timeframe: str = "H1",
                    criterion: str = "composite", jobs: int = 1,
                    slippage_pips: float = 0.3,
                    spread_pips: float | None = None) -> dict:
    """Strategy x symbol matrix: which ticker fits which strategy. SLOW
    (~10-60s depending on universe). Returns best_per_strategy and
    best_per_symbol tables."""
    from analysis.universe_screener import discover_symbols, screen
    strategies = json.loads(strategies_json)
    symbols = json.loads(symbols_json) if symbols_json else discover_symbols(timeframe)
    rep = screen(strategies, symbols=symbols, timeframe=timeframe,
                 criterion=criterion, jobs=jobs, verbose=False,
                 exec_cfg={"slippage_pips": slippage_pips,
                           **({"spread_pips": spread_pips} if spread_pips is not None else {})})
    return _clean(rep.to_dict())


@mcp.tool()
def study_criterion(symbol: str = "EURUSD", timeframe: str = "H1", strategy: str = "adx",
                    grid_json: str = "{}", criteria_json: str | None = None,
                    capital: float = 10000.0, source: str = "auto",
                    slippage_pips: float = 0.3,
                    spread_pips: float | None = None) -> dict:
    """Which selection criterion picks OOS winners? SLOW (~30-60s). Returns
    criteria ranked by top-1 regret (lower is better) with rank correlations."""
    from analysis.criterion_study import study_criteria
    df = _load(symbol, timeframe, source)
    grid = json.loads(grid_json) if isinstance(grid_json, str) else grid_json
    if isinstance(grid, dict):
        grid = [grid]
    if not grid:
        return {"error": "empty param grid (pass a list of param dicts)"}

    def run_bt(params, d):
        m, _ = _backtest_df(d, strategy, params, capital, symbol,
                            slippage_pips, spread_pips)
        return m

    rep = study_criteria(df, run_bt, grid,
                         criteria=json.loads(criteria_json) if criteria_json else None,
                         verbose=False)
    return _clean(rep.to_dict())


@mcp.tool()
def fine_tune(symbol: str = "EURUSD", timeframe: str = "H1", strategy: str = "adx",
              space_json: str = "{}",
              n_trials: int = 20, seed: int = 7, capital: float = 10000.0,
              source: str = "auto", slippage_pips: float = 0.3,
              spread_pips: float | None = None) -> dict:
    """Optuna TPE parameter search with honest train/test reporting. SLOW
    (~20-120s). Returns best params + train/test Sharpe + overfit gap."""
    from analysis.fine_tuner import fine_tune as _tune
    df = _load(symbol, timeframe, source)
    space = json.loads(space_json)
    space = {k: tuple(v) for k, v in space.items()}

    def run_bt(params, d):
        m, t = _backtest_df(d, strategy, params, capital, symbol,
                            slippage_pips, spread_pips)
        m = dict(m)
        m["trades"] = len(t)
        return m

    rep = _tune(df, run_bt, space, n_trials=n_trials, seed=seed, verbose=False)
    return _clean(rep.to_dict())


@mcp.tool()
def permutation(symbol: str = "EURUSD", timeframe: str = "H1", strategy: str = "adx",
                params_json: str = "{}", n_permutations: int = 2000,
                capital: float = 10000.0, source: str = "auto",
                slippage_pips: float = 0.3,
                spread_pips: float | None = None) -> dict:
    """Is this Sharpe luck? Reshuffles trades, counts how often noise wins.
    Fast (seconds)."""
    from analysis.permutation_test import permutation_pvalue
    params = _parse_params(params_json)
    df = _load(symbol, timeframe, source)
    metrics, trades = _backtest_df(df, strategy, params, capital, symbol,
                                   slippage_pips, spread_pips)
    pnl_col = next((c for c in ("pnl", "PnL", "profit", "Profit", "net_pnl")
                    if c in trades.columns), None)
    if pnl_col is None or len(trades) < 10:
        return {"error": f"need >=10 trades with PnL (have {len(trades)})"}
    rep = permutation_pvalue(trades[pnl_col].values.astype(float),
                             n_permutations=n_permutations, verbose=False)
    return _clean(rep.to_dict())


@mcp.tool()
def review_gate(backtest_json: str, wf_verdict: str = "UNKNOWN",
                wf_passed: int = 0, wf_windows: int = 0,
                psr_verdict: str = "UNKNOWN", dsr_verdict: str = "UNKNOWN",
                perm_verdict: str = "UNKNOWN", pbo_verdict: str = "UNKNOWN",
                stress_verdict: str = "UNKNOWN", trades: int = 0) -> dict:
    """Independent PASS/FAIL before paper. backtest_json: {net_pnl, sharpe,
    max_drawdown}. Returns pass flag with named reasons. Instant."""
    from analysis.review_gate import review
    metrics = json.loads(backtest_json) if isinstance(backtest_json, str) else backtest_json
    return _clean(review(backtest_metrics=metrics, wf_verdict=wf_verdict,
                         wf_passed=wf_passed, wf_windows=wf_windows,
                         psr_verdict=psr_verdict, dsr_verdict=dsr_verdict,
                         perm_verdict=perm_verdict, pbo_verdict=pbo_verdict,
                         stress_verdict=stress_verdict, trades=trades))


@mcp.tool()
def competition(focus: str = "all") -> dict:
    """Scored Newmeta/LEAN/MT5 matrix + gaps. focus: all|leads|gaps|code.
    Instant — no data needed."""
    from analysis import competition_study as cs
    if focus == "leads":
        return {"our_leads": cs.our_leads(), "matrix": cs.matrix()}
    if focus == "gaps":
        return {"gaps": cs.gaps()}
    if focus == "code":
        return {"code_closable": cs.code_closable()}
    return {"matrix": cs.matrix(), "our_leads": cs.our_leads(), "gaps": cs.gaps()}


if __name__ == "__main__":
    mcp.run(transport="stdio")
