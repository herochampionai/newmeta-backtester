"""Newmeta Backtester production pipeline.

Single entrypoint that chains every analysis stage:

    data load -> quality -> backtest -> walk-forward -> sensitivity ->
    significance -> stress -> journal/tax -> observability

Usage:
    python -m run_pipeline --symbol EURUSD --timeframe H1
    python -m run_pipeline --symbol EURUSD --strategy adx --bars-calculate 10
    python -m run_pipeline --symbol XAUUSD --skip-stress  # faster
    python -m run_pipeline --list-data  # show cached datasets

Exit codes:
    0 - all stages passed
    1 - any stage failed
    2 - data quality gate failed (intentional; user can override with --force)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Allow running as `python -m run_pipeline` from project root, or `python run_pipeline.py`
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def list_cached_data():
    """Show all cached datasets."""
    cache = Path("data/cache")
    if not cache.exists():
        print("No cache directory found at data/cache")
        return
    print(f"\nCached datasets in {cache}:")
    print(f"{'file':<55} {'rows':>10} {'range':<32}")
    print("-" * 100)
    for p in sorted(cache.glob("*.parquet")):
        meta = p.with_suffix(".meta.json")
        if meta.exists():
            m = json.loads(meta.read_text())
            print(f"{p.name:<55} {m.get('rows', '?'):>10} {m.get('start', '?')[:10]} -> {m.get('end', '?')[:10]}")
        else:
            df = pd.read_parquet(p)
            print(f"{p.name:<55} {len(df):>10} (no meta)")


def find_cache(symbol: str, timeframe: str) -> Path | None:
    """Find a cached parquet file for symbol/timeframe."""
    cache = Path("data/cache")
    if not cache.exists():
        return None
    # Newer convention: SYMBOL_TF_<hash>.parquet
    matches = sorted(cache.glob(f"{symbol.upper()}_{timeframe.upper()}*.parquet"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _pip_size(symbol: str) -> float:
    s = symbol.upper()
    if "JPY" in s:
        return 0.01
    if s.startswith(("XAU", "XAG")):
        return 0.01
    return 0.0001


def _resample_rule(timeframe: str) -> str:
    return {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
            "H1": "1h", "D1": "1D"}.get(timeframe.upper(), "1h")


def load_dukascopy(symbol: str, timeframe: str, days: int = 90) -> pd.DataFrame | None:
    """R019: Dukascopy ticks -> resampled OHLCV with REAL volume + spread.

    Downloads hourly .bi5 tick files, converts to parquet, resamples mid-price
    to OHLC and sums real tick volumes. Cached in data/cache/ as
    {SYM}_{TF}_duka_{hash}.parquet so re-runs are instant.
    """
    from backtester.tick_pipeline import (
        download_dukascopy_range, convert_bi5_to_parquet, load_ticks_parquet,
    )
    import hashlib

    sym = symbol.upper().replace("/", "")
    # Dukascopy feed uses a slash for metals (XAU/USD); 6-letter FX pairs are plain.
    feed_sym = sym
    if sym.startswith(("XAU", "XAG")) and "/" not in symbol:
        feed_sym = f"{sym[:3]}/{sym[3:]}"

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days)
    start_s, end_s = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    print(f"  [data] Dukascopy {feed_sym} {timeframe}: {start_s} -> {end_s} ({days}d)")

    raw_dir = Path("data/ticks_raw") / sym
    pq_dir = Path("data/ticks_parquet")
    results = download_dukascopy_range(
        feed_sym, start_s, end_s, raw_dir, max_hours=days * 24 + 48)
    ok = sum(1 for r in results if r.ok)
    print(f"  [data] downloaded {ok}/{len(results)} hourly files")
    if ok == 0:
        print(f"  [data] Dukascopy returned nothing (bad symbol or blocked?)")
        return None

    conv = convert_bi5_to_parquet(raw_dir, pq_dir, sym, partition_by="month")
    if not conv.get("ok"):
        print(f"  [data] bi5 conversion failed: {conv.get('error')}")
        return None
    print(f"  [data] parsed {conv.get('rows', 0)} ticks")

    ticks = load_ticks_parquet(pq_dir, sym, start_s, end_s)
    if ticks is None or len(ticks) == 0:
        print(f"  [data] no ticks loaded from parquet")
        return None

    # Resample mid-price to OHLC, sum REAL volumes, mean spread
    ticks = ticks.sort_values("timestamp")
    ticks["mid"] = (ticks["bid"] + ticks["ask"]) / 2.0
    ticks = ticks.set_index("timestamp")
    rule = _resample_rule(timeframe)
    ohlc = ticks["mid"].resample(rule).ohlc()
    ohlc.columns = ["open", "high", "low", "close"]
    ohlc["volume"] = (ticks["bid_volume"] + ticks["ask_volume"]).resample(rule).sum()
    ohlc["spread_pips"] = ticks["spread"].resample(rule).mean() / _pip_size(symbol)
    df = ohlc.dropna(subset=["open"])
    if len(df) == 0:
        print(f"  [data] resample produced no bars")
        return None

    # Cache in the standard OHLCV format (+spread_pips bonus for DQ spread profile)
    cache = Path("data/cache")
    cache.mkdir(parents=True, exist_ok=True)
    content = f"{sym}_{timeframe}_duka_{len(df)}_{df.index[0]}_{df.index[-1]}".encode()
    h = hashlib.md5(content).hexdigest()[:16]
    out = cache / f"{sym}_{timeframe}_duka_{h}.parquet"
    df.to_parquet(out)
    meta = {"symbol": sym, "timeframe": timeframe, "rows": len(df),
            "start": str(df.index[0]), "end": str(df.index[-1]), "source": "dukascopy"}
    (cache / f"{sym}_{timeframe}_duka_{h}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  [data] {len(df)} {timeframe} bars with real volume, cached as {out.name}")
    return df


def load_data(symbol: str, timeframe: str, source: str = "auto",
              duka_days: int = 90) -> pd.DataFrame | None:
    """Load data: cache -> Dukascopy (primary, R019) -> Yahoo (fallback)."""
    sym = symbol.upper().replace("/", "")

    def _load_cached(exclude_duka=False):
        if exclude_duka:
            cands = sorted(
                (p for p in Path("data/cache").glob(f"{sym}_{timeframe.upper()}*.parquet")
                 if "_duka_" not in p.name),
                key=lambda p: p.stat().st_mtime, reverse=True)
            cached = cands[0] if cands else None
        else:
            cached = find_cache(sym, timeframe)
        if cached:
            print(f"  [data] Loading cached: {cached.name}")
            df = pd.read_parquet(cached)
            print(f"  [data] {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")
            return df
        return None

    if source == "yahoo":
        df = _load_cached(exclude_duka=True)
        if df is not None:
            return df
    elif source == "dukascopy":
        # Prefer a dukascopy cache file; else download fresh ticks.
        duka = sorted(Path("data/cache").glob(f"{sym}_{timeframe.upper()}_duka_*.parquet"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if duka:
            print(f"  [data] Loading cached: {duka[0].name}")
            df = pd.read_parquet(duka[0])
            print(f"  [data] {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")
            return df
        return load_dukascopy(sym, timeframe, days=duka_days)
    else:  # auto: duka cache -> any cache -> duka download (FX) -> yahoo
        duka = sorted(Path("data/cache").glob(f"{sym}_{timeframe.upper()}_duka_*.parquet"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if duka:
            print(f"  [data] Loading cached: {duka[0].name}")
            df = pd.read_parquet(duka[0])
            print(f"  [data] {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")
            return df
        df = _load_cached()
        if df is not None:
            return df
        # No cache at all: try Dukascopy first for FX spot, else Yahoo.
        if len(sym) == 6 and sym.isalpha():
            df = load_dukascopy(sym, timeframe, days=duka_days)
            if df is not None:
                return df

    # Try Yahoo fetch
    print(f"  [data] No cache for {symbol} {timeframe}, fetching from Yahoo...")
    cache = Path("data/cache")
    try:
        import yfinance as yf
        tf_map = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
                  "H1": "1h", "D1": "1d"}
        interval = tf_map.get(timeframe.upper(), "1h")
        # Yahoo limits 1m to 7 days, 1h to 730 days
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
        if interval == "1h":
            start = (pd.Timestamp.now() - pd.Timedelta(days=720)).strftime("%Y-%m-%d")
        elif interval == "1d":
            start = (pd.Timestamp.now() - pd.Timedelta(days=365*5)).strftime("%Y-%m-%d")
        else:
            start = (pd.Timestamp.now() - pd.Timedelta(days=59)).strftime("%Y-%m-%d")

        raw = yf.download(f"{symbol}=X", start=start, end=end, interval=interval, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
        df = raw[keep].dropna()

        # Cache it
        import hashlib
        cache.mkdir(parents=True, exist_ok=True)
        content = f"{symbol}_{timeframe}_{len(df)}_{df.index[0]}_{df.index[-1]}".encode()
        h = hashlib.md5(content).hexdigest()[:16]
        out = cache / f"{symbol}_{timeframe}_{h}.parquet"
        df.to_parquet(out)
        meta = {"symbol": symbol, "timeframe": timeframe, "rows": len(df),
                "start": str(df.index[0]), "end": str(df.index[-1]), "source": "yahoo"}
        (cache / f"{symbol}_{timeframe}_{h}.meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  [data] Fetched {len(df)} bars, cached as {out.name}")
        return df
    except Exception as e:
        print(f"  [data] Fetch failed: {e}")
        return None


def _make_strategy(strategy_name: str, params: dict):
    """Factory for strategy classes."""
    from strategies.adx import ADX_Strategy
    if strategy_name == "adx":
        return ADX_Strategy(name="adx", params=params)
    raise ValueError(f"Unknown strategy: {strategy_name}")


def run_pipeline(args) -> int:
    """Execute the full chain. Returns exit code."""
    start_time = time.time()

    # ---- Imports (lazy to keep --list-data fast) ----
    from backtester.data_quality import analyze_data_quality, gate_check
    from backtester.engine_full import run_full
    from backtester.trade_journal_v2 import TradeJournal
    from backtester.observability import MetricsRegistry, StructuredLogger
    from analysis.walkforward_v2 import walk_forward_v2
    from analysis.parameter_sensitivity import analyze_parameter_sensitivity
    from analysis.stat_tests import compare_strategies
    from analysis.statistical_significance import (
        probabilistic_sharpe_ratio, deflated_sharpe_ratio,
    )
    from analysis.adversarial_stress import run_stress_test
    from backtester.portfolio import MultiCurrencyPortfolio

    # ---- 0. Observability setup ----
    run_ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    registry = MetricsRegistry()
    logger = StructuredLogger(
        name="pipeline",
        log_path=f"output/logs/pipeline_{run_ts}.jsonl",
        console=False,
    )
    # Consolidated report artifact (IMP-2): every stage records here,
    # written to output/reports/pipeline_<ts>.json at the end.
    report: dict = {
        "run_id": f"pipeline_{run_ts}",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "strategy": args.strategy,
        "params": json.loads(args.params) if args.params else {},
        "capital": args.capital,
        "stages": {},
    }

    print("=" * 75)
    print(f"PIPELINE: {args.strategy} on {args.symbol}/{args.timeframe}")
    print("=" * 75)

    # ---- 1. Data load ----
    print(f"\n[1/8] Loading {args.symbol} {args.timeframe} (source={args.source})…")
    df = load_data(args.symbol, args.timeframe,
                   source=args.source, duka_days=args.duka_days)
    report["data_source_requested"] = args.source
    if df is None or len(df) < 200:
        print(f"  FAIL: insufficient data ({len(df) if df is not None else 0} bars)")
        return 1
    registry.counter("data_loaded_total", labels={"symbol": args.symbol}).inc()
    registry.gauge("data_bars").set(len(df))

    # ---- 2. Data quality ----
    print("\n[2/8] R011 Data Quality…")
    dq = analyze_data_quality(df, args.symbol, args.timeframe)
    gate = gate_check(dq, min_grade=args.min_grade)
    print(f"  Grade: {dq.grade} (gate: {'PASS' if gate['passed'] else 'FAIL'})")
    print(f"  Gaps: {dq.gaps.total_gaps} ({dq.gaps.gap_pct:.2f}%), "
          f"Outliers: {dq.outliers.count} ({dq.outliers.pct:.2f}%), "
          f"Stale days: {dq.stale.days_since_last:.0f}")
    registry.gauge("dq_grade_score").set(ord(dq.grade))
    if not gate["passed"] and not args.force:
        print(f"  Data quality gate failed. Use --force to override.")
        return 2

    # ---- 3. Backtest with user params ----
    print("\n[3/8] Backtest…")
    user_params = json.loads(args.params) if args.params else {}

    def run_bt(params):
        """Run one full backtest, return (metrics, trades)."""
        strat = _make_strategy(args.strategy, params)
        sg = strat.generate(df)
        r = run_full(df,
                     {args.strategy: (sg.entries.values.astype(int),
                                      sg.exits.values.astype(int))},
                     init_cash=args.capital, symbol=args.symbol, strict_data=False)
        return r.get("metrics", {}), r.get("trades", pd.DataFrame())

    metrics, trades = run_bt(user_params)
    print(f"  Net PnL: ${metrics.get('net_pnl', 0):.2f}")
    # max_drawdown is stored as a fraction (-0.04 = -4%); convert to percent.
    max_dd_pct = metrics.get('max_drawdown', 0) * 100
    print(f"  Sharpe: {metrics.get('sharpe', 0):.3f}, Max DD: {max_dd_pct:.2f}%")
    print(f"  Trades: {len(trades)}, Win rate: {metrics.get('win_rate', 0):.1%}")
    registry.gauge("backtest_net_pnl").set(metrics.get("net_pnl", 0))
    registry.gauge("backtest_sharpe").set(metrics.get("sharpe", 0))

    # ---- 4. Walk-Forward v2 ----
    print("\n[4/8] R004 Walk-Forward v2…")
    t0 = time.time()

    def criterion_fn(p):
        s = _make_strategy(args.strategy, p)
        sg = s.generate(df)
        return run_full(df,
                        {args.strategy: (sg.entries.values.astype(int), sg.exits.values.astype(int))},
                        init_cash=args.capital, symbol=args.symbol, strict_data=False
                        ).get("metrics", {}).get("sharpe", 0)

    # Build param spec: vary each user param ±50% in 5 steps.
    # Int params stay ints (midpoints rounded) so strategies doing int()
    # get the intended values instead of silently truncated floats.
    def _grid(v):
        if isinstance(v, int):
            low = max(1, int(v * 0.5))
            high = int(v * 1.5) + 1
            return [low, round((low + v) / 2), v, round((v + high) / 2), high]
        low = v * 0.5
        high = v * 1.5
        return [low, (low + v) / 2, v, (v + high) / 2, high]

    param_spec = {}
    for k, v in user_params.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0:
            param_spec[k] = _grid(v)

    wf_n_trials = 0
    wf = None
    if param_spec:
        wf = walk_forward_v2(
            df=df, strategy_name=args.strategy,
            param_spec=param_spec, criterion_fn=criterion_fn,
            train_months=args.wf_train, test_months=args.wf_test,
            roll_months=1, n_trials=5,
            anchored=args.wf_anchored, min_train_bars=500, min_test_bars=200,
        )
        print(f"  Windows: {len(wf.windows)}, passed: {wf.passed_count}, failed: {wf.failed_count}")
        print(f"  Verdict: {wf.overall_verdict}  ({time.time()-t0:.1f}s)")
        registry.counter("wf_verdict", labels={"verdict": wf.overall_verdict}).inc()
        wf_n_trials = len(wf.windows) * 5
        report["stages"]["walkforward"] = {
            "windows": len(wf.windows), "passed": wf.passed_count,
            "failed": wf.failed_count, "verdict": wf.overall_verdict,
        }
    else:
        print(f"  SKIPPED (no numeric params to vary)")
        report["stages"]["walkforward"] = {"skipped": "no numeric params"}

    # ---- 4b. R024 Auto-iterate: OVERFIT -> shrink bounds -> WF again ----
    if args.auto_iterate > 0 and param_spec and wf is not None \
            and wf.overall_verdict != "ACCEPT":
        from analysis.auto_iterate import auto_iterate
        print(f"\n[4b/8] R024 Auto-iterate (max {args.auto_iterate} refinement rounds)…")
        t_iter = time.time()

        def _wf_adapter(spec):
            rw = walk_forward_v2(
                df=df, strategy_name=args.strategy,
                param_spec=spec, criterion_fn=criterion_fn,
                train_months=args.wf_train, test_months=args.wf_test,
                roll_months=1, n_trials=5,
                anchored=args.wf_anchored, min_train_bars=500, min_test_bars=200,
            )
            wins = list(rw.windows or [])
            pool = [w for w in wins if w.passed] or wins

            def _score(w):
                s = w.test_score
                return s if s is not None else float("-inf")

            best_w = max(pool, key=_score) if pool else None
            return {
                "verdict": rw.overall_verdict,
                "passed": rw.passed_count, "failed": rw.failed_count,
                "n_windows": len(wins),
                "best_params": dict(best_w.params) if best_w and best_w.params else {},
                "best_oos_sharpe": _score(best_w) if best_w else 0.0,
            }

        it = auto_iterate(df, args.strategy, user_params, _wf_adapter,
                          max_rounds=args.auto_iterate, verbose=True)
        it_path = it.save(f"output/reports/{report['run_id']}_iterate.json")
        print(f"  Iterate: accepted={it.accepted}, stop={it.stop_reason}")
        print(f"  Audit: {it_path}  ({time.time()-t_iter:.1f}s)")
        wf_n_trials += sum(1 for _ in it.rounds) * (len(wf.windows) or 1) * 5
        report["stages"]["auto_iterate"] = it.to_dict()
        # Adopt ONLY on strict OOS gain vs round 0 (status-quo bias):
        # a "refined" config with equal-or-worse OOS Sharpe must not
        # silently replace the backtested params.
        r0_score = it.rounds[0].best_oos_sharpe if it.rounds else 0.0
        if it.best_params and it.best_oos_sharpe > r0_score:
            print(f"  Adopting refined params: {it.best_params} "
                  f"(OOS {r0_score:.3f} -> {it.best_oos_sharpe:.3f})")
            user_params = dict(it.best_params)
            report["params"] = dict(user_params)
            report["stages"]["auto_iterate"]["adopted"] = True
            metrics, trades = run_bt(user_params)
            print(f"  Re-run backtest: PnL=${metrics.get('net_pnl', 0):.2f}, "
                  f"Sharpe={metrics.get('sharpe', 0):.3f}, trades={len(trades)}")
        else:
            report["stages"]["auto_iterate"]["adopted"] = False
            print(f"  Keeping original params (no OOS gain: "
                  f"best {it.best_oos_sharpe:.3f} vs round-0 {r0_score:.3f})")
    else:
        report["stages"]["auto_iterate"] = {
            "skipped": ("off (--auto-iterate 0)" if args.auto_iterate <= 0
                        else ("ACCEPT already" if wf is not None else "no WF run"))
        }

    # ---- 5. Sobol sensitivity ----
    if not args.skip_sensitivity:
        print("\n[5/8] R010 Parameter Sensitivity (Sobol)…")
        t0 = time.time()
        specs = []
        for k, v in user_params.items():
            if isinstance(v, (int, float)) and v != 0:
                low = max(1, int(v * 0.5)) if isinstance(v, int) else v * 0.5
                high = int(v * 1.5) + 1 if isinstance(v, int) else v * 1.5
                specs.append({"name": k, "low": low, "high": high,
                              "type": "int" if isinstance(v, int) else "float"})
        if specs:
            sens = analyze_parameter_sensitivity(
                param_specs=specs, evaluator=criterion_fn, n_samples=32)
            # Result has 'importance_ranking' with {param, total_importance} per param.
            ranking = sens.get("importance_ranking", [])
            sobol = sens.get("sobol", {})
            print(f"  Total-order indices (Sobol ST):")
            for row in ranking:
                name = row.get("param", "?")
                ti = row.get("total_importance", 0)
                first_order = sobol.get("S", [0] * len(ranking))[ranking.index(row)] if sobol.get("S") else 0
                print(f"    {name}: ST={ti:.3f}, S={first_order:.3f}")
            print(f"  Time: {time.time()-t0:.1f}s")
            report["stages"]["sensitivity"] = {
                r.get("param"): {"ST": r.get("total_importance"),
                                 "S": sobol.get("S", [0] * len(ranking))[i]
                                 if sobol.get("S") else 0}
                for i, r in enumerate(ranking)
            }
        else:
            print(f"  SKIPPED (no numeric params)")
    else:
        print("\n[5/8] R010 Parameter Sensitivity: SKIPPED (--skip-sensitivity)")
        report["stages"]["sensitivity"] = {"skipped": "--skip-sensitivity"}

    # ---- 6. Stat significance (R012 + R003 PSR/DSR) ----
    pnl_col = next((c for c in ("pnl", "PnL", "profit", "Profit", "net_pnl")
                    if c in trades.columns), None)
    if not args.skip_significance and len(trades) >= 10 and pnl_col:
        print("\n[6/8] R012 Statistical Significance + R003 PSR/DSR…")
        returns = trades[pnl_col].values
        v = compare_strategies(returns, returns * 0.5)
        # result is nested: {t_test: {p_value, cohens_d, ...}, verdict, ...}
        t = v.get('t_test', {})
        mw = v.get('mann_whitney', {})
        print(f"  A vs A-50%: verdict={v.get('verdict', 'n/a')}")
        print(f"    t-test: t={t.get('t_statistic', 0):.3f}, "
              f"p={t.get('p_value', 0):.4f}, d={t.get('cohens_d', 0):.3f}")
        print(f"    mann-whitney: U={mw.get('u_statistic', 0):.1f}, "
              f"p={mw.get('p_value', 0):.4f}")
        # IMP-3: PSR/DSR correct the backtest Sharpe for multiple testing.
        # n_trials = configs actually evaluated (WF windows × trials), min 1.
        n_t = max(1, wf_n_trials)
        obs_sr = float(metrics.get("sharpe", 0) or 0)
        rets = pd.Series(returns, dtype=float)
        skew = float(rets.skew()) if len(rets) > 2 else 0.0
        kurt = float(rets.kurtosis() + 3.0) if len(rets) > 3 else 3.0
        psr = probabilistic_sharpe_ratio(obs_sr, n_trials=n_t, skewness=skew, kurtosis=kurt)
        dsr = deflated_sharpe_ratio(obs_sr, n_trials=n_t, skewness=skew, kurtosis=kurt)
        print(f"  PSR(SR>{0}, {n_t} trials): {psr.get('psr', 0):.3f} [{psr.get('verdict', 'n/a')}]")
        print(f"  DSR({n_t} trials): {dsr.get('dsr', 0):.3f} [{dsr.get('verdict', 'n/a')}]")
        report["stages"]["significance"] = {
            "ab_verdict": v.get("verdict"), "t_p": t.get("p_value"),
            "cohens_d": t.get("cohens_d"),
            "psr": psr.get("psr"), "psr_verdict": psr.get("verdict"),
            "dsr": dsr.get("dsr"), "dsr_verdict": dsr.get("verdict"),
            "n_trials": n_t,
        }
    else:
        reason = (f"need ≥10 trades (have {len(trades)})"
                  if len(trades) < 10 else f"no PnL column (have {list(trades.columns)[:5]})"
                  if not pnl_col else "--skip-significance")
        print(f"\n[6/8] R012 Stat Significance: SKIPPED ({reason})")
        report["stages"]["significance"] = {"skipped": reason}

    # ---- 7. Adversarial stress ----
    if not args.skip_stress:
        print("\n[7/8] R009 Adversarial Stress…")
        t0 = time.time()

        def stress_strat_fn(p):
            def fn(d):
                st = _make_strategy(args.strategy, p)
                sg = st.generate(d)
                return run_full(d,
                                {args.strategy: (sg.entries.values.astype(int),
                                                 sg.exits.values.astype(int))},
                                init_cash=args.capital, symbol=args.symbol,
                                strict_data=False).get("metrics", {})
            return fn

        stress = run_stress_test(df=df, strategy_fn=stress_strat_fn(user_params))
        print(f"  Verdict: {stress.get('verdict', 'n/a')}, "
              f"worst: {stress.get('worst_scenario', 'n/a')}  ({time.time()-t0:.1f}s)")
        report["stages"]["stress"] = {
            "verdict": stress.get("verdict"), "worst_scenario": stress.get("worst_scenario"),
        }
    else:
        print("\n[7/8] R009 Stress: SKIPPED (--skip-stress)")
        report["stages"]["stress"] = {"skipped": "--skip-stress"}

    # ---- 8. Portfolio VaR + Journal + tax + observability ----
    print("\n[8/8] R005 Portfolio VaR + R015 Journal/Tax + R014 Metrics…")
    # IMP-1: feed real trade returns into the portfolio so VaR/CVaR compute
    # (previously compute_metrics() got no returns → VaR stayed $0).
    trade_returns = None
    if pnl_col and len(trades) > 0:
        trade_returns = trades[pnl_col].values.astype(float)
        port = MultiCurrencyPortfolio(base_currency="USD", init_balance=args.capital)
        port.update_equity(float(metrics.get("net_pnl", 0) or 0))
        pm = port.compute_metrics(returns=trade_returns)
        print(f"  Portfolio equity: ${pm.equity:.2f}, VaR95: ${pm.var_95:.2f}, "
              f"VaR99: ${pm.var_99:.2f}, ES: ${pm.expected_shortfall:.2f}")
        registry.gauge("portfolio_var_95").set(pm.var_95 or 0)
        report["stages"]["portfolio"] = {
            "equity": pm.equity, "total_pnl": pm.total_pnl,
            "var_95": pm.var_95, "var_99": pm.var_99,
            "expected_shortfall": pm.expected_shortfall,
        }
    else:
        print(f"  Portfolio: SKIPPED (no trade returns)")
        report["stages"]["portfolio"] = {"skipped": "no trade returns"}

    journal = TradeJournal(run_id=report["run_id"])
    journal.log_run_meta(strategy=args.strategy, symbol=args.symbol, params=user_params,
                         dq_grade=dq.grade)
    if len(trades) > 0:
        journal.log_trades(trades, source="backtest")
        us_tax = journal.tax_report("US")
        print(f"  Net PnL: ${us_tax['totals']['gross_pnl']:.2f}, "
              f"US tax: ${us_tax['totals']['estimated_tax']:.2f}, "
              f"net after tax: ${us_tax['totals']['net_after_tax']:.2f}")
        report["stages"]["journal"] = {
            "trades_logged": len(trades),
            "gross_pnl": us_tax["totals"]["gross_pnl"],
            "us_tax": us_tax["totals"]["estimated_tax"],
            "net_after_tax": us_tax["totals"]["net_after_tax"],
        }
    else:
        report["stages"]["journal"] = {"trades_logged": 0}

    # Backtest + DQ summary for the report
    report["stages"]["data_quality"] = {
        "grade": dq.grade, "score": dq.score, "gate_passed": bool(gate["passed"]),
    }
    report["stages"]["backtest"] = {
        "net_pnl": float(metrics.get("net_pnl", 0) or 0),
        "sharpe": float(metrics.get("sharpe", 0) or 0),
        "max_drawdown": float(metrics.get("max_drawdown", 0) or 0),
        "trades": len(trades),
        "win_rate": float(metrics.get("win_rate", 0) or 0),
    }
    report["elapsed_sec"] = round(time.time() - start_time, 1)

    # IMP-2: single consolidated artifact for audit/proof trail
    reports_dir = Path("output/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{report['run_id']}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"  Report: {report_path}")

    # Finalize metrics
    registry.counter("pipeline_runs_total").inc()
    snap = registry.snapshot()
    logger.info("pipeline_complete", symbol=args.symbol, strategy=args.strategy,
                net_pnl=float(metrics.get("net_pnl", 0)),
                sharpe=float(metrics.get("sharpe", 0)),
                trades=len(trades), dq_grade=dq.grade,
                total_time_sec=time.time() - start_time)

    print(f"\n{'=' * 75}")
    print(f"PIPELINE COMPLETE in {time.time()-start_time:.1f}s")
    print(f"Metrics: {len(snap['counters'])} counters, {len(snap['gauges'])} gauges")
    print(f"Log: {logger.log_path}")
    print(f"{'=' * 75}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Newmeta Backtester pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m run_pipeline --symbol EURUSD --timeframe H1
  python -m run_pipeline --symbol XAUUSD --strategy adx --params '{"bars_calculate": 10}'
  python -m run_pipeline --list-data
  python -m run_pipeline --symbol EURUSD --skip-stress --skip-sensitivity
        """)
    parser.add_argument("--symbol", default="EURUSD", help="Trading symbol")
    parser.add_argument("--timeframe", default="H1", help="Bar timeframe (M1/M5/M15/M30/H1/D1)")
    parser.add_argument("--strategy", default="adx", help="Strategy name (default: adx)")
    parser.add_argument("--params", default="{}", help="JSON params, e.g. '{\"bars_calculate\": 10}'")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial cash")
    parser.add_argument("--source", default="auto",
                        choices=["auto", "dukascopy", "yahoo"],
                        help="Data source: auto = duka cache -> any cache -> duka download (FX) -> yahoo")
    parser.add_argument("--duka-days", type=int, default=90,
                        help="Days of tick history to pull from Dukascopy when downloading")
    parser.add_argument("--min-grade", default="C", choices=["A", "B", "C", "D", "F"],
                        help="Minimum DQ grade to pass gate (default: C)")
    parser.add_argument("--force", action="store_true", help="Override DQ gate failure")
    parser.add_argument("--wf-train", type=int, default=4, help="WF train window (months)")
    parser.add_argument("--wf-test", type=int, default=1, help="WF test window (months)")
    parser.add_argument("--wf-anchored", action="store_true", default=True, help="Anchored WF")
    parser.add_argument("--auto-iterate", type=int, default=0, metavar="N",
                        help="R024: if WF verdict != ACCEPT, run up to N refinement rounds "
                             "and adopt the best OOS params (0 = off)")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-significance", action="store_true")
    parser.add_argument("--list-data", action="store_true", help="Show cached datasets and exit")

    args = parser.parse_args()

    if args.list_data:
        list_cached_data()
        return 0

    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
