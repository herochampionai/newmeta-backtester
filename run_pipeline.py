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


def load_data(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Load data from cache, fetching from Yahoo if missing."""
    cached = find_cache(symbol, timeframe)
    if cached:
        print(f"  [data] Loading cached: {cached.name}")
        df = pd.read_parquet(cached)
        print(f"  [data] {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")
        return df

    # Try Yahoo fetch
    print(f"  [data] No cache for {symbol} {timeframe}, fetching from Yahoo...")
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
    from analysis.adversarial_stress import run_stress_test

    # ---- 0. Observability setup ----
    registry = MetricsRegistry()
    logger = StructuredLogger(
        name="pipeline",
        log_path=f"output/logs/pipeline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        console=False,
    )

    print("=" * 75)
    print(f"PIPELINE: {args.strategy} on {args.symbol}/{args.timeframe}")
    print("=" * 75)

    # ---- 1. Data load ----
    print(f"\n[1/8] Loading {args.symbol} {args.timeframe}…")
    df = load_data(args.symbol, args.timeframe)
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
    strategy = _make_strategy(args.strategy, user_params)
    sig = strategy.generate(df)
    signals = {args.strategy: (sig.entries.values.astype(int), sig.exits.values.astype(int))}
    bt_result = run_full(df, signals, init_cash=args.capital, symbol=args.symbol, strict_data=False)
    metrics = bt_result.get("metrics", {})
    trades = bt_result.get("trades", pd.DataFrame())
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

    # Build param spec: vary each user param ±50% in 3 steps
    param_spec = {}
    for k, v in user_params.items():
        if isinstance(v, (int, float)) and v != 0:
            low = max(1, int(v * 0.5)) if isinstance(v, int) else v * 0.5
            high = int(v * 1.5) + 1 if isinstance(v, int) else v * 1.5
            param_spec[k] = [low, (low + v) / 2, v, (v + high) / 2, high]

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
    else:
        print(f"  SKIPPED (no numeric params to vary)")

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
        else:
            print(f"  SKIPPED (no numeric params)")
    else:
        print("\n[5/8] R010 Parameter Sensitivity: SKIPPED (--skip-sensitivity)")

    # ---- 6. Stat significance ----
    pnl_col = next((c for c in ("pnl", "PnL", "profit", "Profit", "net_pnl")
                    if c in trades.columns), None)
    if not args.skip_significance and len(trades) >= 10 and pnl_col:
        print("\n[6/8] R012 Statistical Significance…")
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
    else:
        reason = (f"need ≥10 trades (have {len(trades)})"
                  if len(trades) < 10 else f"no PnL column (have {list(trades.columns)[:5]})"
                  if not pnl_col else "--skip-significance")
        print(f"\n[6/8] R012 Stat Significance: SKIPPED ({reason})")

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
    else:
        print("\n[7/8] R009 Stress: SKIPPED (--skip-stress)")

    # ---- 8. Journal + tax + observability ----
    print("\n[8/8] R015 Journal + Tax + R014 Metrics…")
    journal = TradeJournal(run_id=f"pipeline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    journal.log_run_meta(strategy=args.strategy, symbol=args.symbol, params=user_params,
                         dq_grade=dq.grade)
    if len(trades) > 0:
        journal.log_trades(trades, source="backtest")
        us_tax = journal.tax_report("US")
        print(f"  Net PnL: ${us_tax['totals']['gross_pnl']:.2f}, "
              f"US tax: ${us_tax['totals']['estimated_tax']:.2f}, "
              f"net after tax: ${us_tax['totals']['net_after_tax']:.2f}")

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
    parser.add_argument("--min-grade", default="C", choices=["A", "B", "C", "D", "F"],
                        help="Minimum DQ grade to pass gate (default: C)")
    parser.add_argument("--force", action="store_true", help="Override DQ gate failure")
    parser.add_argument("--wf-train", type=int, default=4, help="WF train window (months)")
    parser.add_argument("--wf-test", type=int, default=1, help="WF test window (months)")
    parser.add_argument("--wf-anchored", action="store_true", default=True, help="Anchored WF")
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
