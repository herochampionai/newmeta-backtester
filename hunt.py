"""hunt.py — automated edge hunt: screen -> tune -> walk-forward -> gate.

Chains the existing stages with pipeline-identical semantics (exec honesty,
5-step WF grids, embargo, seeded RNGs) and surfaces ONLY walk-forward
survivors. A run with zero survivors is a valid outcome — it means
"no trade", not "broken hunt".

Usage:
    python hunt.py --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
        --strategies 'adx:{"bars_calculate":14}|bb_rsi:{}' \\
        --tune 20 --max-candidates 5 --seed 7
    python hunt.py --symbols auto   # discover cached symbols
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')
import pandas as pd


def _grid(v):
    """5-step grid, pipeline rule: ints stay ints."""
    if isinstance(v, int):
        low = max(1, int(v * 0.5))
        high = int(v * 1.5) + 1
        return [low, round((low + v) / 2), v, round((v + high) / 2), high]
    low, high = v * 0.5, v * 1.5
    return [low, (low + v) / 2, v, (v + high) / 2, high]


def _tune_space(params: dict) -> dict:
    """TPE space from screened params, pipeline ±50% rule."""
    space = {}
    for k, v in (params or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v == 0:
            continue
        if isinstance(v, int):
            space[k] = (max(1, int(v * 0.5)), int(v * 1.5) + 1, "int")
        else:
            space[k] = (v * 0.5, v * 1.5, "float")
    return space


def main() -> int:
    ap = argparse.ArgumentParser(description="Automated edge hunt")
    ap.add_argument("--symbols", default="auto",
                    help="'auto' = discover cached, or csv list")
    ap.add_argument("--strategies", default='adx:{"bars_calculate": 14}',
                    help='|-separated name:param_json '
                         '(use | because JSON contains commas)')
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--min-sharpe", type=float, default=0.3,
                    help="IS screen floor to become a candidate")
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--min-volume", type=float, default=1.0,
                    help="tradability filter (0 = off)")
    ap.add_argument("--tune", type=int, default=20, help="TPE trials (0 = skip)")
    ap.add_argument("--max-candidates", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--wf-embargo", type=int, default=24)
    args = ap.parse_args()

    from backtester.repro import resolve_seed, set_global_seed
    from run_pipeline import load_data, _make_strategy, _sig_tuple
    from backtester.engine_full import run_full
    from analysis.universe_screener import screen
    from analysis.fine_tuner import fine_tune
    from analysis.walkforward_v2 import walk_forward_v2
    from analysis.permutation_test import permutation_pvalue
    from analysis.review_gate import review

    seed = set_global_seed(resolve_seed(args.seed))
    t0 = time.time()

    strats = []
    for item in args.strategies.split("|"):
        name, _, pjson = item.partition(":")
        strats.append({"name": name.strip(),
                       "params": json.loads(pjson) if pjson else {}})
    symbols = None if args.symbols == "auto" else [s.strip() for s in args.symbols.split(",")]

    print("=" * 75)
    print(f"HUNT (seed={seed}): {[s['name'] for s in strats]} "
          f"on {args.symbols}/{args.timeframe}")
    print("=" * 75)

    # ---- Stage 1: screen ----
    print("\n[1/4] Screening…")
    rep = screen(strats, symbols=symbols, timeframe=args.timeframe,
                 capital=args.capital, min_avg_volume=args.min_volume,
                 verbose=False)
    if rep.rejected_symbols:
        print(f"  rejected (untradable): {rep.rejected_symbols}")
    _base = {(s["name"]): dict(s["params"]) for s in strats}
    params_by_cell = {(r["strategy"], r["symbol"]): dict(_base.get(r["strategy"], {}))
                      for r in rep.rows}
    cands = [r for r in rep.rows
             if r.get("ok") and (r.get("sharpe") or -9) >= args.min_sharpe
             and (r.get("trades") or 0) >= args.min_trades]
    cands.sort(key=lambda r: -r.get("sharpe", -9))
    cands = cands[:args.max_candidates]
    print(f"  {len(cands)} candidate(s) at IS sharpe >= {args.min_sharpe}")
    for r in cands:
        print(f"    {r['strategy']} x {r['symbol']}: "
              f"Sharpe={r.get('sharpe')} n={r.get('trades')}")

    # ---- Stages 2-4 per candidate ----
    survivors, autopsies = [], []
    for r in cands:
        name, sym = r["strategy"], r["symbol"]
        base_params = params_by_cell.get((name, sym), {})
        print(f"\n--- {name} x {sym} ---")
        df = load_data(sym, args.timeframe, source="auto")
        if df is None or len(df) < 500:
            autopsies.append({"cell": f"{name}x{sym}", "fate": "no-data"})
            continue

        def _exec(d):
            spread = (float(d["spread_pips"].mean()) if "spread_pips" in d.columns else 0.5)
            return {"commission_pips": 0.7, "slippage_pips": 0.3,
                    "spread_pips": spread, "symbol": sym, "leverage": 30.0}

        def _run(params, d):
            s = _make_strategy(name, params)
            sg = s.generate(d)
            out = run_full(d, {name: _sig_tuple(sg)}, init_cash=args.capital,
                           strict_data=False, **_exec(d))
            m = out.get("metrics", {})
            t = out.get("trades", pd.DataFrame())
            # NOTE: "trades" is the INT count — fine_tune int()s this field.
            # The frame rides along separately for the gate stage.
            return {"sharpe": m.get("sharpe", 0),
                    "max_drawdown": m.get("max_drawdown", 0),
                    "net_pnl": m.get("net_pnl", 0),
                    "trades": len(t), "frame": t}

        # Stage 2: tune (TPE train/holdout, guardrails)
        params = dict(base_params)
        if args.tune > 0:
            space = _tune_space(base_params)
            if space:
                tr = fine_tune(df, _run, space,
                               n_trials=args.tune, min_trades=args.min_trades,
                               seed=seed, verbose=False)
                if tr.best_params:
                    print(f"  tune: {tr.best_params} "
                          f"(train {tr.best_value})")
                    params = dict(tr.best_params)
                else:
                    print("  tune: no winner passed guardrails — kept screen params")

        # Stage 3: embargoed walk-forward around the tuned point
        spec = {k: _grid(v) for k, v in params.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0}
        if not spec:
            autopsies.append({"cell": f"{name}x{sym}", "fate": "no-numeric-params"})
            continue
        wf = walk_forward_v2(
            df=df, strategy_name=name, param_spec=spec,
            criterion_fn=lambda p: _run(p, df)["sharpe"],
            train_months=4, test_months=1, roll_months=1, n_trials=3,
            anchored=True, min_train_bars=500, min_test_bars=200,
            embargo_bars=args.wf_embargo, seed=seed)
        print(f"  WF: {len(wf.windows)} windows, passed {wf.passed_count} "
              f"-> {wf.overall_verdict}")

        # Stage 4: gate (IS metrics + WF + bootstrap leg)
        isr = _run(params, df)
        pnl_col = next((c for c in ("PnL", "pnl", "profit", "net_pnl")
                        if c in isr["frame"].columns), None)
        perm_verdict = "UNKNOWN"
        if pnl_col is not None and isr["trades"] >= 10:
            perm_verdict = permutation_pvalue(
                isr["frame"][pnl_col].values.astype(float),
                n_permutations=2000, seed=seed, verbose=False).verdict()
        gate = review(
            backtest_metrics={"net_pnl": isr["net_pnl"], "sharpe": isr["sharpe"],
                              "max_drawdown": isr["max_drawdown"]},
            wf_verdict=wf.overall_verdict, wf_passed=wf.passed_count,
            wf_windows=len(wf.windows), psr_verdict="UNKNOWN",
            dsr_verdict="UNKNOWN", perm_verdict=perm_verdict,
            stress_verdict="UNKNOWN", trades=isr["trades"])
        ok = gate["pass"] and wf.overall_verdict in ("ACCEPT", "ROBUST")
        print(f"  gate: {'SURVIVOR' if ok else 'reject'} "
              f"({gate['summary']}; boot={perm_verdict})")
        entry = {"cell": f"{name}x{sym}", "params": params,
                 "is_sharpe": round(isr["sharpe"], 3), "is_trades": isr["trades"],
                 "wf": {"windows": len(wf.windows), "passed": wf.passed_count,
                        "verdict": wf.overall_verdict},
                 "perm_verdict": perm_verdict, "gate": gate["summary"]}
        (survivors if ok else autopsies).append(entry)
        if not ok:
            entry["fate"] = "gate-reject"
            entry["reasons"] = gate["reasons"]

    report = {"hunt": {"seed": seed, "symbols": args.symbols,
                       "strategies": [s["name"] for s in strats],
                       "elapsed_sec": round(time.time() - t0, 1)},
              "survivors": survivors, "autopsies": autopsies}
    out = Path("output/reports") / f"hunt_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print("\n" + "=" * 75)
    print(f"HUNT COMPLETE in {report['hunt']['elapsed_sec']}s: "
          f"{len(survivors)} survivor(s), {len(autopsies)} reject(s)")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
