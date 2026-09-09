"""Per-strategy training: baseline + Optuna tuning with criterion study.

For each of the 6 strategies:
  1. Run with current MQL5 defaults → baseline metrics
  2. Run Optuna with multi-objective criterion → best params
  3. Run with best params → tuned metrics
  4. Report improvement + criterion study
  5. Save full results

Uses 1 year EURUSD H1 by default (configurable).
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from datetime import datetime
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from data.cache import load as load_cache
from strategies import STRATEGY_REGISTRY
from backtester.engine_full import run_full, GRID_NONE
from backtester.metrics_v2 import compute_all
from backtester.trade_journal import trades_to_dataframe, export_to_csv
from analysis.optuna_optimizer import optimize_strategy, best_params
from analysis.composite_criterion import (
    composite_score, CRITERION_PRESETS,
    SHARPE_REF, CALMAR_REF, PF_REF, MDD_REF,
)


# MQL5 default parameters (matching the .mq5 source after our patches)
MQL5_DEFAULTS = {
    "ac_ao": dict(open_orders_type=1, close_orders_type=0, level_open_orders=80,
                  level_close_orders=70, use_acceleration_filter=False,
                  use_ao_synchronization=False, acceleration_bars=3,
                  min_acceleration=0.0005, min_ao_synchronization=0.0003),
    "adx": dict(open_orders_type=1, close_orders_type=4, level_open_orders_1=55,
                level_open_orders_2=15, level_close_orders_1=15,
                level_close_orders_2=5, use_di_crossover=True,
                crossover_lookback=3, min_crossover_gap=5, bars_calculate=20),
    "dem": dict(open_orders_type=3, close_orders_type=0, level_open_orders=75,
                level_close_orders=70, bars_calculate=20),
    "fbb": dict(open_orders_type_1=1, open_orders_type_2=0, close_orders_type_1=0,
                close_orders_type_2=0, level_open_orders_1=0,
                level_open_orders_2=50, level_close_orders_1=40,
                level_close_orders_2=40, bars_calculate=20, deviation=1.8),
    "mfi": dict(open_orders_type=3, close_orders_type=0, level_open_orders=70,
                level_close_orders=70, use_slope_filter=False,
                use_divergence=False, use_hidden_divergence=False,
                bars_calculate=12, slope_lookback=5, min_slope_strength=3,
                divergence_bars=10),
    "ms": dict(open_orders_type_1=8, open_orders_type_2=0, close_orders_type_1=0,
               close_orders_type_2=0, level_open_orders_1=20,
               level_open_orders_2=80, level_close_orders_1=50,
               level_close_orders_2=65, use_confluence_filter=False,
               use_macd_divergence=False, use_stoch_divergence=False,
               use_histogram_divergence=False, fast_ema_period=3,
               slow_ema_period=9, signal_period=2, k_period=5, d_period=3,
               slowing_period=12),
}


def run_one_strategy(df: pd.DataFrame, name: str, params: dict,
                     init_cash: float = 10000) -> dict:
    """Run a single strategy and return full metrics + trade summary."""
    cls = STRATEGY_REGISTRY[name]
    strat = cls(params=params)
    sig = strat.generate(df)
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    result = run_full(df, {name: (entries, direction)}, grid_mode=GRID_NONE,
                       base_lot=0.1, init_cash=init_cash)
    m = result["metrics"]
    return {
        "strategy": name,
        "params": params,
        "metrics": m,
        "n_trades": m.get("n_trades", 0),
        "win_rate": m.get("win_rate", 0),
        "max_win": m.get("largest_win", 0),
        "max_loss": m.get("largest_loss", 0),
        "max_dd": m.get("max_drawdown", 0),
        "net_pnl": m.get("net_pnl", 0),
        "sharpe": m.get("sharpe", 0),
        "calmar": m.get("calmar", 0),
        "profit_factor": m.get("profit_factor", 0),
        "trades_per_year": result.get("annual_trades", 0),
        "final_equity": m.get("final_equity", 0),
        "equity": result["equity"],
        "trades": result["trades"],
    }


def evaluate_with_criterion(metrics: dict, weights: dict) -> float:
    """Score a trial using the composite criterion with given weights."""
    return composite_score(metrics, weights)


def criterion_study(df: pd.DataFrame, name: str, params_default: dict,
                     spec: dict, criteria: dict, n_trials: int = 30) -> pd.DataFrame:
    """Run Optuna with multiple criteria in parallel; show which one finds best params."""
    print(f"\n  Criterion study for {name} ({n_trials} trials per criterion)...")
    rows = []
    for crit_name, crit_fn in criteria.items():
        study = optimize_strategy(name, df, spec, n_trials=n_trials)
        best = best_params(study, metric="sharpe")
        strat = STRATEGY_REGISTRY[name](params=best)
        sig = strat.generate(df)
        entries = sig.entries.fillna(False).astype(bool)
        direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
        result = run_full(df, {name: (entries, direction)}, grid_mode=GRID_NONE)
        m = result["metrics"]
        score = crit_fn(m)
        rows.append({
            "criterion": crit_name,
            "best_params": best,
            "score": score,
            "sharpe": m["sharpe"],
            "calmar": m["calmar"],
            "profit_factor": m["profit_factor"],
            "max_dd": m["max_drawdown"],
            "win_rate": m.get("win_rate", 0),
            "n_trades": m.get("n_trades", 0),
            "net_pnl": m.get("net_pnl", 0),
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def main():
    # Load 1Y H1 data
    df_full, _ = load_cache("EURUSD", "H1")
    end_date = df_full.index[-1]
    start_date = end_date - pd.DateOffset(years=1)
    df = df_full[df_full.index >= start_date].copy()
    print(f"Period: {df.index[0].date()} -> {df.index[-1].date()} ({len(df):,} bars)")
    print(f"Timeframe: H1\n")

    # Load spec from strategies.yaml for Optuna ranges
    with open(ROOT / "config" / "strategies.yaml") as f:
        all_specs = yaml.safe_load(f)

    criteria = {
        "Sharpe only": CRITERION_PRESETS["Sharpe"],
        "Calmar only": CRITERION_PRESETS["Calmar"],
        "Composite (Balanced)": CRITERION_PRESETS["Composite (Balanced)"],
        "Composite (Conservative)": CRITERION_PRESETS["Composite (Conservative)"],
        "Composite (Aggressive)": CRITERION_PRESETS["Composite (Aggressive)"],
    }

    print("=" * 90)
    print("STAGE 1 — Per-strategy baseline (MQL5 defaults)")
    print("=" * 90)
    baseline_results = {}
    for name in ["ac_ao", "adx", "dem", "fbb", "mfi", "ms"]:
        result = run_one_strategy(df, name, MQL5_DEFAULTS[name])
        baseline_results[name] = result
        m = result["metrics"]
        status = "PROFIT" if result["net_pnl"] > 0 else "LOSS"
        print(f"  {name:8s} | trades={result['n_trades']:>4d} | WR={result['win_rate']:.1%} | "
              f"Sharpe={result['sharpe']:>+6.2f} | PF={result['profit_factor']:>5.2f} | "
              f"MaxDD={result['max_dd']:>7.2%} | Net=${result['net_pnl']:>+8.2f} | {status}")

    print("\n" + "=" * 90)
    print("STAGE 2 — Per-strategy criterion study (Optuna with 5 criteria)")
    print("=" * 90)
    criterion_results = {}
    for name in ["ac_ao", "adx", "dem", "fbb", "mfi", "ms"]:
        spec = all_specs.get(name)
        if not spec:
            continue
        study_df = criterion_study(df, name, MQL5_DEFAULTS[name], spec, criteria, n_trials=20)
        criterion_results[name] = study_df
        best = study_df.iloc[0]
        print(f"\n  {name}:")
        print(f"    Best criterion: {best['criterion']} (score={best['score']:.2f})")
        print(f"    Tuned Sharpe: {best['sharpe']:+.2f}  PF: {best['profit_factor']:.2f}  "
              f"WR: {best['win_rate']:.1%}  MaxDD: {best['max_dd']:.2%}")
        print(f"    Best params: {dict(best['best_params'])}")

    print("\n" + "=" * 90)
    print("STAGE 3 — Per-strategy baseline vs tuned comparison")
    print("=" * 90)
    comparison = []
    for name in baseline_results:
        base = baseline_results[name]
        if name not in criterion_results:
            continue
        crit_df = criterion_results[name]
        best = crit_df.iloc[0]
        delta_sharpe = best["sharpe"] - base["sharpe"]
        delta_pnl = best["net_pnl"] - base["net_pnl"]
        print(f"\n  {name}:")
        print(f"    Baseline:  Sharpe={base['sharpe']:+.2f}  Net=${base['net_pnl']:+.2f}  WR={base['win_rate']:.1%}")
        print(f"    Tuned:     Sharpe={best['sharpe']:+.2f}  Net=${best['net_pnl']:+.2f}  WR={best['win_rate']:.1%}")
        print(f"    Delta:     Sharpe {delta_sharpe:+.2f}  Net ${delta_pnl:+.2f}")
        comparison.append({
            "strategy": name,
            "baseline_sharpe": base["sharpe"],
            "tuned_sharpe": best["sharpe"],
            "delta_sharpe": delta_sharpe,
            "baseline_pnl": base["net_pnl"],
            "tuned_pnl": best["net_pnl"],
            "delta_pnl": delta_pnl,
        })

    # Save full report
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "per_strategy_training.json"
    report = {
        "period": f"{df.index[0].date()} -> {df.index[-1].date()}",
        "n_bars": len(df),
        "baseline": {n: {k: v for k, v in r.items() if k not in ("equity", "trades")}
                       for n, r in baseline_results.items()},
        "criterion_studies": {n: df_.to_dict(orient="records")
                                  for n, df_ in criterion_results.items()},
        "comparison": comparison,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n\nFull report: {report_path}")

    # Best strategy overall
    best = max(comparison, key=lambda x: x["tuned_sharpe"])
    print(f"\n*** BEST STRATEGY: {best['strategy']} (tuned Sharpe {best['tuned_sharpe']:+.2f}) ***")

    # Per-strategy CSV exports
    for name, r in baseline_results.items():
        if len(r["trades"]) > 0:
            csv_path = output_dir / f"per_strategy_{name}_trades.csv"
            trades_j = trades_to_dataframe(r["trades"], df.index)
            trades_j.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()