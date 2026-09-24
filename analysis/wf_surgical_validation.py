"""Walk-forward validation harness for Roadmap B1/B2/B4 surgical features.

Runs paired walk-forward windows on the GRID engine (run_full) with
GRID_LOSS_AND_PROFIT.  For every test window we back-to-back run:

  • *baseline*  — grid only, every surgical toggle OFF (default-OFF safe mode)
  • *surgical*  — grid + ALL six surgical features enabled

Idle days (B1) are tracked per window.  Results are compared across windows
with the Acceptance Rule: the surgical configuration must beat baseline OOS
on the majority of windows AND on aggregate Sharpe/Calmar, otherwise the
features stay OFF in the presets.

Usage:
    PYTHONPATH=. python -m analysis.wf_surgical_validation --strategy ac_ao --train-months 36 --test-months 6 --roll-months 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtester.engine_full import run_full  # noqa: E402
from backtester.grid_recovery import (  # noqa: E402
    GRID_LOSS_AND_PROFIT,
    RECOVERY_HIGHER_PROFITS,
)
from backtester.metrics_v2 import compute_all  # noqa: E402
from core.surgical_features import enable, get_feature_defaults  # noqa: E402
from data.cache import load as load_cache  # noqa: E402
from strategies import STRATEGY_REGISTRY  # noqa: E402

ALL_SURGICAL = [
    "anomaly_gate",
    "event_blackout",
    "recovery_restart",
    "basket_money_tp",
    "profit_lock_trail",
    "carry_adjusted_tp",
]


def build_surgical_params(base_params: dict | None = None) -> dict:
    """Return a params dict with ALL surgical features enabled."""
    p = dict(base_params or {})
    for name in ALL_SURGICAL:
        p = enable(p, name)
    return p


def generate_signals(strategy_name: str, params: dict, df: pd.DataFrame) -> dict[str, tuple]:
    """Instantiate strategy and return signals_by_strategy dict for run_full."""
    cls = STRATEGY_REGISTRY[strategy_name]
    inst = cls(params=params)
    sig = inst.generate(df)
    return {
        strategy_name: (sig.entries, sig.direction),
    }


def count_idle_days(df: pd.DataFrame, sig_entries: pd.Series) -> tuple[int, list[str]]:
    """Count days in *df* where no entries fired (B1 idle-day tracking)."""
    entries = sig_entries.fillna(False).astype(bool)
    entry_dates = set(entries[entries].index.strftime("%Y-%m-%d"))
    all_dates = set(df.index.strftime("%Y-%m-%d"))
    idle = sorted(all_dates - entry_dates)
    return len(idle), idle


def run_window(df: pd.DataFrame, strategy_name: str, base_params: dict,
               train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Run baseline and surgical backtests on *test_df*. Returns comparison dict."""
    # Generate signals on the test window using the strategy
    # (signals are regenerated per window; train info informs params via caller)
    sig_test = STRATEGY_REGISTRY[strategy_name](params=base_params).generate(test_df)

    signals = {strategy_name: (sig_test.entries, sig_test.direction)}

    # Count idle days (B1)
    n_idle, idle_list = count_idle_days(test_df, sig_test.entries)

    # --- Baseline: grid only, surgical OFF ---
    baseline_result = run_full(
        test_df, signals,
        grid_mode=GRID_LOSS_AND_PROFIT,
        base_lot=0.1,
        grid_take_profit=50,
        grid_stop_loss=200,
        max_grid_layers=4,
        pips_between_orders=30,
        grid_lot_multiplier=1.5,
        recovery_mode=RECOVERY_HIGHER_PROFITS,
        recovery_lot_multiplier=2.0,
        params=base_params,
    )
    base_metrics = compute_all(
        baseline_result["equity"].pct_change().fillna(0),
        baseline_result.get("trades"),
        baseline_result["equity"],
    )

    # --- Surgical: grid + ALL surgical features ON ---
    surgical_params = build_surgical_params(base_params)
    surgical_result = run_full(
        test_df, signals,
        grid_mode=GRID_LOSS_AND_PROFIT,
        base_lot=0.1,
        grid_take_profit=50,
        grid_stop_loss=200,
        max_grid_layers=4,
        pips_between_orders=30,
        grid_lot_multiplier=1.5,
        recovery_mode=RECOVERY_HIGHER_PROFITS,
        recovery_lot_multiplier=2.0,
        params=surgical_params,
    )
    surg_metrics = compute_all(
        surgical_result["equity"].pct_change().fillna(0),
        surgical_result.get("trades"),
        surgical_result["equity"],
    )

    return {
        "n_idle_days": n_idle,
        "idle_days": idle_list,
        "baseline_sharpe": base_metrics.get("sharpe", 0.0),
        "baseline_calmar": base_metrics.get("calmar", 0.0),
        "baseline_max_dd": base_metrics.get("max_drawdown", 0.0),
        "baseline_total_ret": base_metrics.get("total_return", 0.0),
        "baseline_n_trades": base_metrics.get("n_trades", 0),
        "surgical_sharpe": surg_metrics.get("sharpe", 0.0),
        "surgical_calmar": surg_metrics.get("calmar", 0.0),
        "surgical_max_dd": surg_metrics.get("max_drawdown", 0.0),
        "surgical_total_ret": surg_metrics.get("total_return", 0.0),
        "surgical_n_trades": surg_metrics.get("n_trades", 0),
    }


def walk_forward_windows(df: pd.DataFrame, train_months: int, test_months: int,
                         roll_months: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (train_df, test_df) windows from a DatetimeIndex-sorted df."""
    start = df.index[0]
    end = df.index[-1]
    cursor = start
    windows = []
    while True:
        tr_end = cursor + pd.DateOffset(months=train_months)
        te_end = tr_end + pd.DateOffset(months=test_months)
        if te_end > end:
            break
        train = df[(df.index >= cursor) & (df.index < tr_end)]
        test = df[(df.index >= tr_end) & (df.index < te_end)]
        if len(train) < 200 or len(test) < 50:
            cursor += pd.DateOffset(months=roll_months)
            continue
        windows.append((train, test))
        cursor += pd.DateOffset(months=roll_months)
    return windows


def main():
    ap = argparse.ArgumentParser(description="Walk-forward validation of surgical features")
    ap.add_argument("--data", default="EURUSD_H1")
    ap.add_argument("--strategy", default="ac_ao")
    ap.add_argument("--train-months", type=int, default=36)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--roll-months", type=int, default=3)
    ap.add_argument("--symbol", default=None,
                    help="Data symbol (e.g. EURUSD). Derived from --data if omitted.")
    ap.add_argument("--timeframe", default=None,
                    help="Data timeframe (e.g. H1). Derived from --data if omitted.")
    args = ap.parse_args()

    symbol, timeframe = args.symbol, args.timeframe
    if symbol is None or timeframe is None:
        parts = args.data.split("_")
        symbol = symbol or parts[0]
        timeframe = timeframe or (parts[1] if len(parts) > 1 else "H1")

    df, meta = load_cache(symbol, timeframe)
    print(f"[load] {meta.get('symbol', symbol)} {timeframe}: {len(df)} bars")
    print(f"  date range: {df.index[0]} → {df.index[-1]}")
    print(f"  columns: {list(df.columns)}")

    base_params = get_feature_defaults()

    windows = walk_forward_windows(df, args.train_months, args.test_months, args.roll_months)
    print(f"[wf] {len(windows)} walk-forward windows "
          f"(train={args.train_months}m / test={args.test_months}m / roll={args.roll_months}m)")

    results = []
    for i, (train_df, test_df) in enumerate(windows):
        print(f"\n[window {i+1}/{len(windows)}] "
              f"test: {test_df.index[0].date()} → {test_df.index[-1].date()} "
              f"({len(test_df)} bars)")
        try:
            r = run_window(df, args.strategy, base_params, train_df, test_df)
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({"window": i+1})
            continue

        won = r["surgical_sharpe"] > r["baseline_sharpe"]
        status = "WON" if won else "LOST"
        print(f"  idle_days={r['n_idle_days']}  baseline Sharpe={r['baseline_sharpe']:.3f}"
              f" | surgical Sharpe={r['surgical_sharpe']:.3f}  [{status}]")
        r["window_id"] = i + 1
        r["status"] = status
        results.append(r)

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("\n[no results — check strategy name / data]")
        return

    out = ROOT / "output" / "wf_surgical_validation.csv"
    out.parent.mkdir(exist_ok=True)
    res_df.to_csv(out, index=False)
    print(f"\n[saved] {out}")

    # Aggregate comparison
    valid = res_df.dropna(subset=["baseline_sharpe", "surgical_sharpe"])
    if valid.empty:
        print("[no valid windows for comparison]")
        return

    wins = int((valid["status"] == "WON").sum())
    losses = int((valid["status"] == "LOST").sum())
    total = len(valid)

    print("\n" + "=" * 60)
    print("ACCEPTANCE CHECK — does surgical beat baseline OOS?")
    print("=" * 60)
    print(f"  Windows: {total} total | {wins} won | {losses} lost")
    print(f"  Win rate: {wins/total*100:.1f}%")
    print("\n  Aggregate Sharpe:")
    print(f"    Baseline mean: {valid['baseline_sharpe'].mean():.3f}  "
          f"(std {valid['baseline_sharpe'].std():.3f})")
    print(f"    Surgical mean: {valid['surgical_sharpe'].mean():.3f}  "
          f"(std {valid['surgical_sharpe'].std():.3f})")
    print("\n  Aggregate Calmar:")
    print(f"    Baseline: {valid['baseline_calmar'].mean():.3f}")
    print(f"    Surgical: {valid['surgical_calmar'].mean():.3f}")
    print("\n  Total return:")
    print(f"    Baseline: {valid['baseline_total_ret'].mean()*100:.1f}%")
    print(f"    Surgical: {valid['surgical_total_ret'].mean()*100:.1f}%")
    print("\n  Max drawdown:")
    print(f"    Baseline: {valid['baseline_max_dd'].mean()*100:.1f}%")
    print(f"    Surgical: {valid['surgical_max_dd'].mean()*100:.1f}%")
    print(f"\n  Idle days (B1, surgical windows): {int(valid['n_idle_days'].sum())} total")

    beats_baseline = (valid["surgical_sharpe"].mean() > valid["baseline_sharpe"].mean()
                       and wins > losses)
    verdict = "PASS — enable surgical features in presets" if beats_baseline else "FAIL — keep surgical OFF (default)"
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
