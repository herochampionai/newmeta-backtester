"""Portfolio-level walk-forward analysis.

Composes per-strategy walk-forward results into a multi-strategy portfolio
with risk-parity or equal-weight allocation. Uses the existing
``walk_forward()`` per-strategy engine for optimization, then aggregates
per-strategy test-window returns via ``run_portfolio()``.

Reports aggregate metrics, per-strategy contribution, and statistical
significance of OOS Sharpe via bootstrap.

Usage:
    PYTHONPATH=. python -m analysis.portfolio_wf \\
        --strategies ac_ao,adx,dem,fbb,bb_rsi,ms,macd_confluence \\
        --symbols EURUSD,GBPUSD \\
        --timeframe H1 --train-months 36 --test-months 6 --roll-months 3 \\
        --allocation risk_parity --n_trials 20

    PYTHONPATH=. python -m analysis.portfolio_wf \\
        --strategies crypto_9_breakout,crypto_9_reversal,crypto_9_bounce \\
        --symbols BTCUSDT,ETHUSDT \\
        --timeframe H1 --allocation equal
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.data_loader import load_asset
from analysis.walkforward import walk_forward
from analysis.wf_parallel import (
    _resolve_strategy_name, _default_params_for_strategy,
    list_available_strategies,
)
from backtester.engine import run_direction, run_portfolio
from backtester.metrics_v2 import compute_all


@dataclass
class PortfolioWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: dict
    test_metrics: dict
    strategy_weights: dict
    per_strategy_test_metrics: dict
    idle_day_count: int = 0


def _risk_parity_weights(train_returns_by_strategy: dict[str, pd.Series]) -> dict[str, float]:
    """Compute risk-parity weights: w_i = (1/σ_i) / Σ(1/σ_j)."""
    vols = {}
    for name, rets in train_returns_by_strategy.items():
        r = rets.dropna()
        v = float(r.std()) if len(r) > 1 else 0.0
        vols[name] = v if v > 0 else 1.0
    inv_vol = {k: 1.0 / v for k, v in vols.items()}
    total = sum(inv_vol.values())
    return {k: v / total for k, v in inv_vol.items()}


def _equal_weights(names: list[str]) -> dict[str, float]:
    w = 1.0 / len(names)
    return {n: w for n in names}


def _bootstrap_sharpe(returns: pd.Series, n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """Bootstrap Sharpe ratio significance test.

    Returns {sharpe, p_value, ci_lower, ci_upper}.
    Tests H0: Sharpe <= 0.
    """
    r = returns.dropna().values
    if len(r) < 10:
        return {"sharpe": 0.0, "p_value": 1.0, "ci_lower": 0.0, "ci_upper": 0.0}

    rng = np.random.default_rng(seed)
    std_obs = np.std(r, ddof=1)
    sharpe_observed = float(np.mean(r) / (std_obs / np.sqrt(len(r)))) if std_obs > 0 else 0.0

    boot_sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(r, size=len(r), replace=True)
        s_std = np.std(sample, ddof=1)
        s = float(np.mean(sample) / (s_std / np.sqrt(len(sample)))) if s_std > 0 else 0.0
        boot_sharpes.append(s)

    boot_sharpes = np.array(boot_sharpes)
    if sharpe_observed > 0:
        p_value = float(np.mean(boot_sharpes <= 0))
    else:
        p_value = float(np.mean(boot_sharpes >= 0))
    ci_lower = float(np.percentile(boot_sharpes, 2.5))
    ci_upper = float(np.percentile(boot_sharpes, 97.5))

    return {"sharpe": sharpe_observed, "p_value": p_value, "ci_lower": ci_lower, "ci_upper": ci_upper}


def portfolio_walk_forward(
    strategies: list[str],
    df: pd.DataFrame,
    timeframe: str,
    train_months: int = 36,
    test_months: int = 6,
    roll_months: int = 3,
    n_trials: int = 20,
    allocation_method: str = "risk_parity",
) -> list[PortfolioWindow]:
    """Run portfolio-level walk-forward across all strategies on one symbol."""
    from strategies import STRATEGY_REGISTRY
    from analysis.optuna_optimizer import _sample
    import optuna

    results: list[PortfolioWindow] = []
    start = df.index[0]
    end = df.index[-1]
    cursor = start
    periods_per_year = 252 * 24 if "H1" in timeframe.upper() else 252

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

        strategy_returns_train: dict[str, pd.Series] = {}
        strategy_returns_test: dict[str, pd.Series] = {}
        strategy_params: dict[str, dict] = {}
        per_strategy_test: dict[str, dict] = {}
        train_metrics_all: dict[str, dict] = {}

        for strat_label in strategies:
            try:
                strat_name = _resolve_strategy_name(strat_label)
                cls = STRATEGY_REGISTRY[strat_name]
                params = _default_params_for_strategy(strat_label)

                # Optimize on train window via Optuna
                study = optuna.create_study(direction="maximize",
                                           sampler=optuna.samplers.TPESampler(seed=42))

                def obj(trial):
                    p = _sample(trial, params)
                    try:
                        sig = cls(params=p).generate(train)
                        _, m = run_direction(train, sig.entries, sig.direction)
                        return m.get("sharpe", -10)
                    except Exception:
                        return -10

                study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
                best = study.best_params
                strategy_params[strat_label] = best

                # Train
                sig_tr = cls(params=best).generate(train)
                _, train_metrics = run_direction(train, sig_tr.entries, sig_tr.direction)
                train_metrics_all[strat_label] = train_metrics
                # Bar-level returns on train (for allocation volatility)
                train_ret = train["close"].pct_change().fillna(0)
                strategy_returns_train[strat_label] = sig_tr.direction * train_ret

                # Test
                sig_te = cls(params=best).generate(test)
                _, test_metrics = run_direction(test, sig_te.entries, sig_te.direction)
                test_ret = test["close"].pct_change().fillna(0)
                strategy_returns_test[strat_label] = sig_te.direction * test_ret
                per_strategy_test[strat_label] = test_metrics

            except Exception as e:
                print(f"  [SKIP] {strat_label}: {e}")
                traceback.print_exc()
                continue

        if not strategy_returns_train:
            cursor += pd.DateOffset(months=roll_months)
            continue

        # Portfolio allocation
        if allocation_method == "risk_parity":
            weights = _risk_parity_weights(strategy_returns_train)
        else:
            weights = _equal_weights(list(strategy_returns_train.keys()))

        # Build returns matrix for run_portfolio
        ret_matrix = pd.DataFrame({s: strategy_returns_test[s] for s in strategy_returns_test}).fillna(0)
        weight_vec = np.array([weights.get(c, 0.0) for c in ret_matrix.columns])

        # Portfolio metrics
        port_summary = run_portfolio(ret_matrix, weight_vec, init_cash=10_000)
        port_equity = (1 + (ret_matrix.values * weight_vec).sum(axis=1)).cumprod() * 10_000
        port_equity = pd.Series(port_equity, index=ret_matrix.index)

        # Bootstrap significance test
        port_returns = (ret_matrix.values * weight_vec).sum(axis=1)
        port_returns = pd.Series(port_returns, index=ret_matrix.index).dropna()

        # Idle day tracking
        test_entries_any = pd.Series(False, index=test.index)
        for strat_label in strategy_returns_test:
            strat_name_resolved = _resolve_strategy_name(strat_label)
            cls = STRATEGY_REGISTRY[strat_name_resolved]
            best = strategy_params.get(strat_label, {})
            sig_te = cls(params=best).generate(test)
            test_entries_any |= sig_te.entries.fillna(False).astype(bool)

        entry_dates = set(test_entries_any[test_entries_any].index.strftime("%Y-%m-%d"))
        all_dates = set(test.index.strftime("%Y-%m-%d"))
        idle_days = sorted(all_dates - entry_dates)

        # Use compute_all for richer metrics
        portfolio_metrics = compute_all(
            port_returns, None, port_equity, periods_per_year=periods_per_year
        )

        avg_train_sharpe = float(np.mean([
            train_metrics_all[s].get("sharpe", 0) for s in train_metrics_all
        ])) if train_metrics_all else 0.0

        results.append(PortfolioWindow(
            train_start=cursor, train_end=tr_end,
            test_start=tr_end, test_end=te_end,
            train_metrics={"avg_train_sharpe": avg_train_sharpe},
            test_metrics=portfolio_metrics,
            strategy_weights=weights,
            per_strategy_test_metrics=per_strategy_test,
            idle_day_count=len(idle_days),
        ))

        cursor += pd.DateOffset(months=roll_months)

    return results


def main():
    ap = argparse.ArgumentParser(description="Portfolio-level walk-forward analysis")
    ap.add_argument("--strategies", default=None,
                    help="Comma-separated strategy labels (required). E.g. ac_ao,adx,fbb OR crypto_9_breakout,crypto_9_reversal")
    ap.add_argument("--symbols", default="EURUSD,BTCUSDT,ETHUSDT,SOLUSDT",
                    help="Comma-separated symbols")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--train-months", type=int, default=36)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--roll-months", type=int, default=3)
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--allocation", default="risk_parity",
                    choices=["risk_parity", "equal"],
                    help="Portfolio allocation method")
    ap.add_argument("--list-strategies", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=4)
    args = ap.parse_args()

    if args.list_strategies:
        for n in list_available_strategies():
            print(f"  {n}")
        return 0

    if not args.strategies:
        print("ERROR: --strategies is required")
        print(f"Available: {', '.join(list_available_strategies())}")
        return 1

    strategy_list = [s.strip() for s in args.strategies.split(",")]
    symbols = [s.strip() for s in args.symbols.split(",")]

    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    all_rows: list[dict] = []

    for symbol in symbols:
        print(f"\n[portfolio_wf] Symbol: {symbol} — strategies: {', '.join(strategy_list)}")
        try:
            df = load_asset(symbol, args.timeframe, args.start, args.end)
            if df is None or len(df) < 200:
                print(f"  [SKIP] {symbol}: insufficient data ({len(df) if df is not None else 0} bars)")
                continue
        except Exception as e:
            print(f"  [SKIP] {symbol}: {e}")
            continue

        try:
            windows = portfolio_walk_forward(
                strategy_list, df, args.timeframe,
                train_months=args.train_months, test_months=args.test_months,
                roll_months=args.roll_months, n_trials=args.n_trials,
                allocation_method=args.allocation,
            )
        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            traceback.print_exc()
            continue

        for w in windows:
            all_rows.append({
                "strategy": f"portfolio_{args.allocation}",
                "symbol": symbol,
                "timeframe": args.timeframe,
                "train_start": str(w.train_start.date()),
                "train_end": str(w.train_end.date()),
                "test_start": str(w.test_start.date()),
                "test_end": str(w.test_end.date()),
                "test_sharpe": w.test_metrics.get("sharpe", 0),
                "test_calmar": w.test_metrics.get("calmar", 0),
                "test_max_dd": w.test_metrics.get("max_drawdown", 0),
                "test_total_return": w.test_metrics.get("total_return", 0),
                "test_n_trades": w.test_metrics.get("n_trades", 0),
                "test_win_rate": w.test_metrics.get("win_rate", 0),
                "test_final_equity": w.test_metrics.get("final_equity", 0),
                "test_sortino": w.test_metrics.get("sortino", 0),
                "test_recovery_factor": w.test_metrics.get("recovery_factor", 0),
                "test_stability": w.test_metrics.get("stability", 0),
                "train_avg_sharpe": w.train_metrics.get("avg_train_sharpe", 0),
                "weights": json.dumps(w.strategy_weights),
                "per_strategy": json.dumps(w.per_strategy_test_metrics, default=str),
                "idle_days": w.idle_day_count,
            })

    if not all_rows:
        print("[portfolio_wf] No results — check data availability and strategy specs")
        return 1

    df_results = pd.DataFrame(all_rows)
    csv_path = output_dir / f"portfolio_wf_results_{args.allocation}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[portfolio_wf] Saved {len(df_results)} rows → {csv_path}")

    # Per-symbol summary
    summary = df_results.groupby("symbol").agg(
        mean_oos_sharpe=("test_sharpe", "mean"),
        mean_oos_calmar=("test_calmar", "mean"),
        mean_max_dd=("test_max_dd", "mean"),
        mean_total_return=("test_total_return", "mean"),
        mean_sortino=("test_sortino", "mean"),
        total_windows=("test_sharpe", "count"),
    ).sort_values("mean_oos_sharpe", ascending=False)

    print("\n=== Portfolio per-symbol OOS summary ===")
    print(summary.to_string())

    summary_path = output_dir / f"portfolio_wf_summary_{args.allocation}.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved summary → {summary_path}")

    # Overall portfolio summary
    overall = df_results.agg(
        mean_oos_sharpe=("test_sharpe", "mean"),
        mean_oos_calmar=("test_calmar", "mean"),
        mean_max_dd=("test_max_dd", "mean"),
        mean_total_return=("test_total_return", "mean"),
        total_windows=("test_sharpe", "count"),
    )
    print("\n=== Portfolio overall OOS summary ===")
    print(overall.to_string())

    overall_path = output_dir / f"portfolio_wf_overall_{args.allocation}.csv"
    overall.to_csv(overall_path, header=True)
    print(f"\nSaved overall → {overall_path}")

    # Significance test across all windows
    all_sharpes = df_results["test_sharpe"].values
    if len(all_sharpes) >= 5:
        boot = _bootstrap_sharpe(pd.Series(all_sharpes))
        print(f"\n=== Bootstrap Sharpe significance (across {len(all_sharpes)} windows) ===")
        print(f"  Observed Sharpe: {boot['sharpe']:.4f}")
        print(f"  p-value (H0: Sharpe <= 0): {boot['p_value']:.4f}")
        print(f"  95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
        if boot["p_value"] < 0.05 and boot["sharpe"] > 0:
            print("  ✅ Statistically significant (p < 0.05, Sharpe > 0)")
        else:
            print("  ⚠️  Not statistically significant")

    avg_sharpe = overall["mean_oos_sharpe"]
    if avg_sharpe > 0.3:
        print(f"\n✅ Portfolio Sharpe {avg_sharpe:+.3f} — beat threshold 0.3")
    elif avg_sharpe > 0:
        print(f"\n⚠️  Portfolio Sharpe {avg_sharpe:+.3f} — positive but below 0.3 threshold")
    else:
        print(f"\n❌ Portfolio Sharpe {avg_sharpe:+.3f} — negative, revisit core alpha")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())