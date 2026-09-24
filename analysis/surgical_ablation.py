"""Per-feature surgical ablation — measure the real impact of each toggle.

Hypothesis from AGENTS.md:
  Each of the 6 surgical features (anomaly_gate, event_blackout,
  recovery_restart, basket_money_tp, profit_lock_trail, carry_adjusted_tp)
  must beat baseline OOS on the walk-forward harness before being flipped
  to default-ON.

Setup:
  - Single asset: EURUSD H1 (cached, 2022-01-03 to 2024-12-30, 18,644 bars)
  - Strategy: fbb (Fractal Breakout Bands — produces clear grid entries)
  - Grid engine with GRID_LOSS_AND_PROFIT (so surgical features actually fire)
  - Train: 2022-01 → 2024-06 (30 months)  ← baseline + per-feature tested here
  - Test:  2024-07 → 2024-12 (6 months)   ← OOS acceptance test

For each of 8 configurations (baseline + 6 individual + all-on):
  - Run backtest on OOS window
  - Record: net_pnl, total_return, sharpe, sortino, calmar, max_dd,
            win_rate, profit_factor, n_trades

Output: ranked table by net_pnl with delta-vs-baseline columns.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_LOSS_AND_PROFIT, RECOVERY_HIGHER_PROFITS
from backtester.metrics_v2 import compute_all
from core.surgical_features import SURGICAL_FEATURES, enable, disable, get_feature_defaults
from data.cache import load as load_cache
from strategies import STRATEGY_REGISTRY


FEATURE_LIST = list(SURGICAL_FEATURES.keys())  # 6 features
SYMBOL = "EURUSD"
TIMEFRAME = "H1"
STRATEGY = "fbb"
TRAIN_END = pd.Timestamp("2024-07-01", tz="UTC")  # OOS = after this


def build_params(feature_to_enable: str | None = None) -> dict:
    """Build a params dict.

    Args:
        feature_to_enable: None = baseline (all OFF)
                          "all" = all 6 ON
                          <feature name> = only that feature ON
    """
    base = get_feature_defaults()  # all OFF
    if feature_to_enable is None:
        return base
    if feature_to_enable == "all":
        for name in FEATURE_LIST:
            base = enable(base, name)
        return base
    # Single feature ON
    return enable(base, feature_to_enable)


def run_single(df: pd.DataFrame, params: dict) -> dict:
    """Run one backtest and return key metrics."""
    cls = STRATEGY_REGISTRY[STRATEGY]
    inst = cls(params=params)
    sig = inst.generate(df)
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    signals = {STRATEGY: (entries, direction)}

    result = run_full(
        df, signals,
        grid_mode=GRID_LOSS_AND_PROFIT,
        base_lot=0.1,
        grid_take_profit=50,
        grid_stop_loss=200,
        max_grid_layers=4,
        pips_between_orders=30,
        grid_lot_multiplier=1.5,
        recovery_mode=RECOVERY_HIGHER_PROFITS,
        recovery_lot_multiplier=2.0,
        init_cash=10_000.0,
        commission_pips=0.7,
        slippage_pips=0.3,
        spread_pips=1.0,
        pip_size=0.0001,
        contract_size=100_000,
        params=params,
    )
    metrics = compute_all(
        result["equity"].pct_change().fillna(0),
        result.get("trades"),
        result["equity"],
    )
    return {
        "net_pnl": metrics.get("net_pnl", 0.0),
        "total_return": metrics.get("total_return", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "sortino": metrics.get("sortino", 0.0),
        "calmar": metrics.get("calmar", 0.0),
        "max_dd": metrics.get("max_drawdown", 0.0),
        "win_rate": metrics.get("win_rate", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0),
        "n_trades": metrics.get("n_trades", 0),
        "expectancy": metrics.get("expectancy", 0.0),
        "recovery_factor": metrics.get("recovery_factor", 0.0),
    }


def fmt(v: float, kind: str = "pct") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if kind == "pct":
        return f"{v*100:+7.2f}%"
    if kind == "money":
        return f"${v:+,.0f}"
    if kind == "num":
        return f"{v:+.3f}"
    if kind == "int":
        return f"{int(v)}"
    return f"{v}"


def main():
    print("=" * 80)
    print("SURGICAL FEATURE ABLATION — OOS test (2024-07 → 2024-12, EURUSD H1)")
    print(f"Strategy: {STRATEGY} | Grid: LOSS_AND_PROFIT | 6 month OOS window")
    print("=" * 80)

    df_full, meta = load_cache(SYMBOL, TIMEFRAME)
    print(f"\nLoaded {SYMBOL} {TIMEFRAME}: {len(df_full)} bars "
          f"({df_full.index[0].date()} → {df_full.index[-1].date()})")

    df_oos = df_full[df_full.index >= TRAIN_END].copy()
    print(f"OOS window: {len(df_oos)} bars ({df_oos.index[0].date()} → {df_oos.index[-1].date()})\n")

    # Run all configurations
    configs = [
        ("BASELINE (all OFF)", None),
        ("anomaly_gate ON", "anomaly_gate"),
        ("event_blackout ON", "event_blackout"),
        ("recovery_restart ON", "recovery_restart"),
        ("basket_money_tp ON", "basket_money_tp"),
        ("profit_lock_trail ON", "profit_lock_trail"),
        ("carry_adjusted_tp ON", "carry_adjusted_tp"),
        ("ALL 6 ON", "all"),
    ]

    results = {}
    for label, feature in configs:
        params = build_params(feature)
        print(f"  Running {label}...", end=" ", flush=True)
        m = run_single(df_oos, params)
        results[label] = m
        print(f"PnL ${m['net_pnl']:+,.0f} | Sharpe {m['sharpe']:+.2f} | "
              f"PF {m['profit_factor']:+.2f} | WR {m['win_rate']*100:.1f}%")

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS — sorted by Net PnL")
    print("=" * 80)
    base = results["BASELINE (all OFF)"]
    header = f"{'Config':<26} {'Net PnL':>12} {'Δ PnL':>10} {'Sharpe':>8} {'Sortino':>9} " \
             f"{'Calmar':>8} {'Max DD':>9} {'WR %':>7} {'PF':>6} {'#Tr':>5} {'Expect':>9}"
    print(header)
    print("-" * len(header))

    sorted_configs = sorted(results.items(), key=lambda kv: kv[1]["net_pnl"], reverse=True)
    for label, m in sorted_configs:
        d_pnl = m["net_pnl"] - base["net_pnl"]
        marker = " ← baseline" if "BASELINE" in label else ""
        print(f"{label:<26} "
              f"${m['net_pnl']:>10,.0f} "
              f"${d_pnl:>8,.0f} "
              f"{m['sharpe']:>+7.2f} "
              f"{m['sortino']:>+8.2f} "
              f"{m['calmar']:>+7.2f} "
              f"{m['max_dd']*100:>+8.2f}% "
              f"{m['win_rate']*100:>6.1f}% "
              f"{m['profit_factor']:>5.2f} "
              f"{int(m['n_trades']):>5} "
              f"${m['expectancy']:>+7.2f}{marker}")

    # Per-feature delta vs baseline
    print("\n" + "=" * 80)
    print("PER-FEATURE IMPACT (Δ vs baseline)")
    print("=" * 80)
    print(f"{'Feature':<22} {'ΔPnL':>10} {'ΔSharpe':>10} {'ΔPF':>8} {'ΔWinRate':>10} {'ΔMaxDD':>10} {'Verdict':<14}")
    print("-" * 90)
    verdicts = {}
    for feature in FEATURE_LIST:
        label = f"{feature} ON"
        m = results[label]
        d_pnl = m["net_pnl"] - base["net_pnl"]
        d_sharpe = m["sharpe"] - base["sharpe"]
        d_pf = m["profit_factor"] - base["profit_factor"]
        d_wr = (m["win_rate"] - base["win_rate"]) * 100
        d_dd = (m["max_dd"] - base["max_dd"]) * 100
        # Verdict: PASS if both PnL up AND Sharpe up; NEUTRAL if marginal; FAIL if worse
        if d_pnl > 0 and d_sharpe > 0:
            verdict = "✓ PASS"
        elif d_pnl > 0 and d_sharpe <= 0:
            verdict = "~ neutral"
        else:
            verdict = "✗ FAIL"
        verdicts[feature] = verdict
        print(f"{feature:<22} ${d_pnl:>+8,.0f} {d_sharpe:>+9.2f} {d_pf:>+7.2f} "
              f"{d_wr:>+9.1f}% {d_dd:>+9.2f}% {verdict:<14}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION (per AGENTS.md acceptance barrier)")
    print("=" * 80)
    passing = [f for f, v in verdicts.items() if v == "✓ PASS"]
    failing = [f for f, v in verdicts.items() if v == "✗ FAIL"]
    neutral = [f for f, v in verdicts.items() if v == "~ neutral"]
    if passing:
        print(f"\n  ✓ ENABLE (Pnl↑ AND Sharpe↑): {', '.join(passing)}")
    if neutral:
        print(f"  ~ INVESTIGATE (mixed): {', '.join(neutral)}")
    if failing:
        print(f"  ✗ KEEP OFF (failed acceptance): {', '.join(failing)}")

    all_m = results["ALL 6 ON"]
    print(f"\n  All-6-ON combined: PnL ${all_m['net_pnl']:+,.0f}, Sharpe {all_m['sharpe']:+.2f}")
    if all_m["net_pnl"] > base["net_pnl"] and all_m["sharpe"] > base["sharpe"]:
        print("    → All-6 also passes — could ship as one toggle")
    else:
        print("    → All-6 FAILS combined even if individuals pass → interaction effects")

    # Save results
    out = Path("output/surgical_ablation.csv")
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).T.to_csv(out)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
