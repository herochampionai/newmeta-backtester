"""Multi-window walk-forward ablation — confirm results aren't 1-window artifacts.

Runs 4 OOS windows (2024-Q1, Q2, Q3, Q4) for each of 8 configurations.
Aggregates results across windows for robust verdict.
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
from core.surgical_features import SURGICAL_FEATURES, enable, get_feature_defaults
from data.cache import load as load_cache
from strategies import STRATEGY_REGISTRY


SYMBOL = "EURUSD"
TIMEFRAME = "H1"
STRATEGY = "fbb"
FEATURE_LIST = list(SURGICAL_FEATURES.keys())
WINDOWS = [
    ("2024-Q1", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-04-01", tz="UTC")),
    ("2024-Q2", pd.Timestamp("2024-04-01", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC")),
    ("2024-Q3", pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2024-10-01", tz="UTC")),
    ("2024-Q4", pd.Timestamp("2024-10-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
]
CONFIGS = [
    ("BASELINE", None),
    ("anomaly_gate", "anomaly_gate"),
    ("event_blackout", "event_blackout"),
    ("recovery_restart", "recovery_restart"),
    ("basket_money_tp", "basket_money_tp"),
    ("profit_lock_trail", "profit_lock_trail"),
    ("carry_adjusted_tp", "carry_adjusted_tp"),
    ("ALL 6", "all"),
]


def build_params(feature_to_enable):
    base = get_feature_defaults()
    if feature_to_enable is None:
        return base
    if feature_to_enable == "all":
        for name in FEATURE_LIST:
            base = enable(base, name)
        return base
    return enable(base, feature_to_enable)


def run_single(df, params):
    cls = STRATEGY_REGISTRY[STRATEGY]
    sig = cls(params=params).generate(df)
    signals = {STRATEGY: (sig.entries.fillna(False).astype(bool),
                          pd.Series(sig.direction, index=df.index).fillna(0).astype(int))}
    r = run_full(
        df, signals,
        grid_mode=GRID_LOSS_AND_PROFIT,
        base_lot=0.1, grid_take_profit=50, grid_stop_loss=200,
        max_grid_layers=4, pips_between_orders=30, grid_lot_multiplier=1.5,
        recovery_mode=RECOVERY_HIGHER_PROFITS, recovery_lot_multiplier=2.0,
        init_cash=10_000.0, commission_pips=0.7, slippage_pips=0.3,
        spread_pips=1.0, pip_size=0.0001, contract_size=100_000,
        params=params,
    )
    m = compute_all(r["equity"].pct_change().fillna(0), r.get("trades"), r["equity"])
    return {
        "net_pnl": m.get("net_pnl", 0.0),
        "sharpe": m.get("sharpe", 0.0),
        "win_rate": m.get("win_rate", 0.0),
        "profit_factor": m.get("profit_factor", 0.0),
        "max_dd": m.get("max_drawdown", 0.0),
        "n_trades": m.get("n_trades", 0),
    }


def main():
    df_full, meta = load_cache(SYMBOL, TIMEFRAME)
    print(f"Loaded {SYMBOL} {TIMEFRAME}: {len(df_full)} bars\n")

    # Aggregate across windows
    agg = {label: {"net_pnl": [], "sharpe": [], "wr": [], "pf": [], "dd": [], "trades": []}
           for label, _ in CONFIGS}

    for win_name, t0, t1 in WINDOWS:
        df_win = df_full[(df_full.index >= t0) & (df_full.index < t1)].copy()
        if len(df_win) < 200:
            continue
        print(f"=== {win_name} ({df_win.index[0].date()} → {df_win.index[-1].date()}, "
              f"{len(df_win)} bars) ===")
        for label, feat in CONFIGS:
            params = build_params(feat)
            m = run_single(df_win, params)
            agg[label]["net_pnl"].append(m["net_pnl"])
            agg[label]["sharpe"].append(m["sharpe"])
            agg[label]["wr"].append(m["win_rate"])
            agg[label]["pf"].append(m["profit_factor"])
            agg[label]["dd"].append(m["max_dd"])
            agg[label]["trades"].append(m["n_trades"])
            print(f"  {label:<22} PnL ${m['net_pnl']:+,.0f} | "
                  f"Sharpe {m['sharpe']:+.2f} | PF {m['profit_factor']:.2f} | "
                  f"WR {m['win_rate']*100:.1f}%")
        print()

    # Aggregate table
    print("=" * 90)
    print("AGGREGATED ACROSS 4 OOS WINDOWS (mean ± std)")
    print("=" * 90)
    base = agg["BASELINE"]
    print(f"{'Config':<22} {'PnL':>10} {'Δ PnL':>9} {'Sharpe':>8} {'Δ Sharpe':>10} "
          f"{'WR %':>7} {'PF':>6} {'DD %':>7} {'#Tr':>5}")
    print("-" * 90)

    summary = {}
    for label, _ in CONFIGS:
        a = agg[label]
        pnl_mean = np.mean(a["net_pnl"])
        pnl_std = np.std(a["net_pnl"])
        sh_mean = np.mean(a["sharpe"])
        wr_mean = np.mean(a["wr"]) * 100
        pf_mean = np.mean(a["pf"])
        dd_mean = np.mean(a["dd"]) * 100
        trades = int(np.sum(a["trades"]))
        d_pnl = pnl_mean - np.mean(base["net_pnl"])
        d_sharpe = sh_mean - np.mean(base["sharpe"])
        summary[label] = (pnl_mean, d_pnl, sh_mean, d_sharpe, wr_mean, pf_mean, dd_mean, trades)
        marker = " ← BASE" if label == "BASELINE" else ""
        print(f"{label:<22} ${pnl_mean:>8,.0f} ${d_pnl:>+7,.0f} {sh_mean:>+7.2f} "
              f"{d_sharpe:>+9.2f} {wr_mean:>6.1f}% {pf_mean:>5.2f} "
              f"{dd_mean:>+6.2f}% {trades:>5}{marker}")

    # Verdict
    print("\n" + "=" * 90)
    print("VERDICT (4-window walk-forward, mean PnL Δ)")
    print("=" * 90)
    print(f"{'Feature':<22} {'ΔPnL':>9} {'ΔSharpe':>9} {'ΔPF':>7} {'ΔWR':>7} {'Verdict':<15}")
    print("-" * 75)
    verdicts = {}
    for feat in FEATURE_LIST:
        pnl_mean, d_pnl, sh_mean, d_sharpe, wr_mean, pf_mean, dd_mean, _ = summary[feat]
        base_pnl = np.mean(base["net_pnl"])
        base_sh = np.mean(base["sharpe"])
        base_pf = np.mean(base["pf"])
        base_wr = np.mean(base["wr"]) * 100
        d_pf = pf_mean - base_pf
        d_wr = wr_mean - base_wr
        if d_pnl > 0 and d_sharpe > 0:
            verdict = "✓ PASS"
        elif d_pnl > 0:
            verdict = "~ neutral"
        else:
            verdict = "✗ FAIL"
        verdicts[feat] = verdict
        print(f"{feat:<22} ${d_pnl:>+7,.0f} {d_sharpe:>+8.2f} {d_pf:>+6.2f} "
              f"{d_wr:>+6.1f}% {verdict:<15}")

    all_pnl, all_d_pnl, all_sh, all_d_sh, all_wr, all_pf, all_dd, all_tr = summary["ALL 6"]
    print(f"\n{'ALL 6 combined':<22} ${all_d_pnl:>+7,.0f} {all_d_sh:>+8.2f} "
          f"{all_pf - np.mean(base['pf']):>+6.2f} "
          f"{all_wr - np.mean(base['wr'])*100:>+6.1f}% "
          f"{'✓ PASS' if all_d_pnl > 0 and all_d_sh > 0 else '✗ FAIL'}")

    passing = [f for f, v in verdicts.items() if v == "✓ PASS"]
    failing = [f for f, v in verdicts.items() if v == "✗ FAIL"]
    neutral = [f for f, v in verdicts.items() if v == "~ neutral"]
    print("\n" + "=" * 90)
    print("FINAL RECOMMENDATION")
    print("=" * 90)
    if passing:
        print(f"\n  ✓ ENABLE in presets: {', '.join(passing)}")
    if neutral:
        print(f"  ~ Investigate further: {', '.join(neutral)}")
    if failing:
        print(f"  ✗ KEEP OFF (degrade OOS): {', '.join(failing)}")
    print()


if __name__ == "__main__":
    main()
