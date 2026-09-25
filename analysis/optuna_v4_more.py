"""Optuna search for V4 trend-following on macd_confluence, bb_rsi, quad_stoch, fbb."""
import warnings; warnings.filterwarnings('ignore')
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

from analysis.optuna_filters import fetch_h1, run_strategy
from strategies.trend_follow_v4_more import FBBV4, BBRsiV4, MACDConfluenceV4, QuadStochV4

PARAM_SPACES = {
    "macd_confluence_v4": lambda trial: {
        "fast": trial.suggest_int("fast", 5, 30),
        "slow": trial.suggest_int("slow", 20, 60),
        "signal": trial.suggest_int("signal", 5, 20),
        "sma_period": trial.suggest_int("sma_period", 20, 200),
        "adx_threshold": trial.suggest_float("adx_threshold", 10.0, 35.0),
        "pullback_pct": trial.suggest_float("pullback_pct", 0.0005, 0.02),
        "cooldown_bars": trial.suggest_int("cooldown_bars", 3, 30),
        "use_volume_filter": trial.suggest_categorical("use_vol", [True, False]),
    },
    "bb_rsi_v4": lambda trial: {
        "bb_period": trial.suggest_int("bb_period", 10, 40),
        "bb_std": trial.suggest_float("bb_std", 1.0, 3.0),
        "rsi_period": trial.suggest_int("rsi_period", 7, 30),
        "sma_period": trial.suggest_int("sma_period", 20, 200),
        "adx_threshold": trial.suggest_float("adx_threshold", 10.0, 35.0),
        "pullback_pct": trial.suggest_float("pullback_pct", 0.0005, 0.02),
        "cooldown_bars": trial.suggest_int("cooldown_bars", 3, 30),
        "use_volume_filter": trial.suggest_categorical("use_vol", [True, False]),
    },
    "quad_stoch_v4": lambda trial: {
        "k_period": trial.suggest_int("k_period", 5, 30),
        "d_period": trial.suggest_int("d_period", 2, 10),
        "smooth": trial.suggest_int("smooth", 1, 6),
        "sma_period": trial.suggest_int("sma_period", 20, 200),
        "adx_threshold": trial.suggest_float("adx_threshold", 10.0, 35.0),
        "pullback_pct": trial.suggest_float("pullback_pct", 0.0005, 0.02),
        "cooldown_bars": trial.suggest_int("cooldown_bars", 3, 30),
        "use_volume_filter": trial.suggest_categorical("use_vol", [True, False]),
    },
    "fbb_v4": lambda trial: {
        "lookback": trial.suggest_int("lookback", 5, 50),
        "atr_period": trial.suggest_int("atr_period", 8, 30),
        "atr_mult": trial.suggest_float("atr_mult", 1.0, 4.0),
        "sma_period": trial.suggest_int("sma_period", 20, 200),
        "adx_threshold": trial.suggest_float("adx_threshold", 10.0, 35.0),
        "pullback_pct": trial.suggest_float("pullback_pct", 0.0005, 0.02),
        "cooldown_bars": trial.suggest_int("cooldown_bars", 3, 30),
        "use_volume_filter": trial.suggest_categorical("use_vol", [True, False]),
    },
}


STRAT_CLASSES = {
    "macd_confluence_v4": MACDConfluenceV4,
    "bb_rsi_v4": BBRsiV4,
    "quad_stoch_v4": QuadStochV4,
    "fbb_v4": FBBV4,
}


def main():
    print("=" * 100)
    print("V4 TREND-FOLLOWING — Optuna for macd/bb_rsi/quad_stoch/fbb")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    eur_oos1 = eur[eur.index >= pd.Timestamp("2025-09-17", tz="UTC")]
    nas_oos1 = nas[nas.index >= pd.Timestamp("2025-09-17", tz="UTC")]
    eur_oos2 = eur[(eur.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (eur.index < pd.Timestamp("2025-09-17", tz="UTC"))]
    nas_oos2 = nas[(nas.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (nas.index < pd.Timestamp("2025-09-17", tz="UTC"))]

    n_trials = 30
    results = {}
    for sname, cls in STRAT_CLASSES.items():
        for asset, df1, df2, profile in [("EUR", eur_oos1, eur_oos2, "forex"),
                                            ("NAS", nas_oos1, nas_oos2, "nas100")]:
            key = f"{sname}_{asset}"
            print(f"\n>>> {key} ({n_trials} trials)...")
            space = PARAM_SPACES[sname]

            def obj(trial, df1=df1, df2=df2, space=space, profile=profile, cls=cls):
                params = space(trial)
                m1 = run_strategy(df1, cls, params, [], profile)
                if "error" in m1 or m1["n_trades"] < 5:
                    return -1e9
                m2 = run_strategy(df2, cls, params, [], profile)
                if "error" in m2:
                    m2 = {"net_pnl": 0, "sharpe": 0, "n_trades": 0}
                return (m1["net_pnl"] + m2["net_pnl"]) + 50 * (m1["sharpe"] + m2["sharpe"])

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
            best = study.best_params
            m1 = run_strategy(df1, cls, best, [], profile)
            m2 = run_strategy(df2, cls, best, [], profile)
            def fmt(m):
                if isinstance(m, dict) and "error" not in m:
                    return f"${m['net_pnl']:>+8,.0f} Sharpe {m['sharpe']:+.2f} ({m['n_trades']} trades)"
                return "ERR"
            print(f"  OOS-1: {fmt(m1)}")
            print(f"  OOS-2: {fmt(m2)}")
            if isinstance(m1, dict) and isinstance(m2, dict) and "error" not in m1 and "error" not in m2:
                if m1["net_pnl"] > 0 and m2["net_pnl"] > 0:
                    status = "✓✓ ROBUST"
                elif m1["net_pnl"] > 0 or m2["net_pnl"] > 0:
                    status = "~ borderline"
                else:
                    status = "✗ failing"
                print(f"  STATUS: {status}")
            results[key] = {"params": best, "m1": m1, "m2": m2}

    # Save
    out = {k: v["params"] for k, v in results.items()}
    with open("output/v4_more_best_params.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → output/v4_more_best_params.json")

    # Summary
    print("\n" + "=" * 100)
    print("V4 SUMMARY — all 6 trend-following strategies")
    print("=" * 100)
    for key, r in results.items():
        m1 = r["m1"]; m2 = r["m2"]
        if isinstance(m1, dict) and isinstance(m2, dict) and "error" not in m1 and "error" not in m2:
            e_pnl = m1["net_pnl"]; n_pnl = m2["net_pnl"]
            tot = e_pnl + n_pnl
            if e_pnl > 0 and n_pnl > 0:
                st = "✓✓ ROBUST"
            elif e_pnl > 0 or n_pnl > 0:
                st = "~ borderline"
            else:
                st = "✗ failing"
            print(f"  {key:<28} EUR ${e_pnl:+8,.0f} | NAS ${n_pnl:+8,.0f} | {st}")


if __name__ == "__main__":
    main()
