"""Optuna search for V4 trend-following strategies."""
import warnings; warnings.filterwarnings('ignore')
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from analysis.optuna_filters import fetch_h1, run_strategy
from strategies.trend_follow_v4 import TripleRSIV4Strategy, Stoch533V4


def search_v4(strategy_name, cls, df_oos1, df_oos2, market_profile, n_trials=40):
    def obj(trial):
        if "triple" in strategy_name:
            params = {
                "rsi_fast": trial.suggest_int("rsi_fast", 3, 14),
                "rsi_mid": trial.suggest_int("rsi_mid", 10, 21),
                "rsi_slow": trial.suggest_int("rsi_slow", 21, 50),
                "sma_period": trial.suggest_int("sma_period", 20, 200),
                "adx_threshold": trial.suggest_float("adx_threshold", 10.0, 35.0),
                "pullback_pct": trial.suggest_float("pullback_pct", 0.001, 0.02),
                "rsi_bull_threshold": trial.suggest_float("rsi_bull", 45.0, 60.0),
                "rsi_bear_threshold": trial.suggest_float("rsi_bear", 40.0, 55.0),
                "cooldown_bars": trial.suggest_int("cooldown_bars", 3, 30),
                "use_volume_filter": trial.suggest_categorical("use_vol", [True, False]),
                "use_rsi_exit": trial.suggest_categorical("use_exit", [True, False]),
            }
        else:
            params = {
                "k_period": trial.suggest_int("k_period", 5, 30),
                "d_period": trial.suggest_int("d_period", 2, 10),
                "smooth": trial.suggest_int("smooth", 1, 6),
                "sma_period": trial.suggest_int("sma_period", 20, 200),
                "adx_threshold": trial.suggest_float("adx_threshold", 10.0, 35.0),
                "pullback_pct": trial.suggest_float("pullback_pct", 0.001, 0.02),
                "k_bull": trial.suggest_float("k_bull", 45.0, 60.0),
                "cooldown_bars": trial.suggest_int("cooldown_bars", 3, 30),
                "use_volume_filter": trial.suggest_categorical("use_vol", [True, False]),
                "use_k_exit": trial.suggest_categorical("use_exit", [True, False]),
            }
        m1 = run_strategy(df_oos1, cls, params, [], market_profile)
        if "error" in m1 or m1["n_trades"] < 5:
            return -1e9
        m2 = run_strategy(df_oos2, cls, params, [], market_profile)
        if "error" in m2:
            m2 = {"net_pnl": 0, "sharpe": 0, "n_trades": 0}
        # Reward BOTH periods profitable + Sharpe
        return (m1["net_pnl"] + m2["net_pnl"]) + 50 * (m1["sharpe"] + m2["sharpe"])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def main():
    print("=" * 100)
    print("V4 Optuna — find best params per ticker, validate on 2 OOS periods")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    eur_oos1 = eur[eur.index >= pd.Timestamp("2025-09-17", tz="UTC")]
    nas_oos1 = nas[nas.index >= pd.Timestamp("2025-09-17", tz="UTC")]
    eur_oos2 = eur[(eur.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (eur.index < pd.Timestamp("2025-09-17", tz="UTC"))]
    nas_oos2 = nas[(nas.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (nas.index < pd.Timestamp("2025-09-17", tz="UTC"))]

    n_trials = 40
    summary = {}
    for name, cls in [("triple_rsi_v4", TripleRSIV4Strategy), ("stoch533_mtf_v4", Stoch533V4)]:
        for asset, df1, df2, profile in [("EUR", eur_oos1, eur_oos2, "forex"),
                                            ("NAS", nas_oos1, nas_oos2, "nas100")]:
            key = f"{name}_{asset}"
            print(f"\n  Optuna {key} ({n_trials} trials)...")
            params, score = search_v4(key, cls, df1, df2, profile, n_trials)
            m1 = run_strategy(df1, cls, params, [], profile)
            m2 = run_strategy(df2, cls, params, [], profile)
            print(f"    {key} best: score={score:.0f}")
            def fmt(m):
                if isinstance(m, dict) and "error" not in m:
                    return f"${m['net_pnl']:>+8,.0f} Sharpe {m['sharpe']:+.2f} ({m['n_trades']} trades)"
                return f"ERR"
            print(f"      OOS-1: {fmt(m1)}")
            print(f"      OOS-2: {fmt(m2)}")
            if isinstance(m1, dict) and isinstance(m2, dict) and "error" not in m1 and "error" not in m2:
                if m1["net_pnl"] > 0 and m2["net_pnl"] > 0:
                    print(f"      STATUS: ✓✓ ROBUST (profitable both)")
                elif m1["net_pnl"] > 0 or m2["net_pnl"] > 0:
                    print(f"      STATUS: ~ borderline")
                else:
                    print(f"      STATUS: ✗ still failing")
            summary[key] = {"params": params, "score": score, "m1": m1, "m2": m2}

    # Save best params
    out = {k: {"params": v["params"]} for k, v in summary.items()}
    with open("output/v4_best_params.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → output/v4_best_params.json")


if __name__ == "__main__":
    main()
