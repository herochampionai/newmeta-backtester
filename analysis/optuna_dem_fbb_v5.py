"""Optuna search for DeM V5 + FBB V5 on 2Y OOS."""
import warnings; warnings.filterwarnings('ignore')
import json
import sys

sys.path.insert(0, '.')
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import fetch_h1, run_strategy
from strategies.dem_fbb_v5 import DeMV5Strategy, FBBV5Strategy


def obj_dem(trial, df, profile):
    params = {
        "bars_calculate": trial.suggest_int("bars_calculate", 8, 40),
        "htf_rule": trial.suggest_categorical("htf_rule", ["4h", "1d"]),
        "bull_regime": trial.suggest_float("bull_regime", 45.0, 60.0),
        "bear_regime": 100.0 - trial.suggest_float("bull_regime", 45.0, 60.0),
        "pullback_zone_low": trial.suggest_float("pb_low", 30.0, 50.0),
        "pullback_zone_high": trial.suggest_float("pb_high", 50.0, 70.0),
        "use_adx_filter": trial.suggest_categorical("use_adx", [True, False]),
        "adx_min": trial.suggest_float("adx_min", 15.0, 30.0),
        "use_volume_filter": trial.suggest_categorical("use_vol", [True, False]),
        "cooldown_bars": trial.suggest_int("cd", 3, 30),
    }
    m = run_strategy(df, DeMV5Strategy, params, [], profile)
    if "error" in m or m["n_trades"] < 15:
        return -1e9
    return m["net_pnl"] + 50 * m["sharpe"]


def obj_fbb(trial, df, profile):
    params = {
        "bars_calculate": trial.suggest_int("bars_calculate", 10, 50),
        "deviation": trial.suggest_float("deviation", 1.5, 3.0),
        "htf_rule": trial.suggest_categorical("htf_rule", ["4h", "1d"]),
        "ema_period": trial.suggest_int("ema_period", 50, 300),
        "adx_min": trial.suggest_float("adx_min", 15.0, 30.0),
        "require_close_confirm": trial.suggest_categorical("req_confirm", [True, False]),
        "volume_mult": trial.suggest_float("vol_mult", 0.5, 2.0),
        "cooldown_bars": trial.suggest_int("cd", 3, 30),
    }
    m = run_strategy(df, FBBV5Strategy, params, [], profile)
    if "error" in m or m["n_trades"] < 10:
        return -1e9
    return m["net_pnl"] + 50 * m["sharpe"]


def main():
    print('DeM V5 + FBB V5 Optuna — 2Y OOS')
    print('=' * 80)
    eur = fetch_h1('D:/MT5_EuroPrinter/terminal64.exe', 'EURUSD')
    nas = fetch_h1('D:/MT5_Bybit/terminal64.exe', 'NAS100')
    eur_oos = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
    nas_oos = nas[(nas.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (nas.index < pd.Timestamp('2026-09-17', tz='UTC'))]

    n_trials = 50
    for cls_name, cls, obj_fn in [("dem_v5", DeMV5Strategy, obj_dem),
                                    ("fbb_v5", FBBV5Strategy, obj_fbb)]:
        results = {}
        for asset, df, profile in [("EUR", eur_oos, "forex"), ("NAS", nas_oos, "nas100")]:
            print(f"\n>>> {cls_name}_{asset} ({n_trials} trials)...")
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(lambda t: obj_fn(t, df, profile), n_trials=n_trials, show_progress_bar=False)
            best = study.best_params
            m = run_strategy(df, cls, best, [], profile)
            if 'error' not in m:
                st = 'ROBUST' if (m['net_pnl'] > 0 and m['sharpe'] > 0) else 'failing'
                print(f"  PnL ${m['net_pnl']:+,.0f} | Sharpe {m['sharpe']:+.2f} | WR {m['win_rate']*100:.1f}% | PF {m['profit_factor']:.2f} | {m['n_trades']} trades | {st}")
                print(f"  best: {best}")
                results[f"{cls_name}_{asset}"] = {"params": best, "metrics": m}
        # Save
        with open(f'output/{cls_name}_best.json', 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
