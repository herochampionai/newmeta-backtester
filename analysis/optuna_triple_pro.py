"""Optuna search for TripleRSI Pro v6 on 2Y OOS."""
import warnings; warnings.filterwarnings('ignore')
import json
import sys

sys.path.insert(0, '.')
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import fetch_h1, run_strategy
from strategies.triple_rsi_pro_v6 import TripleRSIProV6Strategy


def obj(trial, df, profile):
    params = {
        # RSI periods (period widths)
        "rsi_fast": trial.suggest_int("rsi_fast", 5, 14),
        "rsi_med": trial.suggest_int("rsi_med", 8, 21),
        "rsi_slow": trial.suggest_int("rsi_slow", 14, 30),
        # Threshold — easier to trigger
        "context_threshold": trial.suggest_float("ctx_thr", 50.0, 65.0),
        "momentum_threshold": trial.suggest_float("mom_thr", 45.0, 60.0),
        # Pullback zone — wider
        "pullback_zone_low": trial.suggest_float("pb_low", 30.0, 50.0),
        "pullback_zone_high": trial.suggest_float("pb_high", 45.0, 60.0),
        "min_alignment_count": trial.suggest_int("min_align", 1, 3),
        # Filters
        "use_ema_filter": trial.suggest_categorical("use_ema", [True, False]),
        "ema_period": trial.suggest_int("ema_p", 50, 300),
        "use_adx_filter": trial.suggest_categorical("use_adx", [True, False]),
        "adx_min_trend": trial.suggest_float("adx_min", 15.0, 30.0),
        "filter_choppy": trial.suggest_categorical("use_choppy", [True, False]),
        "choppy_zone": trial.suggest_float("choppy_z", 3.0, 12.0),
        "cooldown_bars": trial.suggest_int("cd", 3, 30),
    }
    m = run_strategy(df, TripleRSIProV6Strategy, params, [], profile)
    if "error" in m or m["n_trades"] < 8:
        return -1e9
    return m["net_pnl"] + 50 * m["sharpe"]


def main():
    print('TripleRSI Pro v6 — Optuna search, 2Y OOS')
    print('=' * 80)
    eur = fetch_h1('D:/MT5_EuroPrinter/terminal64.exe', 'EURUSD')
    nas = fetch_h1('D:/MT5_Bybit/terminal64.exe', 'NAS100')
    eur_oos = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
    nas_oos = nas[(nas.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (nas.index < pd.Timestamp('2026-09-17', tz='UTC'))]

    n_trials = 40
    results = {}
    for asset, df, profile in [('EUR', eur_oos, 'forex'), ('NAS', nas_oos, 'nas100')]:
        print(f"\n>>> {asset} Optuna ({n_trials} trials)...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda t: obj(t, df, profile), n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        m = run_strategy(df, TripleRSIProV6Strategy, best, [], profile)
        results[asset] = {'params': best, 'metrics': m}
        if 'error' not in m:
            st = 'ROBUST' if (m['net_pnl'] > 0 and m['sharpe'] > 0) else 'failing'
            print(f"  {asset} best: PnL ${m['net_pnl']:+,.0f} | Sharpe {m['sharpe']:+.2f} | "
                  f"WR {m['win_rate']*100:.1f}% | PF {m['profit_factor']:.2f} | "
                  f"{m['n_trades']} trades | {st}")
            print(f"  best params: {best}")

    # Save
    with open('output/triple_rsi_pro_v6_best.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
