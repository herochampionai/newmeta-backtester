"""Linda MACD lenient — Optuna on 2Y OOS."""
import warnings; warnings.filterwarnings('ignore')
import sys, json
sys.path.insert(0, '.')
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import fetch_h1, run_strategy
from strategies.linda_macd_lenient import LindaMACDLenientStrategy


def main():
    print('Linda MACD Lenient — Optuna 2Y OOS')
    print('=' * 80)
    eur = fetch_h1('D:/MT5_EuroPrinter/terminal64.exe', 'EURUSD')
    nas = fetch_h1('D:/MT5_Bybit/terminal64.exe', 'NAS100')
    eur_oos = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
    nas_oos = nas[(nas.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (nas.index < pd.Timestamp('2026-09-17', tz='UTC'))]

    def obj(trial, df, profile):
        params = {
            "fast": trial.suggest_int("fast", 5, 30),
            "slow": trial.suggest_int("slow", 20, 60),
            "signal": trial.suggest_int("signal", 5, 20),
            "use_sma_filter": trial.suggest_categorical("use_sma", [True, False]),
            "sma_period": trial.suggest_int("sma_p", 50, 300),
            "use_histogram_momentum": trial.suggest_categorical("use_hist", [True, False]),
            "cooldown": trial.suggest_int("cd", 2, 30),
        }
        m = run_strategy(df, LindaMACDLenientStrategy, params, [], profile)
        if "error" in m or m["n_trades"] < 10:
            return -1e9
        return m["net_pnl"] + 50 * m["sharpe"]

    n_trials = 40
    results = {}
    for asset, df, profile in [("EUR", eur_oos, "forex"), ("NAS", nas_oos, "nas100")]:
        print(f"\n>>> linda_macd_lenient_{asset} ({n_trials} trials)...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda t: obj(t, df, profile), n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        m = run_strategy(df, LindaMACDLenientStrategy, best, [], profile)
        if "error" not in m:
            st = 'ROBUST' if (m['net_pnl'] > 0 and m['sharpe'] > 0) else 'X'
            print(f"  PnL ${m['net_pnl']:+,.0f} | Sharpe {m['sharpe']:+.2f} | WR {m['win_rate']*100:.1f}% | PF {m['profit_factor']:.2f} | {m['n_trades']} trades | {st}")
            print(f"  best: {best}")
            results[asset] = {'params': best, 'metrics': m}

    with open('output/linda_macd_lenient_best.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → output/linda_macd_lenient_best.json")


if __name__ == "__main__":
    main()
