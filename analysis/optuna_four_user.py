"""Test the 4 new user-requested strategies on 2Y OOS + Optuna."""
import warnings; warnings.filterwarnings('ignore')
import json
import sys

sys.path.insert(0, '.')
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import fetch_h1, run_strategy
from strategies.four_user_strategies import (
    BachelierWaveStrategy,
    MA8RibbonStrategy,
    MACDInstitutionalStrategy,
    RC44Strategy,
)


def main():
    print('=' * 100)
    print('FOUR USER STRATEGIES — Bachelier, MACD Institutional, MA8 Ribbon, 4x4 RC')
    print('2Y OOS, EUR + NAS')
    print('=' * 100)
    eur = fetch_h1('D:/MT5_EuroPrinter/terminal64.exe', 'EURUSD')
    nas = fetch_h1('D:/MT5_Bybit/terminal64.exe', 'NAS100')
    eur_oos = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
    nas_oos = nas[(nas.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (nas.index < pd.Timestamp('2026-09-17', tz='UTC'))]

    strats = [
        ("bachelier_wave", BachelierWaveStrategy, {
            "length": [10, 40],
            "smoothing": [5, 30],
            "sensitivity": [0.5, 3.0],
            "ob_level": [0.5, 1.5],
            "os_level": [-1.5, -0.5],
            "strong_mult": [1.0, 2.5],
            "cooldown": [3, 30],
        }),
        ("macd_inst", MACDInstitutionalStrategy, {
            "fast": [5, 30], "slow": [20, 60], "signal": [5, 20],
            "pullback_pct": [0.001, 0.02], "pullback_bars": [5, 30],
            "rsi_period": [7, 30], "rsi_extreme_high": [60, 85], "rsi_extreme_low": [15, 40],
            "cooldown": [3, 30],
        }),
        ("ma8_ribbon", MA8RibbonStrategy, {
            "slope_bars": [3, 15], "cooldown": [3, 30],
        }),
        ("rc44", RC44Strategy, {
            "htf_fast": [20, 100], "htf_slow": [100, 300],
            "pullback_bars": [5, 40], "pullback_pct": [0.001, 0.05],
            "sniper_bars": [2, 10], "cooldown": [3, 30],
        }),
    ]

    n_trials = 40
    all_results = {}
    for strat_name, cls, space in strats:
        for asset, df, profile in [("EUR", eur_oos, "forex"), ("NAS", nas_oos, "nas100")]:
            key = f"{strat_name}_{asset}"
            print(f"\n>>> {key} Optuna ({n_trials} trials)...")
            def obj(trial, df=df, profile=profile, space=space, cls=cls):
                params = {}
                for k, v in space.items():
                    lo, hi = v
                    if isinstance(lo, int) and isinstance(hi, int):
                        params[k] = trial.suggest_int(k, lo, hi)
                    else:
                        params[k] = trial.suggest_float(k, lo, hi)
                # Special case: ma8_ribbon — derive 8 EMAs in ascending order
                if strat_name == "ma8_ribbon":
                    base = params.pop("slope_bars")
                    cd = params.pop("cooldown")
                    ema1 = trial.suggest_int("ema_1", 5, 25)
                    ema2 = trial.suggest_int("ema_2", max(ema1 + 1, 10), 35)
                    ema3 = trial.suggest_int("ema_3", max(ema2 + 1, 15), 45)
                    ema4 = trial.suggest_int("ema_4", max(ema3 + 1, 20), 55)
                    ema5 = trial.suggest_int("ema_5", max(ema4 + 1, 25), 65)
                    ema6 = trial.suggest_int("ema_6", max(ema5 + 1, 30), 75)
                    ema7 = trial.suggest_int("ema_7", max(ema6 + 1, 35), 85)
                    ema8 = trial.suggest_int("ema_8", max(ema7 + 1, 40), 100)
                    params = {"ema_1": ema1, "ema_2": ema2, "ema_3": ema3, "ema_4": ema4,
                              "ema_5": ema5, "ema_6": ema6, "ema_7": ema7, "ema_8": ema8,
                              "slope_bars": base, "cooldown": cd}
                m = run_strategy(df, cls, params, [], profile)
                if "error" in m or m["n_trades"] < 8:
                    return -1e9
                return m["net_pnl"] + 50 * m["sharpe"]
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
            best = study.best_params
            m = run_strategy(df, cls, best, [], profile)
            if "error" not in m:
                st = "ROBUST" if (m["net_pnl"] > 0 and m["sharpe"] > 0) else "X"
                print(f"  PnL ${m['net_pnl']:+,.0f} | Sharpe {m['sharpe']:+.2f} | WR {m['win_rate']*100:.1f}% | PF {m['profit_factor']:.2f} | {m['n_trades']} trades | {st}")
                all_results[key] = {"params": best, "metrics": m}
            else:
                print(f"  ERR: {m['error']}")

    # Save
    with open('output/four_user_strategies_best.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved → output/four_user_strategies_best.json")

    # Summary
    print('\n' + '=' * 100)
    print('SUMMARY — 4 user-requested strategies')
    print('=' * 100)
    print(f"{'Instance':<32} {'PnL':>10} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'Tr':>4} {'Status':<10}")
    print('-' * 85)
    for k in sorted(all_results.keys()):
        m = all_results[k]["metrics"]
        st = 'ROBUST' if (m['net_pnl'] > 0 and m['sharpe'] > 0) else 'X'
        print(f"{k:<32} ${m['net_pnl']:>+9,.0f} {m['sharpe']:>+6.2f} "
              f"{m['win_rate']*100:>4.0f}% {m['profit_factor']:>4.2f} {m['n_trades']:>4} {st:<10}")


if __name__ == "__main__":
    main()
