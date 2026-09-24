"""Test 11 new strategies (10 SCreener setups + Linda MACD) on 2Y OOS."""
import warnings; warnings.filterwarnings('ignore')
import sys, json
sys.path.insert(0, '.')
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from analysis.optuna_filters import fetch_h1, run_strategy
from strategies.screener_and_linda import (
    ScreenerL0Strategy, ScreenerS5Strategy,  # Stochastic
    ScreenerL1Strategy, ScreenerS6Strategy,  # MA Cross
    ScreenerL2Strategy, ScreenerS7Strategy,  # MACD
    ScreenerL3Strategy, ScreenerS8Strategy,  # Bollinger
    ScreenerL4Strategy, ScreenerS9Strategy,  # SuperTrend
    LindaMACDStrategy,  # Linda Raschke
)


STRATS = {
    "sc_l0_stoch": (ScreenerL0Strategy, {"stoch_period": [5, 30], "smooth_k": [1, 5], "smooth_d": [1, 5], "over_sold": [10, 30], "sma_period": [100, 300], "cooldown": [3, 30]}),
    "sc_s5_stoch": (ScreenerS5Strategy, {"stoch_period": [5, 30], "smooth_k": [1, 5], "smooth_d": [1, 5], "over_bought": [70, 90], "sma_period": [100, 300], "cooldown": [3, 30]}),
    "sc_l1_macross": (ScreenerL1Strategy, {"fast": [3, 25], "slow": [10, 60], "sma_period": [100, 300], "cooldown": [3, 30]}),
    "sc_s6_macross": (ScreenerS6Strategy, {"fast": [3, 25], "slow": [10, 60], "sma_period": [100, 300], "cooldown": [3, 30]}),
    "sc_l2_macd": (ScreenerL2Strategy, {"fast": [5, 30], "slow": [20, 60], "signal": [5, 20], "sma_period": [100, 300], "cooldown": [3, 30]}),
    "sc_s7_macd": (ScreenerS7Strategy, {"fast": [5, 30], "slow": [20, 60], "signal": [5, 20], "sma_period": [100, 300], "cooldown": [3, 30]}),
    "sc_l3_bb": (ScreenerL3Strategy, {"bb_period": [10, 40], "bb_std": [1.5, 3.0], "sma_period": [100, 300], "cooldown": [3, 30]}),
    "sc_s8_bb": (ScreenerS8Strategy, {"bb_period": [10, 40], "bb_std": [1.5, 3.0], "sma_period": [100, 300], "cooldown": [3, 30]}),
    "sc_l4_supertrend": (ScreenerL4Strategy, {"atr_period": [5, 30], "factor": [2.0, 5.0], "sma_period": [100, 300], "cooldown": [3, 30]}),
    "sc_s9_supertrend": (ScreenerS9Strategy, {"atr_period": [5, 30], "factor": [2.0, 5.0], "sma_period": [100, 300], "cooldown": [3, 30]}),
    "linda_macd": (LindaMACDStrategy, {"fast": [2, 15], "slow": [5, 30], "signal": [5, 30], "stc_length": [5, 30], "stc_fast": [10, 60], "stc_slow": [30, 100], "diff_threshold": [0.05, 1.0], "sma_period": [50, 300], "cooldown": [3, 30]}),
}


def main():
    print('=' * 100)
    print('SCREENER SETUPS + LINDA MACD — 11 new strategies, 2Y OOS')
    print('=' * 100)
    eur = fetch_h1('D:/MT5_EuroPrinter/terminal64.exe', 'EURUSD')
    nas = fetch_h1('D:/MT5_Bybit/terminal64.exe', 'NAS100')
    eur_oos = eur[(eur.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (eur.index < pd.Timestamp('2026-09-17', tz='UTC'))]
    nas_oos = nas[(nas.index >= pd.Timestamp('2024-09-17', tz='UTC')) & (nas.index < pd.Timestamp('2026-09-17', tz='UTC'))]

    n_trials = 40
    all_results = {}
    for strat_name, (cls, space) in STRATS.items():
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
    with open('output/screener_linda_best.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → output/screener_linda_best.json")

    # Summary
    print('\n' + '=' * 100)
    print('SUMMARY — 11 new strategies on 2Y OOS')
    print('=' * 100)
    print(f"{'Instance':<32} {'PnL':>10} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'Tr':>4} {'Status':<10}")
    print('-' * 85)
    locked = []
    for k in sorted(all_results.keys()):
        m = all_results[k]["metrics"]
        st = 'ROBUST' if (m['net_pnl'] > 0 and m['sharpe'] > 0) else 'X'
        if st == 'ROBUST':
            locked.append(k)
        print(f"{k:<32} ${m['net_pnl']:>+9,.0f} {m['sharpe']:>+6.2f} "
              f"{m['win_rate']*100:>4.0f}% {m['profit_factor']:>4.2f} {m['n_trades']:>4} {st:<10}")
    print(f"\n  Locked: {len(locked)}/22 instances")


if __name__ == "__main__":
    main()
