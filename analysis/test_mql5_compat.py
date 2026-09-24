"""Critical: Test ADX with ONLY MQL5-compatible params (no zone logic).

If this loses while my Python Optuna (with zone logic) wins, there's a huge
discrepancy between Python and MT5 — user needs to update the EA.
"""
import warnings; warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '.')
import pandas as pd
from analysis.optuna_filters import fetch_h1, run_strategy
from strategies.adx import ADX_Strategy


def main():
    print("=" * 100)
    print("CRITICAL TEST: ADX MQL5-compatible vs Python with zone logic")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    eur_oos = eur[(eur.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (eur.index < pd.Timestamp("2026-09-17", tz="UTC"))]
    nas_oos = nas[(nas.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (nas.index < pd.Timestamp("2026-09-17", tz="UTC"))]

    # Python's full params (with zone logic)
    python_full = {
        "bars_calculate": 13, "use_di_cross": True, "crossover_lookback": 7,
        "min_crossover_gap": 1.3986828073000317,
        "adx_zone_low": 24.529232348787147, "adx_zone_high": 53.604182111119634,
        "continuation_level": 37.17672183429483, "reversal_edge": 20.620488333138574,
        "open_orders_type": 3, "level_open_orders_1": 71.16921284392518,
        "level_open_orders_2": 18.615032701879066, "sweep_lookback": 6,
        "level_close_orders_1": 8.154728737910181, "level_close_orders_2": 4.789372175948943,
        "close_orders_type": 1,
    }

    # MQL5-compatible (NO zone, NO sweep — defaults)
    mql5_compat = {
        "bars_calculate": 13, "use_di_crossover": True, "crossover_lookback": 7,
        "min_crossover_gap": 1.4,
        "open_orders_type": 3, "level_open_orders_1": 71.17, "level_open_orders_2": 18.62,
        "close_orders_type": 1, "level_close_orders_1": 8.15, "level_close_orders_2": 4.79,
        # No zone_low/high, no continuation/reversal, no sweep_lookback
    }

    print("\n=== ADX_NAS — Python full vs MQL5-compatible ===")
    print(f"\nPython full (zone logic included):")
    r = run_strategy(nas_oos, ADX_Strategy, python_full, [], "nas100")
    if "error" in r:
        print(f"  ERR: {r['error']}")
    else:
        print(f"  PnL ${r['net_pnl']:+,.0f} | Sharpe {r['sharpe']:+.2f} | "
              f"WR {r['win_rate']*100:.1f}% | PF {r['profit_factor']:.2f} | {r['n_trades']} trades")

    print(f"\nMQL5-compatible (NO zone logic):")
    r = run_strategy(nas_oos, ADX_Strategy, mql5_compat, [], "nas100")
    if "error" in r:
        print(f"  ERR: {r['error']}")
    else:
        print(f"  PnL ${r['net_pnl']:+,.0f} | Sharpe {r['sharpe']:+.2f} | "
              f"WR {r['win_rate']*100:.1f}% | PF {r['profit_factor']:.2f} | {r['n_trades']} trades")

    print("\n=== ADX_EUR — Python full vs MQL5-compatible ===")
    print(f"\nPython full (zone logic included):")
    r = run_strategy(eur_oos, ADX_Strategy, python_full, [], "forex")
    if "error" in r:
        print(f"  ERR: {r['error']}")
    else:
        print(f"  PnL ${r['net_pnl']:+,.0f} | Sharpe {r['sharpe']:+.2f} | "
              f"WR {r['win_rate']*100:.1f}% | PF {r['profit_factor']:.2f} | {r['n_trades']} trades")

    print(f"\nMQL5-compatible (NO zone logic):")
    r = run_strategy(eur_oos, ADX_Strategy, mql5_compat, [], "forex")
    if "error" in r:
        print(f"  ERR: {r['error']}")
    else:
        print(f"  PnL ${r['net_pnl']:+,.0f} | Sharpe {r['sharpe']:+.2f} | "
              f"WR {r['win_rate']*100:.1f}% | PF {r['profit_factor']:.2f} | {r['n_trades']} trades")

    # Re-run Optuna WITHOUT zone logic to find the BEST MQL5-compatible params
    print("\n" + "=" * 100)
    print("Optuna re-search — ADX with ONLY MQL5-compatible params")
    print("=" * 100)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj_mql5(trial, df, profile):
        params = {
            "bars_calculate": trial.suggest_int("bars_calculate", 10, 30),
            "use_di_crossover": True,
            "crossover_lookback": trial.suggest_int("crossover_lookback", 1, 8),
            "min_crossover_gap": trial.suggest_float("min_crossover_gap", 1.0, 15.0),
            "open_orders_type": trial.suggest_int("open_orders_type", 1, 4),
            "level_open_orders_1": trial.suggest_float("level_open_orders_1", 20.0, 80.0),
            "level_open_orders_2": trial.suggest_float("level_open_orders_2", 5.0, 30.0),
            "close_orders_type": trial.suggest_int("close_orders_type", 1, 4),
            "level_close_orders_1": trial.suggest_float("level_close_orders_1", 5.0, 25.0),
            "level_close_orders_2": trial.suggest_float("level_close_orders_2", 1.0, 10.0),
        }
        m = run_strategy(df, ADX_Strategy, params, [], profile)
        if "error" in m or m["n_trades"] < 10:
            return -1e9
        return m["net_pnl"] + 50 * m["sharpe"]

    for asset, df, profile in [("EUR", eur_oos, "forex"), ("NAS", nas_oos, "nas100")]:
        print(f"\n>>> {asset} MQL5-only Optuna (40 trials)...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda t: obj_mql5(t, df, profile), n_trials=40, show_progress_bar=False)
        best = study.best_params
        m = run_strategy(df, ADX_Strategy, best, [], profile)
        if "error" not in m:
            print(f"  score={study.best_value:.0f} | PnL ${m['net_pnl']:+,.0f} | "
                  f"Sharpe {m['sharpe']:+.2f} | WR {m['win_rate']*100:.1f}% | "
                  f"PF {m['profit_factor']:.2f} | {m['n_trades']} trades")
            print(f"  best params: {best}")


if __name__ == "__main__":
    main()
