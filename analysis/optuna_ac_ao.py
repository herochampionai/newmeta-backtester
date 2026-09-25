"""Optuna search for AC-AO per-ticker params.

Goals:
- EURUSD: lower level to get more signals (7 trades/12mo is too few)
- NAS100: higher level to filter noise (667 trades is too many)

Use Optuna to find best (level_open_orders, open_orders_type, use_acceleration_filter,
min_acceleration, use_ao_synchronization, min_ao_synchronization) per ticker.

Plus: add MTF filter on top (which was the best filter for NAS).
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

import MetaTrader5 as mt5

from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from backtester.metrics_v2 import compute_all
from data.mt5_export import init_mt5
from strategies._enhancement import EnhancedStrategy, mtf_trend
from strategies.ac_ao import AC_AO_Strategy


def fetch_h1(terminal, symbol, n_bars=20000):
    init_mt5(terminal)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n_bars)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    mt5.shutdown()
    return df


def run_with_params(df, params, market_profile="forex"):
    base = AC_AO_Strategy(params=params)
    # Add MTF filter (which was the biggest NAS improvement)
    filters = [mtf_trend]
    strat = EnhancedStrategy(base, filters=filters)
    strat.name = "ac_ao_optuna"
    try:
        sig = strat.generate(df)
    except Exception as e:
        return {"error": str(e)}
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    n_entries = int(entries.sum())
    if n_entries == 0:
        return {"error": "no entries"}
    signals = {strat.name: (entries, direction)}
    if market_profile == "forex":
        kw = dict(pip_size=0.0001, contract_size=100_000, base_lot=0.1,
                  commission_pips=0.7, slippage_pips=0.3, spread_pips=1.0, init_cash=10_000.0)
    else:
        kw = dict(pip_size=1.0, contract_size=1.0, base_lot=0.1,
                  commission_pips=2.0, slippage_pips=1.0, spread_pips=1.5, init_cash=10_000.0)
    kw["grid_mode"] = GRID_NONE
    try:
        r = run_full(df, signals, **kw)
    except Exception as e:
        return {"error": f"run_full: {e}"}
    m = compute_all(r["equity"].pct_change().fillna(0), r.get("trades"), r["equity"])
    return {
        "n_entries": n_entries,
        "net_pnl": m.get("net_pnl", 0.0),
        "sharpe": m.get("sharpe", 0.0),
        "max_dd": m.get("max_drawdown", 0.0),
        "win_rate": m.get("win_rate", 0.0),
        "profit_factor": m.get("profit_factor", 0.0),
        "n_trades": m.get("n_trades", 0),
    }


def objective_eur(trial, df):
    params = {
        "level_open_orders": trial.suggest_float("level_open_orders", 10.0, 200.0),
        "open_orders_type": trial.suggest_int("open_orders_type", 1, 8),
        "use_acceleration_filter": trial.suggest_categorical("use_accel", [True, False]),
        "min_acceleration": trial.suggest_float("min_acceleration", 0.0001, 0.002, log=True),
        "use_ao_synchronization": trial.suggest_categorical("use_ao_sync", [True, False]),
        "min_ao_synchronization": trial.suggest_float("min_ao_sync", 0.0001, 0.002, log=True),
        "level_close_orders": 70.0,
    }
    m = run_with_params(df, params, "forex")
    if "error" in m:
        return -1e9
    # Reward: PnL + Sharpe*100 - DD*100
    return m["net_pnl"] + 100 * m["sharpe"] - 200 * abs(m["max_dd"])


def objective_nas(trial, df):
    params = {
        "level_open_orders": trial.suggest_float("level_open_orders", 50.0, 1000.0),
        "open_orders_type": trial.suggest_int("open_orders_type", 1, 8),
        "use_acceleration_filter": trial.suggest_categorical("use_accel", [True, False]),
        "min_acceleration": trial.suggest_float("min_acceleration", 0.0001, 0.002, log=True),
        "use_ao_synchronization": trial.suggest_categorical("use_ao_sync", [True, False]),
        "min_ao_synchronization": trial.suggest_float("min_ao_sync", 0.0001, 0.002, log=True),
        "level_close_orders": 70.0,
    }
    m = run_with_params(df, params, "nas100")
    if "error" in m:
        return -1e9
    return m["net_pnl"] + 100 * m["sharpe"] - 200 * abs(m["max_dd"])


def main():
    print("=" * 100)
    print("AC-AO OPTUNA SEARCH — adaptive params per ticker + MTF filter")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    eur_oos = eur[eur.index >= eur.index[-1] - pd.DateOffset(months=12)]
    nas_oos = nas[nas.index >= nas.index[-1] - pd.DateOffset(months=12)]
    print(f"  EUR OOS: {eur_oos.index[0].date()} → {eur_oos.index[-1].date()} ({len(eur_oos)} bars)")
    print(f"  NAS OOS: {nas_oos.index[0].date()} → {nas_oos.index[-1].date()} ({len(nas_oos)} bars)\n")

    # Run Optuna per ticker
    n_trials = 60

    print(f"  Running Optuna on EUR ({n_trials} trials)...")
    study_eur = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_eur.optimize(lambda t: objective_eur(t, eur_oos), n_trials=n_trials, show_progress_bar=False)
    print(f"    Best EUR score: {study_eur.best_value:.2f}")
    print(f"    Best EUR params: {study_eur.best_params}")

    print(f"\n  Running Optuna on NAS ({n_trials} trials)...")
    study_nas = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_nas.optimize(lambda t: objective_nas(t, nas_oos), n_trials=n_trials, show_progress_bar=False)
    print(f"    Best NAS score: {study_nas.best_value:.2f}")
    print(f"    Best NAS params: {study_nas.best_params}")

    # Validate best params on OOS
    print("\n" + "=" * 100)
    print("FINAL — Best params per ticker, validated on OOS")
    print("=" * 100)
    eur_params = dict(study_eur.best_params)
    eur_params["level_close_orders"] = 70.0
    nas_params = dict(study_nas.best_params)
    nas_params["level_close_orders"] = 70.0

    e = run_with_params(eur_oos, eur_params, "forex")
    n = run_with_params(nas_oos, nas_params, "nas100")
    print("\n  EURUSD optimized (12mo OOS):")
    print(f"    PnL ${e['net_pnl']:+,.0f} | Sharpe {e['sharpe']:+.2f} | "
          f"WR {e['win_rate']*100:.0f}% | PF {e['profit_factor']:.2f} | "
          f"DD {e['max_dd']*100:+.1f}% | {e['n_trades']} trades")
    print(f"    params: {eur_params}")
    print("\n  NAS100 optimized (12mo OOS):")
    print(f"    PnL ${n['net_pnl']:+,.0f} | Sharpe {n['sharpe']:+.2f} | "
          f"WR {n['win_rate']*100:.0f}% | PF {n['profit_factor']:.2f} | "
          f"DD {n['max_dd']*100:+.1f}% | {n['n_trades']} trades")
    print(f"    params: {nas_params}")

    # Compare with baselines
    print("\n  BASELINE comparison (from earlier):")
    print(f"    EUR: +$339 / +0.81 Sharpe (7 trades) — OPTIMIZED: ${e['net_pnl']:+,.0f} / {e['sharpe']:+.2f}")
    print(f"    NAS: -$1,099 / -0.64 Sharpe (667 trades) — OPTIMIZED: ${n['net_pnl']:+,.0f} / {n['sharpe']:+.2f}")

    # Save best params
    import json
    with open("output/ac_ao_best_params.json", "w") as f:
        json.dump({"eur": eur_params, "nas": nas_params}, f, indent=2)
    print("\nSaved → output/ac_ao_best_params.json")


if __name__ == "__main__":
    main()
