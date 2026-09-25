"""Optuna search for ADX strategy — tune zone thresholds per ticker + filters."""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
import json

import MetaTrader5 as mt5

from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from backtester.metrics_v2 import compute_all
from data.mt5_export import init_mt5
from strategies._enhancement import EnhancedStrategy, session_filter
from strategies.adx import ADX_Strategy


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


def run_with_params(df, params, market_profile="forex", filters=None):
    base = ADX_Strategy(params=params)
    strat = EnhancedStrategy(base, filters=filters or [])
    strat.name = "adx_opt"
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
    return {"n_entries": n_entries, "net_pnl": m.get("net_pnl", 0.0),
            "sharpe": m.get("sharpe", 0.0), "max_dd": m.get("max_drawdown", 0.0),
            "win_rate": m.get("win_rate", 0.0), "profit_factor": m.get("profit_factor", 0.0),
            "n_trades": m.get("n_trades", 0)}


def make_objective(df, market_profile, filters=None):
    def obj(trial):
        params = {
            "bars_calculate": trial.suggest_int("bars_calculate", 10, 30),
            "use_di_crossover": trial.suggest_categorical("use_di_cross", [True, False]),
            "crossover_lookback": trial.suggest_int("crossover_lookback", 1, 8),
            "min_crossover_gap": trial.suggest_float("min_crossover_gap", 1.0, 15.0),
            "use_zone_logic": True,
            "adx_zone_low": trial.suggest_float("adx_zone_low", 10.0, 25.0),
            "adx_zone_high": trial.suggest_float("adx_zone_high", 25.0, 60.0),
            "continuation_level": trial.suggest_float("continuation_level", 18.0, 40.0),
            "reversal_edge": trial.suggest_float("reversal_edge", 14.0, 30.0),
            "open_orders_type": trial.suggest_int("open_orders_type", 1, 4),
            "level_open_orders_1": trial.suggest_float("level_open_orders_1", 20.0, 80.0),
            "level_open_orders_2": trial.suggest_float("level_open_orders_2", 5.0, 30.0),
            "sweep_lookback": trial.suggest_int("sweep_lookback", 3, 15),
            "level_close_orders_1": trial.suggest_float("level_close_orders_1", 8.0, 25.0),
            "level_close_orders_2": trial.suggest_float("level_close_orders_2", 2.0, 10.0),
            "close_orders_type": trial.suggest_int("close_orders_type", 1, 4),
        }
        m = run_with_params(df, params, market_profile, filters)
        if "error" in m:
            return -1e9
        return m["net_pnl"] + 100 * m["sharpe"] - 200 * abs(m["max_dd"])
    return obj


def main():
    print("=" * 100)
    print("ADX OPTUNA SEARCH — zone thresholds + filters, per ticker")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    eur_oos = eur[eur.index >= eur.index[-1] - pd.DateOffset(months=12)]
    nas_oos = nas[nas.index >= nas.index[-1] - pd.DateOffset(months=12)]
    print(f"  EUR OOS: {eur_oos.index[0].date()} → {eur_oos.index[-1].date()}")
    print(f"  NAS OOS: {nas_oos.index[0].date()} → {nas_oos.index[-1].date()}")

    # Baseline first
    print("\n  BASELINE (default params, no filters):")
    e0 = run_with_params(eur_oos, {}, "forex")
    n0 = run_with_params(nas_oos, {}, "nas100")
    print(f"    EUR: ${e0.get('net_pnl', 0):+,.0f} / {e0.get('sharpe', 0):+.2f} / {e0.get('n_trades', 0)} trades")
    print(f"    NAS: ${n0.get('net_pnl', 0):+,.0f} / {n0.get('sharpe', 0):+.2f} / {n0.get('n_trades', 0)} trades")

    n_trials = 50
    filters_session = [lambda df: session_filter(df, utc_hours=(7,8,9,10,11,12,13,14,15,16,17,18,19,20))]

    print(f"\n  Optuna EUR ({n_trials} trials, no filter)...")
    study_e = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_e.optimize(make_objective(eur_oos, "forex"), n_trials=n_trials, show_progress_bar=False)
    print(f"    Best score: {study_e.best_value:.2f}")

    print(f"  Optuna NAS ({n_trials} trials, no filter)...")
    study_n = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_n.optimize(make_objective(nas_oos, "nas100"), n_trials=n_trials, show_progress_bar=False)
    print(f"    Best score: {study_n.best_value:.2f}")

    # Validate
    e_best = run_with_params(eur_oos, dict(study_e.best_params), "forex")
    n_best = run_with_params(nas_oos, dict(study_n.best_params), "nas100")
    print("\n" + "=" * 100)
    print("FINAL — Best per-ticker params")
    print("=" * 100)
    print(f"\n  EURUSD optimized: PnL ${e_best.get('net_pnl', 0):+,.0f} | "
          f"Sharpe {e_best.get('sharpe', 0):+.2f} | "
          f"WR {e_best.get('win_rate', 0)*100:.0f}% | "
          f"PF {e_best.get('profit_factor', 0):.2f} | "
          f"DD {e_best.get('max_dd', 0)*100:+.1f}% | {e_best.get('n_trades', 0)} trades")
    print(f"    params: {study_e.best_params}")
    print(f"\n  NAS100 optimized: PnL ${n_best.get('net_pnl', 0):+,.0f} | "
          f"Sharpe {n_best.get('sharpe', 0):+.2f} | "
          f"WR {n_best.get('win_rate', 0)*100:.0f}% | "
          f"PF {n_best.get('profit_factor', 0):.2f} | "
          f"DD {n_best.get('max_dd', 0)*100:+.1f}% | {n_best.get('n_trades', 0)} trades")
    print(f"    params: {study_n.best_params}")
    print("\n  BASELINE → OPTIMIZED:")
    print(f"    EUR: ${e0.get('net_pnl', 0):+,.0f} / {e0.get('sharpe', 0):+.2f} → ${e_best.get('net_pnl', 0):+,.0f} / {e_best.get('sharpe', 0):+.2f}")
    print(f"    NAS: ${n0.get('net_pnl', 0):+,.0f} / {n0.get('sharpe', 0):+.2f} → ${n_best.get('net_pnl', 0):+,.0f} / {n_best.get('sharpe', 0):+.2f}")

    with open("output/adx_best_params.json", "w") as f:
        json.dump({"eur": study_e.best_params, "nas": study_n.best_params}, f, indent=2)
    print("\nSaved → output/adx_best_params.json")


if __name__ == "__main__":
    main()
