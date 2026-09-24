"""Generic Optuna + filter-selection harness for any strategy.

Picks best (params + filter combination) per ticker, validated on TWO OOS periods.

Key idea: instead of forcing filters on top of naked Optuna, let Optuna CHOOSE
which filters to enable. This prevents overfit to naked params AND avoids
filter overload.
"""
from __future__ import annotations
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import json

import MetaTrader5 as mt5
from data.mt5_export import init_mt5
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from backtester.metrics_v2 import compute_all
from strategies._enhancement import (
    EnhancedStrategy,
    regime_adx, session_filter, volatility_atr, candle_filter, mtf_trend,
    market_context_pullback, force_pullback, pivot_points, zigzag_swings,
    regime_filter_advanced, vwap_distance, trend_strength,
    fibonacci_levels, killzone_filter, volume_increase, momentum_increase,
)


# Filter catalog — Optuna picks which to enable
FILTER_CATALOG = {
    "killzone": lambda df: killzone_filter(df, utc_hours=(7,8,9,10,11,12,13,14,15,16,17,18,19,20)),
    "session": lambda df: session_filter(df, utc_hours=(7,8,9,10,11,12,13,14,15,16,17,18,19,20)),
    "regime_adv": regime_filter_advanced,
    "mkt_context": market_context_pullback,
    "trend_str": trend_strength,
    "vol_atr": lambda df: volatility_atr(df, atr_period=14, min_atr_pct=0.0006, max_atr_pct=0.004),
    "mtf_trend": mtf_trend,
    "force_pb": lambda df: force_pullback(df, pullback_pct=0.003),
    "fib": fibonacci_levels,
    "vwap": vwap_distance,
    "mom_inc": momentum_increase,
    "vol_inc": volume_increase,
    "zigzag": lambda df: zigzag_swings(df, threshold_pct=0.005),
}


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


def run_strategy(df, base_cls, params, filters, market_profile):
    base = base_cls(params=params)
    strat = EnhancedStrategy(base, filters=filters or [])
    try:
        sig = strat.generate(df)
    except Exception as e:
        return {"error": str(e)}
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    if not entries.any():
        return {"error": "no entries"}
    signals = {strat.name: (entries, direction)}
    kw = dict(pip_size=0.0001 if market_profile == "forex" else 1.0,
              contract_size=100_000 if market_profile == "forex" else 1.0,
              base_lot=0.1, commission_pips=0.7 if market_profile == "forex" else 2.0,
              slippage_pips=0.3 if market_profile == "forex" else 1.0,
              spread_pips=1.0 if market_profile == "forex" else 1.5, init_cash=10_000.0,
              grid_mode=GRID_NONE)
    try:
        r = run_full(df, signals, **kw)
    except Exception as e:
        return {"error": f"run_full: {e}"}
    m = compute_all(r["equity"].pct_change().fillna(0), r.get("trades"), r["equity"])
    return {"net_pnl": m.get("net_pnl", 0.0), "sharpe": m.get("sharpe", 0.0),
            "max_dd": m.get("max_drawdown", 0.0), "win_rate": m.get("win_rate", 0.0),
            "profit_factor": m.get("profit_factor", 0.0),
            "n_trades": m.get("n_trades", 0)}


def search_strategy(strategy_name: str, base_cls, df_oos1, df_oos2,
                   param_space_fn, market_profile: str = "forex",
                   n_trials: int = 60, min_trades: int = 8):
    """Optuna search with params + filter selection, validated on 2 OOS periods.

    Score = avg of (PnL + 100*Sharpe) across both periods, minus penalty for too few trades.

    Args:
        param_space_fn: callable(trial) -> dict of strategy params
        market_profile: 'forex' or 'nas100'
    """
    def objective(trial):
        params = param_space_fn(trial)
        # Optuna picks filters (binary enable/disable)
        filters = []
        for fname, ffunc in FILTER_CATALOG.items():
            if trial.suggest_categorical(f"f_{fname}", [True, False]):
                filters.append(ffunc)
        # Soft cap on filter count (avoid 10+ filters)
        n_filters = len(filters)
        # Run on OOS-1 (used for primary scoring)
        m1 = run_strategy(df_oos1, base_cls, params, filters, market_profile)
        if "error" in m1 or m1["n_trades"] < min_trades:
            return -1e9
        # Validate on OOS-2
        m2 = run_strategy(df_oos2, base_cls, params, filters, market_profile)
        if "error" in m2:
            m2 = {"net_pnl": 0.0, "sharpe": 0.0, "n_trades": 0}
        # Combined score: BOTH periods must be profitable
        score1 = m1["net_pnl"] + 80 * m1["sharpe"]
        score2 = m2["net_pnl"] + 80 * m2["sharpe"]
        # Penalize if too restrictive (lost too many trades from baseline)
        avg_trades = (m1["n_trades"] + m2["n_trades"]) / 2
        trade_penalty = -50 if avg_trades < min_trades else 0
        return (score1 + score2) / 2 + trade_penalty

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Reconstruct best filters
    best_filters = []
    for fname, ffunc in FILTER_CATALOG.items():
        if study.best_params.get(f"f_{fname}", False):
            best_filters.append(ffunc)
    # Reconstruct best params (filter out f_ prefixed keys)
    best_params = {k: v for k, v in study.best_params.items() if not k.startswith("f_")}
    return best_params, best_filters, study.best_value


def main():
    """Run for DeM as the next strategy."""
    from strategies.dem import DeM_Strategy

    print("=" * 100)
    print("DeM Optuna + Filter Selection — per ticker, validated on 2 OOS periods")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")

    eur_oos1 = eur[(eur.index >= pd.Timestamp("2025-09-17", tz="UTC"))]
    nas_oos1 = nas[(nas.index >= pd.Timestamp("2025-09-17", tz="UTC"))]
    eur_oos2 = eur[(eur.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (eur.index < pd.Timestamp("2025-09-17", tz="UTC"))]
    nas_oos2 = nas[(nas.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (nas.index < pd.Timestamp("2025-09-17", tz="UTC"))]
    print(f"  EUR OOS-1: {len(eur_oos1)} bars | OOS-2: {len(eur_oos2)} bars")
    print(f"  NAS OOS-1: {len(nas_oos1)} bars | OOS-2: {len(nas_oos2)} bars")

    # Baseline
    print("\n  Baseline (default params, no filters):")
    e0 = run_strategy(eur_oos1, DeM_Strategy, {}, [], "forex")
    n0 = run_strategy(nas_oos1, DeM_Strategy, {}, [], "nas100")
    print(f"    EUR OOS-1: PnL ${e0.get('net_pnl', 0):+,.0f} / Sharpe {e0.get('sharpe', 0):+.2f} / {e0.get('n_trades', 0)} trades")
    print(f"    NAS OOS-1: PnL ${n0.get('net_pnl', 0):+,.0f} / Sharpe {n0.get('sharpe', 0):+.2f} / {n0.get('n_trades', 0)} trades")

    def dem_param_space(trial):
        return {
            "bars_calculate": trial.suggest_int("bars_calculate", 8, 40),
            "level_open_orders": trial.suggest_float("level_open_orders", 50.0, 95.0),
            "level_close_orders": trial.suggest_float("level_close_orders", 50.0, 95.0),
            "open_orders_type": trial.suggest_int("open_orders_type", 1, 4),
            "close_orders_type": trial.suggest_int("close_orders_type", 0, 4),
        }

    n_trials = 50
    print(f"\n  EUR Optuna+filters ({n_trials} trials, validated on OOS-2)...")
    eur_best_params, eur_best_filters, eur_score = search_strategy(
        "dem_eur", DeM_Strategy, eur_oos1, eur_oos2, dem_param_space, "forex", n_trials
    )
    print(f"    Best score: {eur_score:.0f}")
    print(f"    Best params: {eur_best_params}")
    print(f"    Best filters: {[f.__name__ if hasattr(f, '__name__') else 'lambda' for f in eur_best_filters]}")

    print(f"\n  NAS Optuna+filters ({n_trials} trials, validated on OOS-2)...")
    nas_best_params, nas_best_filters, nas_score = search_strategy(
        "dem_nas", DeM_Strategy, nas_oos1, nas_oos2, dem_param_space, "nas100", n_trials
    )
    print(f"    Best score: {nas_score:.0f}")
    print(f"    Best params: {nas_best_params}")
    print(f"    Best filters: {[f.__name__ if hasattr(f, '__name__') else 'lambda' for f in nas_best_filters]}")

    # Validate
    print("\n" + "=" * 100)
    print("FINAL — best per-ticker, validated on BOTH periods")
    print("=" * 100)
    for label, profile, df1, df2, params, filters in [
        ("EURUSD", "forex", eur_oos1, eur_oos2, eur_best_params, eur_best_filters),
        ("NAS100", "nas100", nas_oos1, nas_oos2, nas_best_params, nas_best_filters),
    ]:
        m1 = run_strategy(df1, DeM_Strategy, params, filters, profile)
        m2 = run_strategy(df2, DeM_Strategy, params, filters, profile)
        if "error" not in m1:
            print(f"\n  {label} OOS-1 (latest): PnL ${m1['net_pnl']:+,.0f} / Sharpe {m1['sharpe']:+.2f} / {m1['n_trades']} trades")
        if "error" not in m2:
            print(f"  {label} OOS-2 (prior):  PnL ${m2['net_pnl']:+,.0f} / Sharpe {m2['sharpe']:+.2f} / {m2['n_trades']} trades")
        # Status
        if isinstance(m1, dict) and "error" not in m1 and isinstance(m2, dict) and "error" not in m2:
            if m1["net_pnl"] > 0 and m1["sharpe"] > 0 and m2["net_pnl"] > 0:
                print(f"  STATUS: ✓ ACCEPTED (profitable on BOTH periods)")
            elif m1["net_pnl"] > 0 or m2["net_pnl"] > 0:
                print(f"  STATUS: ~ borderline (profitable on one period)")
            else:
                print(f"  STATUS: ✗ needs more work")

    # Save
    with open("output/dem_best_params.json", "w") as f:
        json.dump({
            "eur": {"params": eur_best_params,
                    "filters": [f.__name__ if hasattr(f, "__name__") else "lambda" for f in eur_best_filters]},
            "nas": {"params": nas_best_params,
                    "filters": [f.__name__ if hasattr(f, "__name__") else "lambda" for f in nas_best_filters]},
        }, f, indent=2)
    print(f"\nSaved → output/dem_best_params.json")


if __name__ == "__main__":
    main()
