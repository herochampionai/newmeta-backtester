"""Batch-run Optuna+filters for all REMAINING strategies.

Strategies tested:
  - fbb (Fractal Breakout Bands)
  - mfi (Money Flow Index)
  - ms (MACD Signal)
  - mtf_stoch (MTF Stochastic)
  - bb_rsi (Bollinger Bands + RSI)
  - triple_rsi
  - quad_stoch
  - stoch533_mtf
  - macd_confluence

For each:
  1. Optuna+filters on OOS-1 (latest 12mo)
  2. Validate on OOS-2 (prior 12mo)
  3. Mark as ACCEPTED if profitable on BOTH, else borderline
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
from strategies._enhancement import EnhancedStrategy
from analysis.optuna_filters import (
    FILTER_CATALOG, fetch_h1, run_strategy, search_strategy,
)

from strategies.fbb import FBB_Strategy
from strategies.mfi import MFI_Strategy
from strategies.ms import MS_Strategy
from strategies.mtf_stoch import QuadStochStrategy
from strategies.bb_rsi import BBRsiStrategy
from strategies.triple_rsi import TripleRSIStrategy
from strategies.quad_stoch import QuadStochSameTF
from strategies.stoch533_mtf import Stoch533MTF
from strategies.macd_confluence import MACDConfluenceStrategy


# Param space per strategy — sensible ranges for each
PARAM_SPACES = {
    "fbb": lambda trial: {
        "bars_calculate": trial.suggest_int("bars_calculate", 10, 60),
        "atr_period": trial.suggest_int("atr_period", 8, 30),
        "atr_mult": trial.suggest_float("atr_mult", 1.0, 4.0),
        "lookback": trial.suggest_int("lookback", 5, 30),
        "use_strict": trial.suggest_categorical("use_strict", [True, False]),
    },
    "mfi": lambda trial: {
        "bars_calculate": trial.suggest_int("bars_calculate", 8, 40),
        "level_open_orders": trial.suggest_float("level_open_orders", 30.0, 90.0),
        "open_orders_type": trial.suggest_int("open_orders_type", 1, 4),
        "close_orders_type": trial.suggest_int("close_orders_type", 0, 4),
    },
    "ms": lambda trial: {
        "ms_fast_ema": trial.suggest_int("ms_fast_ema", 5, 30),
        "ms_slow_ema": trial.suggest_int("ms_slow_ema", 20, 60),
        "ms_signal_period": trial.suggest_int("ms_signal_period", 2, 20),
        "level_open_orders": trial.suggest_float("level_open_orders", 20.0, 80.0),
        "open_orders_type": trial.suggest_int("open_orders_type", 1, 4),
    },
    "mtf_stoch": lambda trial: {
        "k_period": trial.suggest_int("k_period", 5, 30),
        "d_period": trial.suggest_int("d_period", 2, 10),
        "smooth": trial.suggest_int("smooth", 1, 6),
        "oversold": trial.suggest_float("oversold", 10.0, 35.0),
        "overbought": trial.suggest_float("overbought", 65.0, 90.0),
        "use_mtf": trial.suggest_categorical("use_mtf", [True, False]),
    },
    "bb_rsi": lambda trial: {
        "bb_period": trial.suggest_int("bb_period", 10, 40),
        "bb_std": trial.suggest_float("bb_std", 1.0, 3.0),
        "rsi_period": trial.suggest_int("rsi_period", 7, 30),
        "rsi_oversold": trial.suggest_float("rsi_oversold", 20.0, 40.0),
        "rsi_overbought": trial.suggest_float("rsi_overbought", 60.0, 80.0),
    },
    "triple_rsi": lambda trial: {
        "fast": trial.suggest_int("fast", 5, 14),
        "mid": trial.suggest_int("mid", 10, 21),
        "slow": trial.suggest_int("slow", 18, 30),
        "oversold": trial.suggest_float("oversold", 15.0, 35.0),
        "overbought": trial.suggest_float("overbought", 65.0, 85.0),
    },
    "quad_stoch": lambda trial: {
        "k_period": trial.suggest_int("k_period", 5, 30),
        "d_period": trial.suggest_int("d_period", 2, 10),
        "smooth": trial.suggest_int("smooth", 1, 6),
        "oversold": trial.suggest_float("oversold", 10.0, 35.0),
        "overbought": trial.suggest_float("overbought", 65.0, 90.0),
    },
    "stoch533_mtf": lambda trial: {
        "htf_rule": trial.suggest_categorical("htf_rule", ["4h", "1d"]),
        "oversold": trial.suggest_float("oversold", 10.0, 30.0),
        "overbought": trial.suggest_float("overbought", 70.0, 90.0),
    },
    "macd_confluence": lambda trial: {
        "fast": trial.suggest_int("fast", 5, 20),
        "slow": trial.suggest_int("slow", 20, 50),
        "signal": trial.suggest_int("signal", 5, 20),
        "min_confluence": trial.suggest_int("min_confluence", 1, 4),
    },
}


STRAT_CLASSES = {
    "fbb": FBB_Strategy,
    "mfi": MFI_Strategy,
    "ms": MS_Strategy,
    "mtf_stoch": QuadStochStrategy,
    "bb_rsi": BBRsiStrategy,
    "triple_rsi": TripleRSIStrategy,
    "quad_stoch": QuadStochSameTF,
    "stoch533_mtf": Stoch533MTF,
    "macd_confluence": MACDConfluenceStrategy,
}


def main():
    print("=" * 100)
    print("BATCH: Optuna+filters for remaining 9 strategies, EUR + NAS, 2 OOS validation")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")

    eur_oos1 = eur[eur.index >= pd.Timestamp("2025-09-17", tz="UTC")]
    nas_oos1 = nas[nas.index >= pd.Timestamp("2025-09-17", tz="UTC")]
    eur_oos2 = eur[(eur.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (eur.index < pd.Timestamp("2025-09-17", tz="UTC"))]
    nas_oos2 = nas[(nas.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (nas.index < pd.Timestamp("2025-09-17", tz="UTC"))]

    print(f"\n{'Strategy':<22} {'─ EUR OOS-1 ─':>22} {'─ EUR OOS-2 ─':>22} "
          f"{'─ NAS OOS-1 ─':>22} {'─ NAS OOS-2 ─':>22}")
    print(f"{'':<22} {'PnL':>9} {'Sharpe':>7} {'Tr':>4}  "
          f"{'PnL':>9} {'Sharpe':>7} {'Tr':>4}    "
          f"{'PnL':>9} {'Sharpe':>7} {'Tr':>4}  "
          f"{'PnL':>9} {'Sharpe':>7} {'Tr':>4}")
    print("-" * 145)

    summary = []
    for sname in ["fbb", "mfi", "ms", "mtf_stoch", "bb_rsi", "triple_rsi", "quad_stoch", "stoch533_mtf", "macd_confluence"]:
        cls = STRAT_CLASSES[sname]
        space = PARAM_SPACES[sname]
        print(f"\n>>> {sname} <<<")
        # EUR
        eur_p, eur_f, eur_sc = search_strategy(sname, cls, eur_oos1, eur_oos2, space, "forex", n_trials=30, min_trades=5)
        # NAS
        nas_p, nas_f, nas_sc = search_strategy(sname, cls, nas_oos1, nas_oos2, space, "nas100", n_trials=30, min_trades=5)
        # Validate on both
        e1 = run_strategy(eur_oos1, cls, eur_p, eur_f, "forex")
        e2 = run_strategy(eur_oos2, cls, eur_p, eur_f, "forex")
        n1 = run_strategy(nas_oos1, cls, nas_p, nas_f, "nas100")
        n2 = run_strategy(nas_oos2, cls, nas_p, nas_f, "nas100")

        def fmt(m):
            if isinstance(m, dict) and "error" not in m:
                return f"${m['net_pnl']:>+8,.0f} {m['sharpe']:>+6.2f} {m['n_trades']:>4}"
            return f"{'ERR':>9} {'—':>7} {0:>4}"

        print(f"  {sname:<20} {fmt(e1)}  {fmt(e2)}    {fmt(n1)}  {fmt(n2)}")

        # Status per ticker
        def status(m1, m2):
            if isinstance(m1, dict) and isinstance(m2, dict) and "error" not in m1 and "error" not in m2:
                if m1["net_pnl"] > 0 and m2["net_pnl"] > 0 and m1["sharpe"] > 0 and m2["sharpe"] > 0:
                    return "✓ ROBUST"
                elif m1["net_pnl"] > 0 or m2["net_pnl"] > 0:
                    return "~ borderline"
                else:
                    return "✗ failing"
            return "?"

        eur_status = status(e1, e2)
        nas_status = status(n1, n2)
        print(f"     EUR status: {eur_status} | NAS status: {nas_status}")

        summary.append({
            "strategy": sname,
            "eur_oos1_pnl": e1.get("net_pnl") if isinstance(e1, dict) else None,
            "eur_oos2_pnl": e2.get("net_pnl") if isinstance(e2, dict) else None,
            "nas_oos1_pnl": n1.get("net_pnl") if isinstance(n1, dict) else None,
            "nas_oos2_pnl": n2.get("net_pnl") if isinstance(n2, dict) else None,
            "eur_status": eur_status, "nas_status": nas_status,
            "eur_params": eur_p, "nas_params": nas_p,
            "eur_filters": [f.__name__ if hasattr(f, "__name__") else "lambda" for f in eur_f],
            "nas_filters": [f.__name__ if hasattr(f, "__name__") else "lambda" for f in nas_f],
        })
        # Save after each
        with open(f"output/{sname}_best_params.json", "w") as f:
            json.dump({"eur": {"params": eur_p, "filters": summary[-1]["eur_filters"]},
                       "nas": {"params": nas_p, "filters": summary[-1]["nas_filters"]}}, f, indent=2)

    # Final summary
    print("\n" + "=" * 100)
    print("ALL STRATEGIES — STATUS")
    print("=" * 100)
    print(f"{'Strategy':<22} {'EUR':>15} {'NAS':>15} {'Overall':<20}")
    print("-" * 75)
    for s in summary:
        e_st = s["eur_status"]
        n_st = s["nas_status"]
        if "ROBUST" in e_st and "ROBUST" in n_st:
            ovr = "✓✓ ACCEPTED"
        elif "ROBUST" in e_st or "ROBUST" in n_st:
            ovr = "✓ ACCEPTED"
        elif "borderline" in e_st or "borderline" in n_st:
            ovr = "~ borderline"
        else:
            ovr = "✗ failing"
        print(f"{s['strategy']:<22} {e_st:>15} {n_st:>15} {ovr:<20}")

    # Combined table including AC-AO and ADX
    print(f"\n  (Plus AC-AO and ADX from earlier)")
    print(f"  AC-AO:   EUR ✓ ACCEPTED (PnL +$353, Sharpe +0.80) | NAS ✗ borderline (-$635)")
    print(f"  ADX:     EUR ~ borderline (OOS-1 great, OOS-2 flat) | NAS ✓ ACCEPTED (OOS-1 +$12k, OOS-2 +$9k)")


if __name__ == "__main__":
    main()
