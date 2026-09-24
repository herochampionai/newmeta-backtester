"""Retry the failed strategies with default params + selective filters.

For each strategy that previously got "no entries" in Optuna batch, try:
  1. Default params with NO filter
  2. Default params with each filter individually (regime + killzone only — least restrictive)
  3. If nothing works, report the strategy as fundamentally limited on the data
"""
from __future__ import annotations
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json

from analysis.optuna_filters import fetch_h1, run_strategy
from strategies.ms import MS_Strategy
from strategies.mtf_stoch import QuadStochStrategy
from strategies.triple_rsi import TripleRSIStrategy
from strategies.quad_stoch import QuadStochSameTF
from strategies.stoch533_mtf import Stoch533MTF
from strategies.macd_confluence import MACDConfluenceStrategy
from strategies._enhancement import (
    regime_filter_advanced, killzone_filter, market_context_pullback,
    mtf_trend, regime_adx, session_filter,
)


STRAT_CLASSES = {
    "ms": MS_Strategy,
    "mtf_stoch": QuadStochStrategy,
    "triple_rsi": TripleRSIStrategy,
    "quad_stoch": QuadStochSameTF,
    "stoch533_mtf": Stoch533MTF,
    "macd_confluence": MACDConfluenceStrategy,
}


def main():
    print("=" * 100)
    print("RETRY — default params + selective filters for 6 failed strategies")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    eur_oos1 = eur[eur.index >= pd.Timestamp("2025-09-17", tz="UTC")]
    nas_oos1 = nas[nas.index >= pd.Timestamp("2025-09-17", tz="UTC")]
    eur_oos2 = eur[(eur.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (eur.index < pd.Timestamp("2025-09-17", tz="UTC"))]
    nas_oos2 = nas[(nas.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (nas.index < pd.Timestamp("2025-09-17", tz="UTC"))]

    filter_variants = [
        ("default (no filter)", []),
        ("regime_advanced", [regime_filter_advanced]),
        ("killzone LDN+NY", [lambda df: killzone_filter(df, utc_hours=(7,8,9,10,11,12,13,14,15,16,17,18,19,20))]),
        ("mtf_trend", [mtf_trend]),
    ]

    print(f"\n{'Strategy':<22} {'Filter':<22} {'─ EUR OOS-1 ─':>22} {'─ EUR OOS-2 ─':>22} {'─ NAS OOS-1 ─':>22} {'─ NAS OOS-2 ─':>22}")
    print(f"{'':<22} {'':<22} {'PnL':>9} {'Sharpe':>7} {'Tr':>4}    {'PnL':>9} {'Sharpe':>7} {'Tr':>4}    {'PnL':>9} {'Sharpe':>7} {'Tr':>4}    {'PnL':>9} {'Sharpe':>7} {'Tr':>4}")
    print("-" * 160)

    summary = {}
    for sname, cls in STRAT_CLASSES.items():
        summary[sname] = {}
        for flabel, filters in filter_variants:
            e1 = run_strategy(eur_oos1, cls, {}, filters, "forex")
            e2 = run_strategy(eur_oos2, cls, {}, filters, "forex")
            n1 = run_strategy(nas_oos1, cls, {}, filters, "nas100")
            n2 = run_strategy(nas_oos2, cls, {}, filters, "nas100")

            def fmt(m):
                if isinstance(m, dict) and "error" not in m:
                    return f"${m['net_pnl']:>+8,.0f} {m['sharpe']:>+6.2f} {m['n_trades']:>4}"
                return f"{'ERR':>9} {'—':>7} {0:>4}"

            print(f"{sname:<22} {flabel:<22} {fmt(e1)}    {fmt(e2)}    {fmt(n1)}    {fmt(n2)}")
            summary[sname][flabel] = {"eur_oos1": e1, "eur_oos2": e2,
                                        "nas_oos1": n1, "nas_oos2": n2}

    # Status per strategy
    print("\n" + "=" * 100)
    print("STATUS — best per strategy per ticker")
    print("=" * 100)
    for sname, results in summary.items():
        # Find best EUR and best NAS
        best_eur = None
        best_nas = None
        for flabel, r in results.items():
            if isinstance(r["eur_oos1"], dict) and isinstance(r["eur_oos2"], dict) and "error" not in r["eur_oos1"] and "error" not in r["eur_oos2"]:
                if r["eur_oos1"]["net_pnl"] > 0 and r["eur_oos2"]["net_pnl"] > 0:
                    if best_eur is None or r["eur_oos1"]["net_pnl"] + r["eur_oos2"]["net_pnl"] > best_eur[1]:
                        best_eur = (flabel, r["eur_oos1"]["net_pnl"] + r["eur_oos2"]["net_pnl"])
            if isinstance(r["nas_oos1"], dict) and isinstance(r["nas_oos2"], dict) and "error" not in r["nas_oos1"] and "error" not in r["nas_oos2"]:
                if r["nas_oos1"]["net_pnl"] > 0 and r["nas_oos2"]["net_pnl"] > 0:
                    if best_nas is None or r["nas_oos1"]["net_pnl"] + r["nas_oos2"]["net_pnl"] > best_nas[1]:
                        best_nas = (flabel, r["nas_oos1"]["net_pnl"] + r["nas_oos2"]["net_pnl"])

        e_st = f"✓ {best_eur[0]} (+${best_eur[1]:,.0f} combined)" if best_eur else "✗ failing both"
        n_st = f"✓ {best_nas[0]} (+${best_nas[1]:,.0f} combined)" if best_nas else "✗ failing both"
        print(f"  {sname:<22} EUR: {e_st}")
        print(f"  {sname:<22} NAS: {n_st}")


if __name__ == "__main__":
    main()
