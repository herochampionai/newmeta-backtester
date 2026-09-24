"""Test V2 rewritten strategies on 2 OOS periods."""
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
from strategies.triple_rsi_v2 import TripleRSIV2Strategy
from strategies.stoch533_mtf_v2 import Stoch533MTFV2


def main():
    print("=" * 100)
    print("V2 REWRITTEN STRATEGIES — validation on 2 OOS periods")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    eur_oos1 = eur[eur.index >= pd.Timestamp("2025-09-17", tz="UTC")]
    nas_oos1 = nas[nas.index >= pd.Timestamp("2025-09-17", tz="UTC")]
    eur_oos2 = eur[(eur.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (eur.index < pd.Timestamp("2025-09-17", tz="UTC"))]
    nas_oos2 = nas[(nas.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (nas.index < pd.Timestamp("2025-09-17", tz="UTC"))]

    strats = {
        "triple_rsi_v2": TripleRSIV2Strategy,
        "stoch533_mtf_v2": Stoch533MTFV2,
    }
    # A few param variants to try
    variants = {
        "triple_rsi_v2": [
            ("default", {}),
            ("tight_thresholds", {"oversold": 25, "overbought": 75, "cooldown_bars": 8}),
            ("loose+cooldown", {"oversold": 35, "overbought": 65, "cooldown_bars": 10}),
            ("no_mtf", {"use_mtf_confirm": False, "cooldown_bars": 8}),
        ],
        "stoch533_mtf_v2": [
            ("default", {}),
            ("tighter_zone", {"oversold": 15, "overbought": 85, "cooldown_bars": 8}),
            ("looser_zone", {"oversold": 25, "overbought": 75, "cooldown_bars": 10}),
            ("with_div", {"use_divergence": True, "cooldown_bars": 8}),
        ],
    }

    print(f"\n{'Strategy':<22} {'Variant':<22} {'─ EUR OOS-1 ─':>22} {'─ EUR OOS-2 ─':>22} {'─ NAS OOS-1 ─':>22} {'─ NAS OOS-2 ─':>22}")
    print(f"{'':<22} {'':<22} {'PnL':>9} {'Sharpe':>7} {'Tr':>4}    {'PnL':>9} {'Sharpe':>7} {'Tr':>4}    {'PnL':>9} {'Sharpe':>7} {'Tr':>4}    {'PnL':>9} {'Sharpe':>7} {'Tr':>4}")
    print("-" * 160)

    for sname, cls in strats.items():
        for vlabel, params in variants[sname]:
            e1 = run_strategy(eur_oos1, cls, params, [], "forex")
            e2 = run_strategy(eur_oos2, cls, params, [], "forex")
            n1 = run_strategy(nas_oos1, cls, params, [], "nas100")
            n2 = run_strategy(nas_oos2, cls, params, [], "nas100")

            def fmt(m):
                if isinstance(m, dict) and "error" not in m:
                    return f"${m['net_pnl']:>+8,.0f} {m['sharpe']:>+6.2f} {m['n_trades']:>4}"
                return f"{'ERR':>9} {'—':>7} {0:>4}"

            print(f"{sname:<22} {vlabel:<22} {fmt(e1)}    {fmt(e2)}    {fmt(n1)}    {fmt(n2)}")

    # Status
    print("\n" + "=" * 100)
    print("STATUS — best per strategy")
    print("=" * 100)


if __name__ == "__main__":
    main()
