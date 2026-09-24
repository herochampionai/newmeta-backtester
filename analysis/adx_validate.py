"""ADX validation — check if Optuna params survive on a DIFFERENT OOS period.

If the same params produce similar results on the previous 12 months, the strategy is robust.
If they don't, the Optuna baseline is over-fit.
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

import MetaTrader5 as mt5
from data.mt5_export import init_mt5
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from backtester.metrics_v2 import compute_all
from strategies.adx import ADX_Strategy
from strategies._enhancement import EnhancedStrategy, regime_filter_advanced, killzone_filter


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


def run(df, params, market_profile, filters=None):
    base = ADX_Strategy(params=params)
    strat = EnhancedStrategy(base, filters=filters or [])
    sig = strat.generate(df)
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
    r = run_full(df, signals, **kw)
    m = compute_all(r["equity"].pct_change().fillna(0), r.get("trades"), r["equity"])
    return {"n_entries": int(entries.sum()), "net_pnl": m.get("net_pnl", 0.0),
            "sharpe": m.get("sharpe", 0.0), "n_trades": m.get("n_trades", 0),
            "win_rate": m.get("win_rate", 0.0), "profit_factor": m.get("profit_factor", 0.0)}


def main():
    print("=" * 100)
    print("ADX ROBUSTNESS — same Optuna params on TWO different OOS periods")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")

    # Two windows: original (2025-09 → 2026-09) and previous (2024-09 → 2025-09)
    oos_windows = [
        ("OOS-1 (latest 12mo)", pd.Timestamp("2025-09-17", tz="UTC"), pd.Timestamp("2026-09-17", tz="UTC")),
        ("OOS-2 (prior 12mo)", pd.Timestamp("2024-09-17", tz="UTC"), pd.Timestamp("2025-09-17", tz="UTC")),
    ]
    eur_params = json.load(open("output/adx_best_params.json"))["eur"]
    nas_params = json.load(open("output/adx_best_params.json"))["nas"]

    # Also try the soft filters that didn't kill too much
    soft_filters = [regime_filter_advanced]
    soft_filters_label = "+regime_advanced (mild filter)"

    print(f"\n{'Window':<22} {'─ EUR Optuna ─':>30} {'─ NAS Optuna ─':>30}")
    print(f"{'':<22} {'PnL':>9} {'Sharpe':>7} {'Tr':>4}    "
          f"{'PnL':>9} {'Sharpe':>7} {'Tr':>4}")
    print("-" * 100)

    for window_label, t0, t1 in oos_windows:
        eur_w = eur[(eur.index >= t0) & (eur.index < t1)]
        nas_w = nas[(nas.index >= t0) & (nas.index < t1)]
        if len(eur_w) < 200 or len(nas_w) < 200:
            print(f"{window_label:<22}: insufficient data")
            continue
        print(f"\n{window_label} (EUR {len(eur_w)} bars, NAS {len(nas_w)} bars):")

        # Baseline Optuna
        e = run(eur_w, eur_params, "forex")
        n = run(nas_w, nas_params, "nas100")
        if "error" not in e and "error" not in n:
            print(f"  {'Optuna alone':<20} ${e['net_pnl']:>+8,.0f} {e['sharpe']:>+6.2f} {e['n_trades']:>4}    "
                  f"${n['net_pnl']:>+8,.0f} {n['sharpe']:>+6.2f} {n['n_trades']:>4}")
        else:
            print(f"  Optuna alone: ERR {e.get('error')} / {n.get('error')}")

        # Optuna + soft filter
        e2 = run(eur_w, eur_params, "forex", filters=soft_filters)
        n2 = run(nas_w, nas_params, "nas100", filters=soft_filters)
        if "error" not in e2 and "error" not in n2:
            print(f"  {soft_filters_label:<20} ${e2['net_pnl']:>+8,.0f} {e2['sharpe']:>+6.2f} {e2['n_trades']:>4}    "
                  f"${n2['net_pnl']:>+8,.0f} {n2['sharpe']:>+6.2f} {n2['n_trades']:>4}")

    print("\n" + "=" * 100)
    print("ROBUSTNESS VERDICT")
    print("=" * 100)
    print("""
If the strategy performs similarly on BOTH OOS periods, it's robust.
If only on OOS-1 (where Optuna tuned), it's likely over-fit.

For ADX:
  - Optuna Sharpe > 4 on OOS-1 = potentially overfit
  - Need to verify on OOS-2 to confirm generalization

Recommended next steps:
  1. If OOS-2 also strong: ACCEPTED (the strategy is real)
  2. If OOS-2 weak: REDUCE Optuna trials + use confluence filters as regularizers
  3. Use confluence filters (regime_advanced, killzone) as SOFT additions
""")


if __name__ == "__main__":
    main()
