"""Test new confluence filters on top of the Optuna-tuned ADX strategy.

Goal: improve robustness WITHOUT killing too many opportunities.
Each filter is tested individually, then best ones combined.
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
from strategies._enhancement import (
    EnhancedStrategy,
    # Old filters
    regime_adx, session_filter, volatility_atr, candle_filter, mtf_trend,
    # New confluence filters
    market_context_pullback, force_pullback, pivot_points, zigzag_swings,
    elliott_wave_proxy, regime_filter_advanced, vwap_distance, trend_strength,
    confluence_score, fibonacci_levels, fvg_bullish, fvg_bearish,
    ifvg_bullish, ifvg_bearish, mss_bullish, mss_bearish, killzone_filter,
    volume_increase, momentum_increase, divergence_bullish, divergence_bearish,
)


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


def run_strategy(df, params, filters, market_profile):
    base = ADX_Strategy(params=params)
    strat = EnhancedStrategy(base, filters=filters)
    strat.name = "adx_test"
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


def main():
    print("=" * 100)
    print("ADX — Optuna params + new confluence filters (EUR + NAS)")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    eur_oos = eur[eur.index >= eur.index[-1] - pd.DateOffset(months=12)]
    nas_oos = nas[nas.index >= nas.index[-1] - pd.DateOffset(months=12)]
    print(f"  EUR OOS: {len(eur_oos)} bars | NAS OOS: {len(nas_oos)} bars")

    eur_params = json.load(open("output/adx_best_params.json"))["eur"]
    nas_params = json.load(open("output/adx_best_params.json"))["nas"]

    # Define filter variants — each tested on top of Optuna params
    variants = [
        ("OPTUNA (baseline)", []),
        # New confluence filters (individual)
        ("+market_context_pullback", [market_context_pullback]),
        ("+force_pullback(0.5%)", [lambda df: force_pullback(df, pullback_pct=0.005)]),
        ("+pivot_points", [pivot_points]),
        ("+zigzag_swings", [zigzag_swings]),
        ("+elliott_wave", [elliott_wave_proxy]),
        ("+regime_advanced", [regime_filter_advanced]),
        ("+vwap_distance", [vwap_distance]),
        ("+trend_strength", [trend_strength]),
        ("+killzone(LDN+NY)", [killzone_filter]),
        # Combinations — soft layering
        ("+pullback+context", [market_context_pullback, force_pullback]),
        ("+regime+context", [regime_filter_advanced, market_context_pullback]),
        ("+trend+pullback", [trend_strength, force_pullback]),
        ("+all pullback-family", [market_context_pullback, force_pullback, trend_strength, regime_filter_advanced]),
    ]

    print(f"\n{'Filter variant':<30} {'─ EUR ─':>30} {'─ NAS ─':>30}")
    print(f"{'':<30} {'PnL':>9} {'Sharpe':>7} {'Tr':>4}  {'PnL':>9} {'Sharpe':>7} {'Tr':>4}")
    print("-" * 100)

    results = []
    for label, filters in variants:
        e = run_strategy(eur_oos, eur_params, filters, "forex")
        n = run_strategy(nas_oos, nas_params, filters, "nas100")
        results.append({"label": label, "filters": [f.__name__ if hasattr(f, "__name__") else str(f) for f in filters], "eur": e, "nas": n})
        if isinstance(e, dict) and "error" not in e and isinstance(n, dict) and "error" not in n:
            print(f"{label:<30} ${e['net_pnl']:>+8,.0f} {e['sharpe']:>+6.2f} {e['n_trades']:>4}  "
                  f"${n['net_pnl']:>+8,.0f} {n['sharpe']:>+6.2f} {n['n_trades']:>4}")
        else:
            err_e = (e or {}).get("error", "?") if isinstance(e, dict) else "?"
            err_n = (n or {}).get("error", "?") if isinstance(n, dict) else "?"
            print(f"{label:<30} ERR_EUR={err_e[:15]} ERR_NAS={err_n[:15]}")

    # Verdict
    print("\n" + "=" * 100)
    print("VERDICT — filter impact on ADX Optuna params")
    print("=" * 100)
    print(f"{'Filter':<30} {'EUR ΔPnL':>12} {'EUR ΔTr':>10} {'NAS ΔPnL':>12} {'NAS ΔTr':>10} {'Combined Δ':>14}")
    print("-" * 100)
    base_e = results[0]["eur"]
    base_n = results[0]["nas"]
    base_total = base_e["net_pnl"] + base_n["net_pnl"]
    base_trades_e = base_e["n_trades"]
    base_trades_n = base_n["n_trades"]
    print(f"{'(baseline Optuna)':<30} ${base_e['net_pnl']:>+10,.0f} {base_trades_e:>10} "
          f"${base_n['net_pnl']:>+10,.0f} {base_trades_n:>10} ${base_total:>+12,.0f}")

    best = {"label": "(baseline)", "total": base_total, "filters": []}
    for r in results[1:]:
        e = r["eur"]; n = r["nas"]
        if "error" in e or "error" in n:
            continue
        d_pnl_e = e["net_pnl"] - base_e["net_pnl"]
        d_pnl_n = n["net_pnl"] - base_n["net_pnl"]
        d_tr_e = e["n_trades"] - base_trades_e
        d_tr_n = n["n_trades"] - base_trades_n
        total = e["net_pnl"] + n["net_pnl"]
        marker = "✓" if (e["net_pnl"] > 0 and e["sharpe"] > 0 and n["net_pnl"] > 0 and n["sharpe"] > 0) else ""
        # Also penalize if trade count drops too much (>50% reduction)
        tr_e_drop_pct = abs(d_tr_e) / max(base_trades_e, 1) * 100
        tr_n_drop_pct = abs(d_tr_n) / max(base_trades_n, 1) * 100
        too_restrictive = (tr_e_drop_pct > 50 or tr_n_drop_pct > 50) and d_pnl_e + d_pnl_n <= 0
        if too_restrictive:
            marker = "✗ killed too much"
        print(f"{r['label']:<30} ${d_pnl_e:>+10,.0f} {d_tr_e:>+10} ${d_pnl_n:>+10,.0f} {d_tr_n:>+10} ${total-base_total:>+12,.0f} {marker}")
        if total > best["total"] and not too_restrictive:
            best = {"label": r["label"], "total": total, "filters": r["filters"]}

    print(f"\n  BEST non-restrictive variant: {best['label']} (combined PnL ${best['total']:+,.0f})")
    if best["filters"]:
        print(f"  Filters: {best['filters']}")

    # Save results
    pd.DataFrame([
        {"filter": r["label"],
         "eur_pnl": r["eur"].get("net_pnl") if isinstance(r["eur"], dict) else None,
         "eur_sharpe": r["eur"].get("sharpe") if isinstance(r["eur"], dict) else None,
         "eur_n": r["eur"].get("n_trades") if isinstance(r["eur"], dict) else None,
         "nas_pnl": r["nas"].get("net_pnl") if isinstance(r["nas"], dict) else None,
         "nas_sharpe": r["nas"].get("sharpe") if isinstance(r["nas"], dict) else None,
         "nas_n": r["nas"].get("n_trades") if isinstance(r["nas"], dict) else None,
        } for r in results
    ]).to_csv("output/adx_confluence.csv", index=False)
    print(f"\nSaved → output/adx_confluence.csv")


if __name__ == "__main__":
    main()
