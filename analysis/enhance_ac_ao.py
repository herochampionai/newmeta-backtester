"""AC-AO enhancement harness.

Tries multiple enhancement combinations on AC-AO across EURUSD + NAS100.
Goal: find combination that improves OOS performance without grid.
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

import MetaTrader5 as mt5
from data.mt5_export import init_mt5
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from backtester.metrics_v2 import compute_all
from strategies import MULTI_STRAT_EA_REGISTRY
from strategies.ac_ao import AC_AO_Strategy
from strategies._enhancement import (
    EnhancedStrategy, regime_adx, session_filter, volatility_atr,
    candle_filter, mtf_trend, volume_filter, day_filter, first_last_hour,
    trend_persistence, rsi_filter, macd_confirm, bollinger_filter,
)


def fetch_h1(terminal: str, symbol: str, n_bars: int = 20000) -> pd.DataFrame:
    if not init_mt5(terminal):
        raise RuntimeError(f"init_mt5 failed for {terminal}")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n_bars)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    mt5.shutdown()
    return df


def run_strategy(df, strat, market_profile="forex"):
    try:
        sig = strat.generate(df)
    except Exception as e:
        return {"error": f"generate failed: {e}"}
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    n_entries = int(entries.sum())
    if n_entries == 0:
        return {"error": "no entries", "n_entries": 0}
    signals = {strat.name: (entries, direction)}
    if market_profile == "forex":
        kw = dict(pip_size=0.0001, contract_size=100_000, base_lot=0.1,
                  commission_pips=0.7, slippage_pips=0.3, spread_pips=1.0,
                  init_cash=10_000.0)
    else:
        kw = dict(pip_size=1.0, contract_size=1.0, base_lot=0.1,
                  commission_pips=2.0, slippage_pips=1.0, spread_pips=1.5,
                  init_cash=10_000.0)
    kw["grid_mode"] = GRID_NONE
    try:
        r = run_full(df, signals, **kw)
    except Exception as e:
        return {"error": f"run_full failed: {e}"}
    m = compute_all(r["equity"].pct_change().fillna(0), r.get("trades"), r["equity"])
    return {
        "n_entries": n_entries,
        "net_pnl": m.get("net_pnl", 0.0),
        "sharpe": m.get("sharpe", 0.0),
        "max_dd": m.get("max_drawdown", 0.0),
        "win_rate": m.get("win_rate", 0.0),
        "profit_factor": m.get("profit_factor", 0.0),
        "n_trades": m.get("n_trades", 0),
        "expectancy": m.get("expectancy", 0.0),
    }


def fmt(m, key, fmt_str="{:+.0f}"):
    if isinstance(m, dict) and "error" in m:
        return m["error"][:10]
    v = m.get(key, 0)
    if fmt_str.endswith("pct"):
        return fmt_str.format(v * 100)
    return fmt_str.format(v)


def main():
    print("=" * 100)
    print("AC-AO ENHANCEMENT — trying multiple filter combinations on EUR + NAS")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    eur_oos = eur[eur.index >= eur.index[-1] - pd.DateOffset(months=12)]
    nas_oos = nas[nas.index >= nas.index[-1] - pd.DateOffset(months=12)]
    print(f"  EUR OOS: {eur_oos.index[0].date()} → {eur_oos.index[-1].date()} ({len(eur_oos)} bars)")
    print(f"  NAS OOS: {nas_oos.index[0].date()} → {nas_oos.index[-1].date()} ({len(nas_oos)} bars)\n")

    base = AC_AO_Strategy()

    # Define enhancement variants
    variants = [
        ("BASELINE (no filter)", []),
        ("regime:ADX<25 (reversal fit)", [
            lambda df: regime_adx(df, adx_period=14, enable_above=25, invert=True)
        ]),
        ("regime:ADX<20 (strict ranging)", [
            lambda df: regime_adx(df, adx_period=14, enable_above=20, invert=True)
        ]),
        ("session:London+NY", [
            lambda df: session_filter(df, utc_hours=(7,8,9,10,11,12,13,14,15,16,17,18,19,20))
        ]),
        ("vol:ATR 0.05%-0.50%", [
            lambda df: volatility_atr(df, atr_period=14, min_atr_pct=0.0005, max_atr_pct=0.005)
        ]),
        ("candle:body>30%", [
            lambda df: candle_filter(df, min_body_pct=0.3)
        ]),
        ("MTF:trend>200sma", [
            lambda df: mtf_trend(df, sma_fast=50, sma_slow=200)
        ]),
        ("RSI:oversold/overbought", [
            lambda df: rsi_filter(df, period=14, buy_below=35, sell_above=65)
        ]),
        ("MACD:confirm", [
            lambda df: macd_confirm(df, fast=12, slow=26, sig=9)
        ]),
        ("Bollinger:mean-revert", [
            lambda df: bollinger_filter(df, period=20, std=2.0, buy_below_pct=0.1, sell_above_pct=0.9)
        ]),
        ("vol_persist:skip dead markets", [
            lambda df: volatility_atr(df, atr_period=14, min_atr_pct=0.0008, max_atr_pct=0.004),
            lambda df: volume_filter(df, lookback=20, mult=0.7),
        ]),
        ("reversal kit:regime+RSI+BB", [
            lambda df: regime_adx(df, adx_period=14, enable_above=25, invert=True),
            lambda df: rsi_filter(df, period=14, buy_below=35, sell_above=65),
            lambda df: bollinger_filter(df, period=20, std=2.0, buy_below_pct=0.1, sell_above_pct=0.9),
        ]),
        ("session+regime+RSI", [
            lambda df: session_filter(df, utc_hours=(7,8,9,10,11,12,13,14,15,16,17,18,19,20)),
            lambda df: regime_adx(df, adx_period=14, enable_above=25, invert=True),
            lambda df: rsi_filter(df, period=14, buy_below=40, sell_above=60),
        ]),
        ("tight reversal:regime+RSI+BB+candle", [
            lambda df: regime_adx(df, adx_period=14, enable_above=22, invert=True),
            lambda df: rsi_filter(df, period=14, buy_below=35, sell_above=65),
            lambda df: bollinger_filter(df, period=20, std=2.0, buy_below_pct=0.1, sell_above_pct=0.9),
            lambda df: candle_filter(df, min_body_pct=0.3),
        ]),
        ("all filters combined", [
            lambda df: regime_adx(df, adx_period=14, enable_above=25, invert=True),
            lambda df: session_filter(df, utc_hours=(7,8,9,10,11,12,13,14,15,16,17,18,19,20)),
            lambda df: volatility_atr(df, atr_period=14, min_atr_pct=0.0006, max_atr_pct=0.004),
            lambda df: rsi_filter(df, period=14, buy_below=35, sell_above=65),
            lambda df: bollinger_filter(df, period=20, std=2.0, buy_below_pct=0.1, sell_above_pct=0.9),
            lambda df: candle_filter(df, min_body_pct=0.3),
        ]),
    ]

    print(f"Testing {len(variants)} variants × 2 assets × (PURE strategy):\n")
    print(f"{'Variant':<40} {'─ EURUSD ─':>30} {'─ NAS100 ─':>30}")
    print(f"{'':<40} {'PnL':>9} {'Sharpe':>7} {'WR':>5} {'Tr':>4}    "
          f"{'PnL':>9} {'Sharpe':>7} {'WR':>5} {'Tr':>4}")
    print("-" * 130)

    results = []
    for label, filters in variants:
        strat = EnhancedStrategy(base, filters=filters)
        strat.name = f"ac_ao+{label[:18]}"
        e = run_strategy(eur_oos, strat, "forex")
        n = run_strategy(nas_oos, strat, "nas100")
        results.append({"label": label, "eur": e, "nas": n})
        if isinstance(e, dict) and "error" not in e and isinstance(n, dict) and "error" not in n:
            print(f"{label:<40} ${e['net_pnl']:>+8,.0f} {e['sharpe']:>+6.2f} "
                  f"{e['win_rate']*100:>4.0f}% {e['n_trades']:>4}    "
                  f"${n['net_pnl']:>+8,.0f} {n['sharpe']:>+6.2f} "
                  f"{n['win_rate']*100:>4.0f}% {n['n_trades']:>4}")
        else:
            err_e = (e or {}).get("error", "?") if isinstance(e, dict) else "?"
            err_n = (n or {}).get("error", "?") if isinstance(n, dict) else "?"
            print(f"{label:<40} ERR_EUR={err_e[:10]} ERR_NAS={err_n[:10]}")

    # Verdict
    print("\n" + "=" * 100)
    print("VERDICT — which variant 'fixes' AC-AO?")
    print("=" * 100)
    print(f"{'Variant':<40} {'EUR':>12} {'NAS':>12} {'Total':>10}")
    print("-" * 80)
    best = None
    best_score = -1e18
    for r in results:
        e = r["eur"]; n = r["nas"]
        if "error" in e or "error" in n:
            continue
        e_pnl, e_sh = e["net_pnl"], e["sharpe"]
        n_pnl, n_sh = n["net_pnl"], n["sharpe"]
        total = e_pnl + n_pnl
        # Score: prefer PnL > 0 AND Sharpe > 0 on both; prefer high combined
        score = total + 1000 * (e_sh + n_sh)
        marker = ""
        if e_pnl > 0 and e_sh > 0 and n_pnl > 0 and n_sh > 0:
            marker = "✓✓ BOTH"
        elif e_pnl > 0 and e_sh > 0:
            marker = "✓ EUR"
        elif n_pnl > 0 and n_sh > 0:
            marker = "✓ NAS"
        else:
            marker = "✗ neither"
        print(f"{r['label']:<40} ${e_pnl:>+10,.0f} ${n_pnl:>+10,.0f} ${total:>+8,.0f} {marker}")
        if score > best_score:
            best_score = score
            best = r

    print(f"\n  BEST variant (by combined score): {best['label']}")
    print(f"  EUR: ${best['eur']['net_pnl']:+,.0f} / Sharpe {best['eur']['sharpe']:+.2f}")
    print(f"  NAS: ${best['nas']['net_pnl']:+,.0f} / Sharpe {best['nas']['sharpe']:+.2f}")

    # Save
    pd.DataFrame([{"label": r["label"],
                    "eur_pnl": r["eur"].get("net_pnl") if isinstance(r["eur"], dict) else None,
                    "eur_sharpe": r["eur"].get("sharpe") if isinstance(r["eur"], dict) else None,
                    "nas_pnl": r["nas"].get("net_pnl") if isinstance(r["nas"], dict) else None,
                    "nas_sharpe": r["nas"].get("sharpe") if isinstance(r["nas"], dict) else None,
                    } for r in results]).to_csv("output/ac_ao_enhancement.csv", index=False)
    print("\nSaved → output/ac_ao_enhancement.csv")


if __name__ == "__main__":
    main()
