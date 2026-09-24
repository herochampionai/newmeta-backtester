"""Per-strategy test using LIVE MT5 data (most accurate possible).

Uses MT5 live OHLCV as the source of truth, runs the same 12 strategies
that exist in the MT5 TwelveStrategies EA + light9 (Python extra) +
crypto_9 (separate PineScript port).

Compares pure (no grid) vs GRID_LOSS_AND_PROFIT for each strategy.
Uses corrected metrics (PF cap 99 instead of 999 sentinel).
"""
from __future__ import annotations
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

import MetaTrader5 as mt5

from data.mt5_export import init_mt5
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE, GRID_LOSS_AND_PROFIT, RECOVERY_HIGHER_PROFITS
from backtester.metrics_v2 import compute_all
from strategies import MULTI_STRAT_EA_REGISTRY, CRYPTO_STRAT_EA_REGISTRY


# MT5 TwelveStrategies EA — verified mapping (12 strategies)
# Base (in TwelveStrategies.mq5): AC+AO, ADX, DeM, FBB, MFI, MS
# New (in TwelveModule.mqh): MSTF, BBR, TRSI, QSTO, S533, MCD
MT5_TWELVE = {
    "ac_ao": "AC+AO (base)",
    "adx": "ADX (base)",
    "dem": "DeM (base)",
    "fbb": "FBB (base)",
    "mfi": "MFI (base)",
    "ms": "MS (base)",
    "mtf_stoch": "MSTF (TwelveModule)",
    "bb_rsi": "BBR (TwelveModule)",
    "triple_rsi": "TRSI (TwelveModule)",
    "quad_stoch": "QSTO (TwelveModule)",
    "stoch533_mtf": "S533 (TwelveModule)",
    "macd_confluence": "MCD (TwelveModule)",
}
PYTHON_EXTRA = {
    "light9": "Python-only (NOT in MT5 EA)",
}
CRYPTO = {
    "crypto_9": "PineScript 9-pattern port",
}


def fetch_mt5_eurusd_h1(terminal: str = "D:/MT5_EuroPrinter/terminal64.exe",
                         n_bars: int = 20000) -> pd.DataFrame:
    if not init_mt5(terminal):
        raise RuntimeError(f"init_mt5 failed for {terminal}")
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, n_bars)
    if rates is None or len(rates) == 0:
        mt5.shutdown()
        raise RuntimeError("copy_rates_from_pos returned empty")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    mt5.shutdown()
    return df


def run_strategy(df, strategy_name, grid_mode):
    cls = MULTI_STRAT_EA_REGISTRY.get(strategy_name) or CRYPTO_STRAT_EA_REGISTRY.get(strategy_name)
    if cls is None:
        return None
    inst = cls()
    try:
        sig = inst.generate(df)
    except Exception as e:
        return {"error": f"generate failed: {e}"}
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    n_entries = int(entries.sum())
    if n_entries == 0:
        return {"error": "no entries", "n_entries": 0}

    signals = {strategy_name: (entries, direction)}
    kw = dict(
        init_cash=10_000.0, commission_pips=0.7, slippage_pips=0.3,
        spread_pips=1.0, pip_size=0.0001, contract_size=100_000,
        base_lot=0.1,
        grid_take_profit=50, grid_stop_loss=200,
        max_grid_layers=4, pips_between_orders=30, grid_lot_multiplier=1.5,
        recovery_mode=RECOVERY_HIGHER_PROFITS, recovery_lot_multiplier=2.0,
    )
    kw["grid_mode"] = grid_mode
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
        "n_losses": m.get("n_losses", 0),
        "pf_undefined": m.get("profit_factor_undefined", False),
        "expectancy": m.get("expectancy", 0.0),
    }


def fmt_pf(pf, undefined):
    if undefined:
        return "—"
    return f"{pf:.2f}"


def main():
    print("=" * 100)
    print("PER-STRATEGY PURE vs GRID — LIVE MT5 DATA (EuroPrinter, EURUSD H1)")
    print("=" * 100)
    print("\nFetching LIVE data from MT5...")
    df = fetch_mt5_eurusd_h1()
    print(f"  Got {len(df):,} bars, {df.index[0]} → {df.index[-1]}")

    df_oos = df[df.index >= pd.Timestamp("2024-07-01", tz="UTC")].copy()
    print(f"  OOS: {len(df_oos):,} bars ({df_oos.index[0].date()} → {df_oos.index[-1].date()})")

    # Run all strategies
    all_strats = list(MT5_TWELVE.keys()) + list(PYTHON_EXTRA.keys()) + list(CRYPTO.keys())
    print(f"\nRunning {len(all_strats)} strategies × {{pure, grid}}...")
    print(f"  {len(MT5_TWELVE)} MT5 TwelveStrategies EA mappings")
    print(f"  {len(PYTHON_EXTRA)} Python-only extras")
    print(f"  {len(CRYPTO)} crypto variants\n")

    results = {}
    for s in all_strats:
        pure = run_strategy(df_oos, s, GRID_NONE)
        grid = run_strategy(df_oos, s, GRID_LOSS_AND_PROFIT)
        results[s] = {"pure": pure, "grid": grid, "source": (
            "MT5 Twelve" if s in MT5_TWELVE
            else "Python extra" if s in PYTHON_EXTRA
            else "Crypto"
        )}

    # Print table grouped by source
    print("=" * 100)
    print("RESULTS — PURE (no grid) vs GRID_LOSS_AND_PROFIT")
    print("=" * 100)
    print(f"{'Strategy':<22} {'Source':<22} {'─ PURE ─':>30} {'─ GRID ─':>30}")
    print(f"{'':<22} {'':<22} {'PnL':>9} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'DD':>7}    "
          f"{'PnL':>9} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'DD':>7}")
    print("-" * 130)

    summary = []
    for source_label, source_strats in [
        ("── MT5 TWELVE ──", list(MT5_TWELVE.keys())),
        ("── PYTHON EXTRA ──", list(PYTHON_EXTRA.keys())),
        ("── CRYPTO ──", list(CRYPTO.keys())),
    ]:
        print(f"\n{source_label}")
        for s in source_strats:
            d = results[s]
            p = d["pure"]; g = d["grid"]
            if not isinstance(p, dict) or "error" in p or not isinstance(g, dict) or "error" in g:
                err = (p or {}).get("error", "?") if isinstance(p, dict) else "?"
                err2 = (g or {}).get("error", "?") if isinstance(g, dict) else "?"
                print(f"  {s:<20} {'ERR':<22} {err[:20]:>20}    {err2[:20]:>20}")
                continue
            pf_p = fmt_pf(p["profit_factor"], p.get("pf_undefined", False))
            pf_g = fmt_pf(g["profit_factor"], g.get("pf_undefined", False))
            print(f"  {s:<20} {d['source']:<22} "
                  f"${p['net_pnl']:>+8,.0f} {p['sharpe']:>+6.2f} {p['win_rate']*100:>4.0f}% "
                  f"{pf_p:>5} {p['max_dd']*100:>+6.2f}%    "
                  f"${g['net_pnl']:>+8,.0f} {g['sharpe']:>+6.2f} {g['win_rate']*100:>4.0f}% "
                  f"{pf_g:>5} {g['max_dd']*100:>+6.2f}%")
            summary.append({
                "strategy": s, "source": d["source"],
                "pure_pnl": p["net_pnl"], "grid_pnl": g["net_pnl"],
                "delta_pnl": g["net_pnl"] - p["net_pnl"],
                "pure_sharpe": p["sharpe"], "grid_sharpe": g["sharpe"],
                "delta_sharpe": g["sharpe"] - p["sharpe"],
                "pure_wr": p["win_rate"], "grid_wr": g["win_rate"],
                "pure_pf": p["profit_factor"] if not p.get("pf_undefined") else None,
                "grid_pf": g["profit_factor"] if not g.get("pf_undefined") else None,
                "pure_dd": p["max_dd"], "grid_dd": g["max_dd"],
                "pure_trades": p["n_trades"], "grid_trades": g["n_trades"],
                "pf_undef_pure": p.get("pf_undefined", False),
                "pf_undef_grid": g.get("pf_undefined", False),
            })

    # Verdict
    print("\n" + "=" * 100)
    print("VERDICT — GRID HELPS / HURTS / NEUTRAL")
    print("=" * 100)
    print(f"{'Strategy':<22} {'Source':<22} {'Δ PnL':>10} {'Δ Sharpe':>10} {'Verdict':<14}")
    print("-" * 80)
    summary.sort(key=lambda r: r["delta_pnl"], reverse=True)
    for r in summary:
        d = r["delta_pnl"]
        ds = r["delta_sharpe"]
        if d > 50 and ds > 0:
            verdict = "✓ GRID BOOSTS"
        elif d > 0 and ds <= 0:
            verdict = "~ PnL only"
        elif abs(d) < 50:
            verdict = "= neutral"
        else:
            verdict = "✗ GRID HURTS"
        print(f"{r['strategy']:<22} {r['source']:<22} ${d:>+8,.0f} {ds:>+9.2f} {verdict:<14}")

    # Save
    out = Path("output/per_strategy_mt5.csv")
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
