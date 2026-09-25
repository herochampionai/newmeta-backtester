"""Fetch NAS100 + EURUSD H1 from MT5 live, then run all 12 strategies
as baseline (no grid) on each asset.

Goal: establish which strategies are already profitable on which asset
and identify weak ones for per-strategy enhancement.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5
import pandas as pd

from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from backtester.metrics_v2 import compute_all
from data.mt5_export import init_mt5
from strategies import MULTI_STRAT_EA_REGISTRY


def fetch_h1(terminal: str, symbol: str, n_bars: int = 20000) -> pd.DataFrame:
    if not init_mt5(terminal):
        raise RuntimeError(f"init_mt5 failed for {terminal}")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n_bars)
    if rates is None or len(rates) == 0:
        mt5.shutdown()
        raise RuntimeError(f"{symbol}: copy_rates_from_pos returned empty")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    mt5.shutdown()
    return df


def run_strategy(df, strategy_name, market_profile="forex"):
    """market_profile: 'forex' (EURUSD) or 'nas100' — affects pip_size / contract_size."""
    cls = MULTI_STRAT_EA_REGISTRY.get(strategy_name)
    if cls is None:
        return None
    try:
        sig = cls().generate(df)
    except Exception as e:
        return {"error": f"generate failed: {e}"}
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    n_entries = int(entries.sum())
    if n_entries == 0:
        return {"error": "no entries", "n_entries": 0}
    signals = {strategy_name: (entries, direction)}

    # Asset-specific sizing
    if market_profile == "forex":
        kw = dict(pip_size=0.0001, contract_size=100_000, base_lot=0.1,
                  commission_pips=0.7, slippage_pips=0.3, spread_pips=1.0,
                  init_cash=10_000.0)
    else:  # nas100 — index CFD: pip=1.0, contract=1, big spread
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
        "total_return": m.get("total_return", 0.0),
        "sharpe": m.get("sharpe", 0.0),
        "sortino": m.get("sortino", 0.0),
        "calmar": m.get("calmar", 0.0),
        "max_dd": m.get("max_drawdown", 0.0),
        "win_rate": m.get("win_rate", 0.0),
        "profit_factor": m.get("profit_factor", 0.0),
        "pf_undef": m.get("profit_factor_undefined", False),
        "n_trades": m.get("n_trades", 0),
        "n_losses": m.get("n_losses", 0),
        "expectancy": m.get("expectancy", 0.0),
        "avg_win": m.get("avg_win", 0.0),
        "avg_loss": m.get("avg_loss", 0.0),
    }


def fmt_pf(pf, undefined):
    return "—" if undefined else f"{pf:.2f}"


def main():
    # Pull live data
    print("=" * 100)
    print("FETCHING LIVE DATA FROM MT5")
    print("=" * 100)
    eur_df = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    print(f"  EURUSD: {len(eur_df):,} bars, {eur_df.index[0].date()} → {eur_df.index[-1].date()}")
    nas_df = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    print(f"  NAS100: {len(nas_df):,} bars, {nas_df.index[0].date()} → {nas_df.index[-1].date()}")

    # OOS: last 12 months on each (consistent window length)
    eur_oos_start = max(eur_df.index[-1] - pd.DateOffset(months=12), eur_df.index[0])
    nas_oos_start = max(nas_df.index[-1] - pd.DateOffset(months=12), nas_df.index[0])
    eur_oos = eur_df[eur_df.index >= eur_oos_start]
    nas_oos = nas_df[nas_df.index >= nas_oos_start]
    print(f"\n  EURUSD OOS: {eur_oos.index[0].date()} → {eur_oos.index[-1].date()} ({len(eur_oos):,} bars)")
    print(f"  NAS100 OOS: {nas_oos.index[0].date()} → {nas_oos.index[-1].date()} ({len(nas_oos):,} bars)")

    # 12 MT5 TwelveStrategies strategies
    strats = [
        "ac_ao", "adx", "dem", "fbb", "mfi", "ms",
        "mtf_stoch", "bb_rsi", "triple_rsi", "quad_stoch",
        "stoch533_mtf", "macd_confluence",
    ]
    print(f"\n  Testing {len(strats)} strategies × 2 assets (NO GRID)\n")

    results = {}
    for s in strats:
        print(f"  [{s:<22}]", end=" ", flush=True)
        eur = run_strategy(eur_oos, s, "forex")
        nas = run_strategy(nas_oos, s, "nas100")
        results[s] = {"eur": eur, "nas": nas}
        if isinstance(eur, dict) and "error" not in eur and isinstance(nas, dict) and "error" not in nas:
            print(f"EUR ${eur['net_pnl']:+,.0f} ({eur['sharpe']:+.2f}) | "
                  f"NAS ${nas['net_pnl']:+,.0f} ({nas['sharpe']:+.2f})")
        else:
            err_e = (eur or {}).get("error", "?") if isinstance(eur, dict) else "?"
            err_n = (nas or {}).get("error", "?") if isinstance(nas, dict) else "?"
            print(f"EUR ERR={err_e} | NAS ERR={err_n}")

    # Print table
    print("\n" + "=" * 100)
    print("BASELINE — NO GRID, last 12 months OOS")
    print("=" * 100)
    print(f"{'Strategy':<22} {'─ EURUSD H1 ─':>40} {'─ NAS100 H1 ─':>40}")
    print(f"{'':<22} {'PnL':>9} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'DD':>7}  "
          f"{'PnL':>9} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'DD':>7}")
    print("-" * 130)

    summary = []
    for s in strats:
        e = results[s]["eur"]
        n = results[s]["nas"]
        if isinstance(e, dict) and "error" not in e and isinstance(n, dict) and "error" not in n:
            print(f"{s:<22} ${e['net_pnl']:>+8,.0f} {e['sharpe']:>+6.2f} "
                  f"{e['win_rate']*100:>4.0f}% {fmt_pf(e['profit_factor'], e['pf_undef']):>5} "
                  f"{e['max_dd']*100:>+6.2f}%  "
                  f"${n['net_pnl']:>+8,.0f} {n['sharpe']:>+6.2f} "
                  f"{n['win_rate']*100:>4.0f}% {fmt_pf(n['profit_factor'], n['pf_undef']):>5} "
                  f"{n['max_dd']*100:>+6.2f}%")
            summary.append({
                "strategy": s,
                "eur_pnl": e["net_pnl"], "eur_sharpe": e["sharpe"],
                "eur_wr": e["win_rate"], "eur_pf": e["profit_factor"],
                "eur_dd": e["max_dd"], "eur_n": e["n_trades"],
                "nas_pnl": n["net_pnl"], "nas_sharpe": n["sharpe"],
                "nas_wr": n["win_rate"], "nas_pf": n["profit_factor"],
                "nas_dd": n["max_dd"], "nas_n": n["n_trades"],
            })

    # Status
    print("\n" + "=" * 100)
    print("STATUS — Profitable standalone (Sharpe > 0 AND PnL > 0)?")
    print("=" * 100)
    print(f"{'Strategy':<22} {'EUR':>10} {'NAS':>10}  {'Verdict':<30}")
    print("-" * 80)
    for r in summary:
        eur_ok = r["eur_pnl"] > 0 and r["eur_sharpe"] > 0
        nas_ok = r["nas_pnl"] > 0 and r["nas_sharpe"] > 0
        e_str = "✓ profit" if eur_ok else "✗ losing"
        n_str = "✓ profit" if nas_ok else "✗ losing"
        if eur_ok and nas_ok:
            verdict = "✓✓ ACCEPTED (both)"
        elif eur_ok:
            verdict = "✓ ACCEPTED (EUR)"
        elif nas_ok:
            verdict = "✓ ACCEPTED (NAS)"
        else:
            verdict = "✗ needs enhancement"
        print(f"{r['strategy']:<22} {e_str:>10} {n_str:>10}  {verdict:<30}")

    # Save
    out = Path("output/baseline_no_grid.csv")
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
