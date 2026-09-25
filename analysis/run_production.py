"""Production multi-instance runner.

Mirrors MT5's multi-chart independent deployment:
  - Each (strategy, ticker) is an INSTANCE
  - Each instance runs independently with its own equity curve + trade log
  - No shared state between instances
  - Aggregated portfolio view at the end

Usage:
  python -m analysis.run_production --profiles "adx_NAS,adx_EUR,ac_ao_EUR"
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys

sys.path.insert(0, '.')

import argparse
import json
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from backtester.metrics_v2 import compute_all
from data.mt5_export import init_mt5


def fetch_h1(term: str, sym: str, n_bars: int = 30000) -> pd.DataFrame:
    init_mt5(term)
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, n_bars)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    mt5.shutdown()
    return df


def get_class(strategy_name: str):
    """Resolve strategy class."""
    import importlib
    v4_map = {
        "triple_rsi_v4": ("strategies.trend_follow_v4", "TripleRSIV4Strategy"),
        "stoch533_mtf_v4": ("strategies.trend_follow_v4", "Stoch533V4"),
        "macd_confluence_v4": ("strategies.trend_follow_v4_more", "MACDConfluenceV4"),
        "bb_rsi_v4": ("strategies.trend_follow_v4_more", "BBRsiV4"),
        "quad_stoch_v4": ("strategies.trend_follow_v4_more", "QuadStochV4"),
        "fbb_v4": ("strategies.trend_follow_v4_more", "FBBV4"),
    }
    if strategy_name in v4_map:
        mod, cls_name = v4_map[strategy_name]
        return getattr(importlib.import_module(mod), cls_name)
    from strategies import CRYPTO_STRAT_EA_REGISTRY, MULTI_STRAT_EA_REGISTRY
    if strategy_name in MULTI_STRAT_EA_REGISTRY:
        return MULTI_STRAT_EA_REGISTRY[strategy_name]
    if strategy_name in CRYPTO_STRAT_EA_REGISTRY:
        return CRYPTO_STRAT_EA_REGISTRY[strategy_name]
    return None


def run_instance(profile: dict, df_data: pd.DataFrame) -> dict:
    """Run one independent instance of (strategy, ticker) with locked params."""
    strategy = profile["strategy"]
    ticker = profile["ticker"]
    timeframe = profile["timeframe"]
    params = profile["params"]

    cls = get_class(strategy)
    if cls is None:
        return {"error": f"unknown strategy: {strategy}"}

    # Filter to OOS window or use full data
    df = df_data.copy()
    if "oos_window" in profile:
        start = pd.Timestamp(profile["oos_window"]["start"], tz="UTC")
        end = pd.Timestamp(profile["oos_window"]["end"], tz="UTC")
        df = df[(df.index >= start) & (df.index < end)]

    if len(df) < 100:
        return {"error": f"insufficient data: {len(df)} bars"}

    # Profile-specific sizing
    profile_kind = "forex" if ticker.upper() in ("EUR", "EURUSD") else "nas100"
    if profile_kind == "forex":
        kw = dict(pip_size=0.0001, contract_size=100_000, base_lot=0.1,
                  commission_pips=0.7, slippage_pips=0.3, spread_pips=1.0, init_cash=10_000.0)
    else:
        kw = dict(pip_size=1.0, contract_size=1.0, base_lot=0.1,
                  commission_pips=2.0, slippage_pips=1.0, spread_pips=1.5, init_cash=10_000.0)
    kw["grid_mode"] = GRID_NONE

    try:
        base = cls(params=params)
        sig = base.generate(df)
        entries = sig.entries.fillna(False).astype(bool)
        direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
        n_entries = int(entries.sum())
        if n_entries == 0:
            return {"error": "no entries"}

        signals = {strategy: (entries, direction)}
        r = run_full(df, signals, **kw)
        m = compute_all(r["equity"].pct_change().fillna(0), r.get("trades"), r["equity"])
        return {
            "strategy": strategy,
            "ticker": ticker,
            "timeframe": timeframe,
            "equity": r["equity"],
            "trades": r.get("trades"),
            "metrics": m,
            "n_entries": n_entries,
            "n_bars": len(df),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=None,
                    help="Comma-separated list of profile names (without .json). "
                         "Default: run all locked winners.")
    ap.add_argument("--start", default=None, help="Override OOS start date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="Override OOS end date YYYY-MM-DD")
    ap.add_argument("--equity-source", default="initial_cash",
                    help="initial_cash (each instance starts at its own $10k) "
                         "or compounded (each instance compounds its own equity)")
    args = ap.parse_args()

    print("=" * 100)
    print("PRODUCTION MULTI-INSTANCE RUNNER")
    print("=" * 100)
    print("Each instance runs INDEPENDENTLY (single chart, single ticker, single timeframe)")
    print("Multiple instances = multiple EA attachments on different charts")
    print()

    # Load profiles
    profiles_dir = Path("output/profiles")
    if args.profiles:
        names = [n.strip() for n in args.profiles.split(",")]
        profiles = [json.load(open(profiles_dir / f"{n}.json")) for n in names]
    else:
        profiles = [json.load(open(p)) for p in sorted(profiles_dir.glob("*.json"))]

    print(f"Loaded {len(profiles)} instances:")
    for p in profiles:
        m = p.get("metrics", {})
        v = p.get("version", "?")
        print(f"  {p['ticker']:<6} {p['strategy']:<25} v={v:<3} PnL ${m.get('net_pnl', 0):+8,.0f} "
              f"Sharpe {m.get('sharpe', 0):+.2f}")
    print()

    # Fetch data once per ticker (cache)
    print("Fetching market data...")
    data_cache = {}
    for p in profiles:
        t = p["ticker"].lower()
        sym = p["ticker"]
        if t in ("eur", "eurusd"):
            term = "D:/MT5_EuroPrinter/terminal64.exe"
            sym_mt5 = "EURUSD"
        elif t in ("nas", "nas100"):
            term = "D:/MT5_Bybit/terminal64.exe"
            sym_mt5 = "NAS100"
        else:
            continue
        key = (t, sym_mt5)
        if key not in data_cache:
            data_cache[key] = fetch_h1(term, sym_mt5)
            print(f"  {sym} (from {term.split('/')[-2]}): {len(data_cache[key])} bars")

    # Run each instance independently
    print("\n" + "=" * 100)
    print("INSTANCE RESULTS")
    print("=" * 100)
    print(f"{'Instance':<32} {'Bars':>6} {'Entries':>8} {'Trades':>7} {'PnL':>10} {'Sharpe':>7} "
          f"{'WR':>6} {'PF':>5} {'MaxDD':>7}")
    print("-" * 100)

    instance_results = []
    total_combined_equity = None
    for p in profiles:
        t = p["ticker"].lower()
        sym = p["ticker"]
        if t in ("eur", "eurusd"):
            sym_mt5 = "EURUSD"
        elif t in ("nas", "nas100"):
            sym_mt5 = "NAS100"
        else:
            continue
        df = data_cache.get((t, sym_mt5))
        if df is None:
            continue
        # Apply date overrides
        if args.start:
            df = df[df.index >= pd.Timestamp(args.start, tz="UTC")]
        if args.end:
            df = df[df.index < pd.Timestamp(args.end, tz="UTC")]
        result = run_instance(p, df)
        if "error" in result:
            print(f"{p['strategy']}_{p['ticker']:<24} ERR: {result['error']}")
            continue
        m = result["metrics"]
        print(f"{p['strategy']}_{p['ticker']:<24} {result['n_bars']:>6} {result['n_entries']:>8} "
              f"{m.get('n_trades', 0):>7} ${m.get('net_pnl', 0):+9,.0f} {m.get('sharpe', 0):>+6.2f} "
              f"{m.get('win_rate', 0)*100:>5.1f}% {m.get('profit_factor', 0):>4.2f} "
              f"{m.get('max_drawdown', 0)*100:>+6.2f}%")
        instance_results.append((p, result))

    # Portfolio aggregation (assumes independent instances, sum their PnL)
    print("\n" + "=" * 100)
    print("PORTFOLIO TOTAL (independent instances, summed)")
    print("=" * 100)
    if instance_results:
        total_pnl = sum(r["metrics"].get("net_pnl", 0) for _, r in instance_results)
        total_trades = sum(r["metrics"].get("n_trades", 0) for _, r in instance_results)
        avg_sharpe = np.mean([r["metrics"].get("sharpe", 0) for _, r in instance_results])
        avg_wr = np.mean([r["metrics"].get("win_rate", 0) for _, r in instance_results])
        avg_pf = np.mean([r["metrics"].get("profit_factor", 0) for _, r in instance_results])
        avg_max_dd = np.mean([r["metrics"].get("max_drawdown", 0) for _, r in instance_results])
        print(f"  Instances: {len(instance_results)}")
        print(f"  Combined PnL: ${total_pnl:+,.0f}")
        print(f"  Combined trades: {total_trades}")
        print(f"  Average Sharpe: {avg_sharpe:.2f}")
        print(f"  Average WR: {avg_wr*100:.1f}%")
        print(f"  Average PF: {avg_pf:.2f}")
        print(f"  Average Max DD: {avg_max_dd*100:.1f}%")

        # Per-year breakdown (if OOS window covers multiple years)
        for p, r in instance_results:
            years = r["n_bars"] / (252 * 24)  # H1 bars per year
            if years > 1:
                annualized = r["metrics"].get("net_pnl", 0) / years
                print(f"  {p['strategy']}_{p['ticker']:<22} annualised PnL: ${annualized:+,.0f} ({years:.1f} yrs)")

    # Save production trade log
    output_log = Path(f"output/production_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    log_rows = []
    for p, r in instance_results:
        trades = r.get("trades")
        if trades is not None and len(trades) > 0:
            for _, tr in trades.iterrows():
                log_rows.append({
                    "strategy": p["strategy"],
                    "ticker": p["ticker"],
                    "version": p["version"],
                    **tr.to_dict(),
                })
    if log_rows:
        pd.DataFrame(log_rows).to_csv(output_log, index=False)
        print(f"\n  Production trade log → {output_log}")
        print(f"  Total trades logged: {len(log_rows)}")


if __name__ == "__main__":
    main()
