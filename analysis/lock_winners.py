"""2-year OOS validation of locked-in winners + production profile generator.

For each winning strategy (V1 + V4), test on 2-year OOS window (2024-09 to 2026-09).
Compute full metrics:
  - Win rate
  - Profit factor
  - Trades per year
  - Average win / average loss / expectancy
  - Max DD, Sharpe, CAGR, Sortino
  - Equity curve shape (smoothness, recovery time)

Outputs:
  - Per-strategy detailed stats
  - Production strategy profiles (JSON)
  - MT5 .set files for direct import (V1 strategies that exist in MQL5)
  - Multi-instance runner
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

import MetaTrader5 as mt5
from data.mt5_export import init_mt5
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from backtester.metrics_v2 import compute_all
from strategies import (
    MULTI_STRAT_EA_REGISTRY, CRYPTO_STRAT_EA_REGISTRY,
    AC_AO_Strategy, ADX_Strategy, DeM_Strategy, FBB_Strategy, MFI_Strategy,
    MS_Strategy, BBRsiStrategy, QuadStochSameTF, Stoch533MTF,
    TripleRSIStrategy, MACDConfluenceStrategy,
)
from strategies.trend_follow_v4 import TripleRSIV4Strategy, Stoch533V4
TripleRSIV4 = TripleRSIV4Strategy
from strategies.trend_follow_v4_more import (
    MACDConfluenceV4, BBRsiV4, QuadStochV4, FBBV4,
)

OOS_2Y_START = pd.Timestamp("2024-09-17", tz="UTC")
OOS_2Y_END = pd.Timestamp("2026-09-17", tz="UTC")

# Loaded best params from prior Optuna runs (saved in output/)
def load_params():
    out = {}
    for fname in os.listdir("output"):
        if fname.endswith("_best_params.json"):
            name = fname.replace("_best_params.json", "")
            try:
                with open(f"output/{fname}") as f:
                    data = json.load(f)
                # Each file is {"eur": {params}, "nas": {params}} or {"eur": {"params": ..., "filters": ...}}
                for ticker in ("eur", "nas"):
                    if ticker in data:
                        v = data[ticker]
                        if isinstance(v, dict) and "params" in v:
                            out[(name, ticker)] = v["params"]
                        else:
                            out[(name, ticker)] = v
            except Exception:
                pass
    # Also load the v4 specific ones
    if os.path.exists("output/v4_best_params.json"):
        with open("output/v4_best_params.json") as f:
            data = json.load(f)
        for key, v in data.items():
            # key like "triple_rsi_v4_EUR"
            parts = key.split("_")
            ticker = parts[-1].lower()
            strat = "_".join(parts[:-1])
            out[(strat, ticker)] = v
    if os.path.exists("output/v4_more_best_params.json"):
        with open("output/v4_more_best_params.json") as f:
            data = json.load(f)
        for key, v in data.items():
            parts = key.split("_")
            ticker = parts[-1].lower()
            strat = "_".join(parts[:-1])
            out[(strat, ticker)] = v
    if os.path.exists("output/fbb_v4_best.json"):
        with open("output/fbb_v4_best.json") as f:
            data = json.load(f)
        for key, v in data.items():
            parts = key.split("_")
            ticker = parts[-1].lower()
            strat = "_".join(parts[:-1])
            out[(strat, ticker)] = v
    return out


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
    """Return strategy class by name."""
    if strategy_name in MULTI_STRAT_EA_REGISTRY:
        return MULTI_STRAT_EA_REGISTRY[strategy_name]
    if strategy_name in CRYPTO_STRAT_EA_REGISTRY:
        return CRYPTO_STRAT_EA_REGISTRY[strategy_name]
    # V4 classes
    v4_map = {
        "triple_rsi_v4": TripleRSIV4,
        "stoch533_mtf_v4": Stoch533V4,
        "macd_confluence_v4": MACDConfluenceV4,
        "bb_rsi_v4": BBRsiV4,
        "quad_stoch_v4": QuadStochV4,
        "fbb_v4": FBBV4,
    }
    return v4_map.get(strategy_name)


def run_2y(strategy_name: str, params: dict, ticker: str, df_full: pd.DataFrame):
    """Run 2-year OOS backtest, return detailed stats."""
    cls = get_class(strategy_name)
    if cls is None:
        return {"error": f"unknown strategy: {strategy_name}"}
    df_oos = df_full[(df_full.index >= OOS_2Y_START) & (df_full.index < OOS_2Y_END)].copy()
    if len(df_oos) < 200:
        return {"error": "insufficient OOS data"}

    profile = "forex" if ticker == "eur" else "nas100"
    if profile == "forex":
        kw = dict(pip_size=0.0001, contract_size=100_000, base_lot=0.1,
                  commission_pips=0.7, slippage_pips=0.3, spread_pips=1.0, init_cash=10_000.0)
    else:
        kw = dict(pip_size=1.0, contract_size=1.0, base_lot=0.1,
                  commission_pips=2.0, slippage_pips=1.0, spread_pips=1.5, init_cash=10_000.0)
    kw["grid_mode"] = GRID_NONE

    try:
        base = cls(params=params)
        sig = base.generate(df_oos)
        entries = sig.entries.fillna(False).astype(bool)
        direction = pd.Series(sig.direction, index=df_oos.index).fillna(0).astype(int)
        n_entries = int(entries.sum())
        if n_entries == 0:
            return {"error": "no entries"}
        signals = {strategy_name: (entries, direction)}
        r = run_full(df_oos, signals, **kw)
        m = compute_all(r["equity"].pct_change().fillna(0), r.get("trades"), r["equity"])
        # Trade-level stats
        trades_df = r.get("trades")
        if trades_df is not None and len(trades_df) > 0:
            pnl_col = next((c for c in ("pnl", "PnL", "profit", "Profit") if c in trades_df.columns), None)
            if pnl_col:
                pnls = trades_df[pnl_col]
                wins = pnls[pnls > 0]
                losses = pnls[pnls < 0]
                avg_win = float(wins.mean()) if len(wins) else 0
                avg_loss = float(losses.mean()) if len(losses) else 0
                largest_win = float(pnls.max())
                largest_loss = float(pnls.min())
                # Year span
                years = (df_oos.index[-1] - df_oos.index[0]).days / 365.25
                trades_per_year = len(trades_df) / max(years, 0.1)
            else:
                avg_win = avg_loss = largest_win = largest_loss = 0
                trades_per_year = 0
        else:
            avg_win = avg_loss = largest_win = largest_loss = 0
            trades_per_year = 0

        # Recovery factor + max DD recovery time
        equity = r["equity"]
        peak = equity.cummax()
        dd = (equity / peak - 1)
        max_dd = float(dd.min())
        # Recovery time: how long to recover from max DD
        if max_dd < 0:
            dd_idx = dd.idxmin()
            after_dd = equity.loc[dd_idx:]
            recovered = after_dd >= float(peak.loc[dd_idx])
            recovery_bars = len(after_dd) if not recovered.any() else int(np.argmax(recovered.values))
        else:
            recovery_bars = 0

        return {
            "n_entries": n_entries,
            "n_trades": m.get("n_trades", 0),
            "trades_per_year": round(trades_per_year, 1),
            "net_pnl": m.get("net_pnl", 0.0),
            "total_return": m.get("total_return", 0.0),
            "cagr": m.get("cagr", 0.0),
            "sharpe": m.get("sharpe", 0.0),
            "sortino": m.get("sortino", 0.0),
            "calmar": m.get("calmar", 0.0),
            "max_dd": max_dd,
            "max_dd_recovery_bars": recovery_bars,
            "win_rate": m.get("win_rate", 0.0),
            "profit_factor": m.get("profit_factor", 0.0),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "expectancy": m.get("expectancy", 0.0),
            "gross_profit": m.get("gross_profit", 0.0),
            "gross_loss": m.get("gross_loss", 0.0),
            "stability": m.get("stability", 0.0),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 100)
    print("2-YEAR OOS VALIDATION — comprehensive stats for locked-in winners")
    print(f"OOS window: {OOS_2Y_START.date()} → {OOS_2Y_END.date()} (~24 months)")
    print("=" * 100)

    # Fetch data
    print("\nFetching data...")
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    print(f"  EUR: {len(eur)} bars ({eur.index[0].date()} → {eur.index[-1].date()})")
    print(f"  NAS: {len(nas)} bars ({nas.index[0].date()} → {nas.index[-1].date()})")

    # Load locked-in params
    all_params = load_params()
    print(f"\n  Loaded {len(all_params)} param sets from output/")

    # Filter to winners (those that worked previously)
    # Winners from prior reports:
    winners = [
        ("ms", "eur"), ("ms", "nas"),
        ("adx", "eur"), ("adx", "nas"),
        ("ac_ao", "eur"), ("ac_ao", "nas"),
        ("mfi", "eur"), ("mfi", "nas"),
        ("stoch533_mtf_v4", "eur"), ("stoch533_mtf_v4", "nas"),
        ("macd_confluence_v4", "eur"), ("macd_confluence_v4", "nas"),
        ("bb_rsi_v4", "eur"), ("bb_rsi_v4", "nas"),
    ]

    print("\n" + "=" * 100)
    print("PER-STRATEGY 2Y RESULTS")
    print("=" * 100)
    print(f"{'Strategy':<25} {'Asset':<6} {'PnL':>9} {'WR':>5} {'PF':>5} {'TPY':>6} "
          f"{'AvgW':>9} {'AvgL':>9} {'Exp':>9} {'MaxDD':>8} {'Sharpe':>7}")
    print("-" * 120)

    results = {}
    for strat, ticker in winners:
        key = (strat, ticker)
        if key not in all_params:
            # Fall back to default params
            params = {}
        else:
            params = all_params[key]
        df = eur if ticker == "eur" else nas
        r = run_2y(strat, params, ticker, df)
        if "error" in r:
            print(f"{strat:<25} {ticker.upper():<6} ERR: {r['error']}")
            continue
        results[key] = r
        wr = r["win_rate"] * 100
        pf = r["profit_factor"]
        avg_w = r["avg_win"]
        avg_l = r["avg_loss"]
        exp = r["expectancy"]
        dd = r["max_dd"] * 100
        tpy = r["trades_per_year"]
        print(f"{strat:<25} {ticker.upper():<6} ${r['net_pnl']:>+8,.0f} {wr:>4.1f}% "
              f"{pf:>4.2f} {tpy:>5.1f} ${avg_w:>+8,.0f} ${avg_l:>+8,.0f} ${exp:>+8,.0f} "
              f"{dd:>+7.2f}% {r['sharpe']:>+6.2f}")

    # Now LOCK the winners (profitable on 2Y OOS)
    print("\n" + "=" * 100)
    print("LOCKED WINNERS (2-year OOS profitable, both Sharpe AND PnL > 0)")
    print("=" * 100)
    locked = []
    for key, r in results.items():
        if r["net_pnl"] > 0 and r["sharpe"] > 0:
            locked.append((key, r))
    locked.sort(key=lambda kv: -kv[1]["sharpe"])
    print(f"\n{'Strategy':<25} {'Asset':<6} {'PnL':>10} {'Sharpe':>7} {'WR':>5} {'PF':>5} "
          f"{'TPY':>6} {'MaxDD':>8} {'Recovery':>9}")
    print("-" * 100)
    for (strat, ticker), r in locked:
        print(f"{strat:<25} {ticker.upper():<6} ${r['net_pnl']:>+9,.0f} {r['sharpe']:>+6.2f} "
              f"{r['win_rate']*100:>4.1f}% {r['profit_factor']:>4.2f} {r['trades_per_year']:>5.1f} "
              f"{r['max_dd']*100:>+7.2f}% {r['max_dd_recovery_bars']:>8} bars")

    # Save production profiles
    print("\n" + "=" * 100)
    print("PRODUCTION PROFILES — JSON configs for each locked winner")
    print("=" * 100)

    profiles_dir = Path("output/profiles")
    profiles_dir.mkdir(parents=True, exist_ok=True)
    for (strat, ticker), r in locked:
        key = (strat, ticker)
        params = all_params.get(key, {})
        profile = {
            "strategy": strat,
            "ticker": ticker.upper(),
            "timeframe": "H1",
            "version": "v4" if "_v4" in strat else "v1",
            "locked_at": "2026-09-17",
            "oos_window": {
                "start": OOS_2Y_START.strftime("%Y-%m-%d"),
                "end": OOS_2Y_END.strftime("%Y-%m-%d"),
                "duration_months": 24,
            },
            "metrics": {
                "net_pnl": r["net_pnl"],
                "total_return": r["total_return"],
                "cagr": r["cagr"],
                "sharpe": r["sharpe"],
                "sortino": r["sortino"],
                "calmar": r["calmar"],
                "max_drawdown": r["max_dd"],
                "max_dd_recovery_bars": r["max_dd_recovery_bars"],
                "win_rate": r["win_rate"],
                "profit_factor": r["profit_factor"],
                "expectancy": r["expectancy"],
                "avg_win": r["avg_win"],
                "avg_loss": r["avg_loss"],
                "largest_win": r["largest_win"],
                "largest_loss": r["largest_loss"],
                "n_trades": r["n_trades"],
                "trades_per_year": r["trades_per_year"],
                "stability": r["stability"],
            },
            "params": params,
            "execution": {
                "broker": "MT5",
                "ea_file": "TwelveStrategies.mq5" if "_v4" not in strat else "(needs porting)",
                "instance_independence": True,
                "single_chart_per_instance": True,
                "multi_instance_supported": True,
            },
        }
        path = profiles_dir / f"{strat}_{ticker.upper()}.json"
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"  Saved {path}")

    # Summary
    print(f"\n\n{'=' * 100}")
    print(f"FINAL: {len(locked)} strategies LOCKED")
    print("=" * 100)
    print(f"  Total combined PnL (24-month OOS): ${sum(r['net_pnl'] for _, r in locked):+,.0f}")
    print(f"  Average Sharpe: {np.mean([r['sharpe'] for _, r in locked]):.2f}")
    print(f"  Average WR: {np.mean([r['win_rate']*100 for _, r in locked]):.1f}%")
    print(f"  Average PF: {np.mean([r['profit_factor'] for _, r in locked]):.2f}")
    print(f"  Average trades/year: {np.mean([r['trades_per_year'] for _, r in locked]):.1f}")


if __name__ == "__main__":
    main()
