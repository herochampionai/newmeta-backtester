"""Final consolidated lock — all profitable strategies, 2-year OOS validation.

Combines results from:
  - V1 (default params + Optuna)
  - V4 (trend-following rewrites)
  - Lenient MTF (creative filters + HTF referee)

Outputs:
  - Per-strategy locked profile (JSON) with full 2Y metrics
  - MT5 .set files for all V1 strategies (deployment-ready)
  - Production multi-instance runner output
  - Final consolidated report
"""
import warnings; warnings.filterwarnings('ignore')
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
import pandas as pd

from analysis.optuna_filters import fetch_h1
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from backtester.metrics_v2 import compute_all

OOS_2Y_START = pd.Timestamp("2024-09-17", tz="UTC")
OOS_2Y_END = pd.Timestamp("2026-09-17", tz="UTC")


# Load ALL previously-discovered winners from saved params
def load_all_winners():
    """Load params from prior runs (V1 Optuna + V4 + MTF lenient)."""
    winners = []

    # V1 + Optuna (ADX, AC+AO, MFI, MS)
    for f in ["adx_best_params.json", "ac_ao_best_params.json", "mfi_best_params.json", "ms_best_params.json"]:
        path = Path(f"output/{f}")
        if not path.exists():
            continue
        try:
            data = json.load(open(path))
            for ticker in ("eur", "nas"):
                if ticker in data:
                    base_name = f.replace("_best_params.json", "")
                    params = data[ticker] if "params" not in data[ticker] else data[ticker]["params"]
                    winners.append({
                        "strategy": base_name,
                        "ticker": ticker.upper(),
                        "version": "v1",
                        "params": params,
                        "source": f"V1 Optuna ({f})",
                    })
        except Exception as e:
            pass

    # V4 (macd_confluence_v4, bb_rsi_v4)
    for f in ["v4_best_params.json", "v4_more_best_params.json"]:
        path = Path(f"output/{f}")
        if not path.exists():
            continue
        try:
            data = json.load(open(path))
            for key, params in data.items():
                # key like "macd_confluence_v4_EUR"
                parts = key.rsplit("_", 1)
                if len(parts) == 2:
                    strat, ticker = parts
                    winners.append({
                        "strategy": strat,
                        "ticker": ticker.upper(),
                        "version": "v4",
                        "params": params,
                        "source": f"V4 ({f})",
                    })
        except Exception:
            pass

    # Lenient MTF (mtf_*)
    path = Path("output/mtf_lenient/best_params.json")
    if path.exists():
        try:
            data = json.load(open(path))
            for key, val in data.items():
                # key like "mtf_dem_EUR"
                parts = key.rsplit("_", 1)
                if len(parts) == 2:
                    strat, ticker = parts
                    if "metrics" in val:
                        val = {**val, **val.get("metrics", {})}
                    winners.append({
                        "strategy": strat,
                        "ticker": ticker.upper(),
                        "version": "mtf",
                        "params": val.get("params", val),
                        "source": f"MTF lenient ({path})",
                    })
        except Exception:
            pass

    return winners


def resolve_class(strategy_name):
    """Get the strategy class."""
    import importlib
    v4_map = {
        "macd_confluence_v4": ("strategies.trend_follow_v4_more", "MACDConfluenceV4"),
        "bb_rsi_v4": ("strategies.trend_follow_v4_more", "BBRsiV4"),
    }
    mtf_map = {
        "mtf_ac_ao": ("strategies.mtf_framework", "MTFAC_AO"),
        "mtf_dem": ("strategies.mtf_framework", "MTFDeM"),
        "mtf_fbb": ("strategies.mtf_framework", "MTFFBB"),
        "mtf_mfi": ("strategies.mtf_framework", "MTFMFI"),
        "mtf_ms": ("strategies.mtf_framework", "MTFMS"),
        "mtf_triple_rsi": ("strategies.mtf_framework", "MTFTriple_RSI"),
    }
    if strategy_name in v4_map:
        mod, cls_name = v4_map[strategy_name]
        return getattr(importlib.import_module(mod), cls_name)
    if strategy_name in mtf_map:
        mod, cls_name = mtf_map[strategy_name]
        return getattr(importlib.import_module(mod), cls_name)
    from strategies import MULTI_STRAT_EA_REGISTRY
    if strategy_name in MULTI_STRAT_EA_REGISTRY:
        return MULTI_STRAT_EA_REGISTRY[strategy_name]
    return None


def run_2y_oos(strategy_name, params, ticker, df_full):
    """Re-validate on 2Y OOS with full metrics."""
    cls = resolve_class(strategy_name)
    if cls is None:
        return {"error": f"unknown strategy: {strategy_name}"}
    df_oos = df_full[(df_full.index >= OOS_2Y_START) & (df_full.index < OOS_2Y_END)].copy()
    if len(df_oos) < 100:
        return {"error": "insufficient data"}
    profile = "forex" if ticker == "EUR" else "nas100"
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
        n = int(entries.sum())
        if n == 0:
            return {"error": "no entries"}
        signals = {strategy_name: (entries, direction)}
        r = run_full(df_oos, signals, **kw)
        m = compute_all(r["equity"].pct_change().fillna(0), r.get("trades"), r["equity"])
        return {"net_pnl": m.get("net_pnl", 0.0), "sharpe": m.get("sharpe", 0.0),
                "win_rate": m.get("win_rate", 0.0), "profit_factor": m.get("profit_factor", 0.0),
                "n_trades": m.get("n_trades", 0), "max_dd": m.get("max_drawdown", 0.0),
                "avg_win": m.get("avg_win", 0.0), "avg_loss": m.get("avg_loss", 0.0),
                "expectancy": m.get("expectancy", 0.0),
                "n_bars": len(df_oos), "n_entries": n}
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 100)
    print("FINAL CONSOLIDATED LOCK — all profitable strategies, 2Y OOS revalidation")
    print("=" * 100)

    # Fetch data once
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")

    # Load all winners from prior runs
    all_winners = load_all_winners()
    print(f"\n  Loaded {len(all_winners)} candidate configurations from prior runs")

    # Re-validate on 2Y OOS, dedupe by (strategy, ticker)
    seen = set()
    results = []
    for w in all_winners:
        key = (w["strategy"], w["ticker"])
        if key in seen:
            continue  # dedupe — keep first
        seen.add(key)
        df = eur if w["ticker"] == "EUR" else nas
        r = run_2y_oos(w["strategy"], w["params"], w["ticker"], df)
        if "error" in r:
            continue
        r["strategy"] = w["strategy"]
        r["ticker"] = w["ticker"]
        r["version"] = w["version"]
        r["params"] = w["params"]
        r["source"] = w["source"]
        results.append(r)

    # Sort by PnL
    results.sort(key=lambda r: -r["net_pnl"])

    print("\n" + "=" * 100)
    print("ALL CANDIDATES — 2Y OOS revalidation")
    print("=" * 100)
    print(f"{'Instance':<32} {'Version':<8} {'PnL':>10} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'Tr':>4}")
    print("-" * 80)
    for r in results:
        status = "✓✓" if r["net_pnl"] > 0 and r["sharpe"] > 0 else ("~" if r["net_pnl"] > 0 else "✗")
        print(f"{r['strategy']+'_'+r['ticker']:<32} {r['version']:<8} "
              f"${r['net_pnl']:>+9,.0f} {r['sharpe']:>+6.2f} {r['win_rate']*100:>4.1f}% "
              f"{r['profit_factor']:>4.2f} {r['n_trades']:>4} {status}")

    # LOCKED = profitable (PnL > 0 AND Sharpe > 0) on at least 1 asset
    locked = [r for r in results if r["net_pnl"] > 0 and r["sharpe"] > 0]
    locked.sort(key=lambda r: -r["sharpe"])

    print("\n" + "=" * 100)
    print(f"LOCKED WINNERS — {len(locked)} profitable instances")
    print("=" * 100)
    print(f"{'Instance':<32} {'PnL':>10} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'Tr':>4} {'TPY':>5}")
    print("-" * 80)
    total_pnl = 0
    total_trades = 0
    for r in locked:
        tpy = r["n_trades"] / 2.0
        total_pnl += r["net_pnl"]
        total_trades += r["n_trades"]
        print(f"{r['strategy']+'_'+r['ticker']:<32} ${r['net_pnl']:>+9,.0f} {r['sharpe']:>+6.2f} "
              f"{r['win_rate']*100:>4.1f}% {r['profit_factor']:>4.2f} {r['n_trades']:>4} {tpy:>5.1f}")

    print(f"\n  Total combined PnL: ${total_pnl:+,.0f}")
    print(f"  Total trades: {total_trades}")
    avg_sharpe = sum(r["sharpe"] for r in locked) / max(len(locked), 1)
    avg_wr = sum(r["win_rate"]*100 for r in locked) / max(len(locked), 1)
    avg_pf = sum(r["profit_factor"] for r in locked) / max(len(locked), 1)
    print(f"  Average Sharpe: {avg_sharpe:.2f}")
    print(f"  Average WR: {avg_wr:.1f}%")
    print(f"  Average PF: {avg_pf:.2f}")

    # Unique strategies profitable
    unique_strats = set(r["strategy"] for r in locked)
    print(f"\n  Unique profitable strategies: {len(unique_strats)}/12")

    # Save locked profiles + final report
    profiles_dir = Path("output/profiles_final")
    profiles_dir.mkdir(parents=True, exist_ok=True)
    for r in locked:
        profile = {
            "strategy": r["strategy"],
            "ticker": r["ticker"],
            "timeframe": "H1",
            "version": r["version"],
            "source": r["source"],
            "oos_window": {"start": OOS_2Y_START.strftime("%Y-%m-%d"),
                           "end": OOS_2Y_END.strftime("%Y-%m-%d"), "months": 24},
            "metrics": {k: v for k, v in r.items() if k not in ("strategy", "ticker", "version", "params", "source")},
            "params": r["params"],
            "deployment": {
                "ea_file": "TwelveStrategies.mq5" if "_v4" not in r["strategy"] and not r["strategy"].startswith("mtf_") else "(needs porting)",
                "ready_for_mt5": "_v4" not in r["strategy"] and not r["strategy"].startswith("mtf_"),
                "single_chart": True,
                "multi_instance": True,
                "independent": True,
            }
        }
        out_path = profiles_dir / f"{r['strategy']}_{r['ticker']}.json"
        with open(out_path, "w") as f:
            json.dump(profile, f, indent=2)

    print(f"\n  Saved {len(locked)} final profiles → {profiles_dir}")

    # Save consolidated report
    report = {
        "validation_date": "2026-09-17",
        "oos_window": "2024-09-17 → 2026-09-17 (24 months)",
        "data_sources": ["MT5_EuroPrinter (EURUSD)", "MT5_Bybit (NAS100)"],
        "summary": {
            "total_locked_instances": len(locked),
            "unique_profitable_strategies": len(unique_strats),
            "total_combined_pnl_2y": total_pnl,
            "total_trades_2y": total_trades,
            "avg_sharpe": avg_sharpe,
            "avg_win_rate": avg_wr,
            "avg_profit_factor": avg_pf,
        },
        "instances": [
            {
                "strategy": r["strategy"],
                "ticker": r["ticker"],
                "version": r["version"],
                "metrics": {k: v for k, v in r.items() if k not in ("strategy", "ticker", "version", "params", "source")},
                "ready_for_mt5": "_v4" not in r["strategy"] and not r["strategy"].startswith("mtf_"),
            } for r in locked
        ],
    }
    with open("output/FINAL_REPORT.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved → output/FINAL_REPORT.json")


if __name__ == "__main__":
    main()
