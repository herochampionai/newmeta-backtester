"""MQL5 equivalence test — runs the Python harness with the same params
the EA uses, and produces a side-by-side comparison report.

This script assumes you've ALREADY run the EA in MT5 Strategy Tester and
exported trades to tester_trades.csv using tools/export_tester_deals.mq5.

If you don't have that CSV yet, this script also runs the harness only,
producing a "harness-only" report so you can validate the harness logic
independently before doing the full side-by-side.

Usage:
    # Side-by-side (you have MT5 tester_trades.csv):
    python -m analysis.mql5_equivalence_test --strategy fbb \\
        --mt5-trades tester_trades.csv --data EURUSD_H1

    # Harness-only (no MT5 yet):
    python -m analysis.mql5_equivalence_test --strategy fbb --data EURUSD_H1 --harness-only
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.live_fetcher import fetch_with_priority
from data.mt5_export import resolve_terminal
from analysis.mt5_trade_parser import parse_mt5_deals_csv
from core.loader import load_any_strategy
from backtester.engine_full import run_full, GRID_NONE
from analysis.mql5_compare import MQL5_DEFAULTS, compare


def run_equivalence(strategy_name: str, df: pd.DataFrame,
                     mt5_csv: str | None = None,
                     grid_mode: int = GRID_NONE) -> dict:
    """Run both Python harness and (optionally) MT5 trades for comparison."""
    # 1. Run Python harness
    cls = STRATEGY_REGISTRY_get(strategy_name)
    p = MQL5_DEFAULTS.get(strategy_name, {})
    strat = cls(params=p)
    sig = strat.generate(df)
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    signals = {"primary": (entries, direction)}
    result = run_full(df, signals, grid_mode=grid_mode, base_lot=0.1)
    py_trades = result["trades"]

    report = {
        "strategy": strategy_name,
        "harness": {
            "n_trades": len(py_trades),
            "net_pnl": float(py_trades["pnl"].sum()) if not py_trades.empty and "pnl" in py_trades.columns else 0,
            "sharpe": result["metrics"]["sharpe"],
            "max_dd": result["metrics"]["max_drawdown"],
        },
        "mt5": None,
        "comparison": None,
    }

    if mt5_csv:
        mt5_trades = parse_mt5_deals_csv(mt5_csv)
        report["mt5"] = {
            "n_trades": len(mt5_trades),
            "net_pnl": float(mt5_trades["pnl"].sum()) if not mt5_trades.empty else 0,
        }
        if not py_trades.empty and not mt5_trades.empty:
            report["comparison"] = compare(mt5_trades, py_trades)
    return report


def STRATEGY_REGISTRY_get(name: str):
    from strategies import STRATEGY_REGISTRY
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}")
    return STRATEGY_REGISTRY[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--data", default="EURUSD_H1")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--mt5-trades", default=None, help="Path to tester_trades.csv from MT5")
    ap.add_argument("--mt5-terminal", default=None)
    ap.add_argument("--harness-only", action="store_true")
    ap.add_argument("--grid", action="store_true", help="Enable grid mode")
    ap.add_argument("--out", default="output/equivalence_report.json")
    args = ap.parse_args()

    sym, tf = args.data.split("_")
    settings = (Path("config/settings.yaml").read_text()
                if Path("config/settings.yaml").exists() else "{}")
    import yaml
    s = yaml.safe_load(settings) if settings else {}
    terminal = args.mt5_terminal or s.get("mt5_terminal")

    print(f"Loading data {sym}/{tf} from {args.start}...")
    df, info = fetch_with_priority(sym, tf, args.start,
                                     args.end if args.end else None,
                                     terminal, allow_synthetic=True)
    print(f"  source: {info.get('source')}, {len(df)} bars")

    print(f"\nRunning Python harness with {args.strategy}...")
    grid_mode = 3 if args.grid else 0  # GRID_LOSS_AND_PROFIT = 3
    report = run_equivalence(args.strategy, df,
                              mt5_csv=args.mt5_trades if not args.harness_only else None,
                              grid_mode=grid_mode)
    print(json.dumps(report, indent=2, default=str))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to {args.out}")


if __name__ == "__main__":
    main()