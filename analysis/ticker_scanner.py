"""Ticker scanner — run a strategy across multiple symbols, rank by fit.

Given a strategy (path to file) and a list of symbols, fetch each symbol's
data, run the backtest, and produce a ranked scoreboard.

This answers "which ticker is the BEST fit for my strategy?" — by Sharpe,
Calmar, net P&L, win rate, etc.

Usage:
    python -m analysis.ticker_scanner --strategy fbb --symbols EURUSD,GBPUSD,USDJPY
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.live_fetcher import fetch_with_priority, load_settings
from data.mt5_export import resolve_terminal
from core.loader import load_any_strategy
from backtester.engine_full import run_full, GRID_NONE
from backtester.grid_recovery import GRID_LOSS_AND_PROFIT, GRID_NONE
from backtester.metrics_v2 import compute_all


# Common FX + crypto universe (can be overridden via CLI)
DEFAULT_UNIVERSE = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "GBPCHF",
    "BTCUSD", "ETHUSD", "XRPUSD", "SOLUSD",
]


def scan_symbols(strategy_path: str | Path,
                   symbols: Iterable[str],
                   timeframe: str = "H1",
                   start: str = "2022-01-01",
                   end: str | None = None,
                   terminal: str | None = None,
                   grid_mode: int = GRID_NONE,
                   base_lot: float = 0.1,
                   progress: bool = True) -> pd.DataFrame:
    """Run strategy on each symbol, return ranked scoreboard.

    Returns DataFrame sorted by composite score (descending).
    """
    settings = load_settings()
    terminal = terminal or settings.get("mt5_terminal")
    rows = []
    sym_list = list(symbols)
    for i, sym in enumerate(sym_list):
        if progress:
            print(f"\n[{i+1}/{len(sym_list)}] {sym} ...", flush=True)
        try:
            # Fetch data
            df, info = fetch_with_priority(sym, timeframe, start, end, terminal,
                                            allow_synthetic=True)
            if df is None or len(df) < 200:
                if progress:
                    print(f"  SKIP (no data)")
                continue
            # Load strategy (handle built-in names)
            from strategies import STRATEGY_REGISTRY
            if strategy_path in STRATEGY_REGISTRY:
                cls = STRATEGY_REGISTRY[strategy_path]
                params = {}
            else:
                cls, params, _ = load_any_strategy(strategy_path)
                if cls is None:
                    continue
            strat = cls(params=params or {})
            sig = strat.generate(df)
            entries = sig.entries.fillna(False).astype(bool)
            direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
            signals = {"primary": (entries, direction)}
            # Run backtest
            result = run_full(df, signals, base_lot=base_lot,
                                grid_mode=grid_mode)
            m = result["metrics"]
            # Composite score (Sharpe + Calmar + PF, weighted)
            sharpe = m.get("sharpe", 0)
            calmar = m.get("calmar", 0)
            pf = min(m.get("profit_factor", 0), 5.0)
            md = abs(m.get("max_drawdown", 0))
            score = (sharpe * 40 + calmar * 30 + pf * 20 + (1 - min(md, 1)) * 10)
            rows.append({
                "symbol": sym,
                "data_source": info.get("source", "?"),
                "n_bars": len(df),
                "n_trades": m.get("n_trades", 0),
                "win_rate": m.get("win_rate", 0),
                "sharpe": sharpe,
                "calmar": calmar,
                "profit_factor": pf,
                "max_drawdown": md,
                "net_pnl": m.get("net_pnl", 0),
                "final_equity": m.get("final_equity", 0),
                "composite_score": round(score, 2),
                "annual_trades": result.get("annual_trades", 0),
            })
            if progress:
                print(f"  trades={rows[-1]['n_trades']}, sharpe={sharpe:+.2f}, "
                      f"score={rows[-1]['composite_score']:.1f}")
        except Exception as e:
            import traceback
            if progress:
                print(f"  ERROR: {e}")
                traceback.print_exc()
            continue

    if not rows:
        return pd.DataFrame()
    df_ranked = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    df_ranked["rank"] = range(1, len(df_ranked) + 1)
    return df_ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, help="Path to strategy file or strategy name")
    ap.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE),
                    help="Comma-separated symbols")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--mt5-terminal", default=None)
    ap.add_argument("--out", default="output/scan_results.csv")
    ap.add_argument("--grid", action="store_true", help="Enable grid mode")
    ap.add_argument("--top", type=int, default=10, help="Show top N")
    args = ap.parse_args()

    # Resolve strategy path
    strat_path = args.strategy
    if strat_path in ("fbb", "ac_ao", "adx", "dem", "mfi", "ms", "mtf_stoch"):
        # Built-in strategy — use it directly
        from strategies import STRATEGY_REGISTRY
        cls = STRATEGY_REGISTRY[strat_path]
        # Save to a temp file so loader can read it
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(f"from strategies import {cls.__name__}\n"
                    f"class {cls.__name__}Strategy({cls.__name__}):\n  pass\n")
            strat_path = f.name
    elif not Path(strat_path).exists():
        print(f"Strategy file not found: {strat_path}")
        sys.exit(1)

    symbols = [s.strip() for s in args.symbols.split(",")]
    grid_mode = GRID_LOSS_AND_PROFIT if args.grid else GRID_NONE
    df = scan_symbols(strat_path, symbols, args.timeframe, args.start,
                       args.end, args.mt5_terminal, grid_mode)
    if df.empty:
        print("No results")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n\n=== TOP {args.top} SYMBOLS (by composite score) ===")
    print(df.head(args.top).to_string(index=False))
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()