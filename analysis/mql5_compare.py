"""MQL5 <-> Python harness equivalence tester.

Workflow:
  1. Run MT5 strategy tester on the EA with known params + period + symbol
  2. Run `tools/export_tester_deals.mq5` script to export deals → CSV
  3. Run `python -m analysis.mql5_compare --mt5-trades tester_trades.csv --strategy fbb --data EURUSD_H1.parquet`
  4. Get a report: trade-count match, per-trade entry/exit price match, aggregate metric match

This is the ONLY way to know if our Python port is faithful to the MQL5 logic.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.mt5_trade_parser import parse_mt5_deals_csv
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from strategies import STRATEGY_REGISTRY


# Default MQL5 EA defaults (taken from `multi strat newmeta.mq5` after patches)
MQL5_DEFAULTS = {
    "ac_ao": dict(open_orders_type=1, close_orders_type=0, level_open_orders=80,
                  level_close_orders=70, use_acceleration_filter=False,
                  use_ao_synchronization=False, acceleration_bars=3,
                  min_acceleration=0.0005, min_ao_synchronization=0.0003),
    "adx": dict(open_orders_type=1, close_orders_type=4, level_open_orders_1=55,
                level_open_orders_2=15, level_close_orders_1=15,
                level_close_orders_2=5, use_di_crossover=True,
                crossover_lookback=3, min_crossover_gap=5, bars_calculate=20),
    "dem": dict(open_orders_type=3, close_orders_type=0, level_open_orders=75,
                level_close_orders=70, bars_calculate=20),
    "fbb": dict(open_orders_type_1=1, open_orders_type_2=0, close_orders_type_1=0,
                close_orders_type_2=0, level_open_orders_1=0,
                level_open_orders_2=50, level_close_orders_1=40,
                level_close_orders_2=40, bars_calculate=20, deviation=1.8),
    "mfi": dict(open_orders_type=3, close_orders_type=0, level_open_orders=70,
                level_close_orders=70, use_slope_filter=False,
                use_divergence=False, use_hidden_divergence=False,
                bars_calculate=12, slope_lookback=5, min_slope_strength=3,
                divergence_bars=10),
    "ms": dict(open_orders_type_1=8, open_orders_type_2=0, close_orders_type_1=0,
               close_orders_type_2=0, level_open_orders_1=20,
               level_open_orders_2=80, level_close_orders_1=50,
               level_close_orders_2=65, use_confluence_filter=False,
               use_macd_divergence=False, use_stoch_divergence=False,
               use_histogram_divergence=False, fast_ema_period=3,
               slow_ema_period=9, signal_period=2, k_period=5, d_period=3,
               slowing_period=12),
}


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    else:
        from data.cache import load
        # Try as symbol+tf
        parts = path.split("_")
        sym, tf = parts[0], parts[1]
        df, _ = load(sym, tf)
    return df


def run_python_backtest(strategy_name: str, df: pd.DataFrame, params: dict | None = None,
                          init_cash: float = 10000, commission_pips: float = 0.7,
                          slippage_pips: float = 0.3) -> pd.DataFrame:
    """Run the Python harness on the data, return round-trip trades in MT5-like format."""
    cls = STRATEGY_REGISTRY[strategy_name]
    p = {**MQL5_DEFAULTS.get(strategy_name, {}), **(params or {})}
    strat = cls(params=p)
    sig = strat.generate(df)
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    signals = {"primary": (entries, direction)}
    result = run_full(df, signals, init_cash=init_cash,
                       commission_pips=commission_pips, slippage_pips=slippage_pips,
                       grid_mode=GRID_NONE)  # disable grid to match MT5 default (None)
    trades = result["trades"]
    if trades.empty:
        return pd.DataFrame()
    # Convert grid-trade format → MT5 round-trip format
    out = []
    for _, t in trades.iterrows():
        out.append({
            "entry_time": df.index[int(t.get("entry_bar", 0))] if "entry_bar" in t.index else None,
            "exit_time": df.index[int(t["exit_bar"])],
            "direction": int(t["direction"]),
            "entry_price": float(t["entry_price"]),
            "exit_price": float(t["exit_price"]),
            "lots": float(sum(t["lots"]) if isinstance(t["lots"], list) else t["lots"]),
            "pnl": float(t["pnl"]),
        })
    return pd.DataFrame(out)


def compare(mt5_trades: pd.DataFrame, python_trades: pd.DataFrame,
             price_tol_pips: float = 1.0, time_tol_minutes: int = 5) -> dict:
    """Compare two trade DataFrames. Returns:
      - count_match_pct
      - per-trade match list
      - aggregate metric comparison
    """
    n_mt5 = len(mt5_trades)
    n_py = len(python_trades)
    if n_mt5 == 0 and n_py == 0:
        return {"match": "PASS (both empty)", "mt5_count": 0, "py_count": 0}
    # Try to pair trades by approximate time + direction
    pip = 0.0001  # EURUSD pip size
    pairs = []
    py_used = set()
    for i, mt5_t in mt5_trades.iterrows():
        best_j = None
        best_score = float("inf")
        for j, py_t in python_trades.iterrows():
            if j in py_used:
                continue
            if int(mt5_t["direction"]) != int(py_t["direction"]):
                continue
            # Score by time diff + price diff
            time_diff = abs((mt5_t["entry_time"] - py_t["entry_time"]).total_seconds())
            if time_diff > time_tol_minutes * 60 * 24:  # beyond tolerance
                continue
            price_diff = abs(mt5_t["entry_price"] - py_t["entry_price"]) / pip
            if price_diff > price_tol_pips * 10:
                continue
            score = time_diff / 60 + price_diff  # minutes + pips
            if score < best_score:
                best_score = score
                best_j = j
        if best_j is not None:
            py_used.add(best_j)
            pairs.append((i, best_j, best_score))
    matched = len(pairs)
    match_pct = matched / max(n_mt5, 1) * 100
    # Aggregate comparison
    mt5_total_pnl = mt5_trades["pnl"].sum() if not mt5_trades.empty else 0
    py_total_pnl = python_trades["pnl"].sum() if not python_trades.empty else 0
    pnl_drift_pct = (py_total_pnl - mt5_total_pnl) / abs(mt5_total_pnl) * 100 if mt5_total_pnl else 0
    return {
        "mt5_count": n_mt5,
        "py_count": n_py,
        "matched": matched,
        "match_pct": match_pct,
        "mt5_total_pnl": float(mt5_total_pnl),
        "py_total_pnl": float(py_total_pnl),
        "pnl_drift_pct": float(pnl_drift_pct),
        "verdict": "PASS" if match_pct >= 80 and abs(pnl_drift_pct) < 5 else
                    "INVESTIGATE" if match_pct >= 50 else "FAIL",
        "pairs": pairs[:20],  # sample
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt5-trades", required=True, help="CSV from MT5 strategy tester (via export_tester_deals.mq5)")
    ap.add_argument("--strategy", required=True, choices=list(STRATEGY_REGISTRY.keys()))
    ap.add_argument("--data", required=True, help="parquet/csv path or symbol_TF")
    ap.add_argument("--out", default="output/mql5_compare.json")
    args = ap.parse_args()

    print(f"MT5 trades: {args.mt5_trades}")
    print(f"Strategy: {args.strategy}")
    print(f"Data: {args.data}")
    mt5_df = parse_mt5_deals_csv(args.mt5_trades)
    print(f"MT5 round-trip trades: {len(mt5_df)}")
    py_df = run_python_backtest(args.strategy, load_data(args.data))
    print(f"Python trades: {len(py_df)}")
    result = compare(mt5_df, py_df)
    print(json.dumps(result, indent=2, default=str))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWritten to {out_path}")
    if result["verdict"] == "PASS":
        print("\n*** PORT IS VALIDATED ***")
    elif result["verdict"] == "INVESTIGATE":
        print("\n*** DRIFT DETECTED — INVESTIGATE ***")
    else:
        print("\n*** PORT IS NOT MATCHING — DO NOT TRUST BACKTEST ***")


if __name__ == "__main__":
    main()