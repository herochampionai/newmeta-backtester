"""Parallel walk-forward analysis across multiple strategies × multiple assets.

Designed to beat QuantConnect's baseline: runs the same TF train/test split
for every strategy on every asset, collecting OOS metrics in parallel.

Uses joblib for parallelism. Each worker runs a single strategy×asset WF
sequence and returns per-window results.

Strategy selection is explicit — user must pass --strategies. To discover
which strategies suit your tickers, run genetic_optimizer.py first for
recommendations.

Usage:
    PYTHONPATH=. python -m analysis.wf_parallel \\
        --strategies fbb,adx,ac_ao,macd_confluence \\
        --symbols EURUSD,BTCUSDT,ETHUSDT,SOLUSDT \\
        --timeframe H1 --n_jobs 8 --train_months 36 --test_months 6

    # Crypto: use any of the 9 crypto_9 pattern variants
    PYTHONPATH=. python -m analysis.wf_parallel \\
        --strategies crypto_9_breakout,crypto_9_reversal\\
        --symbols BTCUSDT,ETHUSDT \\
        --timeframe H1 --n_jobs 4

Output: output/wf_parallel_results.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from dataclasses import asdict

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from strategies import MULTI_STRAT_EA_REGISTRY, STRATEGY_REGISTRY
from strategies import CRYPTO_PATTERN_VARIANTS, CRYPTO_PATTERN_NAMES
from backtester.engine_full import run_full, GRID_NONE
from backtester.metrics_v2 import compute_all
from analysis.data_loader import load_asset
from analysis.walkforward import walk_forward, WFWindow

_ALL_STRATEGY_NAMES = list(STRATEGY_REGISTRY.keys())


def list_available_strategies() -> list[str]:
    """Return all available strategy names (including crypto_9 pattern variants)."""
    names = list(MULTI_STRAT_EA_REGISTRY.keys())
    names.append("crypto_9")
    names.extend(CRYPTO_PATTERN_VARIANTS.keys())
    return names


STRATEGY_PARAM_OVERRIDES: dict[str, dict] = {
    "fbb": {"open_orders_type_1": 2, "open_orders_type_2": 8},
    "adx": {"open_orders_type": 1, "use_zone_logic": True},
    "ac_ao": {"open_orders_type_1": 8},
    "dem": {"open_orders_type": 1},
    "bb_rsi": {},
    "triple_rsi": {},
    "mfi": {},
    "ms": {"use_confluence_filter": True},
    "mtf_stoch": {},
    "quad_stoch": {},
    "stoch533_mtf": {},
    "macd_confluence": {},
    "crypto_9": {},
}


def _resolve_strategy_name(strategy_label: str) -> str:
    """Map a strategy label to the actual strategy class name in the registry.

    Crypto pattern variants (crypto_9_breakout, etc.) map to 'crypto_9'.
    """
    if strategy_label in CRYPTO_PATTERN_VARIANTS:
        return "crypto_9"
    return strategy_label


def _default_params_for_strategy(name: str) -> dict:
    """Return minimal default params for a strategy (from strategies.yaml defaults)."""
    import yaml
    if name in CRYPTO_PATTERN_VARIANTS:
        return dict(CRYPTO_PATTERN_VARIANTS[name])
    if name == "crypto_9":
        from strategies import CRYPTO_DEFAULT_PARAMS
        defaults = dict(CRYPTO_DEFAULT_PARAMS)
        overrides = STRATEGY_PARAM_OVERRIDES.get("crypto_9", {})
        defaults.update(overrides)
        return defaults
    yaml_path = ROOT / "config" / "strategies.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path) as f:
                spec = yaml.safe_load(f).get(name, {})
            defaults: dict = {}
            for k, v in spec.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    if all(isinstance(x, (int, float)) for x in v):
                        defaults[k] = (v[0] + v[1]) / 2 if v[0] != v[1] else v[0]
                    elif len(v) == 2:
                        defaults[k] = v[0]
                    else:
                        defaults[k] = v[0]
                elif isinstance(v, (int, float, bool, str)):
                    if isinstance(v, list):
                        continue
                    defaults[k] = v
            overrides = STRATEGY_PARAM_OVERRIDES.get(name, {})
            defaults.update(overrides)
            return defaults
        except Exception:
            pass
    return STRATEGY_PARAM_OVERRIDES.get(name, {})


def _strategy_signals(df: pd.DataFrame, name: str, params: dict):
    """Generate signals for a strategy with given params."""
    cls = STRATEGY_REGISTRY[name]
    strat = cls(params=params)
    sig = strat.generate(df)
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    return {name: (entries, direction)}


def _run_single_strategy_asset(
    strategy_label: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    train_months: int,
    test_months: int,
    roll_months: int,
    n_trials: int = 20,
) -> list[dict]:
    """Run walk-forward for one strategy×asset. Returns per-window result dicts."""
    results = []
    try:
        df = load_asset(symbol, timeframe, start_date, end_date)
        if df is None or len(df) < 200:
            return []
    except Exception as e:
        print(f"  [SKIP] {strategy_label}/{symbol}: data load failed ({e})")
        return []

    strategy_name = _resolve_strategy_name(strategy_label)
    params = _default_params_for_strategy(strategy_label)

    try:
        windows = walk_forward(
            df, strategy_name, param_spec=params,
            train_months=train_months, test_months=test_months,
            roll_months=roll_months,
            n_trials=n_trials,
            periods_per_year=252 * 24 if "H1" in timeframe or timeframe == "60m" else 252,
        )
    except Exception as e:
        print(f"  [ERROR] {strategy_label}/{symbol}: WF failed ({e})")
        traceback.print_exc()
        return []

    for w in windows:
        tm = w.test_metrics
        results.append({
            "strategy": strategy_label,
            "symbol": symbol,
            "timeframe": timeframe,
            "train_start": str(w.train_start.date()),
            "train_end": str(w.train_end.date()),
            "test_start": str(w.test_start.date()),
            "test_end": str(w.test_end.date()),
            "train_sharpe": tm.get("sharpe", 0),
            "test_sharpe": tm.get("test_sharpe", 0) if "test_sharpe" in tm else tm.get("sharpe", 0),
            "train_calmar": w.train_metrics.get("calmar", 0),
            "test_calmar": w.test_metrics.get("calmar", 0),
            "test_max_dd": w.test_metrics.get("max_drawdown", 0),
            "test_n_trades": w.test_metrics.get("n_trades", 0),
            "test_win_rate": w.test_metrics.get("win_rate", 0),
            "test_total_return": w.test_metrics.get("total_return", 0),
            "test_final_equity": w.test_metrics.get("final_equity", 0),
            "idle_days": w.idle_day_count,
            "params": json.dumps(w.params) if w.params else "{}",
        })

    return results


def run_parallel(
    strategies: list[str],
    symbols: list[str],
    timeframe: str = "H1",
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    train_months: int = 36,
    test_months: int = 6,
    roll_months: int = 3,
    n_jobs: int = 4,
    n_trials: int = 20,
) -> pd.DataFrame:
    """Run WF for all strategy×asset combinations in parallel (manual mode)."""
    from joblib import Parallel, delayed

    all_combos = [(s, sym) for s in strategies for sym in symbols]
    print(f"[wf_parallel] {len(strategies)} strategies × {len(symbols)} symbols = {len(all_combos)} combos ({n_jobs} parallel)")

    def _worker(strategy_name: str, symbol: str) -> list[dict]:
        return _run_single_strategy_asset(
            strategy_name, symbol, timeframe, start_date, end_date,
            train_months, test_months, roll_months, n_trials,
        )

    all_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_worker)(s, sym) for s, sym in all_combos
    )

    flat = [row for batch in all_results for row in batch]
    df_results = pd.DataFrame(flat)
    return df_results


def main():
    ap = argparse.ArgumentParser(description="Parallel walk-forward across strategies × assets")
    ap.add_argument("--strategies", default=None,
                    help="Comma-separated strategy names (required). Crypto variants: crypto_9_breakout, crypto_9_reversal, etc.")
    ap.add_argument("--symbols", default="EURUSD,BTCUSDT,ETHUSDT,SOLUSDT",
                    help="Comma-separated symbols")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--train-months", type=int, default=36)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--roll-months", type=int, default=3)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--list-strategies", action="store_true",
                    help="List all available strategy names and exit")
    args = ap.parse_args()

    if args.list_strategies:
        names = list_available_strategies()
        print("Available strategies:")
        for n in names:
            tag = " (12-core)" if n in MULTI_STRAT_EA_REGISTRY else " (pattern)"
            print(f"  {n}{tag}")
        return 0

    if not args.strategies:
        names = list_available_strategies()
        print("ERROR: --strategies is required.")
        print(f"Available strategies: {', '.join(names)}")
        print("Tip: run with --list-strategies for full list.")
        return 1

    strategy_list = [s.strip() for s in args.strategies.split(",")]
    # Validate
    for s in strategy_list:
        resolved = _resolve_strategy_name(s)
        if resolved not in STRATEGY_REGISTRY:
            print(f"ERROR: unknown strategy '{s}'")
            return 1

    symbols = [s.strip() for s in args.symbols.split(",")]
    combos = [(s, sym) for s in strategy_list for sym in symbols]
    print(f"[wf_parallel] Manual mode: {len(strategy_list)} strategies × {len(symbols)} symbols = {len(combos)} combos")

    df_results = run_parallel(
        strategy_list, symbols,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        train_months=args.train_months,
        test_months=args.test_months,
        roll_months=args.roll_months,
        n_jobs=args.n_jobs,
        n_trials=args.n_trials,
    )

    if df_results.empty:
        print("[wf_parallel] No results — check data availability")
        return 1

    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "wf_parallel_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"[wf_parallel] Saved {len(df_results)} rows → {csv_path}")

    summary = df_results.groupby("strategy").agg(
        mean_oos_sharpe=("test_sharpe", "mean"),
        mean_oos_calmar=("test_calmar", "mean"),
        mean_max_dd=("test_max_dd", "mean"),
        total_trades=("test_n_trades", "sum"),
        total_idle_days=("idle_days", "sum"),
        n_windows=("test_sharpe", "count"),
    ).sort_values("mean_oos_sharpe", ascending=False)

    print("\n=== Per-strategy OOS summary ===")
    print(summary.to_string())

    summary_path = output_dir / "wf_parallel_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved summary → {summary_path}")

    winners = summary[summary["mean_oos_sharpe"] > 0.5]
    if len(winners) > 0:
        print(f"\n✅ Strategies with OOS Sharpe > 0.5: {', '.join(winners.index)}")
    else:
        print("\n⚠️  No strategies beating OOS Sharpe 0.5 — revisit core alpha first")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
