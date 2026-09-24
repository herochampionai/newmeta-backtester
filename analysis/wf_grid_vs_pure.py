"""Walk-forward: grid OFF vs grid ON.

Tests whether adding GRID_LOSS_AND_PROFIT + recovery to a core strategy
actually improves out-of-sample performance.  Each walk-forward window:
  1. Generate signals from the strategy on the test window
  2. Run `run_full(grid_mode=GRID_NONE)` — pure signal performance
  3. Run `run_full(grid_mode=GRID_LOSS_AND_PROFIT)` — same signals + grid overlay
  4. Compare Sharpe, Calmar, total return, n_trades

The grid is an OPTIONAL overlay.  If it doesn't improve OOS, the strategy ships
without it (default).

Usage:
    PYTHONPATH=. python -m analysis.wf_grid_vs_pure --strategy fbb --symbol EURUSD --timeframe H1 --train-months 36 --test-months 6 --roll-months 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtester.engine_full import GRID_LOSS_AND_PROFIT, GRID_NONE, run_full  # noqa: E402
from backtester.grid_recovery import RECOVERY_HIGHER_PROFITS  # noqa: E402
from core.surgical_features import get_feature_defaults  # noqa: E402
from data.cache import load as load_cache  # noqa: E402
from strategies import STRATEGY_REGISTRY  # noqa: E402


def run_window(df: pd.DataFrame, strategy_name: str, params: dict, grid_mode: int) -> dict:
    """Run run_full on a window. Returns metrics dict."""
    cls = STRATEGY_REGISTRY[strategy_name]
    inst = cls(params=params)
    sig = inst.generate(df)
    signals = {strategy_name: (sig.entries, sig.direction)}

    if grid_mode == GRID_NONE:
        result = run_full(df, signals, grid_mode=GRID_NONE, params=params)
    else:
        result = run_full(df, signals,
                           grid_mode=GRID_LOSS_AND_PROFIT,
                           recovery_mode=RECOVERY_HIGHER_PROFITS,
                           base_lot=0.1,
                           grid_take_profit=50,
                           grid_stop_loss=200,
                           max_grid_layers=4,
                           pips_between_orders=30,
                           grid_lot_multiplier=1.5,
                           recovery_lot_multiplier=2.0,
                           params=params)
    return result["metrics"]


def wf_windows(df: pd.DataFrame, train_months: int, test_months: int,
               roll_months: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Rolling train/test windows for walk-forward."""
    start = df.index[0]
    end = df.index[-1]
    cursor = start
    windows = []
    while True:
        tr_end = cursor + pd.DateOffset(months=train_months)
        te_end = tr_end + pd.DateOffset(months=test_months)
        if te_end > end:
            break
        train = df[(df.index >= cursor) & (df.index < tr_end)]
        test = df[(df.index >= tr_end) & (df.index < te_end)]
        if len(train) < 200 or len(test) < 50:
            cursor += pd.DateOffset(months=roll_months)
            continue
        windows.append((train, test))
        cursor += pd.DateOffset(months=roll_months)
    return windows


def main():
    ap = argparse.ArgumentParser(description="Walk-forward: grid OFF vs grid ON")
    ap.add_argument("--strategy", default="fbb")
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--train-months", type=int, default=36)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--roll-months", type=int, default=3)
    args = ap.parse_args()

    df, meta = load_cache(args.symbol, args.timeframe)
    print(f"[load] {meta.get('symbol', args.symbol)} {args.timeframe}: {len(df)} bars")
    print(f"  range: {df.index[0].date()} → {df.index[-1].date()}")

    params = get_feature_defaults()
    windows = wf_windows(df, args.train_months, args.test_months, args.roll_months)
    print(f"[wf] {len(windows)} windows (train={args.train_months}m / test={args.test_months}m)")

    rows = []
    for i, (train_df, test_df) in enumerate(windows):
        print(f"\n[window {i+1}/{len(windows)}] "
              f"test: {test_df.index[0].date()} → {test_df.index[-1].date()}")

        # Grid OFF (pure signal)
        try:
            m_pure = run_window(test_df, args.strategy, params, GRID_NONE)
            pure_sharpe = m_pure.get("sharpe", 0.0)
            pure_calmar = m_pure.get("calmar", 0.0)
            pure_ret = m_pure.get("total_return", 0.0)
            pure_trades = m_pure.get("n_trades", 0)
        except Exception as e:
            print(f"  pure: [ERROR] {e}")
            pure_sharpe = pure_calmar = pure_ret = 0.0
            pure_trades = 0

        # Grid ON
        try:
            m_grid = run_window(test_df, args.strategy, params, GRID_LOSS_AND_PROFIT)
            grid_sharpe = m_grid.get("sharpe", 0.0)
            grid_calmar = m_grid.get("calmar", 0.0)
            grid_ret = m_grid.get("total_return", 0.0)
            grid_trades = m_grid.get("n_trades", 0)
        except Exception as e:
            print(f"  grid: [ERROR] {e}")
            grid_sharpe = grid_calmar = grid_ret = 0.0
            grid_trades = 0

        sharpe_win = grid_sharpe > pure_sharpe
        print(f"  pure Sharpe={pure_sharpe:+.3f} calmar={pure_calmar:.3f} "
              f"ret={pure_ret*100:+.1f}% trades={pure_trades}")
        print(f"  grid Sharpe={grid_sharpe:+.3f} calmar={grid_calmar:.3f} "
              f"ret={grid_ret*100:+.1f}% trades={grid_trades}  "
              f"[{'GRID WINS' if sharpe_win else 'PURE WINS'}]")

        rows.append({
            "window_id": i + 1,
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            "test_bars": len(test_df),
            "pure_sharpe": pure_sharpe,
            "pure_calmar": pure_calmar,
            "pure_return": pure_ret,
            "pure_n_trades": pure_trades,
            "grid_sharpe": grid_sharpe,
            "grid_calmar": grid_calmar,
            "grid_return": grid_ret,
            "grid_n_trades": grid_trades,
            "grid_wins_sharpe": sharpe_win,
            "sharpe_delta": grid_sharpe - pure_sharpe,
        })

    res = pd.DataFrame(rows)
    out = ROOT / "output" / "wf_grid_vs_pure.csv"
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\n[saved] {out}")

    # Summary
    valid = res.dropna(subset=["pure_sharpe", "grid_sharpe"])
    n_windows = len(valid)
    grid_wins = int(valid["grid_wins_sharpe"].sum())
    print(f"\n[summary] {n_windows} windows | grid wins {grid_wins} ({grid_wins/n_windows*100:.0f}%) | pure wins {n_windows-grid_wins}")
    print(f"  pure Sharpe mean: {valid['pure_sharpe'].mean():.3f}")
    print(f"  grid Sharpe mean: {valid['grid_sharpe'].mean():.3f}")
    print(f"  Sharpe delta mean: {valid['sharpe_delta'].mean():.3f}")
    verdict = "GRID IS NET POSITIVE" if valid["sharpe_delta"].mean() > 0 and grid_wins > n_windows/2 else "KEEP PURE (grid not net positive)"
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
