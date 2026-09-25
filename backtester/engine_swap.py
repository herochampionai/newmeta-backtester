"""Swap overlay — adds Wed-3× / holiday-aware swap to equity curve.

This is a POST-PROCESS overlay. Takes a base backtest result, computes the swap
series from each closed trade's direction and lot, and applies it to the equity
curve.

Honest semantics: only runs when swap_enabled=True. The flag in app.py / CLI
controls whether swap is applied at all.
"""
from __future__ import annotations

import pandas as pd

from backtester.swaps import compute_swap_series


def apply_swap(result: dict,
                df: pd.DataFrame,
                long_swap_pips: float = -0.5,
                short_swap_pips: float = 0.2,
                pip_size: float = 0.0001,
                contract_size: float = 100_000,
                triple_day: int = 2,
                spec=None) -> dict:
    """Apply direction-aware swap overlay to a backtest result.

    Reconstructs per-bar direction from trade entry/exit bars (no averaging).
    If spec (SymbolSpec) given, its swap_long/short + triple_day win over args.
    """
    from backtester.engine_pure import infer_periods_per_year
    if spec is not None:
        try:
            long_swap_pips = float(getattr(spec, "swap_long_pips", long_swap_pips))
            short_swap_pips = float(getattr(spec, "swap_short_pips", short_swap_pips))
            triple_day = int(getattr(spec, "triple_day", triple_day))
            pip_size = float(getattr(spec, "pip_size", pip_size))
            contract_size = float(getattr(spec, "contract_size", contract_size))
        except Exception:
            pass
    trades = result.get("trades")
    equity = result.get("equity")
    if trades is None or trades.empty or equity is None:
        result["swap_total"] = 0.0
        result.setdefault("active_overlays", []).append("swap")
        return result
    # Per-bar lots + direction from trade windows (entry_bar..exit_bar)
    n = len(df)
    pos_sizes = pd.Series(0.0, index=df.index)
    pos_dir = pd.Series(0.0, index=df.index)
    try:
        for _, t in trades.iterrows():
            eb = int(t.get("entry_bar", 0)); xb = int(t.get("exit_bar", eb))
            eb = max(0, min(eb, n - 1)); xb = max(eb, min(xb, n - 1))
            d = float(t.get("direction", 1) or 1)
            lots = 0.1
            for lk in ("lots", "lot", "volume"):
                if lk in t and t[lk] is not None:
                    try:
                        v = t[lk]
                        lots = float(sum(v)) if isinstance(v, list) else float(v)
                        break
                    except Exception:
                        pass
            pos_sizes.iloc[eb:xb + 1] += lots
            # direction = signed sum; keep last non-zero for mixed windows
            seg = pos_dir.iloc[eb:xb + 1]
            pos_dir.iloc[eb:xb + 1] = seg + d * lots
        pos_dir = pos_dir.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    except Exception:
        pos_sizes = pd.Series(0.1, index=df.index)
        pos_dir = pd.Series(1.0, index=df.index)
    swap = compute_swap_series(df,
                                long_swap_pips=long_swap_pips,
                                short_swap_pips=short_swap_pips,
                                position_sizes=pos_sizes.abs(),
                                direction_series=pos_dir,
                                triple_day=triple_day,
                                pip_size=pip_size, contract_size=contract_size)
    # Add swap to equity
    new_equity = equity + swap.cumsum()
    result["equity"] = new_equity
    result["swap_total"] = float(swap.sum())
    # Recompute metrics with swap-adjusted equity
    from backtester.metrics_v2 import compute_all
    returns = new_equity.pct_change().fillna(0)
    metrics = compute_all(returns, trades, new_equity, periods_per_year=infer_periods_per_year(df))
    result["metrics"] = metrics
    result.setdefault("active_overlays", []).append("swap")
    return result
