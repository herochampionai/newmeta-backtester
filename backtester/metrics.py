"""Standard backtest metrics. All annualized unless noted."""
from __future__ import annotations
import numpy as np
import pandas as pd


def metrics_from_returns(r: pd.Series, periods_per_year: int = 252 * 8) -> dict:
    """r is bar-level return (e.g., H1 → 8 trading hours/day)."""
    r = r.dropna()
    if r.empty:
        return {"error": "empty returns"}
    equity = (1 + r).cumprod()
    years = len(r) / periods_per_year
    total_ret = equity.iloc[-1] - 1
    cagr = (equity.iloc[-1]) ** (1 / max(years, 1e-9)) - 1
    vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (cagr / vol) if vol > 0 else 0.0
    downside = r[r < 0].std() * np.sqrt(periods_per_year)
    sortino = (cagr / downside) if downside > 0 else 0.0
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0
    wins = (r > 0).sum()
    total = (r != 0).sum()
    win_rate = wins / total if total else 0.0
    return {
        "total_return": float(total_ret),
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "n_bars": int(len(r)),
        "n_positive_bars": int(wins),
        "win_rate": float(win_rate),
        "end_equity": float(equity.iloc[-1]),
    }