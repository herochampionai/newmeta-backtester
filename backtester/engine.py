"""Vectorized backtest via vectorbt Portfolio.from_signals.
Inputs: OHLCV df, Signals dataclass (entries, exits, direction).
Output: vectorbt Portfolio + summary dict (Sharpe, Sortino, Calmar, MaxDD, ...)."""
from __future__ import annotations
import pandas as pd
import numpy as np
import vectorbt as vbt
from .metrics import metrics_from_returns


def run(df: pd.DataFrame, entries: pd.Series, exits: pd.Series,
        init_cash: float = 10_000.0, commission: float = 7e-5,
        slippage: float = 3e-5) -> tuple[vbt.Portfolio, dict]:
    """Vectorized backtest. `entries` is bool (entry moments), `exits` is bool (exit moments).
    Returns (portfolio, summary dict)."""
    price = df["close"]
    pf = vbt.Portfolio.from_signals(
        close=price,
        entries=entries.fillna(False).astype(bool),
        exits=exits.fillna(False).astype(bool),
        init_cash=init_cash,
        fees=commission,
        slippage=slippage,
        freq="h",
    )
    rets = pf.returns()
    summary = metrics_from_returns(rets, periods_per_year=252 * 24)
    summary["trades"] = int(pf.trades.count()) if pf.trades.count() is not None else 0
    summary["final_equity"] = float(pf.final_value())
    return pf, summary


def run_direction(df: pd.DataFrame, entries: pd.Series, direction,
                  init_cash: float = 10_000.0, commission: float = 7e-5,
                  slippage: float = 3e-5) -> tuple[vbt.Portfolio, dict]:
    """Long/short backtest using direction series. direction ∈ {-1, 0, +1}.
    vectorbt 1.1 uses `direction` arg directly. Accepts Series or ndarray."""
    entries = entries.fillna(False).astype(bool)
    if not isinstance(direction, pd.Series):
        direction = pd.Series(direction, index=entries.index)
    direction = direction.fillna(0).astype(int)
    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entries,
        direction=direction,
        init_cash=init_cash,
        fees=commission,
        slippage=slippage,
        freq="h",
    )
    rets = pf.returns()
    summary = metrics_from_returns(rets, periods_per_year=252 * 24)
    summary["trades"] = int(pf.trades.count() or 0)
    summary["final_equity"] = float(pf.final_value())
    return pf, summary


def run_portfolio(returns_matrix: pd.DataFrame, weights: np.ndarray,
                 init_cash: float = 10_000.0) -> dict:
    """Combine per-strategy bar returns using fixed weights.
    returns_matrix: bar x strategy."""
    port_ret = (returns_matrix.values * weights).sum(axis=1)
    port_ret = pd.Series(port_ret, index=returns_matrix.index)
    equity = (1 + port_ret).cumprod() * init_cash
    summary = metrics_from_returns(port_ret, periods_per_year=252 * 24)
    summary["weights"] = dict(zip(returns_matrix.columns, weights.tolist()))
    summary["final_equity"] = float(equity.iloc[-1])
    return summary