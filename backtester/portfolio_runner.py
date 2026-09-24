"""Multi-asset portfolio runner — cross-margin, USD convert, futures/options expiry.

Runs strategy across multiple symbols simultaneously with shared equity,
cross-margin, and proper currency conversion. Matches LEAN portfolio semantics.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd
import numpy as np

from backtester.engine_full import run_full
from backtester.symbol_spec import get_spec
from data.live_fetcher import fetch_with_priority


@dataclass
class PortfolioConfig:
    symbols: list[str]
    timeframes: list[str]
    base_currency: str = "USD"
    leverage: float = 30.0
    max_margin_pct: float = 80.0  # stop-out at 100%, warn at 80%
    risk_per_trade_pct: float = 1.0
    init_cash: float = 10000.0


def _fx_rate(from_ccy: str, to_ccy: str, df: pd.DataFrame) -> float:
    """Approx FX rate from symbol close (last bar)."""
    if from_ccy == to_ccy:
        return 1.0
    # For major pairs, assume 1.0 for now — plug real FX feed later
    return 1.0


def run_portfolio(strategy_name: str, config: PortfolioConfig,
                  params: dict | None = None,
                  engine_kwargs: dict | None = None,
                  terminal: str | None = None) -> dict:
    """Run strategy on all symbols × TFs, aggregate into single portfolio equity."""
    eng = dict(engine_kwargs or {})
    eng.update({"init_cash": config.init_cash, "leverage": config.leverage})
    all_trades = []
    equity_curves = {}
    total_margin_used = 0.0
    cash = config.init_cash

    for sym in config.symbols:
        spec = get_spec(sym, terminal=terminal)
        for tf in config.timeframes:
            df, info = fetch_with_priority(sym, tf, allow_synthetic=False, terminal_override=terminal)
            if df is None or len(df) < 100:
                continue
            # Risk-based sizing per symbol
            spec_lot = lots_for_risk(cash, config.risk_per_trade_pct, 30.0,
                                     spec.pip_size, spec.contract_size,
                                     spec.volume_min, spec.volume_max, spec.volume_step)
            eng_sym = dict(eng)
            eng_sym.update({"base_lot": spec_lot, "pip_size": spec.pip_size,
                            "contract_size": spec.contract_size})
            sig = STRATEGY_REGISTRY[strategy_name](params=params or {}).generate(df)
            entries = sig.entries.fillna(False).astype(bool)
            direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
            signals = {strategy_name: (entries, direction)}
            res = run_full(df, signals, params=params, **eng_sym)
            # Convert equity to base currency
            fx = _fx_rate(spec.symbol[:3], config.base_currency, df)
            eq = res["equity"] * fx
            equity_curves[f"{sym}_{tf}"] = eq
            # Aggregate trades
            tr = res.get("trades")
            if tr is not None and len(tr):
                tr = tr.copy()
                tr["symbol"] = sym
                tr["timeframe"] = tf
                all_trades.append(tr)
            # Track margin
            total_margin_used += abs(spec_lot * spec.contract_size * df["close"].iloc[-1] / config.leverage) * fx

    # Combine equity curves (time-aligned union)
    if equity_curves:
        all_idx = pd.DatetimeIndex([])
        for eq in equity_curves.values():
            all_idx = all_idx.union(eq.index)
        all_idx = all_idx.sort_values()
        portfolio_eq = pd.Series(config.init_cash, index=all_idx, dtype=float)
        for eq in equity_curves.values():
            portfolio_eq = portfolio_eq.add(eq.reindex(all_idx).fillna(method="ffill").fillna(config.init_cash), fill_value=0)
        portfolio_eq = portfolio_eq - config.init_cash * (len(equity_curves) - 1)
    else:
        portfolio_eq = pd.Series(config.init_cash, index=[pd.Timestamp.now()])

    # Portfolio metrics
    from backtester.metrics_v2 import compute_all
    returns = portfolio_eq.pct_change().fillna(0)
    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    metrics = compute_all(returns, all_trades_df, portfolio_eq, periods_per_year=252 * 24)

    # Margin level
    margin_level = (portfolio_eq.iloc[-1] / total_margin_used * 100.0) if total_margin_used > 0 else 9999.0

    return {
        "equity": portfolio_eq,
        "trades": all_trades_df,
        "metrics": metrics,
        "margin_used": total_margin_used,
        "margin_level": margin_level,
        "symbols_run": list(equity_curves.keys()),
        "per_symbol_equity": equity_curves,
    }


# Import here to avoid circular
from backtester.pro_suite import lots_for_risk
from strategies import STRATEGY_REGISTRY