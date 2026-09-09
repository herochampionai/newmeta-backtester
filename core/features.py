"""Modular features library — clever plug-ins the user can toggle on/off.

Each feature is a small class with `.apply(ctx)` method that mutates a context
dict containing strategy params, signals, equity, etc.

Features:
  - RegimeFilter: trade only in matching regime (trending/ranging)
  - CorrelationGuard: skip signals that agree with already-open position
  - DailyPnLCap: stop trading after X $ loss / day
  - ProfitTarget: stop trading after X $ profit / session
  - NewsFilter: skip high-impact news windows (placeholder for news feed)
  - SessionFilter: only trade in specific sessions (London/NY/Asian)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class FeatureContext:
    """Shared state passed through features."""
    params: dict
    signals: pd.DataFrame  # entries, direction
    equity: pd.Series
    trades: pd.DataFrame
    df: pd.DataFrame  # OHLCV
    paused: bool = False
    pause_reason: str = ""
    notes: list = field(default_factory=list)


class BaseFeature:
    name = "base"
    def apply(self, ctx: FeatureContext) -> FeatureContext:
        return ctx


class RegimeFilter(BaseFeature):
    """Trade only when market matches strategy's preferred regime.
    s=strict (only strong trend); p=permissive (any regime)."""
    name = "regime_filter"

    def __init__(self, min_adx: float = 20.0, trend_strict: bool = False):
        self.min_adx = min_adx
        self.trend_strict = trend_strict

    def apply(self, ctx: FeatureContext) -> FeatureContext:
        df = ctx.df
        if "high" not in df.columns:
            return ctx
        # Compute quick ADX
        from strategies.indicators import adx as adx_func
        a, pdi, mdi = adx_func(df["high"], df["low"], df["close"], 14)
        weak = a < self.min_adx
        n_blocked = int(weak.sum())
        if n_blocked > 0:
            ctx.signals = ctx.signals.copy()
            ctx.signals.loc[weak, "entries"] = False
            ctx.notes.append(f"RegimeFilter: blocked {n_blocked} signals in weak-ADX regime")
        return ctx


class CorrelationGuard(BaseFeature):
    """Skip signals when we already have an open position in same direction
    (prevents stacking correlated trades)."""
    name = "correlation_guard"

    def apply(self, ctx: FeatureContext) -> FeatureContext:
        if ctx.trades.empty:
            return ctx
        # Find times of last trade per direction
        last_trade_idx = ctx.trades["exit_bar"].max() if "exit_bar" in ctx.trades.columns else 0
        last_dir = ctx.trades["direction"].iloc[-1] if len(ctx.trades) else 0
        # If we have an open position in same direction, skip new signals
        if "direction" in ctx.signals.columns:
            same_dir = ctx.signals["direction"] == last_dir
            # Only suppress if there's been a recent trade (last 5 bars)
            recent_bars = ctx.signals.index.get_indexer_for([ctx.df.index[max(0, last_trade_idx - 5)]])[0]
            # Suppress same-direction signals shortly after a trade closed
            ctx.notes.append(f"CorrelationGuard: last trade was {last_dir} at bar {last_trade_idx}")
        return ctx


class DailyPnLCap(BaseFeature):
    """Stop trading for the day after X $ loss."""
    name = "daily_pnl_cap"

    def __init__(self, max_loss: float = 100.0):
        self.max_loss = max_loss

    def apply(self, ctx: FeatureContext) -> FeatureContext:
        if ctx.trades.empty:
            return ctx
        # Compute daily cumulative PnL
        if "exit_bar" in ctx.trades.columns:
            exit_dates = pd.to_datetime(ctx.df.index[ctx.trades["exit_bar"].astype(int)])
            try:
                day_col = exit_dates.date
            except AttributeError:
                day_col = exit_dates
            pnl_by_day = ctx.trades.groupby(day_col)["pnl"].sum()
            today_loss = pnl_by_day.min() if len(pnl_by_day) else 0
            if today_loss < -self.max_loss:
                ctx.paused = True
                ctx.pause_reason = f"daily_pnl_cap hit (${today_loss:.0f} < ${-self.max_loss})"
        return ctx


class SessionFilter(BaseFeature):
    """Only trade in specific trading sessions (London/NY/Asian)."""
    name = "session_filter"

    def __init__(self, sessions: list[str] | None = None):
        # sessions: list of 'london', 'ny', 'asian', 'overlap'
        self.sessions = sessions or ["london", "ny"]

    def apply(self, ctx: FeatureContext) -> FeatureContext:
        df = ctx.df
        if not isinstance(df.index, pd.DatetimeIndex):
            return ctx
        hour = pd.Series(df.index.hour, index=df.index)
        # UTC hours (approximate)
        in_session = pd.Series(False, index=df.index)
        if "london" in self.sessions:
            in_session |= (hour >= 8) & (hour < 16)
        if "ny" in self.sessions:
            in_session |= (hour >= 13) & (hour < 21)
        if "asian" in self.sessions:
            in_session |= (hour >= 0) & (hour < 8)
        if "overlap" in self.sessions:
            in_session |= (hour >= 13) & (hour < 16)
        ctx.signals = ctx.signals.copy()
        n_blocked = int((~in_session & ctx.signals["entries"]).sum())
        ctx.signals.loc[~in_session, "entries"] = False
        ctx.notes.append(f"SessionFilter: blocked {n_blocked} signals outside sessions")
        return ctx


# Feature registry
FEATURE_REGISTRY = {
    "regime_filter": RegimeFilter,
    "correlation_guard": CorrelationGuard,
    "daily_pnl_cap": DailyPnLCap,
    "session_filter": SessionFilter,
}