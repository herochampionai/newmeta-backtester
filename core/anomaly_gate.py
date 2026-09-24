"""Anomaly gate — Simons "find the anomaly; if convenient act, else walk away".

The bot is allowed to be **idle/flat**.  No new entry signals are emitted
when a measurable edge does not exist.  Conditions checked:

  * **Spread health** — current spread must be below ``max_spread_pips``
    (or the rolling median of recent spreads must be reasonable).
  * **Regime confidence** — the ``RegimeAwareStrategy.detect_regimes``
    classification must not be ``"unknown"`` or ``"volatile"``
    (configurable confidence threshold).
  * **Macro context** — if ``event_blackout`` is enabled, entries are
    also silenced during event windows.

Idle days are **logged and counted** — never forced to trade.

Usage:
    from core.anomaly_gate import AnomalyGate

    gate = AnomalyGate(
        df=df,
        params=params,               # strategy params dict
        spread_series=spread,        # optional: pd.Series of spread in pips
    )
    sig = gate.apply(sig)             # returns Signals with entries zeroed-out
                                       # when the gate is closed
    gate.idle_days()                  # list of dates with no entry signals
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any
from dataclasses import dataclass

from strategies._base import Signals
from core.surgical_features import is_enabled, get_feature_params
from core.regime import detect_regimes

REGIME_CONFIDENCE_OK = {
    "trending_up", "trending_down", "ranging", "overextended", "choppy",
}


@dataclass
class GateStats:
    """Summary of gate behaviour for a backtest run."""
    total_bars: int = 0
    gated_bars: int = 0
    idle_days: list[str] = None
    idle_day_count: int = 0
    spread_violations: int = 0
    regime_violations: int = 0
    macro_violations: int = 0

    def __post_init__(self):
        if self.idle_days is None:
            self.idle_days = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_bars": self.total_bars,
            "gated_bars": self.gated_bars,
            "idle_day_count": self.idle_day_count,
            "idle_days": self.idle_days,
            "spread_violations": self.spread_violations,
            "regime_violations": self.regime_violations,
            "macro_violations": self.macro_violations,
        }


class AnomalyGate:
    """Silence strategy entry signals when the measurable edge is absent.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe with DatetimeIndex.
    params : dict | None
        Strategy params dict — read for ``anomaly_gate`` and
        ``event_blackout`` sub-dicts.
    spread_series : pd.Series | None
        Per-bar spread in pips.  If None, spread checking is skipped.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        params: dict | None = None,
        spread_series: pd.Series | None = None,
    ):
        self.df = df
        self.params = params or {}
        self.spread = spread_series
        self.regimes = detect_regimes(df)

        gate_params = get_feature_params(self.params, "anomaly_gate")
        self.min_regime_confidence = gate_params.get("min_regime_confidence", 0.6)
        self.max_spread_pips = gate_params.get("max_spread_pips", 2.0)
        self.flatten_before_event_hours = gate_params.get(
            "flatten_before_event_hours", 1.0
        )
        self.resume_after_event_mins = gate_params.get(
            "resume_after_event_mins", 30
        )

        # Lazy import to avoid circular dependency
        from core.macro_events import MacroEventCalendar
        self._macro_calendar = MacroEventCalendar()
        # Gate is a no-op when anomaly_gate feature is not enabled
        self._enabled = is_enabled(self.params, "anomaly_gate")
        self._macro_enabled = is_enabled(self.params, "event_blackout")

        self._gate_mask: pd.Series | None = None
        self._stats: GateStats | None = None

    def _spread_gate(self) -> pd.Series:
        """True where spread is too high (gate should close)."""
        if self.spread is None:
            return pd.Series(False, index=self.df.index)
        return self.spread > self.max_spread_pips

    def _regime_gate(self) -> pd.Series:
        """True where regime is unknown/violent (gate should close)."""
        bad = ~self.regimes.isin(REGIME_CONFIDENCE_OK) | self.regimes.isna()
        return bad

    def _macro_gate(self) -> pd.Series:
        """True where we're inside an event blackout window."""
        if not self._macro_enabled:
            return pd.Series(False, index=self.df.index)
        return self._macro_calendar.in_blackout_window(
            self.df.index,
            flatten_hours=self.flatten_before_event_hours,
            resume_minutes=self.resume_after_event_mins,
        )

    def gate_mask(self) -> pd.Series:
        """Boolean series: True = gate closed (no entries allowed)."""
        if self._gate_mask is not None:
            return self._gate_mask

        if not self._enabled:
            closed = pd.Series(False, index=self.df.index)
        else:
            closed = self._spread_gate() | self._regime_gate() | self._macro_gate()

        spread_violations = int(self._spread_gate().sum()) if self._enabled else 0
        regime_violations = int(self._regime_gate().sum()) if self._enabled else 0
        macro_violations = int(self._macro_gate().sum()) if self._enabled else 0

        self._stats = GateStats(
            total_bars=len(self.df),
            gated_bars=int(closed.sum()),
            spread_violations=spread_violations,
            regime_violations=regime_violations,
            macro_violations=macro_violations,
        )
        self._gate_mask = closed
        return closed

    def apply(self, sig: Signals) -> Signals:
        """Return new Signals with entries zeroed out when gate is closed.

        Existing positions are still managed by the grid engine — only
        *new* entry signals are suppressed.
        """
        closed = self.gate_mask()

        entries = sig.entries & (~closed)
        # Exits are never gated — we always allow closing positions
        return Signals(
            entries=entries,
            exits=sig.exits,
            direction=sig.direction.where(~closed, 0).fillna(0).astype(int),
        )

    def idle_days(self) -> list[str]:
        """Return list of date strings where *no* entry signals fired."""
        closed = self.gate_mask()
        idx = self.df.index
        # Group by date and check if any entries occurred
        dates_with_entries = idx[~closed]
        all_dates = pd.Series(1, index=idx).resample("D").sum()  # all trading days
        if len(dates_with_entries) == 0:
            # All bars were gated — every date is idle
            idle = all_dates.index
        else:
            entry_dates = pd.Series(1, index=dates_with_entries).resample("D").sum()
            idle = all_dates.index.difference(entry_dates.index)

        idle_strs = [d.strftime("%Y-%m-%d") for d in idle]
        if self._stats:
            self._stats.idle_days = idle_strs
            self._stats.idle_day_count = len(idle_strs)
        return idle_strs

    def stats(self) -> GateStats:
        """Return gate statistics (call after apply)."""
        if self._stats is None:
            self.gate_mask()
        return self._stats
