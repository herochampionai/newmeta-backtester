"""Macro-economic event calendar + blackout window overlay.

Roadmap B2: static defaults + editable JSON calendar.  When
``event_blackout`` is enabled in strategy params, entries are silenced
for ``flatten_before_event_hours`` before a scheduled high-impact event
and resume ``resume_after_event_mins`` after the event.

The calendar JSON lives at ``config/events.json`` and can be edited
from the UI.  If the file is missing, a set of static major-event
defaults is used.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
)

DEFAULT_EVENTS: list[dict[str, Any]] = [
    {"ticker": "USD", "event": "FOMC", "month": 1, "day": 29, "hour": 14, "minute": 0, "impact": "high"},
    {"ticker": "USD", "event": "FOMC", "month": 3, "day": 19, "hour": 13, "minute": 0, "impact": "high"},
    {"ticker": "USD", "event": "CPI", "month": 2, "day": 12, "hour": 8, "minute": 30, "impact": "high"},
    {"ticker": "USD", "event": "CPI", "month": 3, "day": 12, "hour": 8, "minute": 30, "impact": "high"},
    {"ticker": "USD", "event": "CPI", "month": 4, "day": 10, "hour": 8, "minute": 30, "impact": "high"},
    {"ticker": "USD", "event": "CPI", "month": 5, "day": 14, "hour": 8, "minute": 30, "impact": "high"},
    {"ticker": "USD", "event": "NFP", "month": 2, "day": 7, "hour": 8, "minute": 30, "impact": "high"},
    {"ticker": "USD", "event": "NFP", "month": 3, "day": 7, "hour": 9, "minute": 30, "impact": "high"},
    {"ticker": "USD", "event": "NFP", "month": 4, "day": 4, "hour": 8, "minute": 30, "impact": "high"},
    {"ticker": "USD", "event": "Jackson Hole", "month": 8, "day": 22, "hour": 14, "minute": 0, "impact": "high"},
    {"ticker": "BTC", "event": "ETF Decision", "month": 10, "day": 1, "hour": 14, "minute": 0, "impact": "medium"},
    {"ticker": "BTC", "event": "Halving", "month": 4, "day": 19, "hour": 0, "minute": 0, "impact": "high"},
]


@dataclass
class EconomicEvent:
    """A single scheduled economic event."""
    ticker: str
    name: str
    timestamp: pd.Timestamp
    impact: str  # "low" | "medium" | "high"
    duration_minutes: int = 30

    def __repr__(self) -> str:
        return (
            f"EconomicEvent({self.ticker}, {self.name}, "
            f"{self.timestamp}, impact={self.impact})"
        )


class MacroEventCalendar:
    """Holds the list of scheduled economic events and provides blackout
    window checks against a time index.

    Load priority:
      1. ``config/events.json`` (user-editable from UI)
      2. ``DEFAULT_EVENTS`` static fallback
    """

    def __init__(self, events: list[EconomicEvent] | None = None,
                 config_path: str | None = None):
        if events is not None:
            self.events = events
        else:
            self.events = self._load(config_path)
        self._sorted = sorted(self.events, key=lambda e: e.timestamp)

    def _load(self, config_path: str | None = None) -> list[EconomicEvent]:
        """Load events from JSON config or fall back to defaults."""
        path = config_path or os.path.join(CONFIG_DIR, "events.json")
        raw: list[dict[str, Any]]

        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "events" in data:
                    raw = data["events"]
                elif isinstance(data, list):
                    raw = data
                else:
                    raw = DEFAULT_EVENTS
            except (json.JSONDecodeError, OSError):
                raw = DEFAULT_EVENTS
        else:
            raw = DEFAULT_EVENTS

        events = []
        for r in raw:
            try:
                if "timestamp" in r:
                    ts = pd.Timestamp(r["timestamp"])
                else:
                    hour_val = r.get("hour", 14)
                    if isinstance(hour_val, str) and not hour_val.isdigit():
                        hour_val = 14
                    else:
                        hour_val = int(hour_val)
                    minute_val = r.get("minute", 0)
                    if isinstance(minute_val, str) and not minute_val.isdigit():
                        minute_val = 0
                    else:
                        minute_val = int(minute_val)
                    ts = pd.Timestamp(
                        year=datetime.now().year,
                        month=int(r["month"]),
                        day=int(r["day"]),
                        hour=hour_val,
                        minute=minute_val,
                    )
                events.append(EconomicEvent(
                    ticker=r.get("ticker", "USD"),
                    name=r.get("event", r.get("name", "Unknown")),
                    timestamp=ts,
                    impact=r.get("impact", "medium"),
                    duration_minutes=r.get("duration_minutes", 30),
                ))
            except (KeyError, ValueError):
                continue
        return events

    @property
    def sorted_events(self) -> list[EconomicEvent]:
        return self._sorted

    def events_between(self, start: pd.Timestamp, end: pd.Timestamp,
                       flatten_hours: float = 2.0,
                       resume_minutes: int = 30) -> list[EconomicEvent]:
        """Return events whose blackout windows overlap [start, end].

        A blackout window spans [event_time - flatten_hours,
        event_time + resume_minutes].  An event is returned if its
        blackout window overlaps the query range at all.
        """
        overlaps: list[EconomicEvent] = []
        for ev in self._sorted:
            window_start = ev.timestamp - pd.Timedelta(hours=flatten_hours)
            window_end = ev.timestamp + pd.Timedelta(minutes=resume_minutes)
            if window_start <= end and window_end >= start:
                overlaps.append(ev)
        return overlaps

    def in_blackout_window(
        self,
        time_index: pd.DatetimeIndex,
        flatten_hours: float = 2.0,
        resume_minutes: int = 30,
    ) -> pd.Series:
        """Return boolean Series aligned to *time_index*:
        True = inside a blackout window (entries should be silenced).

        A bar is in a blackout if:
          - ``time <= event_time + resume_minutes`` (before event: flatten window)
          - ``time >= event_time - flatten_hours`` and ``time <= event_time + resume_minutes``
        """
        blacked = pd.Series(False, index=time_index)
        for ev in self._sorted:
            window_start = ev.timestamp - pd.Timedelta(hours=flatten_hours)
            window_end = ev.timestamp + pd.Timedelta(minutes=resume_minutes)
            mask = (time_index >= window_start) & (time_index <= window_end)
            blacked = blacked | mask
        return blacked

    def count_macro_days(self) -> int:
        """Count distinct dates in the calendar."""
        return len(set(ev.timestamp.strftime("%Y-%m-%d") for ev in self._sorted))


def macro_alloc_multiplier(
    time_index: pd.DatetimeIndex,
    base_multiplier: float = 1.0,
    macro_week_multiplier: float = 0.5,
    calendar: MacroEventCalendar | None = None,
) -> pd.Series:
    """Return per-bar allocation multiplier.

    During a macro-heavy week (defined as a week containing >=2 events),
    apply ``macro_week_multiplier`` to reduce capital allocation.
    """
    if calendar is None:
        calendar = MacroEventCalendar()

    dates_in_index = set(time_index.strftime("%Y-%m-%d"))
    event_dates = set(ev.timestamp.strftime("%Y-%m-%d") for ev in calendar._sorted)

    # Group events by ISO week
    week_counts: dict[str, int] = {}
    for ev in calendar._sorted:
        week_key = ev.timestamp.strftime("%Y-W%U")
        week_counts[week_key] = week_counts.get(week_key, 0) + 1

    macro_weeks = {wk for wk, cnt in week_counts.items() if cnt >= 2}

    multipliers = pd.Series(base_multiplier, index=time_index)
    for ts in time_index:
        week_key = ts.strftime("%Y-W%U")
        if week_key in macro_weeks:
            multipliers.loc[ts] = base_multiplier * macro_week_multiplier
    return multipliers


__all__ = [
    "DEFAULT_EVENTS",
    "EconomicEvent",
    "MacroEventCalendar",
    "macro_alloc_multiplier",
]
