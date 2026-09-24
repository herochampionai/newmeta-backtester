"""Unit tests for core/macro_events.py — economic calendar + blackout windows.

Covers: MacroEventCalendar loading from defaults/config, EconomicEvent,
in_blackout_window (window logic), events_between (fixed overlap check),
macro_alloc_multiplier (macro-week detection).
"""
import json
import os
from unittest.mock import patch

import pandas as pd
import pytest

from core.macro_events import (
    DEFAULT_EVENTS,
    EconomicEvent,
    MacroEventCalendar,
    macro_alloc_multiplier,
)


def make_event(ts: str | pd.Timestamp, impact: str = "high",
               ticker: str = "USD", name: str = "FOMC") -> EconomicEvent:
    return EconomicEvent(
        ticker=ticker, name=name,
        timestamp=pd.Timestamp(ts), impact=impact,
    )


class TestEconomicEvent:
    def test_construction(self) -> None:
        ev = make_event("2024-03-19 13:00")
        assert ev.ticker == "USD"
        assert ev.impact == "high"
        assert ev.duration_minutes == 30  # default

    def test_repr(self) -> None:
        ev = make_event("2024-03-19 13:00", ticker="EUR", name="ECB")
        r = repr(ev)
        assert "EconomicEvent" in r
        assert "EUR" in r
        assert "ECB" in r


class TestMacroEventCalendarInit:
    def test_default_loads(self) -> None:
        """Default-constructed calendar loads from config/events.json (44 events)
        or falls back to DEFAULT_EVENTS (12) if config is missing."""
        cal = MacroEventCalendar()
        assert len(cal.events) >= len(DEFAULT_EVENTS)

    def test_explicit_events(self) -> None:
        events = [make_event("2024-01-01 12:00"), make_event("2024-01-02 12:00")]
        cal = MacroEventCalendar(events=events)
        assert len(cal.events) == 2

    def test_sorted(self) -> None:
        events = [make_event("2024-06-01 12:00"), make_event("2024-01-01 12:00")]
        cal = MacroEventCalendar(events=events)
        assert cal.sorted_events[0].timestamp < cal.sorted_events[1].timestamp

    def test_config_path_load(self, tmp_path) -> None:
        events = [{"ticker": "USD", "event": "FOMC", "timestamp": "2024-03-19T13:00:00"}]
        p = tmp_path / "events.json"
        p.write_text(json.dumps({"events": events}))
        cal = MacroEventCalendar(config_path=str(p))
        assert len(cal.events) == 1
        assert cal.events[0].name == "FOMC"
        assert cal.events[0].timestamp == pd.Timestamp("2024-03-19 13:00")

    def test_missing_config_falls_back_to_defaults(self, tmp_path) -> None:
        cal = MacroEventCalendar(config_path=str(tmp_path / "missing.json"))
        assert len(cal.events) == len(DEFAULT_EVENTS)

    def test_invalid_json_falls_back(self, tmp_path) -> None:
        p = tmp_path / "events.json"
        p.write_text("not json {{{")
        cal = MacroEventCalendar(config_path=str(p))
        assert len(cal.events) == len(DEFAULT_EVENTS)

    def test_month_day_hour_format(self, tmp_path) -> None:
        events = [{"ticker": "USD", "event": "Test", "month": 6, "day": 15, "hour": 14, "minute": 30}]
        p = tmp_path / "events.json"
        p.write_text(json.dumps({"events": events}))
        cal = MacroEventCalendar(config_path=str(p))
        assert cal.events[0].timestamp.hour == 14
        assert cal.events[0].timestamp.minute == 30

    def test_non_digit_hour_falls_back(self, tmp_path) -> None:
        events = [{"ticker": "USD", "event": "Test", "month": 6, "day": 15, "hour": "TBD", "minute": 0}]
        p = tmp_path / "events.json"
        p.write_text(json.dumps({"events": events}))
        cal = MacroEventCalendar(config_path=str(p))
        assert cal.events[0].timestamp.hour == 14  # default fallback

    def test_bad_record_skipped(self, tmp_path) -> None:
        events = [
            {"ticker": "USD", "event": "Test", "month": 6, "day": 15, "hour": 14, "minute": 0},
            {"ticker": "USD"},  # missing month/day
        ]
        p = tmp_path / "events.json"
        p.write_text(json.dumps({"events": events}))
        cal = MacroEventCalendar(config_path=str(p))
        assert len(cal.events) == 1


class TestInBlackoutWindow:
    def test_no_events_no_blackout(self) -> None:
        cal = MacroEventCalendar(events=[])
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        result = cal.in_blackout_window(idx, flatten_hours=1.0, resume_minutes=30)
        assert not result.any()

    def test_bar_inside_window(self) -> None:
        cal = MacroEventCalendar(events=[make_event("2024-03-19 14:00")])
        idx = pd.DatetimeIndex(["2024-03-19 13:30:00", "2024-03-19 14:15:00"])
        result = cal.in_blackout_window(idx, flatten_hours=1.0, resume_minutes=30)
        assert result.all()

    def test_bar_before_window(self) -> None:
        cal = MacroEventCalendar(events=[make_event("2024-03-19 14:00")])
        idx = pd.DatetimeIndex(["2024-03-19 10:00:00"])
        result = cal.in_blackout_window(idx, flatten_hours=2.0, resume_minutes=30)
        assert not result.any()

    def test_bar_after_window(self) -> None:
        cal = MacroEventCalendar(events=[make_event("2024-03-19 14:00")])
        idx = pd.DatetimeIndex(["2024-03-19 16:00:00"])
        result = cal.in_blackout_window(idx, flatten_hours=1.0, resume_minutes=30)
        assert not result.any()  # 16:00 is after 14:30 window end

    def test_bar_at_boundary_start(self) -> None:
        cal = MacroEventCalendar(events=[make_event("2024-03-19 14:00")])
        idx = pd.DatetimeIndex(["2024-03-19 13:00:00"])  # exactly flatten window start
        result = cal.in_blackout_window(idx, flatten_hours=1.0, resume_minutes=30)
        assert result.iloc[0]

    def test_bar_at_boundary_end(self) -> None:
        cal = MacroEventCalendar(events=[make_event("2024-03-19 14:00")])
        idx = pd.DatetimeIndex(["2024-03-19 14:30:00"])  # exactly resume end
        result = cal.in_blackout_window(idx, flatten_hours=1.0, resume_minutes=30)
        assert result.iloc[0]

    def test_overlapping_events(self) -> None:
        cal = MacroEventCalendar(events=[
            make_event("2024-03-19 14:00"),
            make_event("2024-03-20 14:00"),
        ])
        idx = pd.date_range("2024-03-19 10:00", periods=100, freq="h")
        result = cal.in_blackout_window(idx, flatten_hours=2.0, resume_minutes=30)
        assert result.sum() > 0  # at least some bars are in windows

    def test_aligned_to_index(self) -> None:
        cal = MacroEventCalendar(events=[make_event("2024-03-19 14:00")])
        idx = pd.DatetimeIndex(["2024-03-19 13:30:00", "2024-03-19 10:00:00", "2024-03-19 16:00:00"])
        result = cal.in_blackout_window(idx, flatten_hours=1.0, resume_minutes=30)
        assert len(result) == 3
        assert result.iloc[0]  # 13:30 is in window
        assert not result.iloc[1]  # 10:00 is not
        assert not result.iloc[2]  # 16:00 is not


class TestEventsBetween:
    def test_overlapping(self) -> None:
        ev = make_event("2024-03-19 14:00")
        cal = MacroEventCalendar(events=[ev])
        start = pd.Timestamp("2024-03-19 13:00")
        end = pd.Timestamp("2024-03-19 15:00")
        result = cal.events_between(start, end, flatten_hours=1.0, resume_minutes=30)
        assert len(result) == 1
        assert result[0] is ev

    def test_non_overlapping(self) -> None:
        ev = make_event("2024-03-19 14:00")
        cal = MacroEventCalendar(events=[ev])
        start = pd.Timestamp("2024-03-20 10:00")
        end = pd.Timestamp("2024-03-20 12:00")
        result = cal.events_between(start, end, flatten_hours=1.0, resume_minutes=30)
        assert len(result) == 0

    def test_partial_overlap_before(self) -> None:
        ev = make_event("2024-03-19 14:00")
        cal = MacroEventCalendar(events=[ev])
        start = pd.Timestamp("2024-03-19 13:30")
        end = pd.Timestamp("2024-03-19 13:55")
        result = cal.events_between(start, end, flatten_hours=1.0, resume_minutes=30)
        assert len(result) == 1

    def test_multiple_events(self) -> None:
        events = [make_event(f"2024-03-{d} 14:00") for d in range(10, 20)]
        cal = MacroEventCalendar(events=events)
        start = pd.Timestamp("2024-03-12 00:00")
        end = pd.Timestamp("2024-03-17 23:59")
        result = cal.events_between(start, end, flatten_hours=2.0, resume_minutes=30)
        # Events on 12th through 17th (and possibly 18th if window overlaps)
        assert len(result) >= 5


class TestCountMacroDays:
    def test_empty(self) -> None:
        cal = MacroEventCalendar(events=[])
        assert cal.count_macro_days() == 0

    def test_count(self) -> None:
        events = [
            make_event("2024-03-19 14:00", ticker="USD"),
            make_event("2024-03-19 16:00", ticker="EUR"),
            make_event("2024-03-20 14:00", ticker="USD"),
        ]
        cal = MacroEventCalendar(events=events)
        assert cal.count_macro_days() == 2


class TestMacroAllocMultiplier:
    def test_no_events(self) -> None:
        cal = MacroEventCalendar(events=[])
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        result = macro_alloc_multiplier(idx, calendar=cal)
        assert (result == 1.0).all()

    def test_macro_week_reduces_allocation(self) -> None:
        # 3 events in same week → macro week
        events = [
            make_event("2024-03-19 14:00"),
            make_event("2024-03-20 14:00"),
            make_event("2024-03-21 14:00"),
        ]
        cal = MacroEventCalendar(events=events)
        idx = pd.date_range("2024-03-18", periods=10, freq="D")
        result = macro_alloc_multiplier(idx, calendar=cal, macro_week_multiplier=0.5)
        # During macro week, multiplier should be 0.5
        assert (result == 0.5).any()

    def test_non_macro_week_unchanged(self) -> None:
        events = [make_event("2024-03-19 14:00")]
        cal = MacroEventCalendar(events=events)
        idx = pd.date_range("2024-04-01", periods=10, freq="D")
        result = macro_alloc_multiplier(idx, calendar=cal, macro_week_multiplier=0.5)
        assert (result == 1.0).all()

    def test_default_calendar(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        result = macro_alloc_multiplier(idx)  # no calendar arg
        assert len(result) == 10

    def test_base_multiplier(self) -> None:
        events = [
            make_event("2024-03-19 14:00"),
            make_event("2024-03-20 14:00"),
            make_event("2024-03-21 14:00"),
        ]
        cal = MacroEventCalendar(events=events)
        idx = pd.date_range("2024-03-19", periods=5, freq="D")
        result = macro_alloc_multiplier(idx, base_multiplier=0.8, calendar=cal, macro_week_multiplier=0.4)
        assert (abs(result - 0.32) < 1e-6).any()  # 0.8 * 0.4
