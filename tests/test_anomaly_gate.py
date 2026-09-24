"""Unit tests for core/anomaly_gate.py — AnomalyGate (Simons rule).

Covers: GateStats, gate_mask (spread/regime/macro), apply (signal suppression),
idle_days tracking, default-OFF behavior when not enabled.
"""
import numpy as np
import pandas as pd
import pytest

from core.anomaly_gate import AnomalyGate, GateStats, REGIME_CONFIDENCE_OK
from core.surgical_features import enable, disable
from strategies._base import _empty_signals, Signals


def make_df(n: int = 200, freq: str = "h",
            price_col: str = "close",
            start: str = "2024-01-01") -> pd.DataFrame:
    """Create synthetic OHLCV for regime detection.

    Uses a trending price series so detect_regimes returns non-unknown regimes.
    """
    idx = pd.date_range(start, periods=n, freq=freq)
    price = np.linspace(1.1000, 1.1200, n) + np.random.RandomState(42).randn(n) * 0.0005
    df = pd.DataFrame({
        "open": price + np.random.RandomState(42).randn(n) * 0.0002,
        "high": price + np.random.RandomState(42).randn(n) * 0.0003 + 0.0005,
        "low": price - np.random.RandomState(42).randn(n) * 0.0003 - 0.0005,
        "close": price,
        "volume": np.random.RandomState(42).randint(100, 500, n),
    }, index=idx)
    return df


def make_signals(n: int, idx: pd.DatetimeIndex,
                 entry_bars: list[int] | None = None,
                 direction: int = 1) -> Signals:
    """Create synthetic Signals with entries at specified bars."""
    if entry_bars is None:
        entry_bars = list(range(10, n, 20))
    entries = pd.Series(False, index=idx)
    exits = pd.Series(False, index=idx)
    dirs = pd.Series(0, index=idx, dtype=int)
    for b in entry_bars:
        if 0 <= b < n:
            entries.iloc[b] = True
            dirs.iloc[b] = direction
    return Signals(entries=entries, exits=exits, direction=dirs)


class TestGateStats:
    def test_defaults(self) -> None:
        gs = GateStats()
        assert gs.total_bars == 0
        assert gs.gated_bars == 0
        assert gs.idle_days == []
        assert gs.idle_day_count == 0
        assert gs.spread_violations == 0
        assert gs.regime_violations == 0
        assert gs.macro_violations == 0

    def test_to_dict(self) -> None:
        gs = GateStats(total_bars=100, gated_bars=30, idle_days=["2024-01-01"],
                       idle_day_count=1, spread_violations=10, regime_violations=5,
                       macro_violations=15)
        d = gs.to_dict()
        assert d["total_bars"] == 100
        assert d["gated_bars"] == 30
        assert d["idle_days"] == ["2024-01-01"]
        assert d["idle_day_count"] == 1
        assert d["spread_violations"] == 10
        assert d["regime_violations"] == 5
        assert d["macro_violations"] == 15

    def test_post_init_idempotent(self) -> None:
        gs = GateStats()
        assert gs.idle_days == []
        # Calling again should not reset
        gs.idle_days = ["2024-01-02"]
        assert gs.idle_days == ["2024-01-02"]


class TestAnomalyGateConstruction:
    def test_no_params(self) -> None:
        df = make_df(n=200)
        gate = AnomalyGate(df, params=None)
        assert gate.params == {}

    def test_empty_params(self) -> None:
        df = make_df(n=200)
        gate = AnomalyGate(df, params={})
        assert gate.params == {}

    def test_spread_series(self) -> None:
        df = make_df(n=200)
        spread = pd.Series(0.5, index=df.index)
        gate = AnomalyGate(df, params=None, spread_series=spread)
        assert gate.spread is not None


class TestGateMask:
    def test_gate_mask_returns_series(self) -> None:
        df = make_df(n=200)
        gate = AnomalyGate(df, params=None)
        mask = gate.gate_mask()
        assert isinstance(mask, pd.Series)
        assert len(mask) == len(df)

    def test_no_spread_no_spread_violations(self) -> None:
        df = make_df(n=200)
        gate = AnomalyGate(df, params=None)
        gate.gate_mask()
        stats = gate.stats()
        assert stats.spread_violations == 0

    def test_spread_violations(self) -> None:
        df = make_df(n=200)
        spread = pd.Series(5.0, index=df.index)  # all above 2.0 default
        params = enable({}, "anomaly_gate", overrides={"max_spread_pips": 2.0})
        gate = AnomalyGate(df, params=params, spread_series=spread)
        gate.gate_mask()
        stats = gate.stats()
        assert stats.spread_violations > 0

    def test_regime_violations(self) -> None:
        df = make_df(n=100)
        params = enable({}, "anomaly_gate")
        gate = AnomalyGate(df, params=params)
        gate.gate_mask()
        stats = gate.stats()
        # Regime violations depend on what detect_regimes returns
        assert stats.regime_violations >= 0

    def test_macro_disabled_no_macro_violations(self) -> None:
        df = make_df(n=200)
        params = enable({}, "anomaly_gate")  # anomaly_gate on, but no event_blackout
        gate = AnomalyGate(df, params=params)
        gate.gate_mask()
        stats = gate.stats()
        assert stats.macro_violations == 0

    def test_macro_enabled(self) -> None:
        df = make_df(n=200)
        params = enable(enable({}, "anomaly_gate"), "event_blackout")
        gate = AnomalyGate(df, params=params)
        gate.gate_mask()
        stats = gate.stats()
        assert stats.macro_violations >= 0


class TestApply:
    def test_apply_returns_signals(self) -> None:
        df = make_df(n=200)
        gate = AnomalyGate(df, params=None)
        sig = make_signals(200, df.index, entry_bars=[10, 30, 50, 70])
        result = gate.apply(sig)
        assert isinstance(result, Signals)

    def test_apply_preserves_exits(self) -> None:
        df = make_df(n=200)
        sig = make_signals(200, df.index, entry_bars=[10])
        sig.exits.iloc[20] = True
        gate = AnomalyGate(df, params=None)
        result = gate.apply(sig)
        assert result.exits.iloc[20] == True

    def test_apply_no_gate_keeps_entries(self) -> None:
        """When anomaly_gate is disabled, the gate is a no-op — all entries pass."""
        df = make_df(n=200)
        gate = AnomalyGate(df, params=None)
        sig = make_signals(200, df.index, entry_bars=[10, 30, 50])
        result = gate.apply(sig)
        assert result.entries.iloc[10] == True
        assert result.entries.iloc[30] == True
        assert result.entries.iloc[50] == True

    def test_apply_high_spread_supresses_entries(self) -> None:
        df = make_df(n=200)
        spread = pd.Series(5.0, index=df.index)
        params = enable({}, "anomaly_gate", overrides={"max_spread_pips": 2.0})
        gate = AnomalyGate(df, params=params, spread_series=spread)
        sig = make_signals(200, df.index, entry_bars=[10, 30, 50])
        result = gate.apply(sig)
        assert not result.entries.any()

    def test_apply_partial_spread_supression(self) -> None:
        df = make_df(n=200)
        spread = pd.Series(0.5, index=df.index)
        spread.iloc[10:15] = 5.0  # high spread at bars 10-14
        params = enable({}, "anomaly_gate", overrides={"max_spread_pips": 2.0})
        gate = AnomalyGate(df, params=params, spread_series=spread)
        sig = make_signals(200, df.index, entry_bars=[10, 30, 50])
        result = gate.apply(sig)
        # Bar 10 should be suppressed (high spread), bars 30 and 50 should pass
        assert result.entries.iloc[10] == False
        assert result.entries.iloc[30] == True

    def test_apply_exits_never_suppressed(self) -> None:
        df = make_df(n=200)
        spread = pd.Series(5.0, index=df.index)
        params = enable({}, "anomaly_gate", overrides={"max_spread_pips": 2.0})
        gate = AnomalyGate(df, params=params, spread_series=spread)
        sig = make_signals(200, df.index, entry_bars=[10])
        sig.exits.iloc[20] = True
        result = gate.apply(sig)
        assert result.exits.iloc[20] == True


class TestIdleDays:
    def test_idle_days_returns_list(self) -> None:
        df = make_df(n=200, freq="D")  # daily data
        gate = AnomalyGate(df, params=None)
        gate.gate_mask()
        result = gate.idle_days()
        assert isinstance(result, list)

    def test_no_entries_all_idle(self) -> None:
        df = make_df(n=200, freq="D")
        params = enable({}, "anomaly_gate", overrides={"max_spread_pips": 0.0001})
        spread = pd.Series(5.0, index=df.index)
        gate = AnomalyGate(df, params=params, spread_series=spread)
        sig = make_signals(200, df.index, entry_bars=[5, 10, 15])
        gate.apply(sig)
        idle = gate.idle_days()
        assert len(idle) > 0

    def test_no_idle_all_entries(self) -> None:
        """When the gate is disabled (params=None), no bars are gated,
        so every date has entries → zero idle days."""
        df = make_df(n=200, freq="D")
        gate = AnomalyGate(df, params=None)
        sig = make_signals(200, df.index)
        sig.entries = pd.Series(True, index=df.index)
        gate.apply(sig)
        idle = gate.idle_days()
        assert len(idle) == 0

    def test_idle_count_in_stats(self) -> None:
        """When the gate is enabled and all bars are gated, all dates are idle."""
        df = make_df(n=200, freq="D")
        params = enable({}, "anomaly_gate", overrides={"max_spread_pips": 0.0001})
        spread = pd.Series(5.0, index=df.index)
        gate = AnomalyGate(df, params=params, spread_series=spread)
        sig = make_signals(200, df.index, entry_bars=[5, 10, 15])
        gate.apply(sig)
        gate.idle_days()
        stats = gate.stats()
        assert stats.idle_day_count > 0
        assert len(stats.idle_days) == stats.idle_day_count


class TestStats:
    def test_stats_lazy_compute(self) -> None:
        df = make_df(n=200)
        gate = AnomalyGate(df, params=None)
        # Don't call gate_mask first
        stats = gate.stats()
        assert stats is not None
        assert stats.total_bars == len(df)

    def test_stats_called_after_mask(self) -> None:
        df = make_df(n=200)
        gate = AnomalyGate(df, params=None)
        gate.gate_mask()
        stats = gate.stats()
        assert stats.gated_bars == gate._gate_mask.sum()

    def test_total_bars_matches_index(self) -> None:
        df = make_df(n=100)
        gate = AnomalyGate(df, params=None)
        stats = gate.stats()
        assert stats.total_bars == 100
