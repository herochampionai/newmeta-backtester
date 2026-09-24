"""Unit tests for core/surgical_features.py — the B4 feature toggle registry.

Covers: registry integrity, default-OFF contract, is_enabled,
get_feature_params, enable, disable, load_event_calendar.
"""
import json
import os
import tempfile

import pytest

from core.surgical_features import (
    SURGICAL_FEATURES,
    FeatureSpec,
    get_feature_defaults,
    is_enabled,
    get_feature_params,
    enable,
    disable,
    load_event_calendar,
)


ALL_FEATURES = [
    "anomaly_gate",
    "event_blackout",
    "recovery_restart",
    "basket_money_tp",
    "profit_lock_trail",
    "carry_adjusted_tp",
]


class TestRegistryIntegrity:
    """Every feature must be registered with a spec."""

    @pytest.mark.parametrize("name", ALL_FEATURES)
    def test_feature_present(self, name: str) -> None:
        assert name in SURGICAL_FEATURES

    def test_all_specs_are_feature_spec(self) -> None:
        for spec in SURGICAL_FEATURES.values():
            assert isinstance(spec, FeatureSpec)
            assert spec.name in ALL_FEATURES

    def test_all_default_false(self) -> None:
        """The core guarantee: no feature ships enabled by default."""
        for name in ALL_FEATURES:
            assert SURGICAL_FEATURES[name].default is False

    def test_all_have_param_schema(self) -> None:
        """Features that are default-FALSE still carry param defaults for when enabled."""
        for name in ALL_FEATURES:
            assert len(SURGICAL_FEATURES[name].param_schema) > 0


class TestGetFeatureDefaults:
    def test_returns_all_features(self) -> None:
        d = get_feature_defaults()
        for name in ALL_FEATURES:
            assert name in d

    def test_all_disabled(self) -> None:
        d = get_feature_defaults()
        for name in ALL_FEATURES:
            assert d[name] == {"enabled": False}


class TestIsEnabled:
    def test_none_params(self) -> None:
        for name in ALL_FEATURES:
            assert is_enabled(None, name) is False

    def test_empty_params(self) -> None:
        for name in ALL_FEATURES:
            assert is_enabled({}, name) is False

    def test_feature_not_dict(self) -> None:
        assert is_enabled({"anomaly_gate": True}, "anomaly_gate") is False
        assert is_enabled({"anomaly_gate": "yes"}, "anomaly_gate") is False

    def test_enabled_true(self) -> None:
        params = {"anomaly_gate": {"enabled": True}}
        assert is_enabled(params, "anomaly_gate") is True

    def test_enabled_missing_key(self) -> None:
        params = {"anomaly_gate": {"min_regime_confidence": 0.6}}
        assert is_enabled(params, "anomaly_gate") is False

    def test_disabled_explicit(self) -> None:
        params = {"anomaly_gate": {"enabled": False}}
        assert is_enabled(params, "anomaly_gate") is False


class TestGetFeatureParams:
    def test_none_params(self) -> None:
        for name in ALL_FEATURES:
            assert get_feature_params(None, name) == {}

    def test_empty_params(self) -> None:
        for name in ALL_FEATURES:
            assert get_feature_params({}, name) == {}

    def test_feature_not_dict(self) -> None:
        assert get_feature_params({"anomaly_gate": True}, "anomaly_gate") == {}

    def test_returns_params_when_enabled(self) -> None:
        params = {"anomaly_gate": {"enabled": True, "max_spread_pips": 3.0}}
        result = get_feature_params(params, "anomaly_gate")
        assert result["max_spread_pips"] == 3.0

    def test_returns_params_when_disabled(self) -> None:
        """get_feature_params returns the param dict even when disabled;
        is_enabled is the source of truth for enable/disable."""
        params = {"anomaly_gate": {"enabled": False}}
        result = get_feature_params(params, "anomaly_gate")
        assert result == {"enabled": False}


class TestEnable:
    def test_enable_adds_params(self) -> None:
        params = {}
        result = enable(params, "anomaly_gate")
        assert result["anomaly_gate"]["enabled"] is True
        assert "min_regime_confidence" in result["anomaly_gate"]

    def test_enable_non_destructive(self) -> None:
        original = {"strategy": "fbb"}
        result = enable(original, "recovery_restart")
        assert original == {"strategy": "fbb"}
        assert result["strategy"] == "fbb"

    def test_enable_with_overrides(self) -> None:
        params = {}
        result = enable(params, "recovery_restart", overrides={"size_multiplier": 0.3})
        assert result["recovery_restart"]["size_multiplier"] == 0.3
        assert result["recovery_restart"]["enabled"] is True

    def test_enable_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown surgical feature"):
            enable({}, "nonexistent_feature")

    @pytest.mark.parametrize("name", ALL_FEATURES)
    def test_enable_all(self, name: str) -> None:
        result = enable({}, name)
        assert result[name]["enabled"] is True


class TestDisable:
    def test_disable_existing(self) -> None:
        params = enable({}, "anomaly_gate")
        result = disable(params, "anomaly_gate")
        assert result["anomaly_gate"]["enabled"] is False

    def test_disable_preserves_params(self) -> None:
        params = {"anomaly_gate": {"enabled": True, "max_spread_pips": 5.0}}
        result = disable(params, "anomaly_gate")
        assert result["anomaly_gate"]["enabled"] is False
        assert result["anomaly_gate"]["max_spread_pips"] == 5.0

    def test_disable_non_dict_value(self) -> None:
        params = {"anomaly_gate": True}
        result = disable(params, "anomaly_gate")
        assert result["anomaly_gate"] == {"enabled": False}

    def test_disable_missing_is_noop(self) -> None:
        params = {"strategy": "fbb"}
        result = disable(params, "anomaly_gate")
        assert "anomaly_gate" not in result

    def test_disable_non_destructive(self) -> None:
        original = {"strategy": "fbb"}
        result = disable(original, "anomaly_gate")
        assert original == {"strategy": "fbb"}


class TestLoadEventCalendar:
    def test_missing_file_returns_empty(self) -> None:
        assert load_event_calendar("/nonexistent/path.json") == {"events": []}

    def test_valid_json(self, tmp_path) -> None:
        events = [{"ticker": "USD", "event": "FOMC", "month": 1, "day": 29, "hour": 14, "minute": 0, "impact": "high"}]
        p = tmp_path / "events.json"
        p.write_text(json.dumps({"events": events}))
        result = load_event_calendar(str(p))
        assert result == {"events": events}

    def test_list_format(self, tmp_path) -> None:
        events = [{"ticker": "USD", "event": "FOMC", "month": 1, "day": 29, "hour": 14, "minute": 0, "impact": "high"}]
        p = tmp_path / "events.json"
        p.write_text(json.dumps(events))
        result = load_event_calendar(str(p))
        assert result == {"events": events}

    def test_invalid_json_returns_empty(self, tmp_path) -> None:
        p = tmp_path / "events.json"
        p.write_text("not valid json {{{")
        assert load_event_calendar(str(p)) == {"events": []}

    def test_default_path_fallback(self) -> None:
        """Falls back to config/events.json in project."""
        result = load_event_calendar()
        assert "events" in result
        assert len(result["events"]) > 0

    def test_missing_file_with_default_path(self, monkeypatch) -> None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
        monkeypatch.setattr("core.surgical_features.CONFIG_DIR", "/nonexistent")
        assert load_event_calendar() == {"events": []}
