"""Surgical feature toggles — the Roadmap's B4 queue.

Every feature here is **default-OFF** until it beats the baseline
out-of-sample on the walk-forward harness.  The toggle is injected into
strategy ``params`` as a boolean or dict, and read by the strategy / grid
engine / regime engine to short-circuit behaviour.

Features
--------
anomaly_gate      : Simons-rule. Bot may be *idle/flat* — no new entries
                    when spread/stats/regime are unfavourable.
event_blackout    : Macro-economic event calendar. Flatten new entries
                    ``flatten_before_event_hours`` before a scheduled
                    high-impact event; resume after
                    ``resume_after_event_mins``.
recovery_restart   : After a shed, restart smaller & slower (Flash /
                    Venomancer concept).
basket_money_tp    : Energizer-style basket $ TP — close all layers once
                    the group's total $P&L hits a threshold.
profit_lock_trail  : Basket peak-profit trailing — lock a % of the best
                    basket equity as a trailing floor.
carry_adjusted_tp  : Codex-style funding-aware TP (swap/carry adjusted);
                    includes a ``rollover_window``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")


@dataclass
class FeatureSpec:
    """Definition of a single surgical feature."""
    name: str
    default: bool = False
    description: str = ""
    param_schema: dict[str, Any] = field(default_factory=dict)
    validation_note: str = ""

    def defaults(self) -> dict[str, Any]:
        if not self.default:
            return {"enabled": False}
        return {"enabled": True, **self.param_schema}


SURGICAL_FEATURES: dict[str, FeatureSpec] = {
    "anomaly_gate": FeatureSpec(
        name="anomaly_gate",
        default=False,
        description=(
            "Simons anomaly rule: bot may be idle/flat. "
            "No new cycles entered when spread stats, regime quality, or "
            "macro context are unfavourable. Existing inventory still "
            "managed (bank/shed logic intact)."
        ),
        param_schema={
            "min_regime_confidence": 0.6,
            "max_spread_pips": 2.0,
            "flatten_before_event_hours": 1.0,
            "resume_after_event_mins": 30,
        },
        validation_note=(
            "Idle-capable strategy must beat always-on baseline "
            "walk-forward or the gate is reverted."
        ),
    ),
    "event_blackout": FeatureSpec(
        name="event_blackout",
        default=False,
        description=(
            "Macro-economic event calendar overlay. Silences new entries "
            "for flatten_before_event_hours before a scheduled high-impact "
            "event and resumes after resume_after_event_mins post-event."
        ),
        param_schema={
            "calendar_file": "config/events.json",
            "flatten_before_event_hours": 1.5,
            "resume_after_event_mins": 30,
            "macro_week_alloc_multiplier": 0.5,
        },
    ),
    "recovery_restart": FeatureSpec(
        name="recovery_restart",
        default=False,
        description=(
            "After a shed (stop-loss close), restart smaller & slower "
            "instead of full-size re-entry (Flash / Venomancer concept)."
        ),
        param_schema={
            "size_multiplier": 0.5,
            "strictness_bonus": 1,
            "cooldown_bars": 12,
        },
    ),
    "basket_money_tp": FeatureSpec(
        name="basket_money_tp",
        default=False,
        description=(
            "Energizer-style basket $ TP: close ALL layers of a strategy "
            "once cumulative group P&L hits a fixed $ amount."
        ),
        param_schema={
            "basket_take_profit_usd": 50.0,
        },
    ),
    "profit_lock_trail": FeatureSpec(
        name="profit_lock_trail",
        default=False,
        description=(
            "Basket peak-profit trailing: once the basket has banked "
            "basket_peak_profit, trail a profit_lock_% floor behind it "
            "and flatten the residual at market when breached downward."
        ),
        param_schema={
            "profit_lock_pct": 60.0,
        },
    ),
    "carry_adjusted_tp": FeatureSpec(
        name="carry_adjusted_tp",
        default=False,
        description=(
            "Codex-style funding/carry-aware TP: widen TP near funding "
            "rollover windows (Swap/rollover_window) to capture carry."
        ),
        param_schema={
            "rollover_window_hours": 8,
            "tp_extension_pct": 25.0,
        },
    ),
}


def get_feature_defaults() -> dict[str, dict[str, Any]]:
    """Return a dict of {feature_name: {enabled: False, ...params}} for
    every feature that is default-OFF (the safe baseline)."""
    return {name: spec.defaults() for name, spec in SURGICAL_FEATURES.items()}


def is_enabled(params: dict | None, feature_name: str) -> bool:
    """Check if a surgical feature is enabled in a params dict."""
    if not params or not isinstance(params.get(feature_name), dict):
        return False
    return bool(params[feature_name].get("enabled", False))


def get_feature_params(params: dict | None, feature_name: str) -> dict[str, Any]:
    """Return the param dict for a feature, or empty dict if not enabled."""
    if not params or not isinstance(params.get(feature_name), dict):
        return {}
    return params[feature_name]


def enable(params: dict, feature_name: str, overrides: dict | None = None) -> dict:
    """Enable a surgical feature in a params dict (non-destructive)."""
    p = dict(params)
    spec = SURGICAL_FEATURES.get(feature_name)
    if spec is None:
        raise ValueError(f"Unknown surgical feature: {feature_name}")
    val: dict[str, Any] = {"enabled": True, **spec.param_schema}
    if overrides:
        val.update(overrides)
    p[feature_name] = val
    return p


def disable(params: dict, feature_name: str) -> dict:
    """Disable a surgical feature in a params dict (non-destructive)."""
    p = dict(params)
    if feature_name in p and isinstance(p[feature_name], dict):
        p[feature_name] = {**p[feature_name], "enabled": False}
    elif feature_name in p:
        p[feature_name] = {"enabled": False}
    return p


def load_event_calendar(path: str | None = None) -> dict[str, Any]:
    """Load the macro-economic event calendar JSON.
    Falls back to an empty calendar if the file doesn't exist.
    """
    if path is None:
        path = os.path.join(CONFIG_DIR, "events.json")
    if not os.path.exists(path):
        return {"events": []}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"events": data}
    except (json.JSONDecodeError, OSError):
        return {"events": []}


__all__ = [
    "SURGICAL_FEATURES",
    "FeatureSpec",
    "get_feature_defaults",
    "is_enabled",
    "get_feature_params",
    "enable",
    "disable",
    "load_event_calendar",
]
