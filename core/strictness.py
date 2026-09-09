"""Strictness slider (0-10) per strategy.

0 = very permissive (many signals, lots of trades)
10 = very strict (few signals, high-confidence only)

Each strategy exposes a `strictness_to_params(s: int, base: dict)` function that maps
the slider value to concrete indicator thresholds. The mapping is strategy-specific.

Conventions:
  - Permissive (0-3): lower levels, shorter periods → more signals, noisier
  - Balanced (4-6): default MQL5 values
  - Strict (7-10): higher levels, longer periods → fewer signals, higher edge
"""
from __future__ import annotations
import math
from typing import Callable


def ac_ao_strictness(s: int, base: dict) -> dict:
    """Map strictness → AC/AO level + period adjustments.
    s=0 → many signals; s=10 → few.
    """
    p = dict(base)
    # AC level: 30 (permissive) to 95 (strict)
    p["level_open_orders"] = int(round(30 + (s / 10) * 65))
    p["level_close_orders"] = int(round(20 + (s / 10) * 70))
    # Acceleration filter: enable for stricter
    p["use_acceleration_filter"] = s >= 5
    if s >= 5:
        # Higher strictness → larger min acceleration required
        p["min_acceleration"] = round(0.0001 * (1 + (s - 5) * 2), 6)
    # AO sync: enable for stricter
    p["use_ao_synchronization"] = s >= 7
    return p


def adx_strictness(s: int, base: dict) -> dict:
    p = dict(base)
    # ADX main level: 25 → 80
    p["level_open_orders_1"] = int(round(25 + (s / 10) * 55))
    p["level_close_orders_1"] = int(round(10 + (s / 10) * 30))
    # DI crossover gap: smaller for permissive
    p["min_crossover_gap"] = round(2 + (s / 10) * 8, 1)
    return p


def dem_strictness(s: int, base: dict) -> dict:
    p = dict(base)
    # DeM level: 50 → 95
    p["level_open_orders"] = int(round(50 + (s / 10) * 45))
    p["level_close_orders"] = int(round(40 + (s / 10) * 50))
    return p


def fbb_strictness(s: int, base: dict) -> dict:
    p = dict(base)
    # BB deviation: 1.0 (loose) → 3.0 (tight)
    p["deviation"] = round(1.0 + (s / 10) * 2.0, 2)
    # Level open_orders_2: 30 → 90
    p["level_open_orders_2"] = int(round(30 + (s / 10) * 60))
    # Disable Type_2 Force at low strictness (it's noisy)
    if s <= 3:
        p["open_orders_type_2"] = 0
    elif s <= 7:
        p["open_orders_type_2"] = 4
    else:
        p["open_orders_type_2"] = 8
    return p


def mfi_strictness(s: int, base: dict) -> dict:
    p = dict(base)
    # MFI level: 60 → 95
    p["level_open_orders"] = int(round(60 + (s / 10) * 35))
    p["level_close_orders"] = int(round(50 + (s / 10) * 45))
    # Period: shorter (responsive) for permissive, longer (smooth) for strict
    p["bars_calculate"] = int(round(8 + (s / 10) * 12))  # 8 to 20
    # Slope filter + divergence for stricter
    p["use_slope_filter"] = s >= 5
    p["use_divergence"] = s >= 6
    p["use_hidden_divergence"] = s >= 8
    return p


def ms_strictness(s: int, base: dict) -> dict:
    p = dict(base)
    # MACD level: 30 → 80
    p["level_open_orders_1"] = int(round(30 + (s / 10) * 50))
    # Stoch level: 70 → 95
    p["level_open_orders_2"] = int(round(70 + (s / 10) * 25))
    # Confluence filter: stricter means both indicators must agree
    p["use_confluence_filter"] = s >= 4
    # Divergence: tighter divergence with higher strictness
    if s >= 7:
        p["use_macd_divergence"] = True
        p["use_stoch_divergence"] = True
    return p


STRICTNESS_MAP: dict[str, Callable] = {
    "ac_ao": ac_ao_strictness,
    "adx": adx_strictness,
    "dem": dem_strictness,
    "fbb": fbb_strictness,
    "mfi": mfi_strictness,
    "ms": ms_strictness,
}


def apply_strictness(strategy_name: str, strictness: int, base: dict | None = None) -> dict:
    """Apply strictness slider to base params. Returns new params dict."""
    fn = STRICTNESS_MAP.get(strategy_name)
    if fn is None:
        return dict(base or {})
    return fn(int(strictness), dict(base or {}))


# TP / SL widening
def apply_tp_sl_widening(base: dict, widening: int = 5) -> dict:
    """Apply TP/SL widening (0-10).
    0 = very tight (10/30 pips), 5 = default (50/150 pips), 10 = very wide (200/500 pips).
    Returns new params dict with take_profit / stop_loss set.
    """
    p = dict(base)
    tp = int(round(10 + (widening / 10) * 190))  # 10 → 200
    sl = int(round(30 + (widening / 10) * 470))  # 30 → 500
    p["take_profit"] = tp
    p["stop_loss"] = sl
    # Also widen grid TP/SL proportionally
    if "grid_take_profit" in p or "grid_tp" in p:
        p["grid_take_profit"] = int(round(tp * 1.5))
        p["grid_stop_loss"] = int(round(sl * 1.5))
    return p


# Risk profile presets
RISK_PROFILES = {
    "Conservative": dict(
        strictness=7,           # High quality signals only
        tp_widening=8,          # Wide TP/SL — give trades room
        base_lot=0.05,          # Small position
        adaptive=True,          # Reduce on losses
        grid_mode=2,            # Profit-only grid (no averaging into losses)
        recovery_mode=0,        # No martingale
        max_weight=0.20,
    ),
    "Balanced": dict(
        strictness=5,           # Default MQL5 levels
        tp_widening=5,          # Default TP/SL
        base_lot=0.10,
        adaptive=True,
        grid_mode=3,            # Both grids (EA default)
        recovery_mode=1,        # Last close recovery
        max_weight=0.35,
    ),
    "Aggressive": dict(
        strictness=3,           # Many signals
        tp_widening=3,          # Tight TP/SL — quick in/out
        base_lot=0.20,          # Bigger position
        adaptive=False,         # Let winners run
        grid_mode=3,            # Both grids
        recovery_mode=1,        # Aggressive martingale
        max_weight=0.50,
    ),
}


def apply_risk_profile(profile_name: str, base_params: dict | None = None) -> dict:
    """Apply a risk profile preset. Returns updated params dict + side info."""
    profile = RISK_PROFILES[profile_name]
    base = dict(base_params or {})
    # Apply strictness
    from .strictness import apply_strictness as _apply
    # We don't know strategy here, return the profile as-is for the caller to apply
    base["_risk_profile"] = profile_name
    base["_risk_profile_config"] = dict(profile)
    return base