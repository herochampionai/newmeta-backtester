"""Execution realism helpers for Newmeta Research Lab - Backtester.

Keeps market assumptions explicit: pip size, contract size, spread,
commission mode, and slippage. The values are presets, not broker truth.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class MarketProfile:
    name: str
    pip_size: float
    contract_size: float
    default_spread_pips: float
    default_commission_pips: float
    default_slippage_pips: float
    default_long_swap_pips: float
    default_short_swap_pips: float
    commission_mode: str = "pips"  # pips | percent
    commission_pct: float = 0.0     # percent per order when commission_mode=percent

    def to_dict(self) -> dict:
        return asdict(self)


MARKET_PROFILES = {
    "Forex": MarketProfile(
        name="Forex",
        pip_size=0.0001,
        contract_size=100_000,
        default_spread_pips=0.8,
        default_commission_pips=0.7,
        default_slippage_pips=0.3,
        default_long_swap_pips=-0.5,
        default_short_swap_pips=0.2,
    ),
    "XAUUSD": MarketProfile(
        name="XAUUSD",
        pip_size=0.01,
        contract_size=100,
        default_spread_pips=2.5,
        default_commission_pips=0.0,
        default_slippage_pips=1.0,
        default_long_swap_pips=-3.0,
        default_short_swap_pips=1.0,
    ),
    "Crypto": MarketProfile(
        name="Crypto",
        pip_size=0.01,
        contract_size=1,
        default_spread_pips=5.0,
        default_commission_pips=0.0,
        default_slippage_pips=2.0,
        default_long_swap_pips=0.0,
        default_short_swap_pips=0.0,
        commission_mode="percent",
        commission_pct=0.075,
    ),
}


def infer_market(symbol: str) -> str:
    s = (symbol or "").upper()
    if "XAU" in s or "GOLD" in s:
        return "XAUUSD"
    if any(x in s for x in ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE")):
        return "Crypto"
    return "Forex"


def profile_for(market: str, symbol: str | None = None) -> MarketProfile:
    if market == "Auto":
        market = infer_market(symbol or "")
    return MARKET_PROFILES.get(market, MARKET_PROFILES["Forex"])


def pips_to_fee_rate(total_cost_pips: float, pip_size: float, reference_price: float) -> float:
    """Approximate pips as percent fee for vectorized close-price simulation."""
    if reference_price <= 0:
        return 0.0
    return max(total_cost_pips, 0.0) * pip_size / reference_price


def total_cost_pips(commission_pips: float, slippage_pips: float, spread_pips: float) -> float:
    """All pip-denominated execution costs per order side."""
    return max(commission_pips, 0.0) + max(slippage_pips, 0.0) + max(spread_pips, 0.0)
