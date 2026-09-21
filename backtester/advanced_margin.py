"""R016: Advanced margin — futures SPAN/CME + options Greeks.

Features:
- SPAN-style futures margin (simplified): scan price range, return largest loss
- Options Black-Scholes pricing + Greeks (delta/gamma/vega/theta)
- Portfolio Greeks aggregation
- Initial margin per CME futures (simplified)

Usage:
    margin_fut = futures_initial_margin("ES", price=4500, contracts=1, exchange="CME")
    greeks = option_greeks("call", S=100, K=100, T=0.25, r=0.05, sigma=0.20)
    portfolio = aggregate_portfolio_greeks([{"type": "call", ...}, {"type": "put", ...}])
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
from scipy.stats import norm
import numpy as np


# Simplified CME futures initial margin (per contract, USD)
# contract_multiplier = notional $ per 1.0 point move per contract
FUTURES_MARGIN = {
    "ES":  {"initial": 6500, "maintenance": 6000, "tick_value": 12.50, "tick_size": 0.25, "exchange": "CME",   "contract_multiplier": 50.00},
    "NQ":  {"initial": 8000, "maintenance": 7500, "tick_value": 5.00,  "tick_size": 0.25, "exchange": "CME",   "contract_multiplier": 20.00},
    "YM":  {"initial": 5000, "maintenance": 4500, "tick_value": 5.00,  "tick_size": 1.0,  "exchange": "CBOT",  "contract_multiplier": 5.00},
    "RTY": {"initial": 5500, "maintenance": 5000, "tick_value": 5.00,  "tick_size": 0.10, "exchange": "CME",   "contract_multiplier": 50.00},
    "CL":  {"initial": 6000, "maintenance": 5500, "tick_value": 10.00, "tick_size": 0.01, "exchange": "NYMEX", "contract_multiplier": 1000.00},
    "GC":  {"initial": 6500, "maintenance": 6000, "tick_value": 10.00, "tick_size": 0.10, "exchange": "COMEX", "contract_multiplier": 100.00},
    "6E":  {"initial": 2500, "maintenance": 2250, "tick_value": 12.50, "tick_size": 0.0001, "exchange": "CME",  "contract_multiplier": 125000.00},
    "BTC": {"initial": 50000,"maintenance": 45000,"tick_value": 25.00, "tick_size": 5.0, "exchange": "CME",   "contract_multiplier": 5.00},
}


@dataclass
class FuturesMargin:
    """Futures margin requirement."""
    symbol: str
    exchange: str
    contracts: int
    price: float
    initial_margin: float
    maintenance_margin: float
    notional: float
    tick_value: float
    contract_multiplier: float


def futures_initial_margin(symbol: str, price: float, contracts: int = 1) -> FuturesMargin | None:
    """Compute initial + maintenance margin for futures contract."""
    spec = FUTURES_MARGIN.get(symbol.upper())
    if not spec:
        return None
    initial = spec["initial"] * abs(contracts)
    maintenance = spec["maintenance"] * abs(contracts)
    notional = price * abs(contracts) * spec["contract_multiplier"]
    return FuturesMargin(
        symbol=symbol.upper(),
        exchange=spec["exchange"],
        contracts=int(contracts),
        price=price,
        initial_margin=initial,
        maintenance_margin=maintenance,
        notional=notional,
        tick_value=spec["tick_value"],
        contract_multiplier=spec["contract_multiplier"],
    )


def span_margin(entry_price: float, scenario_prices: list[float],
                position_size: int = 1, contract_multiplier: float = 50) -> float:
    """SPAN-style margin: largest loss across scenarios vs entry.

    SPAN scans 16 scenarios (price up/down × vol up/down × time decay).
    This is simplified: worst-case price move × contract_size.

    Args:
        entry_price: position entry price (reference)
        scenario_prices: list of possible future prices
        position_size: number of contracts (long +ve, short -ve)
        contract_multiplier: $ per point move per contract
    """
    if not scenario_prices:
        return 0.0
    losses = []
    for price in scenario_prices:
        # Loss = (price - entry) * position_size * multiplier
        # Negative loss = profit, positive loss = actual loss
        pnl = (price - entry_price) * position_size * contract_multiplier
        losses.append(-pnl)  # store as loss (positive = loss)
    return max(0.0, max(losses))


# ---------- Options Black-Scholes ----------
@dataclass
class OptionGreeks:
    """Black-Scholes Greeks."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def option_greeks(option_type: str, S: float, K: float, T: float, r: float,
                   sigma: float) -> OptionGreeks:
    """Compute Black-Scholes price and Greeks.

    Args:
        option_type: "call" or "put"
        S: spot price
        K: strike
        T: time to expiry (years)
        r: risk-free rate (annualized)
        sigma: volatility (annualized)
    """
    if T <= 0 or sigma <= 0:
        # At expiry, intrinsic only
        if option_type == "call":
            price = max(0.0, S - K)
            delta = 1.0 if S > K else 0.0
        else:
            price = max(0.0, K - S)
            delta = -1.0 if S < K else 0.0
        return OptionGreeks(price=price, delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    else:  # put
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100  # per 1% vol change

    return OptionGreeks(
        price=round(float(price), 4),
        delta=round(float(delta), 4),
        gamma=round(float(gamma), 6),
        vega=round(float(vega), 4),
        theta=round(float(theta), 4),
        rho=round(float(rho), 4),
    )


@dataclass
class PortfolioGreeks:
    """Aggregated portfolio Greeks."""
    n_positions: int
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    total_delta_dollars: float  # delta × 100 (contract size) × spot


def aggregate_portfolio_greeks(positions: list[dict], spot: float = 100.0,
                                contract_size: int = 100) -> PortfolioGreeks:
    """Aggregate Greeks across a portfolio of option positions.

    Args:
        positions: list of dicts with:
          - option_type: "call" | "put"
          - S, K, T, r, sigma
          - quantity (default 1)
        spot: reference spot for dollar delta
        contract_size: shares per contract (default 100)
    """
    if not positions:
        return PortfolioGreeks(0, 0, 0, 0, 0, 0, 0)

    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0
    total_rho = 0.0
    for p in positions:
        g = option_greeks(
            p["option_type"], p.get("S", spot), p["K"],
            p["T"], p.get("r", 0.05), p.get("sigma", 0.20),
        )
        qty = p.get("quantity", 1) * contract_size
        total_delta += g.delta * qty
        total_gamma += g.gamma * qty
        total_vega += g.vega * qty
        total_theta += g.theta * qty
        total_rho += g.rho * qty

    return PortfolioGreeks(
        n_positions=len(positions),
        delta=round(total_delta, 2),
        gamma=round(total_gamma, 4),
        vega=round(total_vega, 2),
        theta=round(total_theta, 2),
        rho=round(total_rho, 2),
        total_delta_dollars=round(total_delta * spot, 2),
    )


# ---------- Self-test ----------
if __name__ == "__main__":
    print("=== Futures Initial Margin ===")
    es_margin = futures_initial_margin("ES", price=4500, contracts=2)
    print(f"ES 2 contracts: ${es_margin.initial_margin} initial, ${es_margin.maintenance_margin} maint")

    btc_margin = futures_initial_margin("BTC", price=65000, contracts=1)
    print(f"BTC 1 contract: ${btc_margin.initial_margin} initial, ${btc_margin.maintenance_margin} maint")

    print()
    print("=== SPAN-style Margin ===")
    scenarios = [4400, 4450, 4500, 4550, 4600]
    span = span_margin(scenarios, position_size=2, contract_multiplier=50)
    print(f"Long 2 ES across scenarios: ${span} margin required")

    print()
    print("=== Option Greeks (ATM call) ===")
    greeks = option_greeks("call", S=100, K=100, T=0.25, r=0.05, sigma=0.20)
    print(f"ATM Call 3M: price=${greeks.price}, delta={greeks.delta}, gamma={greeks.gamma}, "
          f"vega={greeks.vega}, theta={greeks.theta}")

    print()
    print("=== Option Greeks (OTM put) ===")
    greeks_put = option_greeks("put", S=100, K=95, T=0.5, r=0.05, sigma=0.25)
    print(f"OTM Put 6M: price=${greeks_put.price}, delta={greeks_put.delta}, "
          f"gamma={greeks_put.gamma}, vega={greeks_put.vega}, theta={greeks_put.theta}")

    print()
    print("=== Portfolio Greeks ===")
    portfolio = [
        {"option_type": "call", "K": 100, "T": 0.25, "sigma": 0.20, "quantity": 1},
        {"option_type": "put", "K": 95, "T": 0.5, "sigma": 0.25, "quantity": 1},
        {"option_type": "call", "K": 105, "T": 0.25, "sigma": 0.22, "quantity": -1},
    ]
    pg = aggregate_portfolio_greeks(portfolio, spot=100)
    print(f"Portfolio: delta={pg.delta}, gamma={pg.gamma}, vega={pg.vega}, "
          f"theta={pg.theta}, rho={pg.rho}, $delta={pg.total_delta_dollars}")