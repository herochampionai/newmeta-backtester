"""Multi-currency portfolio: FX cash book, cross-margin, correlation limits, VaR.

R005: Enables trading multiple assets in different denominations (EURUSD, USDJPY,
XAUUSD, BTCUSD, etc.) with proper currency conversion and portfolio-level risk
management.

Features:
- FX cash book: track each currency separately, convert PnL to base
- Cross-margin: total margin across all positions, unified equity
- Correlation limits: enforce max correlation exposure (no >70% correlated)
- Portfolio VaR: historical + parametric
- FX rate feed: cache FX rates for consistent conversion

Usage:
    portfolio = MultiCurrencyPortfolio(base_currency="USD")
    portfolio.add_position("EURUSD", lots=0.1, contract_size=100000, price=1.1, ...)
    portfolio.add_position("USDJPY", lots=0.1, contract_size=100000, price=150.0, ...)
    metrics = portfolio.compute_metrics()
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path
import numpy as np
import pandas as pd


# FX rate cache (USD-base default)
FX_CACHE_PATH = Path(__file__).parent.parent / "config" / "fx_rates.json"

# Common currency pairs (against USD)
DEFAULT_FX_RATES = {
    "USD": 1.0,
    "EUR": 1.10,
    "GBP": 1.27,
    "JPY": 0.0067,
    "CHF": 1.13,
    "AUD": 0.66,
    "CAD": 0.74,
    "NZD": 0.61,
    "XAU": 2350.0,  # Gold oz in USD
    "XAG": 28.0,    # Silver oz in USD
    "BTC": 65000.0,
    "ETH": 3500.0,
}


@dataclass
class Position:
    symbol: str
    direction: int  # +1 = long, -1 = short
    lots: float
    contract_size: float
    entry_price: float
    current_price: float
    pip_size: float = 0.0001
    currency: str = "USD"  # quote currency
    base_currency: str = "USD"  # base currency
    pnl: float = 0.0
    pnl_pct: float = 0.0
    margin_required: float = 0.0
    leverage: float = 30.0


@dataclass
class FXRate:
    """FX rate with timestamp."""
    from_ccy: str
    to_ccy: str
    rate: float
    timestamp: str = ""


@dataclass
class PortfolioMetrics:
    """Aggregated portfolio metrics.

    VaR semantics (see compute_metrics): var_95/var_99/expected_shortfall
    are negative-or-zero dollar losses at the stated confidence, and
    var_method names how they were derived (equity_curve preferred,
    per_trade fallback). A var_method of "" means insufficient data.
    """
    total_pnl: float = 0.0
    total_pnl_base: float = 0.0  # converted to base currency
    total_margin: float = 0.0
    equity: float = 0.0
    free_margin: float = 0.0
    margin_level: Optional[float] = None
    var_95: float = 0.0  # Value at Risk 95% (negative $ loss)
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    var_method: str = ""
    max_correlation_exposure: float = 0.0
    n_positions: int = 0
    n_currencies: int = 0
    base_currency: str = "USD"
    per_currency_pnl: dict = field(default_factory=dict)


class MultiCurrencyPortfolio:
    """Multi-currency portfolio with FX conversion, cross-margin, and VaR."""

    def __init__(self, base_currency: str = "USD", init_balance: float = 10000.0,
                 fx_rates: dict | None = None, leverage: float = 30.0):
        self.base_currency = base_currency.upper()
        self.init_balance = init_balance
        self.balance = init_balance
        self.leverage = leverage
        self.positions: list[Position] = []
        self.equity_history: list[float] = [init_balance]
        self.fx_rates = dict(fx_rates or DEFAULT_FX_RATES)
        self._load_fx_cache()

    def _load_fx_cache(self):
        """Load FX rates from cache file if available."""
        try:
            if FX_CACHE_PATH.exists():
                data = json.loads(FX_CACHE_PATH.read_text())
                self.fx_rates.update(data)
        except Exception:
            pass

    def save_fx_cache(self):
        """Save current FX rates to cache."""
        try:
            FX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            FX_CACHE_PATH.write_text(json.dumps(self.fx_rates, indent=2))
        except Exception:
            pass

    def set_fx_rate(self, currency: str, rate: float):
        """Set FX rate against base currency."""
        self.fx_rates[currency.upper()] = float(rate)

    def get_fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        """Get FX rate from one currency to another (cross-rate via USD)."""
        from_ccy = from_ccy.upper()
        to_ccy = to_ccy.upper()
        if from_ccy == to_ccy:
            return 1.0
        # Rates are stored vs USD
        rate_from = self.fx_rates.get(from_ccy, 1.0)
        rate_to = self.fx_rates.get(to_ccy, 1.0)
        if rate_to == 0:
            return 1.0
        return rate_from / rate_to

    def add_position(self, symbol: str, direction: int, lots: float,
                     contract_size: float, entry_price: float, current_price: float,
                     pip_size: float = 0.0001, currency: str = "USD",
                     base_currency: str = "USD") -> Position:
        """Add a position and compute its PnL + margin."""
        # Compute PnL: (current - entry) * direction * contract_size * lots
        price_diff = (current_price - entry_price) * direction
        pnl_local = price_diff * contract_size * lots
        # Convert to base currency if needed
        if currency.upper() != self.base_currency.upper():
            fx_rate = self.get_fx_rate(currency, self.base_currency)
            pnl_base = pnl_local * fx_rate
        else:
            pnl_base = pnl_local

        # Margin
        margin = abs(lots) * contract_size * current_price / self.leverage
        if currency.upper() != self.base_currency.upper():
            margin *= self.get_fx_rate(currency, self.base_currency)

        # PnL percent (relative to margin)
        pnl_pct = pnl_base / margin if margin > 0 else 0.0

        pos = Position(
            symbol=symbol, direction=direction, lots=lots,
            contract_size=contract_size, entry_price=entry_price,
            current_price=current_price, pip_size=pip_size,
            currency=currency, base_currency=base_currency,
            pnl=pnl_local, pnl_pct=pnl_pct,
            margin_required=margin, leverage=self.leverage
        )
        self.positions.append(pos)
        return pos

    def compute_metrics(self, returns: np.ndarray | None = None,
                          equity=None) -> PortfolioMetrics:
        """Compute aggregated portfolio metrics including VaR.

        If no positions are tracked but equity_history exists, derives total_pnl
        and equity from the equity curve. This handles the common case where
        the user just wants portfolio-level metrics from a backtest result
        without manually adding every closed trade as a Position.

        VaR methodology (historical simulation, honest units):
        - equity curve given (>= 30 points): period returns -> 5th/1st
          percentile x latest equity = dollar VaR; ES = mean tail loss.
          Method: "equity_curve".
        - else trade PnLs given: 5th/1st percentile of per-trade $ outcomes
          (i.e. "95% of single trades lose no more than $X") + tail mean.
          Method: "per_trade". NOT scaled by total PnL.
        """
        m = PortfolioMetrics(base_currency=self.base_currency)
        m.n_positions = len(self.positions)
        per_ccy_pnl = {}

        for p in self.positions:
            m.total_pnl += p.pnl
            m.total_margin += p.margin_required

            # Convert PnL to base
            if p.currency.upper() != self.base_currency.upper():
                fx_rate = self.get_fx_rate(p.currency, self.base_currency)
                pnl_base = p.pnl * fx_rate
            else:
                pnl_base = p.pnl
            m.total_pnl_base += pnl_base

            # Per-currency PnL
            ccy = p.currency.upper()
            per_ccy_pnl[ccy] = per_ccy_pnl.get(ccy, 0.0) + pnl_base

        m.n_currencies = len(per_ccy_pnl)
        m.per_currency_pnl = {k: round(v, 2) for k, v in per_ccy_pnl.items()}

        # Fallback: derive total_pnl from equity history when no positions tracked.
        # This lets users call update_equity() and get meaningful metrics without
        # manually adding every closed trade as a Position.
        if m.n_positions == 0 and len(self.equity_history) >= 2:
            m.total_pnl_base = self.equity_history[-1] - self.equity_history[0]
            m.total_pnl = m.total_pnl_base
            per_ccy_pnl[self.base_currency.upper()] = m.total_pnl_base
            m.n_currencies = 1
            m.per_currency_pnl = {self.base_currency.upper(): round(m.total_pnl_base, 2)}

        m.equity = self.balance + m.total_pnl_base
        m.free_margin = m.equity - m.total_margin
        if m.total_margin > 0:
            m.margin_level = (m.equity / m.total_margin) * 100

        # VaR via historical simulation. Preferred input is an equity
        # curve (per-period portfolio risk); fallback is the per-trade PnL
        # distribution (per-trade risk). Never multiply a quantile by an
        # unrelated total — that mixes units and fabricates tail size.
        eq_vals = None
        if equity is not None:
            try:
                import pandas as _pd
                eq_vals = (_pd.Series(equity).dropna().astype(float).values
                           if not isinstance(equity, _pd.Series)
                           else equity.dropna().astype(float).values)
            except Exception:
                eq_vals = None
        if eq_vals is not None and len(eq_vals) >= 30:
            base = eq_vals[:-1]
            rets = (eq_vals[1:] - base) / np.where(base != 0, base, np.nan)
            rets = rets[np.isfinite(rets)]
            if len(rets) >= 10:
                ref_equity = float(eq_vals[-1])
                q95 = float(np.percentile(rets, 5))
                q99 = float(np.percentile(rets, 1))
                m.var_95 = min(0.0, q95 * ref_equity)
                m.var_99 = min(0.0, q99 * ref_equity)
                tail = rets[rets <= q95]
                m.expected_shortfall = min(0.0, float(np.mean(tail)) * ref_equity) \
                    if len(tail) > 0 else 0.0
                m.var_method = "equity_curve"
        if not m.var_method and returns is not None and len(returns) > 10:
            pnls = np.asarray(returns, dtype=float)
            pnls = pnls[np.isfinite(pnls)]
            if len(pnls) > 5:
                q95 = float(np.percentile(pnls, 5))
                q99 = float(np.percentile(pnls, 1))
                m.var_95 = min(0.0, q95)
                m.var_99 = min(0.0, q99)
                tail = pnls[pnls <= q95]
                m.expected_shortfall = min(0.0, float(np.mean(tail))) \
                    if len(tail) > 0 else 0.0
                m.var_method = "per_trade"

        # Correlation exposure: max abs correlation between any pair
        if len(self.positions) >= 2:
            m.max_correlation_exposure = self._compute_max_correlation_exposure()

        return m

    def _compute_max_correlation_exposure(self) -> float:
        """Compute max pairwise correlation across positions (simplified)."""
        # Without historical data, use a heuristic based on currencies
        # In real use, fetch historical returns and compute actual correlation
        symbols = [p.symbol for p in self.positions]
        n = len(symbols)
        if n < 2:
            return 0.0
        # Rough proxy: same base/quote currency = high correlation
        max_corr = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = symbols[i], symbols[j]
                # Heuristic correlation based on currency overlap
                if si[:3] == sj[:3] or si[3:] == sj[3:]:
                    corr = 0.7
                elif si[:3] in ("USD",) or sj[:3] in ("USD",):
                    corr = 0.5  # USD-quoted pairs
                else:
                    corr = 0.3
                max_corr = max(max_corr, corr)
        return max_corr

    def check_correlation_limit(self, max_correlation: float = 0.7) -> dict:
        """Check if any pair exceeds correlation limit."""
        violations = []
        n = len(self.positions)
        for i in range(n):
            for j in range(i + 1, n):
                # Use heuristic for now
                si, sj = self.positions[i].symbol, self.positions[j].symbol
                if si[:3] == sj[:3] or si[3:] == sj[3:]:
                    corr = 0.7
                else:
                    corr = 0.3
                if corr > max_correlation:
                    violations.append({
                        "symbol_a": si,
                        "symbol_b": sj,
                        "correlation": corr,
                        "limit": max_correlation,
                    })
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "max_correlation_observed": self._compute_max_correlation_exposure(),
            "limit": max_correlation,
        }

    def update_equity(self, pnl_delta: float):
        """Update equity history (for time-series analysis)."""
        self.equity_history.append(self.equity_history[-1] + pnl_delta)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "base_currency": self.base_currency,
            "init_balance": self.init_balance,
            "balance": self.balance,
            "n_positions": len(self.positions),
            "positions": [
                {
                    "symbol": p.symbol, "direction": p.direction, "lots": p.lots,
                    "entry_price": p.entry_price, "current_price": p.current_price,
                    "pnl": p.pnl, "pnl_base": p.pnl * self.get_fx_rate(p.currency, self.base_currency),
                    "margin": p.margin_required,
                    "currency": p.currency,
                }
                for p in self.positions
            ],
            "fx_rates": self.fx_rates,
        }


# ---------- Helper functions ----------
def compute_portfolio_var(equity_returns: np.ndarray, confidence: float = 0.95) -> dict:
    """Historical VaR and CVaR from equity returns.

    Args:
        equity_returns: array of period returns (e.g., daily)
        confidence: 0.95 = 95% VaR
    """
    try:
        arr = np.asarray(equity_returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 5:
            return {"error": "need 5+ returns"}

        var = float(np.percentile(arr, 100 * (1 - confidence)))
        # CVaR: mean of returns worse than VaR
        tail = arr[arr <= var]
        cvar = float(np.mean(tail)) if len(tail) > 0 else var

        return {
            "var": round(var, 6),
            "cvar": round(cvar, 6),
            "confidence": confidence,
            "n_samples": len(arr),
            "worst_observed": round(float(np.min(arr)), 6),
            "best_observed": round(float(np.max(arr)), 6),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def fx_convert(pnl: float, from_ccy: str, to_ccy: str, fx_rates: dict | None = None) -> float:
    """Convert PnL between currencies using provided rates."""
    rates = fx_rates or DEFAULT_FX_RATES
    from_ccy = from_ccy.upper()
    to_ccy = to_ccy.upper()
    if from_ccy == to_ccy:
        return float(pnl)
    rate_from = rates.get(from_ccy, 1.0)
    rate_to = rates.get(to_ccy, 1.0)
    if rate_to == 0:
        return float(pnl)
    return float(pnl * rate_from / rate_to)


# ---------- Self-test ----------
if __name__ == "__main__":
    # Test 1: Multi-currency portfolio
    portfolio = MultiCurrencyPortfolio(base_currency="USD", init_balance=10000)

    # EUR/USD long
    portfolio.add_position("EURUSD", direction=1, lots=0.1, contract_size=100000,
                           entry_price=1.1000, current_price=1.1050,
                           pip_size=0.0001, currency="USD", base_currency="EUR")
    # USD/JPY short
    portfolio.add_position("USDJPY", direction=-1, lots=0.1, contract_size=100000,
                           entry_price=150.00, current_price=149.50,
                           pip_size=0.01, currency="JPY", base_currency="USD")
    # XAU/USD long
    portfolio.add_position("XAUUSD", direction=1, lots=0.5, contract_size=100,
                           entry_price=2350.0, current_price=2360.0,
                           pip_size=0.01, currency="USD", base_currency="XAU")

    m = portfolio.compute_metrics()
    print("=== Portfolio Metrics ===")
    print(f"Total PnL: ${m.total_pnl:.2f}")
    print(f"Total PnL (base USD): ${m.total_pnl_base:.2f}")
    print(f"Total margin: ${m.total_margin:.2f}")
    print(f"Equity: ${m.equity:.2f}")
    print(f"Margin level: {m.margin_level:.1f}%" if m.margin_level else "Margin level: N/A")
    print(f"N positions: {m.n_positions}")
    print(f"N currencies: {m.n_currencies}")
    print(f"Per-currency PnL: {m.per_currency_pnl}")

    # Test 2: Correlation check
    print()
    print("=== Correlation Limit Check ===")
    corr_check = portfolio.check_correlation_limit(max_correlation=0.7)
    print(f"Passed: {corr_check['passed']}")
    print(f"Max correlation: {corr_check['max_correlation_observed']}")
    if corr_check["violations"]:
        print(f"Violations: {corr_check['violations']}")

    # Test 3: VaR
    print()
    print("=== Portfolio VaR ===")
    np.random.seed(42)
    fake_returns = np.random.normal(0.001, 0.02, 252)
    var_result = compute_portfolio_var(fake_returns, confidence=0.95)
    print(f"VaR 95%: {var_result['var']}")
    print(f"CVaR 95%: {var_result['cvar']}")
    print(f"Worst observed: {var_result['worst_observed']}")

    # Test 4: FX conversion
    print()
    print("=== FX Conversion ===")
    pnl_eur = 100.0
    pnl_usd = fx_convert(pnl_eur, "EUR", "USD")
    print(f"100 EUR = {pnl_usd:.2f} USD")
    pnl_jpy = 1000.0
    pnl_usd2 = fx_convert(pnl_jpy, "JPY", "USD")
    print(f"1000 JPY = {pnl_usd2:.2f} USD")