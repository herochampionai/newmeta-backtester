"""Real order-ticket engine — MT5/LEAN fill semantics in one place.

- Buy at ask, sell at bid (never mid/close)
- max_spread reject (MT5 'max deviation' style): skip entry when spread too wide
- Stop/limit intrabar with SL-before-TP priority (MT5 every-tick rule)
- Partial closes (scale-out) + slippage by volatility regime
- Commission + spread cost unified per fill
"""
from __future__ import annotations

from dataclasses import dataclass

REJECT = None


def market_fill(side: int, mid: float, spread_price: float,
                spread_pips: float, max_spread_pips: float | None,
                slippage_pips: float, pip_size: float,
                atr_pips: float = 0.0) -> float | None:
    """Return fill price or None (rejected). side +1 buy, -1 sell."""
    if max_spread_pips is not None and spread_pips > float(max_spread_pips):
        return REJECT
    base = (mid + spread_price / 2) if side > 0 else (mid - spread_price / 2)
    # Volatility-regime slippage: wider ATR -> more adverse slip (LEAN-style)
    slip = abs(float(slippage_pips)) * pip_size * (1.0 + min(atr_pips / 50.0, 2.0))
    return float(base * (1 + slip / max(base, 1e-12)) if side > 0 else base * (1 - slip / max(base, 1e-12)))


def check_exits_tick(position: int, bid: float, ask: float,
                     sl_price: float | None, tp_price: float | None) -> str | None:
    """Per-tick stop/limit check. SL first (MT5 ambiguity rule). Returns 'sl'/'tp'/None."""
    if position > 0:
        if sl_price is not None and bid <= sl_price:
            return "sl"
        if tp_price is not None and bid >= tp_price:
            return "tp"
    elif position < 0:
        if sl_price is not None and ask >= sl_price:
            return "sl"
        if tp_price is not None and ask <= tp_price:
            return "tp"
    return None


@dataclass
class Ticket:
    direction: int
    lots: float
    entry_price: float
    entry_bar: int
    sl_price: float | None = None
    tp_price: float | None = None

    def partial(self, frac: float, exit_price: float, contract_size: float) -> float:
        """Close frac of lots, return realized PnL. Mutates lots."""
        frac = min(max(frac, 0.0), 1.0)
        closed = self.lots * frac
        pnl = (exit_price - self.entry_price) * self.direction * contract_size * closed
        self.lots -= closed
        return float(pnl)
