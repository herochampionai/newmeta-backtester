"""Broker-true symbol specs — pulls MT5 symbol_info, caches to JSON.

Kills the averaged-swap bug: each direction uses its own swap_long/swap_short,
triple-day comes from the broker (SYMBOL_SWAP_ROLLOVER3DAYS), not hardcoded Wed.
Falls back to MARKET_PROFILES presets when MT5 offline so backtests never crash.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path

SPEC_CACHE = Path(__file__).parent.parent / "config" / "symbol_specs.json"


@dataclass
class SymbolSpec:
    symbol: str
    digits: int = 5
    point: float = 0.00001
    pip_size: float = 0.0001
    tick_value: float = 1.0
    tick_size: float = 0.00001
    contract_size: float = 100_000
    volume_min: float = 0.01
    volume_max: float = 500.0
    volume_step: float = 0.01
    spread_pips: float = 1.0
    swap_long_pips: float = -0.5
    swap_short_pips: float = 0.2
    triple_day: int = 2  # Monday=0 .. Sunday=6, MT5 default Wednesday=2
    margin_per_lot: float = 0.0
    source: str = "preset"


def _preset_for(symbol: str) -> SymbolSpec:
    from backtester.execution import profile_for, infer_market
    s = (symbol or "EURUSD").upper()
    prof = profile_for("Auto", s)
    digits = 2 if ("XAU" in s or "BTC" in s or "ETH" in s) else 5
    if "JPY" in s:
        digits = 3
    point = 10 ** -digits
    pip = 0.01 if digits in (2, 3) else 0.0001
    tick_value = 1.0 if "Forex" in infer_market(s) else (1.0 if "XAU" in s else 0.1)
    contract = prof.contract_size
    return SymbolSpec(symbol=s, digits=digits, point=point, pip_size=pip,
                      tick_value=tick_value, tick_size=point,
                      contract_size=contract,
                      spread_pips=prof.default_spread_pips,
                      swap_long_pips=prof.default_long_swap_pips,
                      swap_short_pips=prof.default_short_swap_pips,
                      source="preset")


def get_spec(symbol: str, terminal: str | None = None, refresh: bool = False) -> SymbolSpec:
    """MT5 symbol_info first, JSON cache second, preset fallback. Never raises."""
    sym = (symbol or "EURUSD").upper()
    if not refresh:
        try:
            if SPEC_CACHE.exists():
                cache = json.loads(SPEC_CACHE.read_text())
                if sym in cache:
                    d = cache[sym]
                    return SymbolSpec(**d)
        except Exception:
            pass
    try:
        from data.mt5_export import resolve_terminal, init_mt5
        import MetaTrader5 as mt5
        t = terminal or resolve_terminal()
        if t and init_mt5(t):
            info = mt5.symbol_info(sym)
            if info is not None:
                mt5.symbol_select(sym, True)
                tick = mt5.symbol_info_tick(sym)
                # MT5 swap_long/swap_short are in deposit currency per lot per day (already in $)
                # Store as-is for direct use. Also compute pip equivalent for display.
                raw_swap_long = float(getattr(info, "swap_long", 0.0))
                raw_swap_short = float(getattr(info, "swap_short", 0.0))
                tick_val = float(info.trade_tick_value or 1.0)
                tick_sz = float(info.trade_tick_size or info.point)
                contract_sz = float(info.trade_contract_size or 100_000)
                # Pip equivalent: swap $ / (pip_size * contract_size)
                pip_size = 0.01 if info.digits in (2, 3) else 0.0001
                swap_long_pips = raw_swap_long / (pip_size * contract_sz) if contract_sz else -0.5
                swap_short_pips = raw_swap_short / (pip_size * contract_sz) if contract_sz else 0.2
                
                spec = SymbolSpec(
                    symbol=sym, digits=int(info.digits), point=float(info.point),
                    pip_size=pip_size,
                    tick_value=tick_val,
                    tick_size=tick_sz,
                    contract_size=contract_sz,
                    volume_min=float(info.volume_min or 0.01),
                    volume_max=float(info.volume_max or 500.0),
                    volume_step=float(info.volume_step or 0.01),
                    spread_pips=float((tick.ask - tick.bid) / info.point / 10) if tick and tick.ask and tick.bid else 1.0,
                    swap_long_pips=swap_long_pips,
                    swap_short_pips=swap_short_pips,
                    triple_day=int(getattr(info, "swap_rollover3days", 2)),
                    margin_per_lot=float(getattr(info, "margin_initial", 0.0)),
                    source="mt5",
                )
                mt5.shutdown()
                try:
                    SPEC_CACHE.parent.mkdir(parents=True, exist_ok=True)
                    cache = json.loads(SPEC_CACHE.read_text()) if SPEC_CACHE.exists() else {}
                    cache[sym] = asdict(spec)
                    SPEC_CACHE.write_text(json.dumps(cache, indent=2))
                except Exception:
                    pass
                return spec
            mt5.shutdown()
    except Exception:
        pass
    return _preset_for(sym)


def swap_for_day(spec: SymbolSpec, direction: int, lots: float, date) -> float:
    """Direction-aware daily swap in account $ — no averaging. Triple-day from broker.
    
    Swap values in spec are now normalized to pips. Convert to $ using pip_size * contract_size.
    """
    import pandas as pd
    ts = pd.Timestamp(date)
    if ts.weekday() >= 5:
        return 0.0
    mult = 3.0 if ts.weekday() == int(spec.triple_day) else 1.0
    pips = spec.swap_long_pips if direction > 0 else spec.swap_short_pips
    # pips * pip_size * contract_size * lots = $ per day per lot
    return float(pips * mult * spec.pip_size * spec.contract_size * lots)
