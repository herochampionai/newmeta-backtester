"""Per-broker cost model — NDD/ECN tiers, markup, swap, commission schedules.

Loads broker config from config/brokers.json, applies exact fee schedule per symbol.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json


BROKER_PATH = Path(__file__).parent.parent / "config" / "brokers.json"


@dataclass
class BrokerCost:
    name: str
    account_type: str  # standard | raw | ecn | pro
    commission_per_lot_rt: float = 0.0  # round-trip $ per lot
    markup_pips: float = 0.0            # added to raw spread
    swap_long_pips: float = 0.0
    swap_short_pips: float = 0.0
    min_lot: float = 0.01
    max_lot: float = 100.0
    lot_step: float = 0.01
    margin_call_pct: float = 80.0
    stop_out_pct: float = 50.0
    currency: str = "USD"


DEFAULT_BROKERS = {
    "generic": BrokerCost(name="generic", account_type="standard", commission_per_lot_rt=7.0,
                          markup_pips=0.5, swap_long_pips=-0.5, swap_short_pips=0.2),
    "icmarkets_raw": BrokerCost(name="icmarkets", account_type="raw", commission_per_lot_rt=7.0,
                                markup_pips=0.0, swap_long_pips=-1.2, swap_short_pips=0.5),
    "pepperstone_razor": BrokerCost(name="pepperstone", account_type="razor", commission_per_lot_rt=7.0,
                                    markup_pips=0.0, swap_long_pips=-1.0, swap_short_pips=0.4),
    "fxpro_cTrader": BrokerCost(name="fxpro", account_type="ctrader", commission_per_lot_rt=9.0,
                                markup_pips=0.0, swap_long_pips=-0.8, swap_short_pips=0.3),
    "oanda": BrokerCost(name="oanda", account_type="standard", commission_per_lot_rt=0.0,
                        markup_pips=1.2, swap_long_pips=-0.3, swap_short_pips=0.1),
}


def load_brokers() -> dict[str, BrokerCost]:
    try:
        if BROKER_PATH.exists():
            data = json.loads(BROKER_PATH.read_text())
            return {k: BrokerCost(**v) for k, v in data.items()}
    except Exception:
        pass
    return DEFAULT_BROKERS.copy()


def save_brokers(brokers: dict[str, BrokerCost]) -> None:
    BROKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    BROKER_PATH.write_text(json.dumps({k: asdict(v) for k, v in brokers.items()}, indent=2))


def get_broker(name: str) -> BrokerCost:
    return load_brokers().get(name.lower(), DEFAULT_BROKERS["generic"])


def cost_per_trade(broker: BrokerCost, lots: float, spread_pips: float,
                   pip_size: float, contract_size: float) -> float:
    """Total cost per trade (entry + exit) in account currency."""
    # Commission (round-trip already)
    comm = broker.commission_per_lot_rt * lots
    # Spread cost (markup + raw spread) — paid on entry and exit
    total_spread = spread_pips + broker.markup_pips
    spread_cost = total_spread * pip_size * contract_size * lots * 2  # entry + exit
    return comm + spread_cost


def swap_per_day(broker: BrokerCost, direction: int, lots: float,
                 pip_size: float, contract_size: float, triple_day: int = 2) -> float:
    """Daily swap in account currency (direction: +1 long, -1 short)."""
    pips = broker.swap_long_pips if direction > 0 else broker.swap_short_pips
    return pips * pip_size * contract_size * lots


def margin_required(broker: BrokerCost, lots: float, price: float,
                    contract_size: float, leverage: float = 30.0) -> float:
    return abs(lots) * contract_size * price / max(leverage, 1.0)


def margin_level(equity: float, margin_used: float) -> float | None:
    if margin_used <= 0:
        return None
    return equity / margin_used * 100.0


def check_margin_call(broker: BrokerCost, equity: float, margin_used: float) -> str | None:
    lvl = margin_level(equity, margin_used)
    if lvl is None:
        return None
    if lvl <= broker.stop_out_pct:
        return "STOP_OUT"
    if lvl <= broker.margin_call_pct:
        return "MARGIN_CALL"
    return None