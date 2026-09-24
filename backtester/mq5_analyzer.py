"""Enhanced MQ5/MT5 analyzer — parses EA, extracts params, compares to MT5 backtest, auto-corrects.

- Parses .mq5 for input params, OnTester(), strategy logic
- Runs same params through Newmeta (strict parity)
- Detects discrepancies: spread model, swap, commission, tick gen, margin
- Auto-generates corrected .set + patch notes
"""
from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from backtester.parity import strict_parity_test
from backtester.symbol_spec import get_spec
from backtester.broker_cost import get_broker, cost_per_trade
from data.live_fetcher import fetch_with_priority
from strategies import STRATEGY_REGISTRY


@dataclass
class MQ5Params:
    inputs: dict[str, Any]
    on_tester_code: str | None = None
    strategy_name: str | None = None


def parse_mq5(file_path: str | Path) -> MQ5Params:
    """Extract input parameters and OnTester from .mq5 file."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    inputs = {}
    # input double/int/string/bool name = value;
    for m in re.finditer(r'input\s+(double|int|string|bool)\s+(\w+)\s*=\s*([^;]+);', text):
        typ, name, val = m.groups()
        val = val.strip().strip('"')
        try:
            if typ == "int":
                inputs[name] = int(val)
            elif typ == "double":
                inputs[name] = float(val)
            elif typ == "bool":
                inputs[name] = val.lower() in ("true", "1", "yes")
            else:
                inputs[name] = val
        except Exception:
            inputs[name] = val
    # OnTester
    ot_match = re.search(r'double\s+OnTester\s*\(\)\s*\{([^}]+)\}', text, re.DOTALL)
    on_tester = ot_match.group(1).strip() if ot_match else None
    # Strategy name hint
    strat_match = re.search(r'class\s+(\w+)\s*:\s*public\s+\w+', text)
    strat_name = strat_match.group(1) if strat_match else None
    return MQ5Params(inputs=inputs, on_tester_code=on_tester, strategy_name=strat_name)


def extract_mt5_backtest_params(mq5_path: str | Path) -> dict:
    """Parse .set file if exists next to .mq5."""
    set_path = Path(mq5_path).with_suffix(".set")
    if not set_path.exists():
        return {}
    try:
        content = set_path.read_text(encoding="utf-16", errors="ignore")
    except Exception:
        content = set_path.read_text(encoding="utf-8", errors="ignore")
    params = {}
    for line in content.splitlines():
        if "=" in line and not line.strip().startswith(";"):
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"')
            try:
                if "." in v:
                    params[k] = float(v)
                else:
                    params[k] = int(v)
            except Exception:
                params[k] = v
    return params


def analyze_discrepancies(mq5_params: MQ5Params, mt5_params: dict,
                          newmeta_result: dict, mt5_deals: Any,
                          symbol: str, timeframe: str,
                          broker_name: str = "generic") -> dict:
    """Compare Newmeta vs MT5 — detect and explain every discrepancy."""
    issues = []
    fixes = {}

    # 1. Spread model
    nm_spread = newmeta_result.get("execution", {}).get("spread_pips", 0)
    broker = get_broker(broker_name)
    spec = get_spec(symbol)
    if spec and spec.source == "mt5":
        mt5_spread = spec.spread_pips
        if abs(nm_spread - mt5_spread) > 0.5:
            issues.append(f"Spread mismatch: Newmeta {nm_spread:.1f} vs MT5 {mt5_spread:.1f} pips")
            fixes["spread_pips"] = mt5_spread

    # 2. Commission
    nm_comm = newmeta_result.get("execution", {}).get("commission_pips", 0)
    broker_comm = broker.commission_per_lot_rt / 10.0  # approx pips
    if abs(nm_comm - broker_comm) > 0.2:
        issues.append(f"Commission mismatch: Newmeta {nm_comm:.1f} vs Broker {broker_comm:.1f} pips")
        fixes["commission_pips"] = broker_comm

    # 3. Swap
    nm_swap = newmeta_result.get("swap_total", 0)
    # MT5 swap from deals
    mt5_swap = 0.0
    if mt5_deals is not None and hasattr(mt5_deals, "sum"):
        mt5_swap = float(mt5_deals.get("swap", pd.Series([0])).sum())
    if abs(nm_swap - mt5_swap) > 10.0:
        issues.append(f"Swap mismatch: Newmeta ${nm_swap:.0f} vs MT5 ${mt5_swap:.0f}")
        fixes["swap_long_pips"] = broker.swap_long_pips
        fixes["swap_short_pips"] = broker.swap_short_pips

    # 4. Tick generation
    nm_tick = newmeta_result.get("tick_source", "off")
    if nm_tick == "off":
        issues.append("Newmeta ran OHLC-only — MT5 uses every-tick. Enable tick_mode=synthetic/real")
        fixes["tick_mode"] = "synthetic"

    # 5. Margin/stop-out
    nm_min_lvl = newmeta_result.get("min_margin_level")
    if nm_min_lvl and nm_min_lvl < broker.stop_out_pct:
        issues.append(f"Stop-out hit: margin level {nm_min_lvl:.0f}% < broker {broker.stop_out_pct}%")
        fixes["leverage"] = 30.0
        fixes["stopout_level_pct"] = broker.stop_out_pct

    # 6. Parameter drift
    for k, v in mt5_params.items():
        if k in mq5_params.inputs and mq5_params.inputs[k] != v:
            issues.append(f"Param drift: {k} MQ5={mq5_params.inputs[k]} vs .set={v}")
            fixes[k] = v

    return {"issues": issues, "fixes": fixes, "pass": len(issues) == 0}


def auto_correct_set(mq5_path: str | Path, fixes: dict) -> str:
    """Generate corrected .set file with fixes applied."""
    mt5_params = extract_mt5_backtest_params(mq5_path)
    mt5_params.update(fixes)
    lines = []
    for k, v in mt5_params.items():
        if isinstance(v, float):
            lines.append(f"{k}={v:.6f}")
        else:
            lines.append(f"{k}={v}")
    out_path = Path(mq5_path).with_suffix(".corrected.set")
    out_path.write_text("\n".join(lines), encoding="utf-16")
    return str(out_path)


def full_mq5_audit(mq5_path: str | Path, symbol: str, timeframe: str,
                   start: str, end: str, terminal: str | None = None,
                   broker_name: str = "generic") -> dict:
    """End-to-end: parse MQ5, run Newmeta strict parity, compare, output corrected .set."""
    mq5 = parse_mq5(mq5_path)
    mt5_params = extract_mt5_backtest_params(mq5_path)
    strat_name = mq5.strategy_name or "fbb"
    if strat_name not in STRATEGY_REGISTRY:
        strat_name = "fbb"

    # Run Newmeta strict
    df, _ = fetch_with_priority(symbol, timeframe, allow_synthetic=False, terminal_override=terminal)
    if df is None or len(df) < 100:
        return {"error": "insufficient data"}
    spec = get_spec(symbol, terminal=terminal)
    params = {**mq5.inputs, **mt5_params}
    strat = STRATEGY_REGISTRY[strat_name](params=params)
    sig = strat.generate(df)
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    signals = {strat_name: (entries, direction)}
    nm_res = run_full(df, signals, params=params, tick_mode="synthetic", ticks_per_bar=20,
                      pip_size=spec.pip_size, contract_size=spec.contract_size,
                      base_lot=0.1, leverage=30.0, strict_data=True)

    # MT5 deals
    mt5_deals = fetch_mt5_deals(symbol, start, end, terminal)

    # Compare
    disc = analyze_discrepancies(mq5, mt5_params, nm_res, mt5_deals, symbol, timeframe, broker_name)

    # Corrected .set
    corrected_set = auto_correct_set(mq5_path, disc["fixes"]) if disc["fixes"] else ""

    return {
        "mq5_params": mq5.inputs,
        "mt5_params": mt5_params,
        "newmeta_metrics": nm_res.get("metrics", {}),
        "discrepancies": disc["issues"],
        "fixes_applied": disc["fixes"],
        "corrected_set": corrected_set,
        "parity_pass": disc["pass"],
    }


# Import here to avoid circular
import pandas as pd
from backtester.engine_full import run_full
from backtester.parity import fetch_mt5_deals