"""Auto-detect strategy from a dropped file (.mq5 / .py / .txt).
For .mq5: extract input parameters and indicator names, find matching strategy class.
For .py: if it has a class subclassing BaseStrategy, register it.
For .txt: parse for keyword hints (strategy name, params)."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import pandas as pd

from strategies import STRATEGY_REGISTRY


# Map MQL5 indicator names → our strategy class names
INDICATOR_TO_STRATEGY = {
    "iAC": "ac_ao", "iAO": "ac_ao", "Accelerator": "ac_ao", "Awesome": "ac_ao",
    "iADX": "adx", "ADX": "adx", "DI": "adx",
    "iDeMarker": "dem", "DeMarker": "dem",
    "iBands": "fbb", "iForce": "fbb", "Bollinger": "fbb", "Force": "fbb",
    "iMFI": "mfi", "MFI": "mfi", "MoneyFlow": "mfi",
    "iMACD": "ms", "iStochastic": "ms", "Stochastic": "ms", "MACD": "ms",
}

STRATEGY_NAMES = list(STRATEGY_REGISTRY.keys())


def detect_from_extension(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".mq5":
        return detect_from_mq5(p)
    elif suffix == ".py":
        return detect_from_py(p)
    elif suffix in (".txt", ".md"):
        return detect_from_text(p)
    return {"error": f"unsupported file type: {suffix}",
            "suggested_strategy": "fbb",  # safe default
            "params": {}}


def detect_from_mq5(path: Path) -> dict:
    """Parse MQL5 source for indicator names + input params."""
    try:
        text = path.read_text(encoding="utf-16", errors="replace")
    except Exception:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": f"cannot read: {e}", "suggested_strategy": "fbb", "params": {}}
    # Find indicator handles
    indicators = set()
    for kw, strat in INDICATOR_TO_STRATEGY.items():
        if kw in text:
            indicators.add(strat)
    # Find input lines like "input int X = 5;"
    inputs = {}
    for m in re.finditer(r'input\s+(?:int|double|bool|string)\s+(\w+)\s*=\s*([^;]+);',
                          text):
        name, default = m.group(1).strip(), m.group(2).strip()
        # Strip inline comment after =
        default = re.split(r'\s*//', default, maxsplit=1)[0].strip()
        inputs[name] = _parse_default(default)
    # Score strategies: how many indicators match
    chosen = max(indicators, key=lambda s: sum(1 for k, v in INDICATOR_TO_STRATEGY.items()
                                               if v == s and k in text)) if indicators else None
    return {
        "file": str(path),
        "type": "mq5",
        "indicators_found": list(indicators),
        "input_count": len(inputs),
        "inputs": inputs,
        "suggested_strategy": chosen or "fbb",
        "matched_strategies": list(indicators),
    }


def detect_from_py(path: Path) -> dict:
    """For .py: try import + look for BaseStrategy subclass."""
    text = path.read_text(encoding="utf-8", errors="replace")
    info = {"file": str(path), "type": "py", "inputs": {}}
    # Find classes
    classes = re.findall(r'class\s+(\w+)Strategy', text)
    if classes:
        info["classes"] = classes
        info["suggested_strategy"] = classes[0].lower().replace("strategy", "_strategy").strip("_")
        # try direct match
        for c in classes:
            key = c.lower().replace("_strategy", "").replace("strategy", "")
            if key in STRATEGY_NAMES:
                info["suggested_strategy"] = key
                break
    else:
        info["suggested_strategy"] = "fbb"
    # Find any obvious params
    for m in re.finditer(r'(\w+)\s*[:=]\s*([\d.]+)', text):
        name, val = m.group(1), m.group(2)
        if name[0].isupper() and not name.startswith("_"):
            try:
                info["inputs"][name] = float(val) if "." in val else int(val)
            except ValueError:
                pass
    return info


def detect_from_text(path: Path) -> dict:
    """For .txt: search for strategy names + numbers."""
    text = path.read_text(encoding="utf-8", errors="replace")
    info = {"file": str(path), "type": "txt", "inputs": {}}
    lower = text.lower()
    found = [s for s in STRATEGY_NAMES if s in lower]
    info["matched_strategies"] = found
    info["suggested_strategy"] = found[0] if found else "fbb"
    # Extract likely param values
    for m in re.finditer(r'(\w+)\s*[=:]\s*([\d.]+)', text):
        name, val = m.group(1), m.group(2)
        try:
            info["inputs"][name] = float(val) if "." in val else int(val)
        except ValueError:
            pass
    return info


def _parse_default(s: str) -> Any:
    """Parse MQL5 default value: true/false, 1.0, -5, "string", etc."""
    s = s.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def apply_suggested(detection: dict, df: pd.DataFrame, params_override: dict | None = None) -> tuple:
    """Build a strategy instance from detection. Returns (strategy, params, info)."""
    name = detection.get("suggested_strategy", "fbb")
    if name not in STRATEGY_REGISTRY:
        name = "fbb"
    cls = STRATEGY_REGISTRY[name]
    # Build defaults from detection inputs (intersect with strategy's expected params)
    from strategies._base import BaseStrategy
    # Use first strategy's docstring to know expected params? Just use known defaults.
    defaults = {
        "ac_ao": dict(open_orders_type=1, close_orders_type=0, level_open_orders=80,
                      level_close_orders=70, use_acceleration_filter=False,
                      use_ao_synchronization=False, acceleration_bars=3,
                      min_acceleration=0.0005, min_ao_synchronization=0.0003),
        "adx": dict(open_orders_type=1, close_orders_type=4, level_open_orders_1=55,
                    level_open_orders_2=15, level_close_orders_1=15,
                    level_close_orders_2=5, use_di_crossover=True,
                    crossover_lookback=3, min_crossover_gap=5, bars_calculate=20),
        "dem": dict(open_orders_type=3, close_orders_type=0, level_open_orders=75,
                    level_close_orders=70, bars_calculate=20),
        "fbb": dict(open_orders_type_1=1, open_orders_type_2=0, close_orders_type_1=0,
                    close_orders_type_2=0, level_open_orders_1=0,
                    level_open_orders_2=50, level_close_orders_1=40,
                    level_close_orders_2=40, bars_calculate=20, deviation=1.8),
        "mfi": dict(open_orders_type=3, close_orders_type=0, level_open_orders=70,
                    level_close_orders=70, use_slope_filter=False,
                    use_divergence=False, use_hidden_divergence=False,
                    bars_calculate=12, slope_lookback=5, min_slope_strength=3,
                    divergence_bars=10),
        "ms": dict(open_orders_type_1=8, open_orders_type_2=0, close_orders_type_1=0,
                   close_orders_type_2=0, level_open_orders_1=20,
                   level_open_orders_2=80, level_close_orders_1=50,
                   level_close_orders_2=65, use_confluence_filter=False,
                   use_macd_divergence=False, use_stoch_divergence=False,
                   use_histogram_divergence=False, fast_ema_period=3,
                   slow_ema_period=9, signal_period=2, k_period=5, d_period=3,
                   slowing_period=12),
    }
    params = dict(defaults.get(name, {}))
    if params_override:
        params.update(params_override)
    strat = cls(params=params)
    info = {
        "detection": detection,
        "strategy": name,
        "params_used": params,
    }
    return strat, params, info