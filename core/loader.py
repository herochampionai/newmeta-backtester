"""Universal file loader — detects and converts any strategy file to a Strategy class.
Supports .mq5, .py (BaseStrategy subclass), .pine, .txt (params)."""
from __future__ import annotations
import re
import importlib.util
from pathlib import Path
from typing import Any

from strategies import STRATEGY_REGISTRY


def load_any_strategy(path: str | Path, params_override: dict | None = None) -> tuple:
    """Load any strategy file. Returns (strategy_class, params, info) or (None, None, error_info)."""
    p = Path(path)
    if not p.exists():
        return None, None, {"error": "file not found", "file": str(p)}
    suffix = p.suffix.lower()
    info = {"file": str(p), "type": None}
    try:
        if suffix == ".mq5":
            return _load_mq5(p, params_override, info)
        elif suffix == ".py":
            return _load_py(p, params_override, info)
        elif suffix in (".pine", ".txt", ".md"):
            # Try PineScript first
            if suffix == ".pine":
                return _load_pine(p, params_override, info)
            else:
                # .txt — try keyword detection
                return _load_text(p, params_override, info)
    except Exception as e:
        return None, None, {"error": str(e), "type": type(e).__name__, "file": str(p),
                              "traceback": str(e.__traceback__)}
    return None, None, {"error": f"unsupported file type: {suffix}", "file": str(p)}


def _load_mq5(p: Path, params_override: dict | None, info: dict) -> tuple:
    """Parse MQL5 — extract inputs + indicators, build strategy via Pine-like wrapper."""
    from frontend.file_detect import detect_from_extension
    det = detect_from_extension(p)
    info.update(det)
    info["type"] = "mq5"
    suggested = det.get("suggested_strategy", "fbb")
    if suggested not in STRATEGY_REGISTRY:
        suggested = "fbb"
    cls = STRATEGY_REGISTRY[suggested]
    # Use detected inputs as parameter overrides
    merged = _merge_with_defaults(suggested, det.get("inputs", {}))
    if params_override:
        merged.update(params_override)
    return cls, merged, info


def _load_py(p: Path, params_override: dict | None, info: dict) -> tuple:
    """Import Python module that subclasses BaseStrategy."""
    spec = importlib.util.spec_from_file_location("user_strategy", p)
    if not spec or not spec.loader:
        return None, None, {"error": "cannot import Python file", "file": str(p)}
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return None, None, {"error": f"import failed: {e}", "file": str(p)}
    # Find BaseStrategy subclass
    from strategies._base import BaseStrategy
    cls = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
            cls = obj
            info["class"] = name
            break
    if cls is None:
        return None, None, {"error": "no BaseStrategy subclass found in .py", "file": str(p)}
    info["type"] = "py"
    if params_override:
        return cls, params_override, info
    return cls, {}, info


def _load_pine(p: Path, params_override: dict | None, info: dict) -> tuple:
    """Parse PineScript v5, build a UniversalStrategy."""
    from core.pine_parser import parse_pine_to_strategy, parse_pine
    parsed = parse_pine(p)
    spec = parse_pine_to_strategy(parsed)
    if spec is None:
        return None, None, {"error": "no indicators found in PineScript", "file": str(p)}
    info["type"] = "pine"
    info["parsed_indicators"] = [i["type"] for i in spec["required_indicators"]]
    info["n_entry_conditions"] = len(spec["entry_conditions"])
    info["n_close_conditions"] = len(spec["close_conditions"])
    # Build UniversalStrategy
    cls = STRATEGY_REGISTRY["universal"]
    merged_params = dict(spec["inputs"])
    if params_override:
        merged_params.update(params_override)
    instance = cls(name=spec["name"], spec=spec, params=merged_params)
    # Wrap so it can be instantiated again with same spec
    class _WrappedUniversal:
        def __new__(cls2, params=None):
            return cls(name=spec["name"], spec=spec, params={**(merged_params), **(params or {})})
    return _WrappedUniversal, merged_params, info


def _load_text(p: Path, params_override: dict | None, info: dict) -> tuple:
    """Treat .txt/.md as PineScript or keyword config."""
    text = p.read_text(encoding="utf-8", errors="replace")
    if "strategy.entry" in text or "ta.rsi" in text or "ta.ema" in text:
        # Likely PineScript saved as .txt
        from core.pine_parser import parse_pine_text, parse_pine_to_strategy
        parsed = parse_pine_text(text)
        spec = parse_pine_to_strategy(parsed)
        if spec:
            info["type"] = "pine_via_txt"
            cls = STRATEGY_REGISTRY["universal"]
            merged = dict(spec["inputs"])
            if params_override:
                merged.update(params_override)
            class _Wrapped:
                def __new__(cls2, params=None):
                    return cls(name=spec["name"], spec=spec, params={**merged, **(params or {})})
            return _Wrapped, merged, info
    # Keyword-based detection
    from frontend.file_detect import detect_from_text
    det = detect_from_text(p)
    info.update(det)
    suggested = det.get("suggested_strategy", "fbb")
    if suggested not in STRATEGY_REGISTRY:
        suggested = "fbb"
    info["type"] = "txt_keyword"
    cls = STRATEGY_REGISTRY[suggested]
    merged = _merge_with_defaults(suggested, det.get("inputs", {}))
    if params_override:
        merged.update(params_override)
    return cls, merged, info


def _merge_with_defaults(strategy_name: str, detected_inputs: dict) -> dict:
    """Merge detected inputs with strategy defaults, keeping detected values where names match."""
    defaults_map = {
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
    defaults = defaults_map.get(strategy_name, {})
    merged = dict(defaults)
    # Normalize detected keys: drop suffixes like "_0", "_1"
    for k, v in detected_inputs.items():
        clean = re.sub(r'_\d+$', '', k)
        # Find a default key that matches
        for dk in defaults:
            if clean.lower() == dk.lower() or clean.lower() in dk.lower():
                merged[dk] = v
                break
    return merged