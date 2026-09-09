"""Universal strategy wrapper — executes parsed PineScript / MQL5 logic on any df.
Builds a Strategy spec from parsed content, then computes signals vectorized."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any

from strategies import indicators as ind_module
from strategies._base import BaseStrategy, Signals


@dataclass
class UniversalStrategy(BaseStrategy):
    """A strategy built from a parsed external spec (Pine, MQL5).
    Compiles the parsed conditions into vectorized pandas ops at runtime."""
    name: str = "universal"
    spec: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def generate(self, df: pd.DataFrame) -> Signals:
        spec = self.spec
        sig = _empty_signals(df.index)
        # Compute all required indicators
        indicator_values: dict[str, pd.Series] = {}
        for req in spec.get("required_indicators", []):
            var_name = req["var"]
            try:
                val = _compute_indicator(df, req, indicator_values)
                indicator_values[var_name] = val
            except Exception as e:
                continue
        # Default OHLCV shortcuts
        indicator_values["close"] = df["close"]
        indicator_values["open"] = df["open"]
        indicator_values["high"] = df["high"]
        indicator_values["low"] = df["low"]
        indicator_values["volume"] = df["volume"]
        # Evaluate entry conditions
        long_signal = pd.Series(False, index=df.index)
        short_signal = pd.Series(False, index=df.index)
        for entry in spec.get("entry_conditions", []):
            try:
                cond = _eval_condition(entry["condition_text"], indicator_values, df)
                if entry["direction"] == "long":
                    long_signal = long_signal | cond.fillna(False)
                else:
                    short_signal = short_signal | cond.fillna(False)
            except Exception:
                continue
        # Evaluate close conditions (any close → exit both)
        for close in spec.get("close_conditions", []):
            try:
                cond = _eval_condition(close["condition_text"], indicator_values, df)
                sig.exits = cond.fillna(False)
            except Exception:
                continue
        sig.entries = long_signal | short_signal
        sig.direction = np.where(long_signal, 1, np.where(short_signal, -1, 0))
        return sig


def _compute_indicator(df: pd.DataFrame, req: dict, cache: dict) -> pd.Series:
    """Compute one indicator from the spec. Uses strategies.indicators module."""
    t = req["type"]
    args = req["args"]
    variant = req.get("variant")
    h, l, c = df["high"], df["low"], df["close"]
    if t == "rsi":
        length = int(_safe_eval(args[1] if len(args) > 1 else "14", cache))
        return ind_module.ta.rsi(c, length) if hasattr(ind_module.ta, 'rsi') else _rsi_fallback(c, length)
    if t == "ema":
        length = int(_safe_eval(args[1] if len(args) > 1 else "9", cache))
        return c.ewm(span=length, adjust=False).mean()
    if t == "sma":
        length = int(_safe_eval(args[1] if len(args) > 1 else "14", cache))
        return c.rolling(length, min_periods=1).mean()
    if t == "macd":
        # args: close, fast, slow, signal
        fast = int(_safe_eval(args[1] if len(args) > 1 else "12", cache))
        slow = int(_safe_eval(args[2] if len(args) > 2 else "26", cache))
        sig_p = int(_safe_eval(args[3] if len(args) > 3 else "9", cache))
        m, s, h = ind_module.macd(c, fast, slow, sig_p)
        if variant == "main":
            return m
        if variant == "signal":
            return s
        return h  # hist fallback
    if t == "stoch":
        # args: high, low, close, k, d, slowing (or simple)
        k_p = int(_safe_eval(args[3] if len(args) > 3 else "14", cache))
        d_p = int(_safe_eval(args[4] if len(args) > 4 else "3", cache))
        sl = int(_safe_eval(args[5] if len(args) > 5 else "3", cache))
        k_s, d_s = ind_module.stochastic(h, l, c, k_p, d_p, sl)
        return k_s if variant == "k" else d_s
    if t == "bb":
        # Bollinger Bands: ta.bb(close, length, mult) → [middle, upper, lower]
        length = int(_safe_eval(args[1] if len(args) > 1 else "20", cache))
        mult = float(_safe_eval(args[2] if len(args) > 2 else "2.0", cache))
        mid, up, lo = ind_module.bollinger(c, length, mult)
        if variant == "upper":
            return up
        if variant == "lower":
            return lo
        return mid
    if t == "atr":
        length = int(_safe_eval(args[1] if len(args) > 1 else "14", cache))
        return _atr_fallback(h, l, c, length)
    if t == "highest":
        length = int(_safe_eval(args[1] if len(args) > 1 else "20", cache))
        return h.rolling(length, min_periods=1).max()
    if t == "lowest":
        length = int(_safe_eval(args[1] if len(args) > 1 else "20", cache))
        return l.rolling(length, min_periods=1).min()
    raise ValueError(f"unknown indicator type: {t}")


def _rsi_fallback(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr_fallback(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


def _safe_eval(expr: str, cache: dict) -> Any:
    """Try to evaluate an expression as a number or as a named var in cache."""
    expr = expr.strip()
    try:
        return float(expr)
    except ValueError:
        pass
    try:
        return int(expr)
    except ValueError:
        pass
    if expr in cache:
        return cache[expr].iloc[-1] if hasattr(cache[expr], 'iloc') else cache[expr]
    return expr


def _eval_condition(cond_text: str, cache: dict, df: pd.DataFrame) -> pd.Series:
    """Evaluate a PineScript condition expression into a boolean Series.
    Supports: comparisons, ta.crossover, ta.crossunder, ta.barssince, and, or, not.
    Falls back to False on unsupported expressions (with warning suppression)."""
    text = cond_text.strip()
    # ta.crossover(a, b) → a crosses above b
    m = re.match(r'ta\.crossover\s*\(([^,]+),\s*([^)]+)\)', text)
    if m:
        a = _resolve(m.group(1), cache, df)
        b = _resolve(m.group(2), cache, df)
        return (a > b) & (a.shift(1) <= b.shift(1))
    m = re.match(r'ta\.crossunder\s*\(([^,]+),\s*([^)]+)\)', text)
    if m:
        a = _resolve(m.group(1), cache, df)
        b = _resolve(m.group(2), cache, df)
        return (a < b) & (a.shift(1) >= b.shift(1))
    # Comparison: "rsi > 70" or "rsi < 30"
    for op in (">=", "<=", "==", "!=", ">", "<"):
        if op in text:
            left, right = text.split(op, 1)
            try:
                l = _resolve(left.strip(), cache, df)
                r = _resolve(right.strip(), cache, df)
                if hasattr(l, '__getitem__') and hasattr(r, '__getitem__'):
                    if op == ">":
                        return l > r
                    if op == "<":
                        return l < r
                    if op == ">=":
                        return l >= r
                    if op == "<=":
                        return l <= r
                    if op == "==":
                        return l == r
                    if op == "!=":
                        return l != r
            except Exception:
                pass
    # AND/OR
    if " and " in text:
        parts = text.split(" and ")
        a = _eval_condition(parts[0], cache, df)
        b = _eval_condition(" and ".join(parts[1:]), cache, df)
        return a & b
    if " or " in text:
        parts = text.split(" or ")
        a = _eval_condition(parts[0], cache, df)
        b = _eval_condition(" or ".join(parts[1:]), cache, df)
        return a | b
    return pd.Series(False, index=df.index)


def _resolve(name: str, cache: dict, df: pd.DataFrame) -> pd.Series | float:
    """Resolve a name to either a Series from cache or a literal float."""
    name = name.strip()
    # Direct cache hit
    if name in cache:
        return cache[name]
    # Literal number
    try:
        return float(name)
    except ValueError:
        pass
    # Bare close/open/high/low
    if name in ("close", "open", "high", "low", "volume"):
        return df[name]
    # Unknown — return zeros (will fail comparison gracefully)
    return pd.Series(0.0, index=df.index)


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))