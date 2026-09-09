"""PineScript v5 parser — extract inputs, indicators, and signal conditions.
Outputs a dict that the UniversalStrategy wrapper can execute on any OHLCV df.

Supports common PineScript v5 constructs:
  - strategy() declaration
  - input() declarations (int/float/bool/string)
  - indicator calls: ta.rsi, ta.ema, ta.sma, ta.macd, ta.stoch, ta.bb, ta.atr,
    ta.crossover, ta.crossunder, ta.highest, ta.lowest, ta.barssince, ta.sma, ta.ema
  - strategy.entry / strategy.close calls
  - Basic if/else conditionals
  - Variable assignments

NOT supported (yet):
  - User-defined functions (we extract their bodies but don't execute complex ones)
  - Arrays, maps, matrices
  - methods, types
  - varip, dynamic requests
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any


def parse_pine(path: str | Path) -> dict:
    """Parse a PineScript v5 file. Returns dict with:
      - version: '5' or '4' or None
      - title: strategy title
      - overlay: bool
      - inputs: {name: default_value}
      - indicators: list of {type, args, assigned_to}
      - entry_conditions: list of {direction: 'long'|'short', condition: parsed_expr}
      - close_conditions: list of {condition: parsed_expr}
      - raw_text: original source for debugging
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_pine_text(text)


def parse_pine_text(text: str) -> dict:
    """Parse PineScript source directly."""
    result = {
        "version": None,
        "title": None,
        "overlay": False,
        "inputs": {},
        "indicators": [],
        "entry_conditions": [],
        "close_conditions": [],
        "raw_text": text,
    }
    # Version
    m = re.search(r"//\s*@version\s*=\s*(\d+)", text)
    if m:
        result["version"] = m.group(1)
    # Strategy declaration
    m = re.search(r'strategy\s*\(\s*"([^"]+)"', text)
    if m:
        result["title"] = m.group(1)
    result["overlay"] = "overlay=true" in text
    # Inputs
    # input.int(name, default, ...)
    # input.float(name, default, ...)
    # input.bool(name, default)
    for m in re.finditer(r'input\.(int|float|bool|string)\s*\(\s*([^,]+),\s*"([^"]+)"(?:\s*,\s*([^)]+))?\s*\)',
                           text):
        _, name, title, default_part = m.group(1), m.group(2).strip(), m.group(3), m.group(4)
        # Default value
        default = None
        if default_part:
            # Try to extract a literal value
            dp = default_part.strip()
            for cast in ("int", "float"):
                if dp.startswith(cast + "("):
                    dp = dp[6:-1]
                    break
            try:
                if "." in dp:
                    default = float(dp)
                else:
                    default = int(dp)
            except ValueError:
                if dp.lower() == "true":
                    default = True
                elif dp.lower() == "false":
                    default = False
                elif dp.startswith('"') and dp.endswith('"'):
                    default = dp[1:-1]
                else:
                    default = dp
        else:
            # No default — try the first param which is sometimes the default in v5
            if name[0].isupper() or "_" in name:
                try:
                    default = int(name) if "." not in name else float(name)
                except ValueError:
                    default = name
        result["inputs"][title] = default
    # Indicators
    # ta.rsi(close, length) → assigned to var
    for m in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*ta\.(\w+)\s*\(([^)]+)\)', text):
        var_name = m.group(1)
        func = m.group(2)
        args = m.group(3)
        result["indicators"].append({
            "type": func,
            "args": [a.strip() for a in args.split(",")],
            "assigned_to": var_name,
        })
    # MACD is special: [macd_line, signal_line, hist] = ta.macd(...)
    for m in re.finditer(r'\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*=\s*ta\.macd\s*\(([^)]+)\)',
                           text):
        macd_var = m.group(1)
        signal_var = m.group(2)
        hist_var = m.group(3)
        args = [a.strip() for a in m.group(4).split(",")]
        result["indicators"].extend([
            {"type": "macd", "variant": "main", "args": args, "assigned_to": macd_var},
            {"type": "macd", "variant": "signal", "args": args, "assigned_to": signal_var},
            {"type": "macd", "variant": "hist", "args": args, "assigned_to": hist_var},
        ])
    # Stochastic is also multi-output: [k, d] = ta.stoch(...)
    for m in re.finditer(r'\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*=\s*ta\.stoch\s*\(([^)]+)\)',
                           text):
        k_var = m.group(1)
        d_var = m.group(2)
        args = [a.strip() for a in m.group(3).split(",")]
        result["indicators"].extend([
            {"type": "stoch", "variant": "k", "args": args, "assigned_to": k_var},
            {"type": "stoch", "variant": "d", "args": args, "assigned_to": d_var},
        ])
    # Strategy entries: strategy.entry(id, direction) preceded by condition
    # Pattern: if <cond>\n    strategy.entry("X", strategy.long)
    for m in re.finditer(r'(?:if\s+)?(\([^)]*\)|[a-zA-Z_][a-zA-Z0-9_.()\s]*?)\s*(?:\n|\s)+strategy\.entry\s*\(\s*"[^"]+"\s*,\s*strategy\.(long|short)\)',
                           text, re.MULTILINE):
        cond_text = m.group(1).strip()
        direction = m.group(2)
        result["entry_conditions"].append({
            "direction": "long" if direction == "long" else "short",
            "condition_text": cond_text,
        })
    # Strategy closes: strategy.close(id) or strategy.close_all()
    for m in re.finditer(r'(?:if\s+)?(\([^)]*\)|[a-zA-Z_][a-zA-Z0-9_.()\s]*?)\s*(?:\n|\s)+strategy\.(close|close_all)\s*\(',
                           text, re.MULTILINE):
        cond_text = m.group(1).strip()
        result["close_conditions"].append({
            "condition_text": cond_text,
        })
    return result


def parse_pine_to_strategy(parsed: dict):
    """Convert parsed PineScript dict into a UniversalStrategy-compatible spec.
    Returns dict with: name, required_indicators, signal_logic, params.
    """
    if not parsed.get("indicators"):
        return None
    # Map PineScript indicators to our indicator functions
    spec = {
        "name": parsed.get("title") or "pine_strategy",
        "version": parsed.get("version"),
        "inputs": parsed["inputs"],
        "required_indicators": [],
        "entry_conditions": parsed["entry_conditions"],
        "close_conditions": parsed["close_conditions"],
    }
    for ind in parsed["indicators"]:
        spec["required_indicators"].append({
            "type": ind["type"],
            "variant": ind.get("variant"),
            "args": ind["args"],
            "var": ind["assigned_to"],
        })
    return spec