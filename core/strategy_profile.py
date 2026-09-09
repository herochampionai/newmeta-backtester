"""Strategy profile analyzer — reads any strategy file and shows what it has.
Used by the UI to display "this strategy uses grid / recovery / MTF / swaps"
before running the backtest, so the user knows what layers will run."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any


def analyze_strategy_file(path: str | Path, parsed_info: dict | None = None) -> dict:
    """Analyze a strategy file and return its feature profile.

    Returns dict with:
      - has_grid: bool (file mentions grid / martingale / recovery)
      - has_recovery: bool
      - has_adaptive_sizing: bool
      - has_mtf: bool (multi-timeframe)
      - has_divergence: bool
      - has_quad_stoch: bool (4 stochastic with different params)
      - has_swaps: bool (mentions swap)
      - has_session_filter: bool (London/NY/Asian)
      - has_news_filter: bool
      - has_sl_tp: bool (stop loss + take profit)
      - has_break_even: bool
      - has_trailing_stop: bool
      - risk_features: list of detected risk features
      - strategy_name: best-guess name
    """
    p = Path(path)
    text = ""
    for enc in ("utf-16", "utf-8", "latin-1"):
        try:
            text = p.read_text(encoding=enc, errors="replace")
            break
        except Exception:
            continue
    if not text:
        return {"error": f"cannot read {path}"}

    profile = {
        "path": str(p),
        "name": p.stem,
        "size_kb": round(p.stat().st_size / 1024, 1),
        "has_grid": False,
        "has_recovery": False,
        "has_adaptive_sizing": False,
        "has_mtf": False,
        "has_divergence": False,
        "has_quad_stoch": False,
        "has_swaps": False,
        "has_session_filter": False,
        "has_news_filter": False,
        "has_sl_tp": False,
        "has_break_even": False,
        "has_trailing_stop": False,
        "has_confluence": False,
        "risk_features": [],
        "indicators_found": [],
    }

    text_low = text.lower()

    # Grid detection
    if re.search(r"make_?grid|gridmode|grid_mode|pipsbetweenorders|multirecovery", text_low):
        profile["has_grid"] = True
        profile["risk_features"].append("Grid (averaging)")
    if re.search(r"recovery|multirecoverylot|martingale", text_low):
        profile["has_recovery"] = True
        profile["risk_features"].append("Recovery (martingale)")
    if re.search(r"break_?even|breakeven", text_low):
        profile["has_break_even"] = True
    if re.search(r"trailing", text_low):
        profile["has_trailing_stop"] = True
    if re.search(r"stop_?loss.*take_?profit|takeprofit.*stoploss|tp.*sl", text_low):
        profile["has_sl_tp"] = True
    if re.search(r"adaptive|lot_?size|sizer", text_low):
        profile["has_adaptive_sizing"] = True
    if re.search(r"multi.*timeframe|mtf_|resample|htf", text_low):
        profile["has_mtf"] = True
    if re.search(r"diverg", text_low):
        profile["has_divergence"] = True
    if re.search(r"swap.*wed|wed.*swap|3.?x.*swap|overnight.*swap", text_low):
        profile["has_swaps"] = True
    if re.search(r"session|london|ny|asian", text_low):
        profile["has_session_filter"] = True
    if re.search(r"news|economic.*calendar|high.*impact", text_low):
        profile["has_news_filter"] = True
    if re.search(r"confluence", text_low):
        profile["has_confluence"] = True

    # Quad stochastic (4 stochastic with different params)
    stoch_calls = re.findall(r"iStochastic\s*\([^)]+\)", text)
    unique_params = set()
    for s in stoch_calls:
        m = re.search(r"\d+\s*,\s*\d+\s*,\s*\d+", s)
        if m:
            unique_params.add(m.group())
    if len(unique_params) >= 3:
        profile["has_quad_stoch"] = True

    # Detect indicators
    indicators = ["iAC", "iAO", "iADX", "iDeMarker", "iBands", "iForce",
                   "iMFI", "iMACD", "iStochastic", "iRSI", "iMA", "iATR"]
    for ind in indicators:
        if ind in text:
            profile["indicators_found"].append(ind)

    # Detect parameter count
    param_matches = re.findall(r"input\s+(?:int|double|bool|string)\s+(\w+)", text)
    profile["parameter_count"] = len(set(param_matches))

    # Estimate strictness based on level thresholds (higher = stricter)
    levels = re.findall(r"level(?:open|close)?(?:orders|_[12])?\s*=\s*(\d+)", text_low)
    if levels:
        avg_level = sum(int(l) for l in levels) / len(levels)
        if avg_level >= 70:
            profile["strictness_estimate"] = "strict"
        elif avg_level >= 50:
            profile["strictness_estimate"] = "balanced"
        else:
            profile["strictness_estimate"] = "permissive"

    return profile


def format_profile_for_display(profile: dict) -> str:
    """One-line human-readable summary."""
    if "error" in profile:
        return f"Error: {profile['error']}"
    parts = []
    if profile["has_grid"]:
        parts.append("Grid")
    if profile["has_recovery"]:
        parts.append("Recovery")
    if profile["has_adaptive_sizing"]:
        parts.append("Adaptive")
    if profile["has_mtf"]:
        parts.append("MTF")
    if profile["has_quad_stoch"]:
        parts.append("QuadStoch")
    if profile["has_swaps"]:
        parts.append("Swap")
    if profile["has_confluence"]:
        parts.append("Confluence")
    features = " + ".join(parts) if parts else "Pure signals"
    inds = ", ".join(profile["indicators_found"]) or "none"
    n_params = profile.get("parameter_count", 0)
    strictness = profile.get("strictness_estimate", "?")
    return (f"{features} | {len(inds.split(','))} indicators | {n_params} params | "
            f"strictness: {strictness}")