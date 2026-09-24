"""MT5 handoff helpers for optimized Newmeta strategy parameters."""
from __future__ import annotations

from typing import Any, Mapping


def build_mql5_set_text(
    strategy_name: str,
    params: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Build a simple MT5-compatible .set text payload.

    MT5 accepts Name=Value lines. Python strategy parameter names are kept
    unchanged so the export is transparent and easy to map into MQL5 inputs.
    """
    lines = [
        "; Newmeta Research Lab - Backtester",
        f"; Strategy={strategy_name}",
    ]
    for key, value in (metadata or {}).items():
        lines.append(f"; {key}={_format_value(value)}")
    lines.append("")
    for key in sorted(params):
        lines.append(f"{_sanitize_key(str(key))}={_format_value(params[key])}")
    lines.append("")
    return "\n".join(lines)


def _sanitize_key(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in key)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)
