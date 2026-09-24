"""Research packet export helpers for Newmeta Research Lab."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def save_research_packet(
    output_dir: str | Path,
    strategy: str,
    payload: Mapping[str, Any],
) -> Path:
    """Persist a complete research packet JSON and return its path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_strategy = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in strategy)
    ts = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{safe_strategy}_{ts}_research_packet.json"
    path.write_text(json.dumps(_json_safe(dict(payload)), indent=2, default=str), encoding="utf-8")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value

