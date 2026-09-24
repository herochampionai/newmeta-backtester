from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.data_quality import assess_data_quality
from analysis.mql5_set_export import build_mql5_set_text
from analysis.research_packet import save_research_packet


def test_data_quality_passes_clean_ohlc():
    df = pd.DataFrame(
        {
            "open": [1.0] * 1000,
            "high": [1.2] * 1000,
            "low": [0.9] * 1000,
            "close": [1.1] * 1000,
        },
        index=pd.date_range("2024-01-01", periods=1000, freq="h"),
    )
    verdict = assess_data_quality(df, timeframe="H1", source="test")
    assert verdict["status"] == "PASS"
    assert verdict["checks"]["missing_columns"] == []


def test_data_quality_fails_bad_ohlc():
    df = pd.DataFrame(
        {"open": [1.0], "high": [0.9], "low": [1.2], "close": [1.1]},
        index=pd.date_range("2024-01-01", periods=1, freq="h"),
    )
    verdict = assess_data_quality(df, timeframe="H1", source="test")
    assert verdict["status"] == "FAIL"
    assert any("invalid OHLC" in reason for reason in verdict["reasons"])


def test_mql5_set_export_formats_values():
    text = build_mql5_set_text("demo", {"length": 14, "use_filter": True, "risk": 0.125})
    assert "Strategy=demo" in text
    assert "length=14" in text
    assert "use_filter=true" in text
    assert "risk=0.125" in text


def test_save_research_packet_writes_json(tmp_path: Path):
    path = save_research_packet(tmp_path, "demo", {"metrics": {"profit_factor": 1.5}})
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"]["profit_factor"] == 1.5

