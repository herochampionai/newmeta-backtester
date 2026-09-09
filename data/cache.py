"""Parquet cache + symbol lookup. Cache files are immutable (named by sha).
`load()` returns the most recent file matching symbol+timeframe."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"


def list_cache(symbol: str | None = None, timeframe: str | None = None) -> list[Path]:
    out: list[Path] = []
    for p in CACHE_DIR.glob("*.parquet"):
        if symbol and timeframe:
                if p.stem.startswith(f"{symbol}_{timeframe}_"):
                    out.append(p)
        else:
            out.append(p)
    return sorted(out)


def load(symbol: str, timeframe: str) -> tuple[pd.DataFrame, dict]:
    matches = list_cache(symbol, timeframe)
    if not matches:
        raise FileNotFoundError(f"no cache for {symbol} {timeframe}; run data.mt5_export first")
    p = matches[-1]  # newest by name (sha sorts)
    df = pd.read_parquet(p)
    meta = json.loads((p.with_suffix(".meta.json")).read_text())
    return df, meta


def write(df: pd.DataFrame, meta: dict, path: Path | None = None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = path or CACHE_DIR / f"{meta['symbol']}_{meta['timeframe']}_{meta['sha']}.parquet"
    df.to_parquet(p)
    (p.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
    return p