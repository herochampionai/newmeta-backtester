"""Reproducibility helpers — one seed, one data fingerprint per run.

Before: seeds scattered per module (tuner 7, WF 42, Sobol 42, PBO 7,
bootstrap 7) and reports carried no record of which data produced them.
Now: NEWMETA_SEED (or --seed) flows everywhere, and every report embeds
a data fingerprint so two runs can be proven to share inputs.
"""
from __future__ import annotations
import os
import random

DEFAULT_SEED = 7


def resolve_seed(explicit: int | None = None) -> int:
    """CLI --seed wins, else NEWMETA_SEED env, else DEFAULT_SEED."""
    if explicit is not None:
        return int(explicit)
    return int(os.environ.get("NEWMETA_SEED", DEFAULT_SEED))


def set_global_seed(seed: int) -> int:
    """Seed stdlib + numpy global RNGs. Module RNGs take seed explicitly."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    return int(seed)


def data_fingerprint(df, symbol: str, timeframe: str, source: str) -> dict:
    """Lightweight content fingerprint: any bar added/removed/changed
    flips it. Not a file hash (caches rewrite); a content hash."""
    import hashlib
    import pandas as pd
    close = pd.Series(df["close"]).astype(float)
    h = hashlib.sha256()
    h.update(f"{symbol}|{timeframe}|{source}|{len(df)}|".encode())
    h.update(f"{df.index[0]}|{df.index[-1]}|".encode())
    h.update(f"{close.sum():.6f}|{close.iloc[0]:.6f}|{close.iloc[-1]:.6f}".encode())
    return {
        "sha16": h.hexdigest()[:16],
        "symbol": symbol, "timeframe": timeframe, "source_requested": source,
        "rows": int(len(df)),
        "first": str(df.index[0]), "last": str(df.index[-1]),
    }
