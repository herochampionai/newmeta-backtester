"""Confluence scoring — combine multiple strategy signals into a single
confidence-weighted signal.

Each strategy's signal at each bar gets a score in [-1, +1]:
  -1.0 = strong short, +1.0 = strong long, 0.0 = no signal

Confluence score = mean of all active strategy scores (only those that fired).
If only 1 strategy fires: confidence = 1.0 but weight = 0.3 (low alone)
If 3+ strategies agree: confidence approaches 1.0

Output:
  - composite_signal: pd.Series of int (-1, 0, +1)
  - confidence: pd.Series of float [0, 1]
  - n_strategies_agreeing: pd.Series of int
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def confluence(directions: dict[str, pd.Series], weights: dict[str, float] | None = None,
               min_agreement: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Combine multiple strategy direction Series into composite signal.

    directions: {strategy_name: pd.Series of {-1, 0, +1}}
    weights: optional per-strategy weight (default: equal)
    min_agreement: minimum number of strategies that must agree for a trade

    Returns:
      composite: int {-1, 0, +1} — trade signal after confluence filter
      confidence: float [0, 1] — strength of agreement
      n_agreeing: int — number of strategies in same direction
    """
    if not directions:
        raise ValueError("directions dict is empty")
    # Convert Series/ndarray to Series for consistent indexing
    norm = {}
    for name, d in directions.items():
        if isinstance(d, np.ndarray):
            # Use the index of the first Series for back-fill, else integer index
            ref = next((v for v in directions.values() if isinstance(v, pd.Series)), None)
            idx = ref.index if ref is not None else pd.RangeIndex(len(d))
            norm[name] = pd.Series(d, index=idx)
        else:
            norm[name] = d
    idx = list(norm.values())[0].index
    n_strats = len(directions)
    weights = weights or {n: 1.0 for n in directions}
    df = pd.DataFrame(norm).reindex(idx)
    # Count agreement per direction
    n_long = (df > 0).sum(axis=1)
    n_short = (df < 0).sum(axis=1)
    n_total = n_long + n_short  # strategies that fired (non-zero)
    n_agreeing = pd.Series(np.where(n_long > n_short, n_long.values, n_short.values),
                            index=idx)
    # Composite direction: side with more votes, only if >= min_agreement
    composite = pd.Series(0, index=idx, dtype=int)
    long_signal = (n_long >= min_agreement) & (n_long > n_short)
    short_signal = (n_short >= min_agreement) & (n_short > n_long)
    composite = composite.mask(long_signal, 1).mask(short_signal, -1)
    # Confidence: agreement ratio × weighting
    weighted_sum = sum(df[n] * weights[n] for n in df.columns)
    confidence = weighted_sum.abs() / max(sum(weights.values()), 1e-9)
    return composite, confidence, n_agreeing