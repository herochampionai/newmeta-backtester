"""Base strategy: takes OHLCV + params, returns entry/exit signal Series.
Signals:
  entries:  pd.Series[bool]  — True = open
  exits:    pd.Series[bool]  — True = close
  direction: pd.Series[int]  — +1 long, -1 short (NaN = flat)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import numpy as np


@dataclass
class Signals:
    entries: pd.Series
    exits: pd.Series
    direction: pd.Series  # +1 long, -1 short, 0 flat


@dataclass
class BaseStrategy:
    name: str = "base"
    params: dict[str, Any] = field(default_factory=dict)

    def generate(self, df: pd.DataFrame) -> Signals:
        raise NotImplementedError


def _empty_signals(idx: pd.DatetimeIndex) -> Signals:
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))