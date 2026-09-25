"""Base strategy: takes OHLCV + params, returns entry/exit signal Series.
Signals:
  entries:  pd.Series[bool]  — True = open
  exits:    pd.Series[bool]  — True = close
  direction: pd.Series[int]  — +1 long, -1 short (NaN = flat)

Param aliasing: subclasses can declare `param_aliases = {'short_name': 'canonical_name'}`.
If a short_name is passed without canonical_name, generate() transparently renames it.
This prevents silent-failure bugs from naming drift between callers and strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Signals:
    entries: pd.Series
    exits: pd.Series
    direction: pd.Series  # +1 long, -1 short, 0 flat


@dataclass
class BaseStrategy:
    name: str = "base"
    params: dict[str, Any] = field(default_factory=dict)

    def _resolve_params(self) -> dict[str, Any]:
        """Return params with aliases applied (alias -> canonical if canonical missing).

        Subclasses override aliases via class attribute:
            class MyStrategy(BaseStrategy):
                _aliases = {'short_name': 'canonical_name'}
        """
        # Read from class (not instance) so subclass overrides survive dataclass field
        # resolution. Falls back to instance attr for advanced cases.
        aliases = {}
        for cls in type(self).__mro__:
            if '_aliases' in cls.__dict__:
                aliases = cls.__dict__['_aliases']
                break
        if not aliases:
            return self.params
        resolved = dict(self.params)
        for alias, canonical in aliases.items():
            if alias in resolved and canonical not in resolved:
                resolved[canonical] = resolved[alias]
        return resolved

    def generate(self, df: pd.DataFrame) -> Signals:
        raise NotImplementedError


def _empty_signals(idx: pd.DatetimeIndex) -> Signals:
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))
