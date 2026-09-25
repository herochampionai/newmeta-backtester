"""CombinedStrategy — AND-gate of two strategies. Both must signal same direction to enter."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


class CombinedStrategy(BaseStrategy):
    """AND-gate: requires BOTH strategy1 AND strategy2 to signal same direction on the same bar.
    If strategy1 says BUY but strategy2 says nothing -> no entry.
    If strategy1 says BUY and strategy2 says BUY -> entry BUY.
    Conflicting directions (S1=BUY, S2=SELL) -> no entry (and close opposite)."""
    name = "combined"

    def generate(self, df):
        p = self.params
        s1 = p['_strategy1'](self._clone(p['_strategy1_params']))
        s2 = p['_strategy2'](self._clone(p['_strategy2_params']))
        sig1 = s1.generate(df)
        sig2 = s2.generate(df)

        d1 = sig1.direction
        d2 = sig2.direction
        if isinstance(d1, np.ndarray):
            d1 = pd.Series(d1, index=df.index)
        if isinstance(d2, np.ndarray):
            d2 = pd.Series(d2, index=df.index)
        dir1 = d1.values
        dir2 = d2.values
        # AND-gate: only fire when BOTH agree on direction
        direction = np.where((dir1 == dir2) & (dir1 != 0), dir1, 0)
        direction = pd.Series(direction, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig

    def _clone(self, d):
        """Clone a dict (params dict may contain lists/scalars)."""
        out = {}
        for k, v in d.items():
            try:
                out[k] = v.copy()
            except AttributeError:
                out[k] = v
        return out
