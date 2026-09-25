"""LindaMACDStrategy — LENIENT version (drop STC + SMA filters, just MACD cross).

The strict version required MACD cross + STC > 0 + strongDiff + 200 SMA.
This caused 0 entries on EUR and losses on NAS. Try simpler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


class LindaMACDLenientStrategy(BaseStrategy):
    """Linda MACD Simplified — MACD cross + optional 200 SMA + optional histogram momentum."""
    name = "linda_macd_lenient"

    def generate(self, df):
        p = self.params
        fast = int(p.get("fast", 12))
        slow = int(p.get("slow", 26))
        sig_period = int(p.get("signal", 9))
        use_sma_filter = bool(p.get("use_sma_filter", False))
        sma_period = int(p.get("sma_period", 200))
        use_histogram_momentum = bool(p.get("use_histogram_momentum", False))
        cooldown = int(p.get("cooldown", 4))

        ml, sl, hist = ind.macd(df["close"], fast, slow, sig_period)
        hist_prev = hist.shift(1)

        # Cross conditions
        macd_xover = (ml > sl) & (ml.shift(1) <= sl.shift(1))
        macd_xunder = (ml < sl) & (ml.shift(1) >= sl.shift(1))

        # Histogram momentum (optional)
        if use_histogram_momentum:
            hist_rising = (hist > 0) & (hist > hist_prev)
            hist_falling = (hist < 0) & (hist < hist_prev)
        else:
            hist_rising = pd.Series(True, index=df.index)
            hist_falling = pd.Series(True, index=df.index)

        # 200 SMA filter (optional)
        if use_sma_filter:
            sma = df["close"].rolling(sma_period, min_periods=50).mean()
            above_sma = (df["close"] > sma).fillna(False)
            below_sma = (df["close"] < sma).fillna(False)
        else:
            above_sma = pd.Series(True, index=df.index)
            below_sma = pd.Series(True, index=df.index)

        buy = macd_xover.fillna(False) & hist_rising.fillna(False) & above_sma
        sell = macd_xunder.fillna(False) & hist_falling.fillna(False) & below_sma
        # Allow no SMA filter version
        buy_loose = macd_xover.fillna(False) & hist_rising.fillna(False)
        sell_loose = macd_xunder.fillna(False) & hist_falling.fillna(False)

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)),
                               index=df.index, dtype=int)
        if cooldown > 0 and len(direction) > cooldown:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction
        return sig
