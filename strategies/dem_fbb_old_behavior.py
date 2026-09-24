"""Old-behavior DeM (mean-reversion) and FBB (Bollinger breakout) ports — exact replicas of mq5 defaults.

DeM:
  OpenOrdersType=3 (default)
  BUY:  DeMarker[0] < 25 AND DeMarker[1] > 25   (crossing OUT of oversold)
  SELL: DeMarker[0] > 75 AND DeMarker[1] < 75   (crossing OUT of overbought)

FBB:
  OpenOrdersType_1=1 (default)
  BUY:  PriceHigh[1] < LowerBand[1]   (pierces lower band, mean-reversion buy)
  SELL: PriceLow[1] > UpperBand[1]    (pierces upper band, mean-reversion sell)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


class DeMOldBehaviorStrategy(BaseStrategy):
    name = "dem_old_behavior"

    def generate(self, df):
        p = self.params
        bars = int(p.get("bars", 20))
        level = float(p.get("level", 75.0))  # LevelOpenOrders
        cooldown = int(p.get("cooldown", 0))
        dem = ind.dem(df["high"], df["low"], bars)
        # Normalize to 0-100 like MQL5: NormalizeDouble(DeMarker*100, 3)
        dem_pct = dem * 100.0
        level_lo = 100.0 - level
        level_hi = level
        # Type 3: BUY when current < level_lo AND prev > level_lo
        #         SELL when current > level_hi AND prev < level_hi
        buy = (dem_pct < level_lo) & (dem_pct.shift(1) > level_lo)
        sell = (dem_pct > level_hi) & (dem_pct.shift(1) < level_hi)
        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy.fillna(False), 1,
                                       np.where(sell.fillna(False), -1, 0)),
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


class FBBOldBehaviorStrategy(BaseStrategy):
    name = "fbb_old_behavior"

    def generate(self, df):
        p = self.params
        bars = int(p.get("bars", 20))
        deviation = float(p.get("deviation", 1.8))
        cooldown = int(p.get("cooldown", 0))
        upper, middle, lower = ind.bollinger(df["close"], bars, deviation)
        # MQL5 uses PriceHigh[1] and PriceLow[1] from previous (confirmed) bar
        high1 = df["high"].shift(1)
        low1 = df["low"].shift(1)
        # Type 1: BUY when high[1] < lower[1], SELL when low[1] > upper[1]
        buy = high1 < lower.shift(1)
        sell = low1 > upper.shift(1)
        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy.fillna(False), 1,
                                       np.where(sell.fillna(False), -1, 0)),
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
