"""ADX ENHANCED strategy. Mirror of MQL5 ADX_GetSignals.

Key facts:
  1. Open cases 1-4: ADX > lev1, ADX rising/falling, +/- DI dominance, DI > lev2.
  2. Close cases 1-4: different logic — measures (ADX - DI) or (DI - ADX) gap.
  3. DI Crossover: detected in lookback window of CrossoverLookback bars; if a
     cross happened with gap ≥ MinCrossoverGap, Bull/Bear flag is set.
     - For OPENS: if Buy flagged but no bullish DI cross → block buy.
     - For CLOSES: any bearish cross closes buys; bullish cross closes sells.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


class ADX_Strategy(BaseStrategy):
    name = "adx"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        period = int(p.get("bars_calculate", 20))
        adx_v, pdi, mdi = ind.adx(df["high"], df["low"], df["close"], period)
        adx_prev = adx_v.shift(1)
        pdi_prev = pdi.shift(1)
        mdi_prev = mdi.shift(1)

        use_di_cross = bool(p.get("use_di_crossover", True))
        lookback = int(p.get("crossover_lookback", 3))
        min_gap = float(p.get("min_crossover_gap", 5))

        # DI crossover detection over lookback window
        bull_cross = pd.Series(False, index=df.index)
        bear_cross = pd.Series(False, index=df.index)
        if use_di_cross:
            for k in range(lookback):
                pdi_k = pdi.shift(-k) if k > 0 else pdi
                mdi_k = mdi.shift(-k) if k > 0 else mdi
                pdi_kp1 = pdi.shift(-(k + 1))
                mdi_kp1 = mdi.shift(-(k + 1))
                bull_cross |= (pdi_kp1 < mdi_kp1) & (pdi_k > mdi_k) & ((pdi_k - mdi_k) >= min_gap)
                bear_cross |= (pdi_kp1 > mdi_kp1) & (pdi_k < mdi_k) & ((mdi_k - pdi_k) >= min_gap)

        lev1 = float(p.get("level_open_orders_1", 55))
        lev2 = float(p.get("level_open_orders_2", 15))

        sig = _empty_signals(df.index)
        ot = int(p.get("open_orders_type", 1))
        adx_above = adx_v > lev1

        # Open cases 1-4
        if ot > 0 and adx_above.any():
            rising = adx_v > adx_prev
            falling = adx_v < adx_prev
            buy = pd.Series(False, index=df.index)
            sell = pd.Series(False, index=df.index)
            if ot == 1:
                # ADX rising + Plus_DI > Minus_DI + Plus_DI > lev2 → buy
                buy = rising & (pdi > mdi) & (pdi > lev2)
                sell = rising & (mdi > pdi) & (mdi > lev2)
            elif ot == 2:
                # ADX rising + inverse DI (contrarian)
                buy = rising & (mdi > pdi) & (mdi > lev2)
                sell = rising & (pdi > mdi) & (pdi > lev2)
            elif ot == 3:
                # ADX falling + DI dominance (trend losing steam, ride last leg)
                buy = falling & (pdi > mdi) & (pdi > lev2)
                sell = falling & (mdi > pdi) & (mdi > lev2)
            elif ot == 4:
                # ADX falling + inverse DI (contrarian fade)
                buy = falling & (mdi > pdi) & (mdi > lev2)
                sell = falling & (pdi > mdi) & (pdi > lev2)
            # Gate by ADX > lev1
            buy = buy & adx_above
            sell = sell & adx_above
            # Apply DI crossover enhancement
            if use_di_cross:
                buy = buy & bull_cross
                sell = sell & bear_cross
            sig.entries = buy | sell
            sig.direction = np.where(buy, 1, np.where(sell, -1, 0))

        # Close cases 1-4 (only fires when in position)
        ct = int(p.get("close_orders_type", 4))
        if ct > 0:
            lev_c1 = float(p.get("level_close_orders_1", 15))
            lev_c2 = float(p.get("level_close_orders_2", 5))
            adx_above_c = adx_v > lev_c1
            cb = pd.Series(False, index=df.index)
            cs = pd.Series(False, index=df.index)
            if ct == 1:
                # If Plus_DI < Minus_DI and (ADX - Minus_DI) > lev_c2 → close buy
                # If Plus_DI > Minus_DI and (ADX - Plus_DI) > lev_c2 → close sell
                cb = (pdi < mdi) & ((adx_v - mdi) > lev_c2)
                cs = (pdi > mdi) & ((adx_v - pdi) > lev_c2)
            elif ct == 2:
                # Mirror of case 1 (cross signal direction)
                cb = (pdi > mdi) & ((adx_v - pdi) > lev_c2)
                cs = (pdi < mdi) & ((adx_v - mdi) > lev_c2)
            elif ct == 3:
                # (DI - ADX) gap: Plus_DI - ADX > lev_c2 → close buy (Plus_DI weakening)
                cb = (pdi > mdi) & ((pdi - adx_v) > lev_c2)
                cs = (mdi > pdi) & ((mdi - adx_v) > lev_c2)
            elif ct == 4:
                # Inverse of case 3
                cb = (mdi > pdi) & ((mdi - adx_v) > lev_c2)
                cs = (pdi > mdi) & ((pdi - adx_v) > lev_c2)
            cb = cb & adx_above_c
            cs = cs & adx_above_c
            if use_di_cross:
                # Bearish cross closes buys; bullish cross closes sells
                cb = cb | bear_cross
                cs = cs | bull_cross
            sig.exits = cb | cs

        return sig


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))