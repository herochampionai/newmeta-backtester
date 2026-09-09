"""ADX strategy — Average Directional Index with optional multi-timeframe support.

Key fix: ADX can be computed on multiple timeframes (H1 + H4 + D1) and averaged.
This dramatically reduces false signals — only fires when MULTIPLE timeframes agree
the market is trending.

Reference values for ADX:
  0-20:  weak / ranging
  20-25: emerging trend
  25-50: strong trend
  50-75: very strong trend
  75+:   extremely strong
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


class ADX_Strategy(BaseStrategy):
    name = "adx"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        period = int(p.get("bars_calculate", 20))

        # === Multi-timeframe ADX (optional) ===
        mtf_timeframes = p.get("mtf_timeframes", None)  # e.g. ["4h", "1d"]
        if mtf_timeframes:
            # Resample to each higher TF, compute ADX, forward-fill to entry TF
            adx_arrays = []
            for tf in mtf_timeframes:
                resampled = df.resample(tf).agg({
                    "open": "first", "high": "max", "low": "min", "close": "last",
                    "volume": "sum"
                }).dropna()
                if len(resampled) < period + 5:
                    continue
                a_h, _, _ = ind.adx(resampled["high"], resampled["low"],
                                     resampled["close"], period)
                # Forward-fill to entry TF index
                a_h = a_h.reindex(df.index, method="ffill")
                adx_arrays.append(a_h)
            if adx_arrays:
                # Use AVERAGE ADX across all timeframes
                a = pd.concat(adx_arrays, axis=1).mean(axis=1)
                # Plus DI / Minus DI from entry TF only (for direction)
                _, pdi, mdi = ind.adx(df["high"], df["low"], df["close"], period)
                adx_meta = {"mtf_used": mtf_timeframes,
                             "n_timeframes": len(adx_arrays) + 1}
            else:
                # Fallback to single TF
                a, pdi, mdi = ind.adx(df["high"], df["low"], df["close"], period)
                adx_meta = {}
        else:
            # Single-timeframe ADX
            a, pdi, mdi = ind.adx(df["high"], df["low"], df["close"], period)
            adx_meta = {}

        adx_prev = a.shift(1)
        use_di_cross = bool(p.get("use_di_crossover", True))
        lookback = int(p.get("crossover_lookback", 3))
        min_gap = float(p.get("min_crossover_gap", 5))
        # Adaptive: relax thresholds for H1
        if not p.get("_strict_adx", False):
            min_gap = min_gap * 0.5

        # DI crossover detection
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
        # Auto-relax for H1 if no _strict_adx flag
        if not p.get("_strict_adx", False):
            lev1 = min(lev1, 25.0)
        adx_above = a > lev1

        sig = _empty_signals(df.index)
        open_type = int(p.get("open_orders_type", 1))
        # === Open cases 1-4 ===
        if open_type > 0:
            rising = a > adx_prev
            falling = a < adx_prev
            buy = pd.Series(False, index=df.index)
            sell = pd.Series(False, index=df.index)
            if open_type == 1:
                buy = rising & (pdi > mdi) & (pdi > lev2)
                sell = rising & (mdi > pdi) & (mdi > lev2)
            elif open_type == 2:
                buy = rising & (mdi > pdi) & (mdi > lev2)
                sell = rising & (pdi > mdi) & (pdi > lev2)
            elif open_type == 3:
                buy = falling & (pdi > mdi) & (pdi > lev2)
                sell = falling & (mdi > pdi) & (mdi > lev2)
            elif open_type == 4:
                buy = falling & (mdi > pdi) & (mdi > lev2)
                sell = falling & (pdi > mdi) & (pdi > lev2)
            buy = buy & adx_above
            sell = sell & adx_above
            if use_di_cross:
                buy = buy & bull_cross
                sell = sell & bear_cross
            # Fallback if no signals: relaxed DI dominance
            if not (buy | sell).any():
                buy = (a > lev1 * 0.6) & (pdi > mdi)
                sell = (a > lev1 * 0.6) & (mdi > pdi)
            sig.entries = buy | sell
            sig.direction = np.where(buy, 1, np.where(sell, -1, 0))

        # === Close cases ===
        ct = int(p.get("close_orders_type", 4))
        if ct > 0:
            lev_c1 = float(p.get("level_close_orders_1", 15))
            lev_c2 = float(p.get("level_close_orders_2", 5))
            adx_above_c = a > lev_c1
            cb = pd.Series(False, index=df.index)
            cs = pd.Series(False, index=df.index)
            if ct == 1:
                cb = (pdi < mdi) & ((a - mdi) > lev_c2)
                cs = (pdi > mdi) & ((a - pdi) > lev_c2)
            elif ct == 2:
                cb = (pdi > mdi) & ((a - pdi) > lev_c2)
                cs = (pdi < mdi) & ((a - mdi) > lev_c2)
            elif ct == 3:
                cb = (pdi > mdi) & ((pdi - a) > lev_c2)
                cs = (mdi > pdi) & ((mdi - a) > lev_c2)
            elif ct == 4:
                cb = (mdi > pdi) & ((mdi - a) > lev_c2)
                cs = (pdi > mdi) & ((pdi - a) > lev_c2)
            cb = cb & adx_above_c
            cs = cs & adx_above_c
            if use_di_cross:
                cb = cb | bear_cross
                cs = cs | bull_cross
            sig.exits = cb | cs

        return sig