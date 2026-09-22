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
from dataclasses import field
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


def _reversal_context(df: pd.DataFrame, lookback: int) -> tuple[pd.Series, pd.Series]:
    body = (df["close"] - df["open"]).abs()
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    small_body = body <= candle_range * 0.35
    shooting_star = small_body & (upper_wick >= body * 2.0) & (lower_wick <= body * 1.2)
    hammer = small_body & (lower_wick >= body * 2.0) & (upper_wick <= body * 1.2)

    prev_bear = df["close"].shift(1) < df["open"].shift(1)
    prev_bull = df["close"].shift(1) > df["open"].shift(1)
    bull_engulf = prev_bear & (df["close"] > df["open"]) & (df["close"] >= df["open"].shift(1)) & (df["open"] <= df["close"].shift(1))
    bear_engulf = prev_bull & (df["close"] < df["open"]) & (df["open"] >= df["close"].shift(1)) & (df["close"] <= df["open"].shift(1))

    morning_star = (
        (df["close"].shift(2) < df["open"].shift(2))
        & ((df["close"].shift(1) - df["open"].shift(1)).abs() <= candle_range.shift(1) * 0.35)
        & (df["close"] > df["open"])
        & (df["close"] > (df["open"].shift(2) + df["close"].shift(2)) / 2)
    )
    evening_star = (
        (df["close"].shift(2) > df["open"].shift(2))
        & ((df["close"].shift(1) - df["open"].shift(1)).abs() <= candle_range.shift(1) * 0.35)
        & (df["close"] < df["open"])
        & (df["close"] < (df["open"].shift(2) + df["close"].shift(2)) / 2)
    )

    prior_low = df["low"].shift(1).rolling(lookback, min_periods=2).min()
    prior_high = df["high"].shift(1).rolling(lookback, min_periods=2).max()
    bullish_sweep = (df["low"] < prior_low) & (df["close"] > prior_low)
    bearish_sweep = (df["high"] > prior_high) & (df["close"] < prior_high)

    bull_reversal = hammer | bull_engulf | morning_star | bullish_sweep
    bear_reversal = shooting_star | bear_engulf | evening_star | bearish_sweep
    return bull_reversal.fillna(False), bear_reversal.fillna(False)


class ADX_Strategy(BaseStrategy):
    name = "adx"
    # Accept common naming conventions transparently.
    # If callers pass `adx_period` without `bars_calculate`, treat them as equal.
    # Same for `threshold` <-> `level_open_orders_2`.
    _aliases = {
        "adx_period": "bars_calculate",
        "threshold": "level_open_orders_2",
    }

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self._resolve_params()
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
                # Use average ADX across entry TF plus all requested higher TFs.
                a_base, pdi, mdi = ind.adx(df["high"], df["low"], df["close"], period)
                a = pd.concat([a_base] + adx_arrays, axis=1).mean(axis=1)
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
        use_zone_logic = bool(p.get("use_zone_logic", True))
        adx_low = float(p.get("adx_zone_low", 18))
        adx_high = float(p.get("adx_zone_high", 35))
        continuation_level = float(p.get("adx_continuation_level", 24))
        reversal_edge = float(p.get("adx_reversal_edge", 20))
        if use_zone_logic:
            adx_above = (a > adx_low) & (a < adx_high)
            continuation_zone = a >= continuation_level
            reversal_zone = (a >= adx_low) & (a <= reversal_edge)
        else:
            # Auto-relax for H1 if no _strict_adx flag
            if not p.get("_strict_adx", False):
                lev1 = min(lev1, 25.0)
            adx_above = a > lev1
            continuation_zone = adx_above
            reversal_zone = pd.Series(False, index=df.index)
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
            if use_zone_logic:
                sweep_lookback = int(p.get("sweep_lookback", 5))
                bull_reversal, bear_reversal = _reversal_context(df, sweep_lookback)
                bull_cont = continuation_zone & rising & (pdi > mdi) & (pdi > lev2)
                bear_cont = continuation_zone & rising & (mdi > pdi) & (mdi > lev2)
                bull_rev = reversal_zone & (pdi > mdi) & bull_reversal
                bear_rev = reversal_zone & (mdi > pdi) & bear_reversal
                buy = (buy & continuation_zone) | bull_cont | bull_rev
                sell = (sell & continuation_zone) | bear_cont | bear_rev
            buy = buy & adx_above
            sell = sell & adx_above
            if use_di_cross:
                if use_zone_logic:
                    buy = buy & (bull_cross | reversal_zone)
                    sell = sell & (bear_cross | reversal_zone)
                else:
                    buy = buy & bull_cross
                    sell = sell & bear_cross
            # Fallback if no signals: relaxed DI dominance
            if not (buy | sell).any():
                if use_zone_logic:
                    buy = adx_above & (pdi > mdi) & (continuation_zone | reversal_zone)
                    sell = adx_above & (mdi > pdi) & (continuation_zone | reversal_zone)
                else:
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





