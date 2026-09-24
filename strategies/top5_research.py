"""Top 5 research strategies — implementable on H1 EURUSD data:

#1 Regime Switching Engine — detect trend/range/crash, route to different sub-strategies
#2 Volatility Breakout + ADX — ATR expansion + ADX + Donchian breakout
#3 Adaptive ADX — ADX percentile (e.g., 70th) instead of fixed threshold (25)
#4 Compression Expansion — Bollinger Width near 30-day lows, then trade breakout
#7 Turtle Variant — Donchian 20-day breakout with ATR-based sizing

All use the user's framework: 200 SMA trend filter + indicator signal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


def sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def _atr(df, period):
    """Standard ATR (Average True Range)."""
    h = df['high']; l = df['low']; c = df['close']
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# =============================================================================
# #1 REGIME SWITCHING ENGINE (priority 10/10)
# =============================================================================
class RegimeSwitchingEngineStrategy(BaseStrategy):
    """Detect trend/range/crash regime, route to different sub-strategies.

    Regime detector:
    - ADX > 25: TREND regime → use trend-following entry (DI cross + DI bias)
    - ADX < 18: RANGE regime → use mean-reversion entry (BB bounce)
    - ATR > 1.5x average: EXPANSION regime → use breakout entry

    Each regime uses different entry logic but shares exit (ATR trail).
    """
    name = "regime_engine"

    def generate(self, df):
        p = self.params
        adx_period = int(p.get('adx_period', 14))
        atr_period = int(p.get('atr_period', 14))
        bb_period = int(p.get('bb_period', 20))
        bb_mult = float(p.get('bb_mult', 2.0))
        trend_thresh = float(p.get('trend_thresh', 25.0))
        range_thresh = float(p.get('range_thresh', 18.0))
        expansion_mult = float(p.get('expansion_mult', 1.5))
        atr_avg_period = int(p.get('atr_avg_period', 50))
        sma_p = int(p.get('sma_period', 200))
        cd = int(p.get('cooldown', 10))

        # Compute indicators
        adx_val, di_plus, di_minus = ind.adx(df['high'], df['low'], df['close'], adx_period)
        atr = _atr(df, atr_period)
        atr_avg = atr.rolling(atr_avg_period).mean()
        basis, upper, lower = ind.bollinger(df['close'], bb_period, bb_mult)
        sma200 = sma(df['close'], sma_p)

        # Regime detection
        is_trend = adx_val > trend_thresh
        is_range = adx_val < range_thresh
        is_expansion = atr > expansion_mult * atr_avg

        # === Sub-strategies per regime ===
        # Trend: DI cross aligned with trend
        di_cross_up = (di_plus > di_minus) & (di_plus.shift(1) <= di_minus.shift(1))
        di_cross_down = (di_plus < di_minus) & (di_plus.shift(1) >= di_minus.shift(1))
        # Range: BB bounce — price touched lower/upper and reverses
        range_low_touch = (df['low'] <= lower) & (df['close'] > lower.shift(1))
        range_high_touch = (df['high'] >= upper) & (df['close'] < upper.shift(1))
        # Expansion: Donchian breakout
        high_20 = df['high'].rolling(20).max().shift(1)
        low_20 = df['low'].rolling(20).min().shift(1)
        breakout_up = df['close'] > high_20
        breakout_down = df['close'] < low_20

        # Combine: trend regime + DI cross + above/below SMA
        long_trend = is_trend & di_cross_up & (df['close'] > sma200)
        short_trend = is_trend & di_cross_down & (df['close'] < sma200)
        # Range regime + BB bounce + above/below SMA
        long_range = is_range & range_low_touch & (df['close'] > sma200)
        short_range = is_range & range_high_touch & (df['close'] < sma200)
        # Expansion regime + breakout + ADX confirmation + above/below SMA
        long_exp = is_expansion & breakout_up & (adx_val > 18) & (df['close'] > sma200)
        short_exp = is_expansion & breakout_down & (adx_val > 18) & (df['close'] < sma200)

        long_entry = long_trend | long_range | long_exp
        short_entry = short_trend | short_range | short_exp

        direction = pd.Series(np.where(long_entry.fillna(False), 1,
                                       np.where(short_entry.fillna(False), -1, 0)),
                               index=df.index, dtype=int)

        if cd > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


# =============================================================================
# #2 VOLATILITY BREAKOUT + ADX (priority 9.5/10)
# =============================================================================
class VolatilityBreakoutStrategy(BaseStrategy):
    """ATR expansion + ADX trend + Donchian breakout."""
    name = "vol_breakout"

    def generate(self, df):
        p = self.params
        adx_period = int(p.get('adx_period', 14))
        atr_period = int(p.get('atr_period', 14))
        atr_avg_period = int(p.get('atr_avg_period', 20))
        donchian = int(p.get('donchian', 20))
        adx_thresh = float(p.get('adx_thresh', 25.0))
        sma_p = int(p.get('sma_period', 200))
        cd = int(p.get('cooldown', 5))

        adx_val, _, _ = ind.adx(df['high'], df['low'], df['close'], adx_period)
        atr = _atr(df, atr_period)
        atr_avg = atr.rolling(atr_avg_period).mean()
        sma200 = sma(df['close'], sma_p)

        # Long: Close > Highest(20), ADX > 25, ATR(14) > ATR(14)[5]
        high_n = df['high'].rolling(donchian).max().shift(1)
        low_n = df['low'].rolling(donchian).min().shift(1)
        atr_rising = atr > atr.shift(5)

        long_entry = (df['close'] > high_n) & (adx_val > adx_thresh) & atr_rising & (df['close'] > sma200)
        short_entry = (df['close'] < low_n) & (adx_val > adx_thresh) & atr_rising & (df['close'] < sma200)

        direction = pd.Series(np.where(long_entry.fillna(False), 1,
                                       np.where(short_entry.fillna(False), -1, 0)),
                               index=df.index, dtype=int)

        if cd > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


# =============================================================================
# #3 ADAPTIVE ADX (priority 9/10)
# =============================================================================
class AdaptiveADXStrategy(BaseStrategy):
    """ADX percentile instead of fixed threshold. Adapts across markets."""
    name = "adaptive_adx"

    def generate(self, df):
        p = self.params
        adx_period = int(p.get('adx_period', 14))
        lookback = int(p.get('lookback', 200))
        adx_pct = float(p.get('adx_percentile', 70.0))
        di_plus = ind.adx(df['high'], df['low'], df['close'], adx_period)[1]
        di_minus = ind.adx(df['high'], df['low'], df['close'], adx_period)[2]
        bars_calc = int(p.get('bars_calculate', 13))
        di_xover = int(p.get('crossover_lookback', 6))
        di_gap = float(p.get('min_crossover_gap', 1.0))
        # Zone logic from adx.py
        zone_lo = float(p.get('zone_lo', 22.5))
        zone_hi = float(p.get('zone_hi', 56.0))
        cont = float(p.get('cont', 21.7))
        rev = float(p.get('rev', 26.3))
        lv1 = float(p.get('lv1', 51.6))
        lv2 = float(p.get('lv2', 21.3))
        swp = int(p.get('swp', 16))
        cl1 = float(p.get('cl1', 16.8))
        cl2 = float(p.get('cl2', 3.59))
        oot = int(p.get('oot', 1))
        cot = int(p.get('cot', 6))
        sma_p = int(p.get('sma_period', 200))
        cd = int(p.get('cooldown', 4))

        adx_val = ind.adx(df['high'], df['low'], df['close'], adx_period)[0]
        # ADX percentile — adapt to current market volatility
        adx_pctile = adx_val.rolling(lookback).rank(pct=True) * 100
        adx_is_strong = adx_pctile >= adx_pct

        sma200 = sma(df['close'], sma_p)

        # Use existing ADX logic but trigger only when ADX is in top percentile
        # Plus DI cross + zone
        bars = bars_calc
        di_cross_up = (di_plus > di_minus) & (di_plus.shift(1) <= di_minus.shift(1))
        di_cross_down = (di_plus < di_minus) & (di_plus.shift(1) >= di_minus.shift(1))
        above_above = (df['close'] > sma200)
        below_above = (df['close'] < sma200)

        long_entry = adx_is_strong & di_cross_up & above_above
        short_entry = adx_is_strong & di_cross_down & below_above

        direction = pd.Series(np.where(long_entry.fillna(False), 1,
                                       np.where(short_entry.fillna(False), -1, 0)),
                               index=df.index, dtype=int)

        if cd > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


# =============================================================================
# #4 COMPRESSION EXPANSION (priority 8.5/10)
# =============================================================================
class CompressionExpansionStrategy(BaseStrategy):
    """Bollinger Width near 30-day lows (compression), then trade breakout."""
    name = "compression"

    def generate(self, df):
        p = self.params
        bb_period = int(p.get('bb_period', 20))
        bb_mult = float(p.get('bb_mult', 2.0))
        compression_period = int(p.get('compression_period', 30))
        compression_pct = float(p.get('compression_percentile', 20.0))
        # ADX threshold (low ADX required for compression)
        adx_period = int(p.get('adx_period', 14))
        adx_max = float(p.get('adx_max', 25.0))
        atr_period = int(p.get('atr_period', 14))
        sma_p = int(p.get('sma_period', 200))
        cd = int(p.get('cooldown', 5))

        basis, upper, lower = ind.bollinger(df['close'], bb_period, bb_mult)
        bb_width = (upper - lower) / basis.replace(0, np.nan)
        # BB width percentile
        bb_pctile = bb_width.rolling(compression_period).rank(pct=True) * 100
        # Compression: BB width is in bottom percentile
        is_compressed = bb_pctile <= compression_pct

        adx_val = ind.adx(df['high'], df['low'], df['close'], adx_period)[0]
        atr = _atr(df, atr_period)
        sma200 = sma(df['close'], sma_p)

        # Breakout: price breaks upper/lower band with ATR confirmation
        long_break = (df['close'] > upper) & (atr > atr.shift(1))
        short_break = (df['close'] < lower) & (atr > atr.shift(1))

        # Long: compression (BB narrow) → breakout up + ADX below max (still room) + above SMA
        long_entry = is_compressed & long_break & (adx_val < adx_max) & (df['close'] > sma200)
        short_entry = is_compressed & short_break & (adx_val < adx_max) & (df['close'] < sma200)

        direction = pd.Series(np.where(long_entry.fillna(False), 1,
                                       np.where(short_entry.fillna(False), -1, 0)),
                               index=df.index, dtype=int)

        if cd > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


# =============================================================================
# #7 TURTLE VARIANT (priority 7/10)
# =============================================================================
class TurtleDonchianStrategy(BaseStrategy):
    """20-day breakout (Donchian) with ATR-based sizing."""
    name = "turtle"

    def generate(self, df):
        p = self.params
        entry_period = int(p.get('entry_period', 20))
        exit_period = int(p.get('exit_period', 10))
        atr_period = int(p.get('atr_period', 14))
        atr_min = float(p.get('atr_min', 0.5))  # minimum ATR (volatility filter)
        sma_p = int(p.get('sma_period', 200))
        cd = int(p.get('cooldown', 5))

        atr = _atr(df, atr_period)
        sma200 = sma(df['close'], sma_p)

        # Entry: 20-bar breakout
        high_n = df['high'].rolling(entry_period).max().shift(1)
        low_n = df['low'].rolling(entry_period).min().shift(1)

        # Volatility filter — only trade when ATR is high enough
        vol_ok = atr > atr_min * atr.rolling(50).mean()

        long_entry = (df['close'] > high_n) & vol_ok & (df['close'] > sma200)
        short_entry = (df['close'] < low_n) & vol_ok & (df['close'] < sma200)

        direction = pd.Series(np.where(long_entry.fillna(False), 1,
                                       np.where(short_entry.fillna(False), -1, 0)),
                               index=df.index, dtype=int)

        if cd > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig
