"""Port of the SCreener + Setups Pine Script — 10 setups (5 long, 5 short).

From C:\\Users\\youha\\Desktop\\Tradingview Pine\\screeners\\SCreener + Setups.txt

Each setup has a 200 SMA trend filter:
  L0/S5: Stochastic oversold/overbought bounce + cross
  L1/S6: 9/18 SMA crossover
  L2/S7: MACD histogram zero-cross
  L3/S8: Bollinger band re-entry (close cross)
  L4/S9: SuperTrend direction change
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


def _atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


class ScreenerL0Strategy(BaseStrategy):
    """L0: Stochastic oversold bounce (K crosses D + K < 20) + above 200 SMA."""
    name = "sc_l0_stoch"

    def generate(self, df):
        p = self.params
        period = int(p.get("stoch_period", 14))
        smooth_k = int(p.get("smooth_k", 3))
        smooth_d = int(p.get("smooth_d", 3))
        over_sold = float(p.get("over_sold", 20))
        sma_period = int(p.get("sma_period", 200))
        cooldown = int(p.get("cooldown", 8))

        k, d = ind.stochastic(df["high"], df["low"], df["close"], period, smooth_k, smooth_d)
        co = (k > d) & (k.shift(1) <= d.shift(1))  # K crosses D up
        stoch_bull = co.fillna(False) & (k < over_sold).fillna(False)
        sma = df["close"].rolling(sma_period, min_periods=50).mean()
        above_sma = (df["close"] > sma).fillna(False)
        buy = stoch_bull & above_sma

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, 0), index=df.index, dtype=int)
        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


class ScreenerS5Strategy(BaseStrategy):
    """S5: Stochastic overbought fade (K crosses D + K > 80) + below 200 SMA."""
    name = "sc_s5_stoch"

    def generate(self, df):
        p = self.params
        period = int(p.get("stoch_period", 14))
        smooth_k = int(p.get("smooth_k", 3))
        smooth_d = int(p.get("smooth_d", 3))
        over_bought = float(p.get("over_bought", 80))
        sma_period = int(p.get("sma_period", 200))
        cooldown = int(p.get("cooldown", 8))

        k, d = ind.stochastic(df["high"], df["low"], df["close"], period, smooth_k, smooth_d)
        cu = (k < d) & (k.shift(1) >= d.shift(1))
        stoch_bear = cu.fillna(False) & (k > over_bought).fillna(False)
        sma = df["close"].rolling(sma_period, min_periods=50).mean()
        below_sma = (df["close"] < sma).fillna(False)
        sell = stoch_bear & below_sma

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(sell, -1, 0), index=df.index, dtype=int)
        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


class ScreenerL1Strategy(BaseStrategy):
    """L1: 9/18 SMA crossover + above 200 SMA."""
    name = "sc_l1_macross"

    def generate(self, df):
        p = self.params
        fast_period = int(p.get("fast", 9))
        slow_period = int(p.get("slow", 18))
        sma_period = int(p.get("sma_period", 200))
        cooldown = int(p.get("cooldown", 8))

        mafast = df["close"].rolling(fast_period, min_periods=2).mean()
        maslow = df["close"].rolling(slow_period, min_periods=2).mean()
        xover = (mafast > maslow) & (mafast.shift(1) <= maslow.shift(1))
        sma = df["close"].rolling(sma_period, min_periods=50).mean()
        above_sma = (df["close"] > sma).fillna(False)
        buy = xover.fillna(False) & above_sma

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, 0), index=df.index, dtype=int)
        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


class ScreenerS6Strategy(BaseStrategy):
    """S6: 9/18 SMA crossunder + below 200 SMA."""
    name = "sc_s6_macross"

    def generate(self, df):
        p = self.params
        fast_period = int(p.get("fast", 9))
        slow_period = int(p.get("slow", 18))
        sma_period = int(p.get("sma_period", 200))
        cooldown = int(p.get("cooldown", 8))

        mafast = df["close"].rolling(fast_period, min_periods=2).mean()
        maslow = df["close"].rolling(slow_period, min_periods=2).mean()
        xunder = (mafast < maslow) & (mafast.shift(1) >= maslow.shift(1))
        sma = df["close"].rolling(sma_period, min_periods=50).mean()
        below_sma = (df["close"] < sma).fillna(False)
        sell = xunder.fillna(False) & below_sma

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(sell, -1, 0), index=df.index, dtype=int)
        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


class ScreenerL2Strategy(BaseStrategy):
    """L2: MACD histogram zero-cross + above 200 SMA."""
    name = "sc_l2_macd"

    def generate(self, df):
        p = self.params
        fast = int(p.get("fast", 12))
        slow = int(p.get("slow", 26))
        sig_period = int(p.get("signal", 9))
        sma_period = int(p.get("sma_period", 200))
        cooldown = int(p.get("cooldown", 6))

        ml, sl, hist = ind.macd(df["close"], fast, slow, sig_period)
        xover_dt = (hist > 0) & (hist.shift(1) <= 0)
        sma = df["close"].rolling(sma_period, min_periods=50).mean()
        above_sma = (df["close"] > sma).fillna(False)
        buy = xover_dt.fillna(False) & above_sma

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, 0), index=df.index, dtype=int)
        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


class ScreenerS7Strategy(BaseStrategy):
    """S7: MACD histogram zero-crossunder + below 200 SMA."""
    name = "sc_s7_macd"

    def generate(self, df):
        p = self.params
        fast = int(p.get("fast", 12))
        slow = int(p.get("slow", 26))
        sig_period = int(p.get("signal", 9))
        sma_period = int(p.get("sma_period", 200))
        cooldown = int(p.get("cooldown", 6))

        ml, sl, hist = ind.macd(df["close"], fast, slow, sig_period)
        xunder_dt = (hist < 0) & (hist.shift(1) >= 0)
        sma = df["close"].rolling(sma_period, min_periods=50).mean()
        below_sma = (df["close"] < sma).fillna(False)
        sell = xunder_dt.fillna(False) & below_sma

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(sell, -1, 0), index=df.index, dtype=int)
        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


class ScreenerL3Strategy(BaseStrategy):
    """L3: Bollinger band re-entry (close crosses above lower band) + above 200 SMA."""
    name = "sc_l3_bb"

    def generate(self, df):
        p = self.params
        bb_period = int(p.get("bb_period", 20))
        bb_std = float(p.get("bb_std", 2.0))
        sma_period = int(p.get("sma_period", 200))
        cooldown = int(p.get("cooldown", 8))

        basis, upper, lower = ind.bollinger(df["close"], bb_period, bb_std)
        buy_xover = (df["close"] > lower) & (df["close"].shift(1) <= lower.shift(1))
        sma = df["close"].rolling(sma_period, min_periods=50).mean()
        above_sma = (df["close"] > sma).fillna(False)
        buy = buy_xover.fillna(False) & above_sma

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, 0), index=df.index, dtype=int)
        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


class ScreenerS8Strategy(BaseStrategy):
    """S8: Bollinger band breakdown re-entry + below 200 SMA."""
    name = "sc_s8_bb"

    def generate(self, df):
        p = self.params
        bb_period = int(p.get("bb_period", 20))
        bb_std = float(p.get("bb_std", 2.0))
        sma_period = int(p.get("sma_period", 200))
        cooldown = int(p.get("cooldown", 8))

        basis, upper, lower = ind.bollinger(df["close"], bb_period, bb_std)
        sell_xover = (df["close"] < upper) & (df["close"].shift(1) >= upper.shift(1))
        sma = df["close"].rolling(sma_period, min_periods=50).mean()
        below_sma = (df["close"] < sma).fillna(False)
        sell = sell_xover.fillna(False) & below_sma

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(sell, -1, 0), index=df.index, dtype=int)
        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


class ScreenerL4Strategy(BaseStrategy):
    """L4: SuperTrend direction change to up + above 200 SMA."""
    name = "sc_l4_supertrend"

    def generate(self, df):
        p = self.params
        atr_period = int(p.get("atr_period", 10))
        factor = float(p.get("factor", 3.0))
        sma_period = int(p.get("sma_period", 200))
        cooldown = int(p.get("cooldown", 12))

        # Simple SuperTrend approximation
        atr_local = _atr(df, atr_period)
        hl2 = (df["high"] + df["low"]) / 2
        upper_band = hl2 + factor * atr_local
        lower_band = hl2 - factor * atr_local
        # SuperTrend direction: close > upper_band → up
        st_dir = pd.Series(0, index=df.index, dtype=int)
        st_dir[df["close"] > upper_band] = 1
        st_dir[df["close"] < lower_band] = -1
        st_dir = st_dir.replace(0, np.nan).ffill().fillna(0).astype(int)
        ch_dir = st_dir.diff()
        # Buy: direction changed to +1
        buy = (ch_dir > 0).fillna(False)
        sma = df["close"].rolling(sma_period, min_periods=50).mean()
        above_sma = (df["close"] > sma).fillna(False)
        buy = buy & above_sma

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, 0), index=df.index, dtype=int)
        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


class ScreenerS9Strategy(BaseStrategy):
    """S9: SuperTrend direction change to down + below 200 SMA."""
    name = "sc_s9_supertrend"

    def generate(self, df):
        p = self.params
        atr_period = int(p.get("atr_period", 10))
        factor = float(p.get("factor", 3.0))
        sma_period = int(p.get("sma_period", 200))
        cooldown = int(p.get("cooldown", 12))

        atr_local = _atr(df, atr_period)
        hl2 = (df["high"] + df["low"]) / 2
        upper_band = hl2 + factor * atr_local
        lower_band = hl2 - factor * atr_local
        st_dir = pd.Series(0, index=df.index, dtype=int)
        st_dir[df["close"] > upper_band] = 1
        st_dir[df["close"] < lower_band] = -1
        st_dir = st_dir.replace(0, np.nan).ffill().fillna(0).astype(int)
        ch_dir = st_dir.diff()
        sell = (ch_dir < 0).fillna(False)
        sma = df["close"].rolling(sma_period, min_periods=50).mean()
        below_sma = (df["close"] < sma).fillna(False)
        sell = sell & below_sma

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(sell, -1, 0), index=df.index, dtype=int)
        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


# ═══════════════════════════════════════════════════════════════════════
#  Linda MACD Enhanced — from Macd Youha.txt
# ═══════════════════════════════════════════════════════════════════════

class LindaMACDStrategy(BaseStrategy):
    """Linda MACD Enhanced — Meta MACD v2.6 (Linda Raschke-inspired).

    Entry logic (line 501-502 of Macd Youha.txt):
      bullishCondition = MACD crosses ABOVE signal + STC > 0 + strongDiff + strongMomentum
      bearishCondition = MACD crosses BELOW signal + STC < 0 + strongDiff + strongMomentum

    Parameters:
      fastLength = 3, slowLength = 10, signalSmoothing = 16
      STC: length=12, fast=26, slow=50
      diff_threshold = 0.2
    """
    name = "linda_macd"

    def _calc_stc(self, close, length, fast_len, slow_len):
        """Schaff Trend Cycle (STC) — from Macd Youha.txt lines 136-149."""
        factor = 0.5
        # Calc MACD
        ema_fast = close.ewm(span=fast_len, min_periods=5).mean()
        ema_slow = close.ewm(span=slow_len, min_periods=5).mean()
        macd_val = ema_fast - ema_slow
        # PercentK
        lowest_macd = macd_val.rolling(length, min_periods=5).min()
        highest_macd = macd_val.rolling(length, min_periods=5).max()
        rng = (highest_macd - lowest_macd).replace(0, 1e-9)
        percent_k = ((macd_val - lowest_macd) / rng) * 100
        # Smoothed
        smoothed = percent_k.copy()
        for i in range(1, len(smoothed)):
            if pd.notna(smoothed.iloc[i - 1]):
                smoothed.iloc[i] = smoothed.iloc[i - 1] + factor * (percent_k.iloc[i] - smoothed.iloc[i - 1])
        # PercentD
        lowest_k = smoothed.rolling(length, min_periods=5).min()
        highest_k = smoothed.rolling(length, min_periods=5).max()
        rng_k = (highest_k - lowest_k).replace(0, 1e-9)
        percent_d = ((smoothed - lowest_k) / rng_k) * 100
        # Smoothed percent K
        smoothed_pk = percent_d.copy()
        for i in range(1, len(smoothed_pk)):
            if pd.notna(smoothed_pk.iloc[i - 1]):
                smoothed_pk.iloc[i] = smoothed_pk.iloc[i - 1] + factor * (percent_d.iloc[i] - smoothed_pk.iloc[i - 1])
        return smoothed_pk.fillna(50.0)

    def generate(self, df):
        p = self.params
        fast = int(p.get("fast", 3))
        slow = int(p.get("slow", 10))
        sig_period = int(p.get("signal", 16))
        stc_length = int(p.get("stc_length", 12))
        stc_fast = int(p.get("stc_fast", 26))
        stc_slow = int(p.get("stc_slow", 50))
        diff_threshold = float(p.get("diff_threshold", 0.2))
        sma_period = int(p.get("sma_period", 200))
        cooldown = int(p.get("cooldown", 8))

        close = df["close"]

        # MACD (Linda's 3/10/16 - much faster than default 12/26/9)
        ema_fast = close.ewm(span=fast, min_periods=2).mean()
        ema_slow = close.ewm(span=slow, min_periods=2).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=sig_period, min_periods=2).mean()
        hist = macd_line - signal_line

        # STC (centered around 0 for visualization, but raw STC is 0-100)
        stc_raw = self._calc_stc(close, stc_length, stc_fast, stc_slow)
        stc_centered = (stc_raw - 50)  # -50 to +50

        # MACD-Signal diff threshold
        macd_diff = macd_line - signal_line

        # Strong diff conditions
        strong_diff_bull = macd_diff > diff_threshold
        strong_diff_bear = macd_diff < -diff_threshold

        # Momentum conditions: MACD line rising/falling
        macd_rising = macd_line > macd_line.shift(1)
        macd_falling = macd_line < macd_line.shift(1)

        # Strong momentum (faster window)
        fast_macd_diff = (ema_fast.diff(3) - signal_line.diff(3))
        strong_momentum_bull = fast_macd_diff > 0
        strong_momentum_bear = fast_macd_diff < 0

        # Cross conditions
        macd_xover = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        macd_xunder = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        # 200 SMA trend filter (Linda's classic approach)
        sma = close.rolling(sma_period, min_periods=50).mean()
        above_sma = (close > sma).fillna(False)
        below_sma = (close < sma).fillna(False)

        # Linda's combined entry
        buy = macd_xover.fillna(False) & (stc_centered > 0).fillna(False) & \
              strong_diff_bull.fillna(False) & strong_momentum_bull.fillna(False) & above_sma
        sell = macd_xunder.fillna(False) & (stc_centered <= 0).fillna(False) & \
               strong_diff_bear.fillna(False) & strong_momentum_bear.fillna(False) & below_sma
        # Lenient version without SMA filter
        buy_loose = macd_xover.fillna(False) & (stc_centered > 0).fillna(False) & \
                    strong_diff_bull.fillna(False) & strong_momentum_bull.fillna(False)
        sell_loose = macd_xunder.fillna(False) & (stc_centered <= 0).fillna(False) & \
                     strong_diff_bear.fillna(False) & strong_momentum_bear.fillna(False)

        sig = _empty_signals(df.index)
        # Use strict version if SMA filter helps, else loose
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)),
                               index=df.index, dtype=int)
        if cooldown > 0 and len(direction) > cooldown:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction
        return sig
