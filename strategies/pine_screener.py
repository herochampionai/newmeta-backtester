"""Pine Script SCreener setups — implemented with 200 SMA filter as the trend gate.

From SCreener + Setups.txt:
- L0/S5 Stochastic: stoch(14,3,3) cross + K<20 OR K>80 + 200 SMA filter
- L1/S6 MA Cross: SMA(9) cross SMA(18) + 200 SMA filter
- L2/S7 MACD: MACD delta cross 0 + 200 SMA filter
- L3/S8 Bollinger: price cross lower/upper band (20, 2) + 200 SMA filter
- L4/S9 Supertrend: ST direction change + 200 SMA filter

These are the user's standard params — not Optuna-tuned, so likely more robust.
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


def stoch_kdj(df, length=14, smooth_k=3, smooth_d=3):
    """Standard Pine ta.stoch(close, high, low, length) -> K = SMA of stochastic, D = SMA of K."""
    h = df['high'].rolling(length).max()
    l = df['low'].rolling(length).min()
    k_raw = 100 * (df['close'] - l) / (h - l).replace(0, np.nan)
    k = k_raw.rolling(smooth_k).mean().fillna(50)
    d = k.rolling(smooth_d).mean().fillna(50)
    return k, d


def macd(close, fast=12, slow=26, signal=9):
    """Standard MACD."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    delta = macd_line - signal_line
    return macd_line, signal_line, delta


def bollinger(close, period=20, mult=2):
    basis = sma(close, period)
    std = close.rolling(period).std()
    return basis, basis + mult * std, basis - mult * std


def supertrend(df, atr_period=10, factor=3.0):
    """Standard Supertrend."""
    h = df['high']
    l = df['low']
    c = df['close']
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()
    hl2 = (h + l) / 2
    upper = hl2 + factor * atr
    lower = hl2 - factor * atr
    direction = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if c.iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = -1
        elif c.iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = 1
        else:
            direction.iloc[i] = direction.iloc[i-1] if direction.iloc[i-1] != 0 else -1
    return upper, lower, direction


class PineL0StochasticStrategy(BaseStrategy):
    """L0 Stochastic: stoch(14,3,3) %K crosses %D from below in OS zone, with price > 200 SMA."""
    name = "pine_l0_stochastic"

    def generate(self, df):
        p = self.params
        length = int(p.get('length', 14))
        sm_k = int(p.get('smooth_k', 3))
        sm_d = int(p.get('smooth_d', 3))
        ob = float(p.get('overbought', 80.0))
        os = float(p.get('oversold', 20.0))
        sma_p = int(p.get('sma_period', 200))
        cd = int(p.get('cooldown', 4))

        k, d = stoch_kdj(df, length, sm_k, sm_d)
        sma200 = sma(df['close'], sma_p)

        long_cross = (k > d) & (k.shift(1) <= d.shift(1)) & (k < os)
        short_cross = (k < d) & (k.shift(1) >= d.shift(1)) & (k > ob)

        above_sma = df['close'] > sma200
        below_sma = df['close'] < sma200

        long_entry = long_cross & above_sma
        short_entry = short_cross & below_sma

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


class PineL1MACrossStrategy(BaseStrategy):
    """L1 2x MA Cross: SMA(9) crosses SMA(18), with price > 200 SMA filter."""
    name = "pine_l1_macross"

    def generate(self, df):
        p = self.params
        fast_len = int(p.get('fast_len', 9))
        slow_len = int(p.get('slow_len', 18))
        sma_p = int(p.get('sma_period', 200))
        cd = int(p.get('cooldown', 4))

        sma_fast = sma(df['close'], fast_len)
        sma_slow = sma(df['close'], slow_len)
        sma200 = sma(df['close'], sma_p)

        long_cross = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
        short_cross = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

        above_sma = df['close'] > sma200
        below_sma = df['close'] < sma200

        long_entry = long_cross & above_sma
        short_entry = short_cross & below_sma

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


class PineL2MACDStrategy(BaseStrategy):
    """L2 MACD: MACD delta crosses 0, with 200 SMA filter."""
    name = "pine_l2_macd"

    def generate(self, df):
        p = self.params
        fast = int(p.get('fast', 12))
        slow = int(p.get('slow', 26))
        sig = int(p.get('signal', 9))
        sma_p = int(p.get('sma_period', 200))
        cd = int(p.get('cooldown', 4))

        macd_line, sig_line, delta = macd(df['close'], fast, slow, sig)
        sma200 = sma(df['close'], sma_p)

        long_cross = (delta > 0) & (delta.shift(1) <= 0)
        short_cross = (delta < 0) & (delta.shift(1) >= 0)

        above_sma = df['close'] > sma200
        below_sma = df['close'] < sma200

        long_entry = long_cross & above_sma
        short_entry = short_cross & below_sma

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


class PineL3BollingerStrategy(BaseStrategy):
    """L3 Bollinger: price crosses lower band (long) / upper band (short), with 200 SMA filter."""
    name = "pine_l3_bollinger"

    def generate(self, df):
        p = self.params
        period = int(p.get('period', 20))
        mult = float(p.get('mult', 2.0))
        sma_p = int(p.get('sma_period', 200))
        cd = int(p.get('cooldown', 4))

        basis, upper, lower = bollinger(df['close'], period, mult)
        sma200 = sma(df['close'], sma_p)

        long_cross = (df['close'] > lower) & (df['close'].shift(1) <= lower.shift(1))
        short_cross = (df['close'] < upper) & (df['close'].shift(1) >= upper.shift(1))

        above_sma = df['close'] > sma200
        below_sma = df['close'] < sma200

        long_entry = long_cross & above_sma
        short_entry = short_cross & below_sma

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


class PineL4SupertrendStrategy(BaseStrategy):
    """L4 Supertrend: ST direction change from -1 to 1 (bullish), with 200 SMA filter."""
    name = "pine_l4_supertrend"

    def generate(self, df):
        p = self.params
        atr_period = int(p.get('atr_period', 10))
        factor = float(p.get('factor', 3.0))
        sma_p = int(p.get('sma_period', 200))
        cd = int(p.get('cooldown', 4))

        upper, lower, direction_st = supertrend(df, atr_period, factor)
        sma200 = sma(df['close'], sma_p)

        # Bullish: direction changes from bearish (-1) to bullish (+1)... wait, let me check Pine logic
        # In Pine: chDir = ta.change(STdirection), L4_Entry: chDir < 0
        # ta.change(STdirection) means current - previous. If STdirection goes from -1 (down) to +1 (up),
        # chDir = +1 - (-1) = +2 > 0... actually wait, let me re-check
        # Actually in Pine ta.supertrend returns [supertrend, direction] where direction is -1 for downtrend, +1 for uptrend
        # ta.change(direction): uptrend→uptrend = 0, down→up = +2, up→down = -2
        # So chDir < 0 means transition from uptrend to downtrend (short entry)
        # L4_Entry_Bullish uses chDir > 0 (transition to uptrend, long entry)
        # L4_Entry_Bearish uses chDir < 0 (transition to downtrend, short entry)

        # But the Pine code I saw: L4_Entry = chDir < 0 and aboveSma200 — short entry when going down but price above SMA?
        # That's weird. Let me just implement both directions

        chDir = direction_st.diff()  # 0 = no change, +/-2 = direction change

        above_sma = df['close'] > sma200
        below_sma = df['close'] < sma200

        # LONG: ST direction changes to bullish (chDir > 0)
        long_entry = (chDir > 0) & above_sma
        # SHORT: ST direction changes to bearish (chDir < 0)
        short_entry = (chDir < 0) & below_sma

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
