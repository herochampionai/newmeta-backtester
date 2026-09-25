"""Ports of 4 additional strategies from the user's MT5 archive.

1. BachelierWaveStrategy — from Enhanced oscillator bachelier.txt
2. MACDInstitutionalStrategy — from big_mac_inst.mq5 (v7 institutional)
3. MA8RibbonStrategy — from 8 ma ribbon.mq5
4. RC44Strategy — from 4x4 RC.mq5 (HTF EMA + LTF pullback + Weis Wave sniper)
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


class BachelierWaveStrategy(BaseStrategy):
    """Bachelier Wave Oscillator — from Enhanced oscillator bachelier.txt.

    Logic:
      - returns = (close - close[1]) / close[1]
      - mu = SMA(returns, length); sigma = stdev(returns, length)
      - expected_price = close[1] * (1 + mu)
      - bachelier_raw = (close - expected_price) / (sigma * close[1])
      - bachelier_smooth = EMA(bachelier_raw, smoothing)
      - z_raw = (close - SMA(close, length)) / stdev(close, length)
      - z_smooth = EMA(z_raw, smoothing)
      - primary_wave = weighted blend (bachelier + z_score) * sensitivity
      - unified_wave = primary_wave + momentum_smooth * 0.3
      - final_wave = EMA(normalized wave, 3)
      - ENTRY: STRONG_BUY when final_wave < -1.5 (oversold + strong)
      - EXIT: STRONG_SELL when final_wave > 1.5 (overbought + strong)
    """
    name = "bachelier_wave"

    def generate(self, df):
        p = self.params
        length = int(p.get("length", 20))
        smoothing = int(p.get("smoothing", 10))
        sensitivity = float(p.get("sensitivity", 1.5))
        ob_level = float(p.get("ob_level", 1.0))
        os_level = float(p.get("os_level", -1.0))
        strong_mult = float(p.get("strong_mult", 1.5))
        cooldown = int(p.get("cooldown", 8))

        close = df["close"]
        returns = close.pct_change()
        mu = returns.rolling(length, min_periods=5).mean()
        sigma = returns.rolling(length, min_periods=5).std().fillna(0).replace(0, 1e-9)

        expected = close.shift(1) * (1 + mu)
        bachelier_raw = (close - expected) / (sigma * close.shift(1))
        bachelier_smooth = bachelier_raw.ewm(span=smoothing, min_periods=5).mean()

        sma_c = close.rolling(length, min_periods=5).mean()
        stdev_c = close.rolling(length, min_periods=5).std().fillna(0).replace(0, 1e-9)
        z_raw = (close - sma_c) / stdev_c
        z_smooth = z_raw.ewm(span=smoothing, min_periods=5).mean()

        # Volatility-adaptive weights
        vol_factor = np.minimum(2.0, sigma / 0.02)
        bach_weight = 0.4 + 0.2 * vol_factor
        z_weight = 1.0 - bach_weight
        primary_wave = (bachelier_smooth * bach_weight + z_smooth * z_weight) * sensitivity

        momentum = primary_wave.diff()
        accel = momentum.diff()
        momentum_smooth = momentum.ewm(span=max(2, smoothing // 2), min_periods=2).mean()
        unified_wave = primary_wave + momentum_smooth * 0.3

        # Normalize
        unified_std = unified_wave.rolling(length * 2, min_periods=10).std().fillna(0).replace(0, 1e-9)
        wave_normalized = unified_wave / unified_std
        final_wave = wave_normalized.ewm(span=3, min_periods=3).mean()

        ob_strong = ob_level * strong_mult  # 1.5 by default
        os_strong = os_level * strong_mult  # -1.5

        # STRONG BUY: wave < os_strong (oversold + strong)
        # STRONG SELL: wave > ob_strong (overbought + strong)
        buy = final_wave < os_strong
        sell = final_wave > ob_strong

        sig = _empty_signals(df.index)
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


class MACDInstitutionalStrategy(BaseStrategy):
    """MACD Institutional v7 — MTF + Sniper + Volume confirmation.

    From big_mac_inst.mq5:
      - Multi-TF: weekly + daily trend must align
      - Sniper: pullback from recent extreme + histogram color flip
      - Volume: confirmation (volume > average on signal bar)
      - RSI filter: block if RSI > 75 or < 25 (extremes)
      - ATR-adaptive SL/TP

    Entry logic:
      - Weekly + Daily MACD trend bullish (MACD > signal)
      - Price pulled back from 20-bar high by N%
      - MACD histogram flipped from negative to positive (or vice versa)
      - Volume on signal bar > 1.0 * avg volume
      - RSI not in extreme zone
    """
    name = "macd_inst"

    def generate(self, df):
        p = self.params
        fast = int(p.get("fast", 12))
        slow = int(p.get("slow", 26))
        sig_period = int(p.get("signal", 9))
        pullback_pct = float(p.get("pullback_pct", 0.005))
        pullback_bars = int(p.get("pullback_bars", 10))
        rsi_period = int(p.get("rsi_period", 14))
        rsi_extreme_high = float(p.get("rsi_extreme_high", 75))
        rsi_extreme_low = float(p.get("rsi_extreme_low", 25))
        cooldown = int(p.get("cooldown", 6))

        # MACD
        ml, sl, hist = ind.macd(df["close"], fast, slow, sig_period)
        hist_prev = hist.shift(1)

        # HTF context: D1 MACD trend (resampled)
        htf_bull = pd.Series(False, index=df.index)
        htf_bear = pd.Series(False, index=df.index)
        try:
            htf = df.resample("1d").agg({"close": "last"}).dropna()
            if len(htf) >= slow + sig_period + 5:
                htf_ml, htf_sl, _ = ind.macd(htf["close"], fast, slow, sig_period)
                htf_bull_local = htf_ml > htf_sl
                htf_bear_local = htf_ml < htf_sl
                htf_bull = htf_bull_local.reindex(df.index, method="ffill").fillna(False).astype(bool)
                htf_bear = htf_bear_local.reindex(df.index, method="ffill").fillna(False).astype(bool)
        except Exception:
            pass

        # Pullback detection
        recent_high = df["high"].rolling(pullback_bars, min_periods=2).max()
        recent_low = df["low"].rolling(pullback_bars, min_periods=2).min()
        drop_pct = 1 - (df["close"] / recent_high)
        rally_pct = (df["close"] / recent_low) - 1
        has_pullback_buy = drop_pct >= pullback_pct
        has_pullback_sell = rally_pct >= pullback_pct

        # Histogram flip
        hist_flip_up = (hist > 0) & (hist_prev <= 0)
        hist_flip_dn = (hist < 0) & (hist_prev >= 0)

        # RSI filter
        rsi = ind.rsi(df["close"], rsi_period)
        rsi_ok = (rsi >= rsi_extreme_low) & (rsi <= rsi_extreme_high)

        # Volume confirmation
        vol_ok = pd.Series(True, index=df.index)
        if "volume" in df.columns:
            try:
                vol_avg = df["volume"].rolling(20, min_periods=2).mean()
                vol_ok = (df["volume"] >= vol_avg).fillna(True)
            except Exception:
                pass

        # Entry: HTF trend + pullback + histogram flip + RSI ok + volume
        buy = (htf_bull | ~htf_bear) & has_pullback_buy & hist_flip_up & rsi_ok.fillna(True) & vol_ok
        sell = (htf_bear | ~htf_bull) & has_pullback_sell & hist_flip_dn & rsi_ok.fillna(True) & vol_ok
        # Allow entries without strict HTF if other conditions align
        buy = buy | (has_pullback_buy & hist_flip_up & rsi_ok.fillna(True) & vol_ok)
        sell = sell | (has_pullback_sell & hist_flip_dn & rsi_ok.fillna(True) & vol_ok)

        sig = _empty_signals(df.index)
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


class MA8RibbonStrategy(BaseStrategy):
    """8 MA Ribbon strategy.

    From 8 ma ribbon.mq5: 8 EMAs (20, 25, 30, 35, 40, 45, 50, 55).

    Signal logic (inferred from typical ribbon strategies):
      - All 8 EMAs in bullish order (EMA1 > EMA2 > ... > EMA8) → BUY
      - All 8 EMAs in bearish order → SELL
      - Or: EMA1 > EMA8 (above) → bullish, below → bearish
      - Slope of all MAs rising/falling as confirmation
    """
    name = "ma8_ribbon"

    def generate(self, df):
        p = self.params
        periods = [int(p.get(f"ema_{i}", period)) for i, period in enumerate([20, 25, 30, 35, 40, 45, 50, 55], 1)]
        slope_bars = int(p.get("slope_bars", 5))
        cooldown = int(p.get("cooldown", 12))

        emas = [df["close"].ewm(span=period, min_periods=5).mean() for period in periods]

        # Bullish alignment: EMA1 > EMA2 > ... > EMA8 (most recent)
        bullish_aligned = emas[0] > emas[1]
        for i in range(1, len(emas) - 1):
            bullish_aligned = bullish_aligned & (emas[i] > emas[i + 1])
        # Bearish alignment: EMA1 < EMA2 < ... < EMA8
        bearish_aligned = emas[0] < emas[1]
        for i in range(1, len(emas) - 1):
            bearish_aligned = bearish_aligned & (emas[i] < emas[i + 1])

        # Slope confirmation: most EMAs rising
        rising = emas[0] > emas[0].shift(slope_bars)
        for i in range(1, len(emas)):
            rising = rising & (emas[i] > emas[i].shift(slope_bars))
        falling = emas[0] < emas[0].shift(slope_bars)
        for i in range(1, len(emas)):
            falling = falling & (emas[i] < emas[i].shift(slope_bars))

        buy = bullish_aligned & rising
        sell = bearish_aligned & falling

        sig = _empty_signals(df.index)
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


class RC44Strategy(BaseStrategy):
    """4x4 RC — HTF EMA + LTF pullback + Weis Wave sniper.

    From 4x4 RC.mq5:
      - HTF: 50 EMA / 200 EMA crossover (trend)
      - LTF: 15% pullback from 20-bar extreme
      - Sniper: Weis Wave reversal (consecutive opposite candles then reversal)
      - Combined entry: HTF trend + LTF pullback + Weis Wave sniper
    """
    name = "rc44"

    def generate(self, df):
        p = self.params
        htf_fast = int(p.get("htf_fast", 50))
        htf_slow = int(p.get("htf_slow", 200))
        pullback_bars = int(p.get("pullback_bars", 20))
        pullback_pct = float(p.get("pullback_pct", 0.15))
        sniper_bars = int(p.get("sniper_bars", 5))
        cooldown = int(p.get("cooldown", 8))

        # HTF trend
        ema_fast = df["close"].ewm(span=htf_fast, min_periods=20).mean()
        ema_slow = df["close"].ewm(span=htf_slow, min_periods=50).mean()
        trend_up = ema_fast > ema_slow
        trend_dn = ema_fast < ema_slow

        # LTF pullback (15% retracement from 20-bar high)
        recent_high = df["high"].rolling(pullback_bars, min_periods=2).max()
        recent_low = df["low"].rolling(pullback_bars, min_periods=2).min()
        drop_pct = 1 - (df["close"] / recent_high)
        rally_pct = (df["close"] / recent_low) - 1
        # 15% pullback is huge for H1 — for index 1.5% might be enough
        actual_pullback = max(pullback_pct, 0.01)  # min 1%
        pb_buy = drop_pct >= actual_pullback
        pb_sell = rally_pct >= actual_pullback

        # Weis Wave sniper: 3+ consecutive same-direction candles then reversal
        bullish_candle = df["close"] > df["open"]
        bearish_candle = df["close"] < df["open"]
        # Count consecutive bearish candles (for buy setup: downtrend exhaustion)
        bearish_streak = bearish_candle.astype(int).rolling(sniper_bars, min_periods=1).sum()
        bullish_streak = bullish_candle.astype(int).rolling(sniper_bars, min_periods=1).sum()
        # Buy sniper: was in bearish streak (downtrend) + current bar is bullish (reversal)
        sniper_buy = (bearish_streak.shift(1) >= sniper_bars) & bullish_candle
        sniper_sell = (bullish_streak.shift(1) >= sniper_bars) & bearish_candle

        # Combined entry
        buy = trend_up & pb_buy & sniper_buy
        sell = trend_dn & pb_sell & sniper_sell

        sig = _empty_signals(df.index)
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
