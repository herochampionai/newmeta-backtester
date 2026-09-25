"""V4 trend-following variants for macd_confluence, bb_rsi, fbb, quad_stoch.

Same template as triple_rsi_v4 / stoch533_mtf_v4:
  - Use the indicator as a TREND FILTER, not reversal signal
  - HTF trend alignment: only trade with the trend
  - Wait for pullback to enter
  - ADX trending filter
  - Volume confirmation

This flips the paradigm: instead of "indicator at extreme = reversal", use
"indicator aligned with trend = confirmation".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


def _trend_follow_v4(df: pd.DataFrame, regime_bull: pd.Series, regime_bear: pd.Series,
                     sma_period: int, adx_threshold: float,
                     pullback_pct: float, use_volume: bool = True,
                     cd: int = 8) -> tuple[pd.Series, pd.Series]:
    """Common V4 trend-following logic."""
    sma = df["close"].rolling(sma_period, min_periods=10).mean()
    adx, _, _ = ind.adx(df["high"], df["low"], df["close"], 14)
    adx_trending = (adx > adx_threshold).fillna(False).astype(bool)
    above_sma = (df["close"] > sma).fillna(False).astype(bool)
    below_sma = (df["close"] < sma).fillna(False).astype(bool)

    recent_high = df["high"].rolling(20, min_periods=5).max()
    recent_low = df["low"].rolling(20, min_periods=5).min()
    drop_pct = 1 - (df["close"] / recent_high)
    rally_pct = (df["close"] / recent_low) - 1
    has_pullback = ((drop_pct >= pullback_pct) | (rally_pct >= pullback_pct)).fillna(False).astype(bool)

    vol_ok = pd.Series(True, index=df.index)
    if "volume" in df.columns and use_volume:
        try:
            obv_up, cvd_up, _ = ind.volume_rising(df["close"], df["high"], df["low"],
                                                   df["volume"], 5)
            vol_ok = (obv_up | cvd_up).fillna(True)
        except Exception:
            pass

    regime_bull = regime_bull.fillna(False).astype(bool)
    regime_bear = regime_bear.fillna(False).astype(bool)

    buy = regime_bull & adx_trending & above_sma & has_pullback & vol_ok
    sell = regime_bear & adx_trending & below_sma & has_pullback & vol_ok
    return buy, sell


def _apply_cooldown(buy, sell, cd: int):
    """Apply cooldown to avoid signal spam."""
    direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)), index=buy.index, dtype=int)
    if cd > 0 and len(direction) > cd:
        new_dir = direction.values.copy()
        last_signal_idx = -999
        for i in range(len(direction)):
            if new_dir[i] != 0:
                if i - last_signal_idx < cd:
                    new_dir[i] = 0
                else:
                    last_signal_idx = i
        direction = pd.Series(new_dir, index=buy.index, dtype=int)
    return direction != 0, direction


class MACDConfluenceV4(BaseStrategy):
    """MACD as trend filter (line above signal = bullish, below = bearish)."""
    name = "macd_confluence_v4"

    def generate(self, df):
        p = self.params
        fast = int(p.get("fast", 12))
        slow = int(p.get("slow", 26))
        signal = int(p.get("signal", 9))
        sma_period = int(p.get("sma_period", 50))
        adx_threshold = float(p.get("adx_threshold", 20))
        pullback_pct = float(p.get("pullback_pct", 0.002))
        cd = int(p.get("cooldown_bars", 8))
        use_volume = bool(p.get("use_volume_filter", True))

        ml, sl, hist = ind.macd(df["close"], fast, slow, signal)
        # Trend regime: MACD line above signal AND histogram positive = bullish
        regime_bull = (ml > sl) & (hist > 0)
        regime_bear = (ml < sl) & (hist < 0)

        buy, sell = _trend_follow_v4(df, regime_bull, regime_bear, sma_period,
                                       adx_threshold, pullback_pct, use_volume, cd)
        sig = _empty_signals(df.index)
        entries, direction = _apply_cooldown(buy, sell, cd)
        sig.entries = entries
        sig.direction = direction
        return sig


class BBRsiV4(BaseStrategy):
    """Bollinger as trend filter + RSI confirmation."""
    name = "bb_rsi_v4"

    def generate(self, df):
        p = self.params
        bb_period = int(p.get("bb_period", 20))
        bb_std = float(p.get("bb_std", 2.0))
        rsi_period = int(p.get("rsi_period", 14))
        sma_period = int(p.get("sma_period", 50))
        adx_threshold = float(p.get("adx_threshold", 20))
        pullback_pct = float(p.get("pullback_pct", 0.002))
        cd = int(p.get("cooldown_bars", 8))
        use_volume = bool(p.get("use_volume_filter", True))

        sma_b, upper, lower = ind.bollinger(df["close"], bb_period, bb_std)
        rsi = ind.rsi(df["close"], rsi_period)
        # Trend regime: price above middle BB AND RSI > 50 = bullish
        regime_bull = (df["close"] > sma_b) & (rsi > 50)
        regime_bear = (df["close"] < sma_b) & (rsi < 50)

        buy, sell = _trend_follow_v4(df, regime_bull, regime_bear, sma_period,
                                       adx_threshold, pullback_pct, use_volume, cd)
        sig = _empty_signals(df.index)
        entries, direction = _apply_cooldown(buy, sell, cd)
        sig.entries = entries
        sig.direction = direction
        return sig


class QuadStochV4(BaseStrategy):
    """Quad stochastic as trend filter (above 50 = bullish)."""
    name = "quad_stoch_v4"

    def generate(self, df):
        p = self.params
        k_period = int(p.get("k_period", 14))
        d_period = int(p.get("d_period", 3))
        smooth = int(p.get("smooth", 3))
        sma_period = int(p.get("sma_period", 50))
        adx_threshold = float(p.get("adx_threshold", 20))
        pullback_pct = float(p.get("pullback_pct", 0.002))
        cd = int(p.get("cooldown_bars", 8))
        use_volume = bool(p.get("use_volume_filter", True))

        k_e, d_e = ind.stochastic(df["high"], df["low"], df["close"], k_period, d_period, smooth)
        # Regime: K > D AND K > 50 = bullish
        regime_bull = (k_e > d_e) & (k_e > 50)
        regime_bear = (k_e < d_e) & (k_e < 50)

        buy, sell = _trend_follow_v4(df, regime_bull, regime_bear, sma_period,
                                       adx_threshold, pullback_pct, use_volume, cd)
        sig = _empty_signals(df.index)
        entries, direction = _apply_cooldown(buy, sell, cd)
        sig.entries = entries
        sig.direction = direction
        return sig


class FBBV4(BaseStrategy):
    """Fractal Breakout Bands as trend filter (price > recent swing high + ADX trending = bullish)."""
    name = "fbb_v4"

    def generate(self, df):
        p = self.params
        lookback = int(p.get("lookback", 20))
        atr_period = int(p.get("atr_period", 14))
        sma_period = int(p.get("sma_period", 50))
        adx_threshold = float(p.get("adx_threshold", 20))
        pullback_pct = float(p.get("pullback_pct", 0.002))
        cd = int(p.get("cooldown_bars", 8))
        use_volume = bool(p.get("use_volume_filter", True))

        # Trend regime: rolling max trend — current close above recent max = bullish breakout
        roll_high = df["high"].rolling(lookback, min_periods=5).max()
        roll_low = df["low"].rolling(lookback, min_periods=5).min()
        regime_bull = df["close"] >= roll_high.shift(1)
        regime_bear = df["close"] <= roll_low.shift(1)

        buy, sell = _trend_follow_v4(df, regime_bull, regime_bear, sma_period,
                                       adx_threshold, pullback_pct, use_volume, cd)
        sig = _empty_signals(df.index)
        entries, direction = _apply_cooldown(buy, sell, cd)
        sig.entries = entries
        sig.direction = direction
        return sig
