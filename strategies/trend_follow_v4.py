"""Triple RSI V4 + Stoch V4 — TREND-FOLLOWING using the same indicators.

Key insight: RSI/Stochastic are usually used as reversal signals, but in
trending markets they LOSE. The fix: USE THEM AS TREND FILTERS instead.

V4 logic:
  1. All 3 RSIs aligned ABOVE 50 = bullish regime; below 50 = bearish regime
  2. Wait for ADX > 20 (trending market) to enable signals
  3. BUY: regime bullish + ADX trending + price > 200 SMA + pullback to SMA
  4. SELL: regime bearish + ADX trending + price < 200 SMA + pullback to SMA
  5. RSI used as EXIT: if RSI drops below 40 in a buy, exit (instead of relying on TP/SL)

Character: pure trend-following. Works in trending markets (NAS, FX-trending periods).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


class TripleRSIV4Strategy(BaseStrategy):
    """Trend-following pullback using triple RSI as trend filter."""
    name = "triple_rsi_v4"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        pf = int(p.get("rsi_fast", 7))
        pm = int(p.get("rsi_mid", 14))
        ps = int(p.get("rsi_slow", 28))
        sma_period = int(p.get("sma_period", 50))
        adx_threshold = float(p.get("adx_threshold", 20))
        pullback_pct = float(p.get("pullback_pct", 0.002))
        rsi_bull = float(p.get("rsi_bull_threshold", 50))
        rsi_bear = float(p.get("rsi_bear_threshold", 50))
        cd = int(p.get("cooldown_bars", 8))

        # RSI triple
        rf = ind.rsi(df["close"], pf)
        rm = ind.rsi(df["close"], pm)
        rs = ind.rsi(df["close"], ps)

        # SMA for trend
        sma = df["close"].rolling(sma_period, min_periods=10).mean()

        # ADX for trend strength
        adx, pdi, mdi = ind.adx(df["high"], df["low"], df["close"], 14)

        # Trend regime (cast to bool to avoid NaN issues with ~)
        bullish_regime = ((rf > rsi_bull) & (rm > rsi_bull) & (rs > rsi_bear)).fillna(False).astype(bool)
        bearish_regime = ((rf < rsi_bull) & (rm < rsi_bull) & (rs < rsi_bear)).fillna(False).astype(bool)
        adx_trending = (adx > adx_threshold).fillna(False).astype(bool)

        # Price position relative to SMA
        above_sma = df["close"] > sma
        below_sma = df["close"] < sma

        # Pullback from recent high/low
        recent_high = df["high"].rolling(20, min_periods=5).max()
        recent_low = df["low"].rolling(20, min_periods=5).min()
        drop_pct = 1 - (df["close"] / recent_high)
        rally_pct = (df["close"] / recent_low) - 1
        has_pullback = (drop_pct >= pullback_pct) | (rally_pct >= pullback_pct)

        # Volume
        vol_ok = pd.Series(True, index=df.index)
        if "volume" in df.columns and bool(p.get("use_volume_filter", True)):
            try:
                wmp = 5
                obv_up, cvd_up, _ = ind.volume_rising(df["close"], df["high"], df["low"],
                                                       df["volume"], wmp)
                vol_ok = obv_up | cvd_up
            except Exception:
                pass

        # === ENTRY ===
        # BUY: bullish regime + ADX trending + price above SMA + pullback occurred
        buy = (bullish_regime & adx_trending & above_sma & has_pullback
               & vol_ok.fillna(False))
        # SELL: bearish regime + ADX trending + price below SMA + pullback occurred
        sell = (bearish_regime & adx_trending & below_sma & has_pullback
                & vol_ok.fillna(False))

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)), index=df.index, dtype=int)

        # Cooldown
        if cd > 0 and len(direction) > cd:
            new_dir = direction.values.copy()
            last_signal_idx = -999
            for i in range(len(df)):
                if new_dir[i] != 0:
                    if i - last_signal_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_signal_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction

        # Exit: when RSI regime changes, close
        if bool(p.get("use_rsi_exit", True)):
            regime_changed_long = bullish_regime & ~bullish_regime.shift(1).fillna(False).astype(bool)
            regime_changed_short = bearish_regime & ~bearish_regime.shift(1).fillna(False).astype(bool)
            sig.exits = regime_changed_long | regime_changed_short
        return sig


class Stoch533V4(BaseStrategy):
    """Trend-following using multi-TF stochastic as trend filter."""
    name = "stoch533_mtf_v4"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        k_period = int(p.get("k_period", 5))
        d_period = int(p.get("d_period", 3))
        smooth = int(p.get("smooth", 3))
        sma_period = int(p.get("sma_period", 50))
        adx_threshold = float(p.get("adx_threshold", 20))
        pullback_pct = float(p.get("pullback_pct", 0.002))
        k_bull = float(p.get("k_bull", 50))
        cd = int(p.get("cooldown_bars", 8))

        # Stochastic
        k_e, d_e = ind.stochastic(df["high"], df["low"], df["close"],
                                  k_period, d_period, smooth)

        # SMA + ADX
        sma = df["close"].rolling(sma_period, min_periods=10).mean()
        adx, pdi, mdi = ind.adx(df["high"], df["low"], df["close"], 14)

        # Trend regime: stochastic above 50 = bullish, below 50 = bearish
        bullish_regime = k_e > k_bull
        bearish_regime = k_e < k_bull
        adx_trending = adx > adx_threshold
        above_sma = df["close"] > sma
        below_sma = df["close"] < sma

        # Multi-TF confluence: how many TFs agree
        mtf_timeframes = p.get("htf_timeframes", ["15min", "30min", "1h", "4h"])
        htf_bull = pd.Series(0, index=df.index, dtype=int)
        htf_bear = pd.Series(0, index=df.index, dtype=int)
        for rule in mtf_timeframes:
            try:
                htf_x = df.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
                if len(htf_x) < k_period + d_period + smooth + 5:
                    continue
                kx, _ = ind.stochastic(htf_x["high"], htf_x["low"], htf_x["close"],
                                       k_period, d_period, smooth)
                kx = kx.reindex(df.index, method="ffill")
                htf_bull = htf_bull + (kx > k_bull).fillna(False).astype(int)
                htf_bear = htf_bear + (kx < k_bull).fillna(False).astype(int)
            except Exception:
                continue

        # Pullback
        recent_high = df["high"].rolling(20, min_periods=5).max()
        recent_low = df["low"].rolling(20, min_periods=5).min()
        drop_pct = 1 - (df["close"] / recent_high)
        rally_pct = (df["close"] / recent_low) - 1
        has_pullback = (drop_pct >= pullback_pct) | (rally_pct >= pullback_pct)

        # Volume
        vol_ok = pd.Series(True, index=df.index)
        if "volume" in df.columns and bool(p.get("use_volume_filter", True)):
            try:
                obv_up, cvd_up, _ = ind.volume_rising(df["close"], df["high"], df["low"],
                                                       df["volume"], 5)
                vol_ok = obv_up | cvd_up
            except Exception:
                pass

        # === ENTRY ===
        # BUY: bullish regime + ADX trending + above SMA + HTF mostly bullish + pullback
        buy = (bullish_regime & adx_trending & above_sma
               & (htf_bull >= 2) & has_pullback & vol_ok.fillna(False))
        # SELL: bearish regime + ADX trending + below SMA + HTF mostly bearish + pullback
        sell = (bearish_regime & adx_trending & below_sma
                & (htf_bear >= 2) & has_pullback & vol_ok.fillna(False))

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)), index=df.index, dtype=int)

        # Cooldown
        if cd > 0 and len(direction) > cd:
            new_dir = direction.values.copy()
            last_signal_idx = -999
            for i in range(len(df)):
                if new_dir[i] != 0:
                    if i - last_signal_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_signal_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)
        sig.entries = direction != 0
        sig.direction = direction

        # Exit: stochastic crosses 50 in opposite direction
        if bool(p.get("use_k_exit", True)):
            cross_50 = (k_e > 50) & (k_e.shift(1) <= 50) | (k_e < 50) & (k_e.shift(1) >= 50)
            sig.exits = cross_50
        return sig
