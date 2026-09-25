"""Triple RSI V3 — TREND+PULLBACK (HTF trend + LTF reversal entry).

The breakthrough realization: pure RSI reversal loses on trending NAS
because overbought/oversold signals fire AGAINST the trend.

V3 changes:
  1. HTF trend detection: H4 SMA200 direction = uptrend/downtrend/range
  2. Trade-with-trend pullbacks:
     - In HTF uptrend: ONLY buy when LTF RSI is oversold (buy the dip)
     - In HTF downtrend: ONLY sell when LTF RSI is overbought (sell the rally)
     - In range: both directions allowed
  3. RSI hook pattern: RSI must be turning (rising for buy, falling for sell)
  4. Multi-RSI triple confirmation (V1 idea kept, but as SCORE not gate)
  5. Volume + HTF context alignment
  6. Cooldown to avoid signal spam

Character: trend-following pullback (works on both FX and indices).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


class TripleRSIV3Strategy(BaseStrategy):
    """Trend-following pullback using triple RSI."""
    name = "triple_rsi_v3"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        pf = int(p.get("rsi_fast", 7))
        pm = int(p.get("rsi_mid", 14))
        ps = int(p.get("rsi_slow", 28))
        os_base = float(p.get("oversold", 35))  # looser (we're trading with trend, not against)
        ob_base = float(p.get("overbought", 65))
        htf_period = int(p.get("htf_sma_period", 200))
        htf_rule = str(p.get("htf_rule", "4h"))
        min_pullback_pct = float(p.get("min_pullback_pct", 0.001))

        # === HTF TREND DETECTION ===
        try:
            htf = df.resample(htf_rule).agg({"close": "last", "high": "max", "low": "min"}).dropna()
            if len(htf) >= htf_period:
                htf_sma = htf["close"].rolling(htf_period, min_periods=50).mean()
                htf_trend_up = htf["close"] > htf_sma
                htf_trend_dn = htf["close"] < htf_sma
                # Forward-fill to H1 index
                htf_trend_up = htf_trend_up.reindex(df.index, method="ffill").fillna(False)
                htf_trend_dn = htf_trend_dn.reindex(df.index, method="ffill").fillna(False)
            else:
                # Fallback: use H1 SMA if HTF insufficient
                h1_sma = df["close"].rolling(htf_period, min_periods=50).mean()
                htf_trend_up = df["close"] > h1_sma
                htf_trend_dn = df["close"] < h1_sma
        except Exception:
            h1_sma = df["close"].rolling(htf_period, min_periods=50).mean()
            htf_trend_up = df["close"] > h1_sma
            htf_trend_dn = df["close"] < h1_sma
        htf_ranging = ~htf_trend_up & ~htf_trend_dn

        # === LTF RSI ===
        rf = ind.rsi(df["close"], pf)
        rm = ind.rsi(df["close"], pm)
        rs = ind.rsi(df["close"], ps)

        # Adaptive thresholds
        atr_local = _atr(df, 14)
        atr_pct = atr_local / df["close"]
        vol_adj = (atr_pct - atr_pct.rolling(100, min_periods=20).mean()).fillna(0) * 100
        os_level = (os_base - vol_adj).clip(15, 45)
        ob_level = (ob_base + vol_adj).clip(55, 85)

        # Pullback from recent high/low
        recent_high = df["high"].rolling(20, min_periods=5).max()
        recent_low = df["low"].rolling(20, min_periods=5).min()
        drop_from_high = 1 - (df["close"] / recent_high)
        rally_from_low = (df["close"] / recent_low) - 1
        has_pullback = (drop_from_high >= min_pullback_pct) | (rally_from_low >= min_pullback_pct)

        # RSI hook: is fast RSI turning?
        rf_hook_up = (rf > rf.shift(1)) & (rf.shift(1) <= rf.shift(2))
        rf_hook_dn = (rf < rf.shift(1)) & (rf.shift(1) >= rf.shift(2))

        # Triple RSI confluence score (0-3)
        buy_score = (rf < os_level).astype(int) + (rm < 50).astype(int) + (rs < 55).astype(int)
        sell_score = (rf > ob_level).astype(int) + (rm > 50).astype(int) + (rs > 45).astype(int)

        # Volume: OBV rising OR CVD rising
        vol_ok = pd.Series(True, index=df.index)
        if "volume" in df.columns and bool(p.get("use_volume_filter", True)):
            try:
                wmp = int(p.get("volume_wma_period", 5))
                obv_up, cvd_up, _ = ind.volume_rising(df["close"], df["high"], df["low"],
                                                       df["volume"], wmp)
                vol_ok = obv_up | cvd_up
            except Exception:
                pass

        # === TREND-ALIGNED ENTRY LOGIC ===
        # BUY: HTF uptrend + LTF RSI oversold + RSI hook up + pullback + volume
        # In range: allow both sides
        buy_trend = (htf_trend_up & (rf < os_level) & rf_hook_up
                     & has_pullback & vol_ok.fillna(False))
        sell_trend = (htf_trend_dn & (rf > ob_level) & rf_hook_dn
                      & has_pullback & vol_ok.fillna(False))
        # Range: looser conditions
        buy_range = (htf_ranging & (buy_score >= 2) & rf_hook_up & vol_ok.fillna(False))
        sell_range = (htf_ranging & (sell_score >= 2) & rf_hook_dn & vol_ok.fillna(False))

        buy = buy_trend | buy_range
        sell = sell_trend | sell_range

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)), index=df.index, dtype=int)

        # Cooldown
        cd = int(p.get("cooldown_bars", 8))
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
        return sig


class Stoch533MTFV3(BaseStrategy):
    """Trend-following pullback using multi-TF stochastic."""
    name = "stoch533_mtf_v3"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        k_period = int(p.get("k_period", 5))
        d_period = int(p.get("d_period", 3))
        smooth = int(p.get("smooth", 3))
        os_base = float(p.get("oversold", 25))
        ob_base = float(p.get("overbought", 75))
        htf_period = int(p.get("htf_sma_period", 200))
        htf_rule = str(p.get("htf_rule", "4h"))
        min_pullback_pct = float(p.get("min_pullback_pct", 0.001))

        # === HTF TREND ===
        try:
            htf = df.resample(htf_rule).agg({"close": "last", "high": "max", "low": "min"}).dropna()
            if len(htf) >= htf_period:
                htf_sma = htf["close"].rolling(htf_period, min_periods=50).mean()
                htf_trend_up = htf["close"] > htf_sma
                htf_trend_dn = htf["close"] < htf_sma
                htf_trend_up = htf_trend_up.reindex(df.index, method="ffill").fillna(False)
                htf_trend_dn = htf_trend_dn.reindex(df.index, method="ffill").fillna(False)
            else:
                h1_sma = df["close"].rolling(htf_period, min_periods=50).mean()
                htf_trend_up = df["close"] > h1_sma
                htf_trend_dn = df["close"] < h1_sma
        except Exception:
            h1_sma = df["close"].rolling(htf_period, min_periods=50).mean()
            htf_trend_up = df["close"] > h1_sma
            htf_trend_dn = df["close"] < h1_sma
        htf_ranging = ~htf_trend_up & ~htf_trend_dn

        # Adaptive zones
        atr_local = _atr(df, 14)
        atr_pct = atr_local / df["close"]
        vol_adj = (atr_pct - atr_pct.rolling(100, min_periods=20).mean()).fillna(0) * 100
        os_level = (os_base - vol_adj).clip(10, 40)
        ob_level = (ob_base + vol_adj).clip(60, 90)

        # Entry-TF stochastic
        k_e, d_e = ind.stochastic(df["high"], df["low"], df["close"],
                                  k_period, d_period, smooth)

        # Pullback
        recent_high = df["high"].rolling(20, min_periods=5).max()
        recent_low = df["low"].rolling(20, min_periods=5).min()
        drop_from_high = 1 - (df["close"] / recent_high)
        rally_from_low = (df["close"] / recent_low) - 1
        has_pullback = (drop_from_high >= min_pullback_pct) | (rally_from_low >= min_pullback_pct)

        # K hook: turning
        k_hook_up = (k_e > k_e.shift(1)) & (k_e.shift(1) <= k_e.shift(2))
        k_hook_dn = (k_e < k_e.shift(1)) & (k_e.shift(1) >= k_e.shift(2))

        # Partial MTF: at least 1 HTF in same zone
        mtf_timeframes = p.get("htf_timeframes", ["15min", "30min", "1h", "4h"])
        htf_in_os = pd.Series(0, index=df.index, dtype=int)
        htf_in_ob = pd.Series(0, index=df.index, dtype=int)
        for rule in mtf_timeframes:
            try:
                htf_x = df.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
                if len(htf_x) < k_period + d_period + smooth + 5:
                    continue
                kx, _ = ind.stochastic(htf_x["high"], htf_x["low"], htf_x["close"],
                                       k_period, d_period, smooth)
                kx = kx.reindex(df.index, method="ffill")
                htf_in_os = htf_in_os + (kx < os_level).fillna(False).astype(int)
                htf_in_ob = htf_in_ob + (kx > ob_level).fillna(False).astype(int)
            except Exception:
                continue

        # Volume
        vol_ok = pd.Series(True, index=df.index)
        if "volume" in df.columns and bool(p.get("use_volume_filter", True)):
            try:
                wmp = int(p.get("volume_wma_period", 5))
                obv_up, cvd_up, _ = ind.volume_rising(df["close"], df["high"], df["low"],
                                                       df["volume"], wmp)
                vol_ok = obv_up | cvd_up
            except Exception:
                pass

        # === TREND-ALIGNED ENTRY ===
        # BUY: HTF uptrend + LTF K in oversold + K hook up + pullback + volume
        buy_trend = (htf_trend_up & (k_e < os_level) & k_hook_up
                     & has_pullback & vol_ok.fillna(False))
        sell_trend = (htf_trend_dn & (k_e > ob_level) & k_hook_dn
                      & has_pullback & vol_ok.fillna(False))
        # Range: allow any direction with strong oversold/overbought
        buy_range = (htf_ranging & (k_e < os_level) & k_hook_up
                     & (htf_in_os >= 1) & vol_ok.fillna(False))
        sell_range = (htf_ranging & (k_e > ob_level) & k_hook_dn
                      & (htf_in_ob >= 1) & vol_ok.fillna(False))

        buy = buy_trend | buy_range
        sell = sell_trend | sell_range

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)), index=df.index, dtype=int)

        # Cooldown
        cd = int(p.get("cooldown_bars", 8))
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
        return sig
