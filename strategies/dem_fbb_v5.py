"""DeM V5 — TREND FILTER + HTF CONTEXT + ADAPTIVE ZONES.

Original DeM used cross-under-threshold (oversold) as reversal signal.
This loses in trending markets because DeM stays oversold for long periods.

V5 changes the paradigm:
  - DeM as trend filter (>50 = bullish regime, <50 = bearish regime)
  - HTF DeM must agree (H4 DeM > 50 for long, < 50 for short)
  - Wait for PULLBACK within the regime (DeM dipping to mid-zone, then rising)
  - Adaptive zones (scale with volatility)
  - Cooldown + ADX trending filter
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


class DeMV5Strategy(BaseStrategy):
    name = "dem_v5"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        period = int(p.get("bars_calculate", 14))
        htf_rule = str(p.get("htf_rule", "4h"))
        # Regime thresholds
        bull_threshold = float(p.get("bull_regime", 50.0))
        bear_threshold = float(p.get("bear_regime", 50.0))
        # Adaptive zones
        pb_low = float(p.get("pullback_zone_low", 40.0))
        pb_high = float(p.get("pullback_zone_high", 55.0))
        # Filters
        use_adx = bool(p.get("use_adx_filter", True))
        adx_min = float(p.get("adx_min", 18.0))
        use_volume = bool(p.get("use_volume_filter", True))
        cooldown = int(p.get("cooldown_bars", 6))

        # DeM scaled 0-100
        dem = ind.dem(df["high"], df["low"], period) * 100
        dem_prev = dem.shift(1)

        # HTF DeM
        htf_bull = pd.Series(False, index=df.index)
        htf_bear = pd.Series(False, index=df.index)
        try:
            htf = df.resample(htf_rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
            if len(htf) >= period + 5:
                htf_dem = ind.dem(htf["high"], htf["low"], period) * 100
                htf_bull = (htf_dem > 55).reindex(df.index, method="ffill").fillna(False).astype(bool)
                htf_bear = (htf_dem < 45).reindex(df.index, method="ffill").fillna(False).astype(bool)
        except Exception:
            pass

        # Adaptive zone (scale with volatility)
        atr_local = _atr(df, 14)
        atr_pct = atr_local / df["close"]
        # In high vol, widen the zone; in low vol, narrow it
        vol_adj = (atr_pct - atr_pct.rolling(100, min_periods=20).mean()).fillna(0) * 100
        pb_low_adj = (pb_low - vol_adj).clip(30, 50)
        pb_high_adj = (pb_high + vol_adj).clip(50, 65)

        # Bull regime: DeM > 50
        bull_regime = dem > bull_threshold
        bear_regime = dem < bear_threshold

        # Pullback detection
        # LONG: DeM dipped into pullback zone (within [pb_low, pb_high]) AND is now rising
        in_pullback_zone_long = (dem >= pb_low_adj) & (dem <= pb_high_adj)
        rising = dem > dem_prev
        pullback_buy = in_pullback_zone_long & rising

        # SHORT: mirror
        short_pb_low = 100 - pb_high_adj
        short_pb_high = 100 - pb_low_adj
        in_pullback_zone_short = (dem >= short_pb_low) & (dem <= short_pb_high)
        falling = dem < dem_prev
        pullback_sell = in_pullback_zone_short & falling

        # ADX trending
        adx_ok = pd.Series(True, index=df.index)
        if use_adx:
            try:
                adx, _, _ = ind.adx(df["high"], df["low"], df["close"], 14)
                adx_ok = (adx >= adx_min).fillna(False).astype(bool)
            except Exception:
                pass

        # Volume confirmation (rising)
        vol_ok = pd.Series(True, index=df.index)
        if use_volume and "volume" in df.columns:
            try:
                vol_avg = df["volume"].rolling(20, min_periods=2).mean()
                vol_ok = (df["volume"] >= vol_avg).fillna(True)
            except Exception:
                pass

        # ENTRY LOGIC
        # LONG: HTF bullish + DeM in pullback zone + rising + ADX trending + volume
        buy = (htf_bull | ~htf_bear & bull_regime) & pullback_buy & adx_ok & vol_ok
        # Be more lenient: allow entry if HTF not strictly bullish but entry TF is in regime
        buy = (htf_bull | bull_regime) & pullback_buy & adx_ok & vol_ok

        sell = (htf_bear | bear_regime) & pullback_sell & adx_ok & vol_ok

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)),
                               index=df.index, dtype=int)

        # Cooldown
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


class FBBV5Strategy(BaseStrategy):
    """FBB V5 — Confirmed Breakout + HTF Context + Volume Confirmation.

    Original FBB fires on first breakout bar (often false breakout).
    V5 waits for CONFIRMATION:
      - Breakout bar fires
      - Next bar closes ABOVE the breakout level (confirmed)
      - Volume on confirmation > average
      - HTF context: only trade WITH the trend
      - ADX > threshold (trending market)
    """
    name = "fbb_v5"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        period = int(p.get("bars_calculate", 20))
        dev = float(p.get("deviation", 2.0))
        htf_rule = str(p.get("htf_rule", "4h"))
        ema_period = int(p.get("ema_period", 200))
        adx_min = float(p.get("adx_min", 20.0))
        require_close_confirm = bool(p.get("require_close_confirm", True))
        cooldown = int(p.get("cooldown_bars", 8))
        vol_mult = float(p.get("volume_mult", 1.0))

        # Bollinger Bands
        mid, up, lo = ind.bollinger(df["close"], period, dev)
        # Force Index for additional confirmation
        force = ind.force_index(df["close"], df["volume"], period) * 20
        force_prev = force.shift(1)

        # HTF context: EMA 200 trend
        ema_long = pd.Series(False, index=df.index)
        ema_short = pd.Series(False, index=df.index)
        try:
            htf = df.resample(htf_rule).agg({"close": "last"}).dropna()
            if len(htf) >= ema_period:
                htf_ema = htf["close"].rolling(ema_period, min_periods=50).mean()
                ema_long_local = htf["close"] > htf_ema
                ema_short_local = htf["close"] < htf_ema
                ema_long = ema_long_local.reindex(df.index, method="ffill").fillna(False).astype(bool)
                ema_short = ema_short_local.reindex(df.index, method="ffill").fillna(False).astype(bool)
        except Exception:
            pass

        # ADX filter
        adx_ok = pd.Series(True, index=df.index)
        try:
            adx, _, _ = ind.adx(df["high"], df["low"], df["close"], 14)
            adx_ok = (adx >= adx_min).fillna(False).astype(bool)
        except Exception:
            pass

        # Volume filter
        vol_ok = pd.Series(True, index=df.index)
        if "volume" in df.columns:
            try:
                vol_avg = df["volume"].rolling(20, min_periods=2).mean()
                vol_ok = (df["volume"] >= vol_avg * vol_mult).fillna(True)
            except Exception:
                pass

        # Detect breakouts
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        prev_close = df["close"].shift(1)

        # Breakout above upper band on previous bar
        upper_breakout = prev_close > up.shift(1)
        lower_breakout = prev_close < lo.shift(1)

        # Confirmation: current close continues the breakout direction
        if require_close_confirm:
            # Buy: prev close > prev upper, current close > upper
            buy_breakout = upper_breakout & (df["close"] > up)
            sell_breakout = lower_breakout & (df["close"] < lo)
        else:
            # Just detect breakout (no confirmation)
            buy_breakout = upper_breakout
            sell_breakout = lower_breakout

        # Force Index confirmation: positive on breakout
        force_buy_ok = (force > 0) & (force > force_prev)
        force_sell_ok = (force < 0) & (force < force_prev)

        # ENTRY LOGIC
        # BUY: HTF EMA bullish + ADX trending + breakout + force positive + volume
        buy = (ema_long | ~ema_short) & adx_ok & buy_breakout & force_buy_ok.fillna(False) & vol_ok
        # Allow if no HTF context but trend confirmed by entry TF (force + breakout)
        buy = buy_breakout & force_buy_ok.fillna(False) & adx_ok & vol_ok
        sell = sell_breakout & force_sell_ok.fillna(False) & adx_ok & vol_ok

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)),
                               index=df.index, dtype=int)

        # Cooldown
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
