"""MACD + confluences — port of Pine 'Meta MACD Enhanced v2.6'.

13-condition confluence table, each row counts bull-exclusive or bear-exclusive:
  1 Divergence (regular hist-extreme + swing pivot via generic div on hist)
  2 Higher Low / Lower High (hist lowest/highest 5 vs 10)
  3 Engulfing pattern
  4 Retest failure (crossover 1 bar ago + pullback holding)
  5 Zero-line rejection
  6 Steep slope (adaptive vol-adjusted)
  7 Histogram slope
  8 MACD-Signal diff (>= diff_threshold)
  9 Momentum (ROC 14 > 0)
  10 MACD ROC 5-bar (> 0)
  11 Strong crossover (cross + diff)
  12 STC zero cross (stcCentered)
  13 Awesome Oscillator (ao > 0 rising)

Entry (Pine base + score gate):
  BUY:  crossover(macd,signal) AND stcC > 0 AND strongDiff AND strongMom
        AND bullish_count >= min_confluence_score
  SELL: mirror.
  Optional last-signal state filter (Pine default) avoids repeat same-direction
  fires. Score-only mode available via require_base_trigger=False.

Defaults match Pine v2.6 inputs. MACD 3/10/16, STC 12/26/50 x5, AO x0.3.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals
from .triple_rsi import _detect_divergence


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


class MACDConfluenceStrategy(BaseStrategy):
    name = "macd_confluence"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        close, high, low = df["close"], df["high"], df["low"]
        op = df["open"] if "open" in df.columns else close

        fast = int(p.get("fast_ema", 3))
        slow = int(p.get("slow_ema", 10))
        sig_len = int(p.get("signal_ema", 16))
        macd_mult = float(p.get("macd_multiplier", 1.0))
        stc_len = int(p.get("stc_length", 12))
        stc_fast = int(p.get("stc_fast", 26))
        stc_slow = int(p.get("stc_slow", 50))
        stc_mult = float(p.get("stc_multiplier", 5.0))
        ao_mult = float(p.get("ao_multiplier", 0.3))
        diff_th = float(p.get("diff_threshold", 0.2))
        mom_len = int(p.get("momentum_length", 14))
        lb_div = int(p.get("lookback_div", 20))
        base_steep = float(p.get("base_steep", 0.1))
        avg_hist_th = float(p.get("avg_hist_threshold", 0.5))
        min_score = int(p.get("min_confluence_score", 6))

        # Pine thresholds are absolute price units (tuned for BTC/indices).
        # On forex MACD moves ~0.0005 so 0.2 never fires. Auto-scale by ATR.
        if bool(p.get("auto_scale_thresholds", True)):
            atr14 = (high - low).rolling(14, min_periods=1).mean().ffill().fillna((high - low).mean())
            scale = atr14
        else:
            scale = pd.Series(1.0, index=df.index)
        diff_th_s = diff_th * scale
        avg_hist_th_s = avg_hist_th * scale
        base_steep_s = base_steep * scale

        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()
        macd = (fast_ema - slow_ema) * macd_mult
        signal = macd.ewm(span=sig_len, adjust=False).mean()
        hist = macd - signal

        hl2 = (high + low) / 2
        ao = (hl2.rolling(5, min_periods=1).mean()
              - hl2.rolling(34, min_periods=1).mean()) * ao_mult

        stc_raw = ind.stc(close, stc_len, stc_fast, stc_slow)
        stcC = (stc_raw - 50) * stc_mult

        macd_p = macd.shift(1)
        sig_p = signal.shift(1)
        hist_p = hist.shift(1)
        stcC_p = stcC.shift(1)
        ao_p = ao.shift(1)

        cross_up = (macd > signal) & (macd_p <= sig_p)
        cross_dn = (macd < signal) & (macd_p >= sig_p)

        # --- Row 1: divergence ---
        hist_high = hist.rolling(lb_div, min_periods=1).max()
        hist_low = hist.rolling(lb_div, min_periods=1).min()
        reg_bull = (low < low.shift(1)) & (hist > hist_low) & (hist_low < hist_low.shift(1))
        reg_bear = (high > high.shift(1)) & (hist < hist_high) & (hist_high > hist_high.shift(1))
        sw_bull, sw_bear = _detect_divergence(close, hist, lb_div)
        div_bull = (reg_bull | sw_bull).fillna(False)
        div_bear = (reg_bear | sw_bear).fillna(False)

        # --- Row 2: higher low / lower high ---
        hl_bull = hist.rolling(5, min_periods=1).min() > hist.rolling(10, min_periods=1).min()
        hl_bear = hist.rolling(5, min_periods=1).max() < hist.rolling(10, min_periods=1).max()

        # --- Row 3: engulfing ---
        bull_eng = (close > op) & (close.shift(1) < op.shift(1)) & (close > op.shift(1)) & ((close - op) > (op.shift(1) - close.shift(1)).abs())
        bear_eng = (close < op) & (close.shift(1) > op.shift(1)) & (close < op.shift(1)) & ((op - close) > (close.shift(1) - op.shift(1)).abs())

        # --- Row 4: retest failure ---
        retest_bull = cross_up.shift(1).fillna(False) & (macd < macd_p) & (macd > signal)
        retest_bear = cross_dn.shift(1).fillna(False) & (macd > macd_p) & (macd < signal)

        # --- Row 5: zero-line rejection ---
        zero_bull = (macd > 0) & (macd_p < macd) & (macd_p > 0)
        zero_bear = (macd < 0) & (macd_p > macd) & (macd_p < 0)

        # --- Row 6: steep slope (adaptive) ---
        macd_slope = macd - macd_p
        vol_factor = (close.rolling(20, min_periods=1).std(ddof=0)
                      / close.rolling(20, min_periods=1).mean().replace(0, np.nan) * 10).fillna(0)
        steep_up_lvl = base_steep_s * (1 + vol_factor)
        steep_bull = macd_slope > steep_up_lvl
        steep_bear = macd_slope < -steep_up_lvl

        # --- Row 7: histogram slope ---
        hist_bull = (hist - hist_p) > 0
        hist_bear = (hist - hist_p) < 0

        # --- Row 8: macd-signal diff ---
        diff_bull = (macd - signal) >= diff_th_s
        diff_bear = (macd - signal) <= -diff_th_s

        # --- Row 9: momentum ROC14 ---
        mom = (macd - macd.shift(mom_len)) / macd.shift(mom_len).abs().replace(0, np.nan) * 100
        mom = mom.fillna(0)
        mom_bull = mom > 0
        mom_bear = mom < 0

        # --- Row 10: macd ROC 5 ---
        roc5 = (macd - macd.shift(5)) / macd.shift(5).abs().replace(0, 0.01)
        roc_bull = roc5.fillna(0) > 0
        roc_bear = roc5.fillna(0) < 0

        # --- Row 11: strong crossover ---
        cross_bull = cross_up & ((macd - signal) > diff_th_s)
        cross_bear = cross_dn & ((macd - signal) < -diff_th_s)

        # --- Row 12: STC zero cross ---
        stc_bull = (stcC > 0) & (stcC_p <= 0)
        stc_bear = (stcC < 0) & (stcC_p >= 0)
        stc_trend_bull = stcC > 0
        stc_trend_bear = stcC <= 0

        # --- Row 13: AO ---
        ao_bull = (ao > 0) & (ao > ao_p)
        ao_bear = (ao < 0) & (ao < ao_p)

        rows_b = [div_bull, hl_bull, bull_eng, retest_bull, zero_bull,
                  steep_bull, hist_bull, diff_bull, mom_bull, roc_bull,
                  cross_bull, stc_bull, ao_bull.fillna(False)]
        rows_r = [div_bear, hl_bear, bear_eng, retest_bear, zero_bear,
                  steep_bear, hist_bear, diff_bear, mom_bear, roc_bear,
                  cross_bear, stc_bear, ao_bear.fillna(False)]

        bull_cnt = pd.Series(0, index=df.index)
        bear_cnt = pd.Series(0, index=df.index)
        for b, r in zip(rows_b, rows_r):
            b = b.fillna(False)
            r = r.fillna(False)
            bull_cnt = bull_cnt + (b & ~r).astype(int)
            bear_cnt = bear_cnt + (r & ~b).astype(int)

        # Pine base triggers
        base_bull = cross_up & stc_trend_bull & diff_bull & mom_bull
        base_bear = cross_dn & stc_trend_bear & diff_bear & mom_bear

        if bool(p.get("require_base_trigger", True)):
            raw_bull = base_bull & (bull_cnt >= min_score)
            raw_bear = base_bear & (bear_cnt >= min_score)
        else:
            raw_bull = (bull_cnt >= min_score) & (bull_cnt > bear_cnt)
            raw_bear = (bear_cnt >= min_score) & (bear_cnt > bull_cnt)

        # Pine lastSignal state filter (no repeat same direction)
        if bool(p.get("use_last_signal_filter", True)):
            bull = pd.Series(False, index=df.index)
            bear = pd.Series(False, index=df.index)
            last = 0
            rb = raw_bull.values
            rs = raw_bear.values
            for i in range(len(df)):
                if rb[i] and last <= 0:
                    bull.iloc[i] = True
                    last = 1
                elif rs[i] and last >= 0:
                    bear.iloc[i] = True
                    last = -1
        else:
            bull, bear = raw_bull.fillna(False), raw_bear.fillna(False)

        sig = _empty_signals(df.index)
        sig.entries = bull | bear
        sig.direction = np.where(bull, 1, np.where(bear, -1, 0))
        return sig
