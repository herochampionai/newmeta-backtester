"""BB + RSI divergence — mean-reversion only.

Per user spec:
  bb mean reversion + rsi div  → this file
  bb breakout/breakdown + force index → fbb.py

Logic:
  BUY:  price tags lower band (low <= lower) AND (rsi_fast oversold OR bullish div)
        AND rsi_mid < 50 AND fast hooking up
  SELL: price tags upper band (high >= upper) AND (rsi_fast overbought OR bearish div)
        AND rsi_mid > 50 AND fast hooking down

Params:
  bb_period=20, bb_dev=2.0
  rsi_fast=7, rsi_mid=14, rsi_slow=28
  oversold=30, overbought=70
  divergence_lookback=30, require_divergence=False
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


class BBRsiStrategy(BaseStrategy):
    name = "bb_rsi"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        bb_p = int(p.get("bb_period", 20))
        bb_d = float(p.get("bb_dev", 2.0))
        pf = int(p.get("rsi_fast", 7))
        pm = int(p.get("rsi_mid", 14))
        ps = int(p.get("rsi_slow", 28))
        os_ = float(p.get("oversold", 30))
        ob_ = float(p.get("overbought", 70))

        _, up, lo = ind.bollinger(df["close"], bb_p, bb_d)
        rf = ind.rsi(df["close"], pf)
        rm = ind.rsi(df["close"], pm)
        rs = ind.rsi(df["close"], ps)

        lb = int(p.get("divergence_lookback", 30))
        bull_div, bear_div = _detect_divergence(df["close"], rm, lb)

        hook_up = rf > rf.shift(1)
        hook_dn = rf < rf.shift(1)

        tag_lo = df["low"] <= lo
        tag_hi = df["high"] >= up

        if bool(p.get("require_divergence", False)):
            trig_b = bull_div
            trig_s = bear_div
        else:
            trig_b = (rf < os_) | bull_div
            trig_s = (rf > ob_) | bear_div

        buy = tag_lo & trig_b & (rm < 50) & (rs < 60) & hook_up
        sell = tag_hi & trig_s & (rm > 50) & (rs > 40) & hook_dn

        sig = _empty_signals(df.index)
        sig.entries = buy | sell
        sig.direction = np.where(buy, 1, np.where(sell, -1, 0))
        return sig
