"""Triple RSI strategy — three RSIs (fast/mid/slow) + divergence.

This was discussed but never implemented — now added.

Logic:
  BUY:  rsi_fast in oversold AND rsi_mid < 50 AND rsi_slow < 55
        AND fast hooking up AND (bullish divergence OR deep oversold)
  SELL: mirror.

Params:
  rsi_fast=7, rsi_mid=14, rsi_slow=28
  oversold=30, overbought=70
  use_divergence=True, divergence_lookback=30
  require_divergence=False  (if True: divergence mandatory, very few signals)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


def _detect_divergence(price: pd.Series, osc: pd.Series, lookback: int = 30):
    idx = price.index
    bull = pd.Series(False, index=idx)
    bear = pd.Series(False, index=idx)
    if lookback < 2 or len(price) <= lookback:
        return bull, bear
    p = price.values
    o = osc.values
    n = len(p)
    for i in range(lookback, n):
        win_p = p[i - lookback:i]
        win_o = o[i - lookback:i]
        prev_min_idx = int(win_p[:-1].argmin())
        prev_max_idx = int(win_p[:-1].argmax())
        if p[i] < win_p[prev_min_idx] and o[i] > win_o[prev_min_idx]:
            bull.iloc[i] = True
        if p[i] > win_p[prev_max_idx] and o[i] < win_o[prev_max_idx]:
            bear.iloc[i] = True
    return bull, bear


class TripleRSIStrategy(BaseStrategy):
    name = "triple_rsi"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        pf = int(p.get("rsi_fast", 7))
        pm = int(p.get("rsi_mid", 14))
        ps = int(p.get("rsi_slow", 28))
        os_ = float(p.get("oversold", 30))
        ob_ = float(p.get("overbought", 70))

        rf = ind.rsi(df["close"], pf)
        rm = ind.rsi(df["close"], pm)
        rs = ind.rsi(df["close"], ps)

        hook_up = rf > rf.shift(1)
        hook_dn = rf < rf.shift(1)

        sig = _empty_signals(df.index)
        if bool(p.get("use_divergence", True)):
            lb = int(p.get("divergence_lookback", 30))
            bull_div, bear_div = _detect_divergence(df["close"], rm, lb)
        else:
            bull_div = pd.Series(False, index=df.index)
            bear_div = pd.Series(False, index=df.index)

        req_div = bool(p.get("require_divergence", False))
        if req_div:
            trig_b = bull_div
            trig_s = bear_div
        else:
            trig_b = (rf < os_) | bull_div
            trig_s = (rf > ob_) | bear_div

        buy = trig_b & (rm < 50) & (rs < 55) & hook_up
        sell = trig_s & (rm > 50) & (rs > 45) & hook_dn

        sig.entries = buy | sell
        sig.direction = np.where(buy, 1, np.where(sell, -1, 0))
        return sig
