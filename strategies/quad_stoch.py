"""Quad Stoch (same timeframe, multi inputs).

SPLIT 1 of 2 — same-TF quad:
  Four stochastics with different inputs on the SAME timeframe:
    (5,3,3), (14,3,3), (40,4,6), (60,6,10)
  BUY:  all four in oversold AND bullish divergence on fastest
        AND fastest hooking up AND k/d cross
        AND volume ok (OBV rising OR CVD rising OR OBV/CVD bull div)
  SELL: mirror.
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


class QuadStochSameTF(BaseStrategy):
    name = "quad_stoch"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        sig = _empty_signals(df.index)

        quad = p.get("quad_params", [(5, 3, 3), (14, 3, 3), (40, 4, 6), (60, 6, 10)])
        ks, ds = [], []
        for k, d, sl in quad:
            k_s, d_s = ind.stochastic(df["high"], df["low"], df["close"], k, d, sl)
            ks.append(k_s)
            ds.append(d_s)
        k_fast, d_fast = ks[0], ds[0]

        os_ = float(p.get("oversold", 20))
        ob_ = float(p.get("overbought", 80))

        all_os = pd.Series(True, index=df.index)
        all_ob = pd.Series(True, index=df.index)
        for k_s in ks:
            all_os = all_os & (k_s < os_)
            all_ob = all_ob & (k_s > ob_)

        lb = int(p.get("divergence_lookback", 30))
        bull_div, bear_div = _detect_divergence(df["close"], k_fast, lb)

        hook_up = k_fast > k_fast.shift(1)
        hook_dn = k_fast < k_fast.shift(1)
        cross_up = (k_fast > d_fast) & (k_fast.shift(1) <= d_fast.shift(1))
        cross_dn = (k_fast < d_fast) & (k_fast.shift(1) >= d_fast.shift(1))

        if bool(p.get("use_volume_filter", True)):
            wmp = int(p.get("volume_wma_period", 5))
            vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
            obv_up, cvd_up, _ = ind.volume_rising(df["close"], df["high"], df["low"], vol, wmp)
            o_series = ind.obv(df["close"], vol)
            c_series = ind.cvd(df["close"], df["high"], df["low"], vol)
            obv_bd, obv_rd = _detect_divergence(df["close"], o_series, lb)
            cvd_bd, cvd_rd = _detect_divergence(df["close"], c_series, lb)
            vol_buy = obv_up | cvd_up | obv_bd | cvd_bd
            vol_sell = obv_up | cvd_up | obv_rd | cvd_rd
        else:
            vol_buy = pd.Series(True, index=df.index)
            vol_sell = pd.Series(True, index=df.index)

        buy = (all_os & bull_div & hook_up & cross_up
               & (k_fast < 50) & vol_buy)
        sell = (all_ob & bear_div & hook_dn & cross_dn
                & (k_fast > 50) & vol_sell)

        # Permissive fallback (off by default): if strict combo gives nothing,
        # allow hook+cross in zone with volume pass.
        if not (buy | sell).any() and bool(p.get("fallback_on_empty", False)):
            buy = (k_fast < os_) & hook_up & cross_up & vol_buy
            sell = (k_fast > ob_) & hook_dn & cross_dn & vol_sell

        sig.entries = buy | sell
        sig.direction = np.where(buy, 1, np.where(sell, -1, 0))
        return sig
