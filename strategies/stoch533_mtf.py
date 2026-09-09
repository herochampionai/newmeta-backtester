"""Stoch 5-3-3 MTF — same stochastic on multiple timeframes.

SPLIT 2 of 2 — multi-timeframe 5/3/3:
  Stochastic(5,3,3) on m15, m30, h1, h4 (resampled + ffilled).
  Finer-than-entry TFs fall back to entry-TF values (H1-only safe).

  BUY:  all four were oversold within N bars (in) AND composite
        average crossed above exit level (out, default 30)
        AND bullish divergence on fastest entry AND entry hooking up
        AND volume ok (OBV rising OR CVD rising OR OBV/CVD bull div)
  SELL: mirror (were overbought, composite crossed below 70,
        bear div, hooking down, OBV|CVD rising or bear div).

Precomputed columns htf_0_k..htf_3_k honoured if present.
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


def _htf_stoch_533(df: pd.DataFrame, rule: str):
    try:
        htf = df.resample(rule).agg({"open": "first", "high": "max",
                                     "low": "min", "close": "last"})
    except Exception:
        return None
    htf = htf.dropna()
    if len(htf) < 5 + 3 + 3 + 5:
        return None
    if len(htf) >= len(df):
        return None
    kk, dd = ind.stochastic(htf["high"], htf["low"], htf["close"], 5, 3, 3)
    return kk.reindex(df.index, method="ffill"), dd.reindex(df.index, method="ffill")


class Stoch533MTF(BaseStrategy):
    name = "stoch533_mtf"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        sig = _empty_signals(df.index)

        tfs = p.get("htf_timeframes", ["15min", "30min", "1h", "4h"])
        k_e, d_e = ind.stochastic(df["high"], df["low"], df["close"], 5, 3, 3)

        htf_ks = []
        for i, rule in enumerate(tfs):
            col = f"htf_{i}_k"
            if col in df.columns:
                htf_ks.append(df[col])
                continue
            res = _htf_stoch_533(df, rule)
            htf_ks.append(k_e if res is None else res[0])

        os_ = float(p.get("oversold", 20))
        ob_ = float(p.get("overbought", 80))
        exit_l = float(p.get("composite_exit_long", 30))
        exit_s = float(p.get("composite_exit_short", 70))
        n_inout = int(p.get("inout_lookback", 5))

        all_os = pd.Series(True, index=df.index)
        all_ob = pd.Series(True, index=df.index)
        for k_s in htf_ks:
            all_os = all_os & (k_s < os_)
            all_ob = all_ob & (k_s > ob_)

        composite = pd.concat(htf_ks, axis=1).mean(axis=1)
        # In-and-out: was inside zone within N bars, now crossing exit level
        was_os = (all_os | (composite < os_)).rolling(n_inout, min_periods=1).max() > 0
        was_ob = (all_ob | (composite > ob_)).rolling(n_inout, min_periods=1).max() > 0
        cross_out_os = (composite > exit_l) & (composite.shift(1) <= exit_l)
        cross_out_ob = (composite < exit_s) & (composite.shift(1) >= exit_s)

        lb = int(p.get("divergence_lookback", 30))
        bull_div, bear_div = _detect_divergence(df["close"], k_e, lb)

        hook_up = k_e > k_e.shift(1)
        hook_dn = k_e < k_e.shift(1)

        # Volume: OBV rising OR CVD rising OR relevant divergence
        vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
        if bool(p.get("use_volume_filter", True)):
            wmp = int(p.get("volume_wma_period", 5))
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

        buy = (was_os & cross_out_os & bull_div & hook_up
               & (k_e < 50) & vol_buy)
        sell = (was_ob & cross_out_ob & bear_div & hook_dn
                & (k_e > 50) & vol_sell)

        if not (buy | sell).any() and bool(p.get("fallback_on_empty", False)):
            buy = (was_os & cross_out_os & hook_up & vol_buy)
            sell = (was_ob & cross_out_ob & hook_dn & vol_sell)

        sig.entries = buy | sell
        sig.direction = np.where(buy, 1, np.where(sell, -1, 0))
        return sig
