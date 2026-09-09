"""Stoch 5-3-3 MTF — same stochastic on multiple timeframes.

SPLIT 2 of 2 — multi-timeframe 5/3/3:
  Stochastic(5,3,3) computed on each of m15, m30, h1, h4 (resampled
  from entry TF, forward-filled). Finer-than-entry TFs fall back to
  entry-TF values so the strat also runs on H1-only data.
  BUY:  all four TFs oversold AND bullish divergence on entry
        AND entry hooking up AND k/d cross up AND (OBV rising OR CVD rising)
  SELL: mirror.

Precomputed columns htf_0_k..htf_3_k are honoured if present
(same convention as legacy mtf_stoch).
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
    """Stoch(5,3,3) on resampled `rule`, ffilled to df.index.

    Returns None if rule is finer than entry TF or not enough bars.
    """
    try:
        htf = df.resample(rule).agg({"open": "first", "high": "max",
                                     "low": "min", "close": "last"})
    except Exception:
        return None
    htf = htf.dropna()
    if len(htf) < 5 + 3 + 3 + 5:
        return None
    # Finer-than-entry guard: resampled bars must be fewer than entry bars
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

        # Entry-TF 5/3/3 (sniper)
        k_e, d_e = ind.stochastic(df["high"], df["low"], df["close"], 5, 3, 3)

        # HTF 5/3/3s
        htf_ks = []
        for i, rule in enumerate(tfs):
            col = f"htf_{i}_k"
            if col in df.columns:
                htf_ks.append(df[col])
                continue
            res = _htf_stoch_533(df, rule)
            if res is None:
                htf_ks.append(k_e)  # fallback: entry-TF value
            else:
                htf_ks.append(res[0])

        os_ = float(p.get("oversold", 20))
        ob_ = float(p.get("overbought", 80))

        all_os = pd.Series(True, index=df.index)
        all_ob = pd.Series(True, index=df.index)
        for k_s in htf_ks:
            all_os = all_os & (k_s < os_)
            all_ob = all_ob & (k_s > ob_)

        lb = int(p.get("divergence_lookback", 30))
        bull_div, bear_div = _detect_divergence(df["close"], k_e, lb)

        hook_up = k_e > k_e.shift(1)
        hook_dn = k_e < k_e.shift(1)
        cross_up = (k_e > d_e) & (k_e.shift(1) <= d_e.shift(1))
        cross_dn = (k_e < d_e) & (k_e.shift(1) >= d_e.shift(1))

        if bool(p.get("use_volume_filter", True)):
            wmp = int(p.get("volume_wma_period", 5))
            _, _, vol_pass = ind.volume_rising(df["close"], df["high"], df["low"],
                                              df["volume"] if "volume" in df.columns
                                              else pd.Series(1.0, index=df.index),
                                              wmp)
        else:
            vol_pass = pd.Series(True, index=df.index)

        buy = (all_os & bull_div & hook_up & cross_up
               & (k_e < 50) & vol_pass)
        sell = (all_ob & bear_div & hook_dn & cross_dn
                & (k_e > 50) & vol_pass)

        if not (buy | sell).any() and bool(p.get("fallback_on_empty", False)):
            buy = (k_e < os_) & hook_up & cross_up & vol_pass
            sell = (k_e > ob_) & hook_dn & cross_dn & vol_pass

        sig.entries = buy | sell
        sig.direction = np.where(buy, 1, np.where(sell, -1, 0))
        return sig
