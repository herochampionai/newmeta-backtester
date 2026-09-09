"""Quad Stoch (same timeframe, multi inputs) — refined with Pine MultiStoch spec.

Pine-faithful params (stoch Length / K / D):
  S1 (9,3,1), S2 (14,3,1), S3 (40,1,4), S4 (60,1,10)
  -> our stochastic(k=Length, d=D, slowing=K)

Uses D lines for OB/OS + pivot divergence (exactly like Pine
stoch_divergence: pivotlow/high with left=right=div_len, range 5..60).

Trigger = Pine B/S re-entry labels:
  B: re-entry into all-OS (same-color OS->OS, cooldown>=3) + >=minDivs
     BULLISH divs accumulated in the OS window (opposite-only)
  S: mirror with BEARISH divs in OB window.

Optional permissive OBV/CVD gate (OBV|CVD rising or relevant div).
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


def _pivot_divergence(low: np.ndarray, high: np.ndarray, osc: np.ndarray,
                      left: int = 10, right: int = 10,
                      range_lower: int = 5, range_upper: int = 60):
    """Pine stoch_divergence port. Returns (bearCond, bullCond) bool arrays
    aligned to confirmation bar (no repaint offset)."""
    n = len(osc)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    # pivot indices (position of pivot bar) confirmed at i
    last_pl_pos = -1
    last_pl_osc = np.nan
    last_pl_price = np.nan
    last_ph_pos = -1
    last_ph_osc = np.nan
    last_ph_price = np.nan
    for i in range(left + right, n):
        piv = i - right
        # pivot low of osc?
        w_lo = osc[piv - left:piv + right + 1]
        w_hi = osc[piv - left:piv + right + 1]
        is_pl = osc[piv] == w_lo.min()
        is_ph = osc[piv] == w_hi.max()
        if is_pl:
            if last_pl_pos >= 0:
                bars = piv - last_pl_pos
                if range_lower <= bars <= range_upper:
                    if low[piv] < last_pl_price and osc[piv] > last_pl_osc:
                        bull[i] = True
            last_pl_pos = piv
            last_pl_osc = osc[piv]
            last_pl_price = low[piv]
        if is_ph:
            if last_ph_pos >= 0:
                bars = piv - last_ph_pos
                if range_lower <= bars <= range_upper:
                    if high[piv] > last_ph_price and osc[piv] < last_ph_osc:
                        bear[i] = True
            last_ph_pos = piv
            last_ph_osc = osc[piv]
            last_ph_price = high[piv]
    return bear, bull


class QuadStochSameTF(BaseStrategy):
    name = "quad_stoch"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        sig = _empty_signals(df.index)

        quad = p.get("quad_params", [(9, 1, 3), (14, 1, 3), (40, 4, 1), (60, 10, 1)])
        ds = []
        for k, d, sl in quad:
            _, d_s = ind.stochastic(df["high"], df["low"], df["close"], k, d, sl)
            ds.append(d_s.values)
        d1, d2, d3, d4 = ds

        os_ = float(p.get("oversold", 20))
        ob_ = float(p.get("overbought", 80))
        div_len = int(p.get("div_len", 10))
        rng_lo = int(p.get("div_range_lower", 5))
        rng_hi = int(p.get("div_range_upper", 60))

        low_v = df["low"].values
        high_v = df["high"].values

        bull_conds, bear_conds = [], []
        for d_s in ds:
            be, bu = _pivot_divergence(low_v, high_v, d_s, div_len, div_len, rng_lo, rng_hi)
            bull_conds.append(bu)
            bear_conds.append(be)

        all_ob = (d1 > ob_) & (d2 > ob_) & (d3 > ob_) & (d4 > ob_)
        all_os = (d1 < os_) & (d2 < os_) & (d3 < os_) & (d4 < os_)
        all_ob_v = np.asarray(all_ob)
        all_os_v = np.asarray(all_os)

        min_out = int(p.get("min_outside_bars", 3))
        cooldown_neutral = bool(p.get("cooldown_neutral", False))
        require_div = bool(p.get("require_div", True))
        min_divs = int(p.get("min_divs", 1))

        n = len(df)
        mode = str(p.get("entry_mode", "alert")).lower()
        div_win = int(p.get("div_window", 5))

        bull_any = pd.Series(bull_conds[0] | bull_conds[1] | bull_conds[2] | bull_conds[3], index=df.index)
        bear_any = pd.Series(bear_conds[0] | bear_conds[1] | bear_conds[2] | bear_conds[3], index=df.index)
        bull_recent = bull_any.rolling(div_win, min_periods=1).max() > 0
        bear_recent = bear_any.rolling(div_win, min_periods=1).max() > 0

        if mode == "alert":
            # Pine unified alerts: all in zone + any divergence (windowed for
            # confirmation delay). Works on H1 where episodes are 1-bar flickers.
            re_b = (all_os & bull_recent).values
            re_s = (all_ob & bear_recent).values
        else:
            # Pine B/S re-entry labels (strict; needs multi-bar episodes,
            # best on m5/m15). Same-color repeat + cooldown + opposite divs.
            re_b = np.zeros(n, dtype=bool)
            re_s = np.zeros(n, dtype=bool)
            prev_ob = False
            prev_os = False
            last_hl = 0
            bars_ob = 0
            bars_os = 0
            ob_win = False
            os_win = False
            ob_div = [False] * 4
            os_div = [False] * 4
            for i in range(n):
                cur_ob = bool(all_ob_v[i])
                cur_os = bool(all_os_v[i])
                start_ob = cur_ob and not prev_ob
                end_ob = prev_ob and not cur_ob
                start_os = cur_os and not prev_os
                end_os = prev_os and not cur_os

                neutral = (not cur_ob) and (not cur_os)
                bars_ob = bars_ob + 1 if (neutral if cooldown_neutral else not cur_ob) else 0
                bars_os = bars_os + 1 if (neutral if cooldown_neutral else not cur_os) else 0
                out_ob = bars_ob - (1 if (neutral if cooldown_neutral else not cur_ob) else 0)
                out_os = bars_os - (1 if (neutral if cooldown_neutral else not cur_os) else 0)

                base_ob = start_ob and (last_hl == 1) and (out_ob >= min_out)
                base_os = start_os and (last_hl == -1) and (out_os >= min_out)

                if end_ob:
                    ob_win = True
                    ob_div = [False] * 4
                    last_hl = 1
                if end_os:
                    os_win = True
                    os_div = [False] * 4
                    last_hl = -1
                if start_ob:
                    ob_win = False
                if start_os:
                    os_win = False
                if ob_win:
                    for j in range(4):
                        if bear_conds[j][i]:
                            ob_div[j] = True
                if os_win:
                    for j in range(4):
                        if bull_conds[j][i]:
                            os_div[j] = True
                ob_cnt = sum(1 for x in ob_div if x)
                os_cnt = sum(1 for x in os_div if x)
                ok_ob = (not require_div) or (ob_cnt >= min_divs)
                ok_os = (not require_div) or (os_cnt >= min_divs)

                if base_ob and ok_ob:
                    re_s[i] = True
                if base_os and ok_os:
                    re_b[i] = True

                prev_ob = cur_ob
                prev_os = cur_os

        buy = pd.Series(re_b, index=df.index)
        sell = pd.Series(re_s, index=df.index)

        if bool(p.get("use_volume_filter", False)):
            wmp = int(p.get("volume_wma_period", 5))
            vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
            obv_up, cvd_up, _ = ind.volume_rising(df["close"], df["high"], df["low"], vol, wmp)
            lb = int(p.get("divergence_lookback", 30))
            o_s = ind.obv(df["close"], vol)
            c_s = ind.cvd(df["close"], df["high"], df["low"], vol)
            obd_b, obd_r = _detect_divergence(df["close"], o_s, lb)
            cvd_b, cvd_r = _detect_divergence(df["close"], c_s, lb)
            buy = buy & (obv_up | cvd_up | obd_b | cvd_b)
            sell = sell & (obv_up | cvd_up | obd_r | cvd_r)

        sig.entries = buy | sell
        sig.direction = np.where(buy, 1, np.where(sell, -1, 0))
        return sig
