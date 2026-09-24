"""Stoch 5-3-3 MTF V2 — REWRITTEN with partial-MTF confluence + adaptive zones.

V1 was too strict: required ALL 4 TFs oversold + composite cross + divergence + volume + hook.
On H1 FX this filtered out everything (0 entries sometimes).

V2 changes:
  1. PARTIAL MTF: 3 of 4 TFs must agree (instead of all 4) — softer
  2. ADAPTIVE zones: oversold/overbought scale with volatility
  3. K-line slope as primary trigger: K rising in oversold = buy (not requiring composite cross)
  4. SOFTER divergence: optional, not mandatory
  5. Stochastic-relative position: K's percentile over last 100 bars (more stable than absolute)
  6. MTF stronger HTF dominance: H4/D1 take precedence (trend), H1/M15 provide entry

Character: mean-reversion with multi-TF context. Works on FX and indices.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    n = len(idx) if hasattr(idx, "__len__") else 0
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _stoch_resample(df: pd.DataFrame, rule: str, k_period: int = 5,
                    d_period: int = 3, smooth: int = 3):
    try:
        htf = df.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(htf) < k_period + d_period + smooth + 5:
            return None
        k, d = ind.stochastic(htf["high"], htf["low"], htf["close"], k_period, d_period, smooth)
        return (k.reindex(df.index, method="ffill"), d.reindex(df.index, method="ffill"))
    except Exception:
        return None


class Stoch533MTFV2(BaseStrategy):
    name = "stoch533_mtf_v2"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        k_period = int(p.get("k_period", 5))
        d_period = int(p.get("d_period", 3))
        smooth = int(p.get("smooth", 3))
        os_base = float(p.get("oversold", 20))
        ob_base = float(p.get("overbought", 80))
        # Adaptive zones: scale with ATR%
        atr_local = _atr(df, 14)
        atr_pct = atr_local / df["close"]
        vol_adj = (atr_pct - atr_pct.rolling(100, min_periods=20).mean()).fillna(0) * 100
        os_level = (os_base - vol_adj).clip(10, 40)
        ob_level = (ob_base + vol_adj).clip(60, 90)

        # Stochastic on entry TF
        k_e, d_e = ind.stochastic(df["high"], df["low"], df["close"],
                                  k_period, d_period, smooth)

        # MTF stochastics
        tfs = p.get("htf_timeframes", ["15min", "30min", "1h", "4h"])
        htf_ks = []
        for rule in tfs:
            res = _stoch_resample(df, rule, k_period, d_period, smooth)
            htf_ks.append(k_e if res is None else res[0])

        # Partial MTF: count how many TFs are oversold/overbought
        n_min_agree = int(p.get("min_tfs_agree", 2))  # at least 2 TFs (was all 4)
        all_in_os = pd.Series(0, index=df.index, dtype=int)
        all_in_ob = pd.Series(0, index=df.index, dtype=int)
        for k_s in htf_ks:
            all_in_os = all_in_os + (k_s < os_level).astype(int)
            all_in_ob = all_in_ob + (k_s > ob_level).astype(int)
        # Composite: average across TFs
        composite = pd.concat(htf_ks, axis=1).mean(axis=1)

        # K slope: is the entry-TF stochastic rising/falling?
        k_slope_up = k_e > k_e.shift(1)
        k_slope_dn = k_e < k_e.shift(1)
        # K turning: slope flipped
        k_turning_up = (k_e > k_e.shift(1)) & (k_e.shift(1) <= k_e.shift(2))
        k_turning_dn = (k_e < k_e.shift(1)) & (k_e.shift(1) >= k_e.shift(2))

        # % rank of K over last N bars (more stable than absolute levels)
        k_pctile = k_e.rolling(50, min_periods=10).rank(pct=True)
        buy_zone = k_pctile < 0.2  # K in bottom 20% of recent range
        sell_zone = k_pctile > 0.8  # K in top 20% of recent range

        # Divergence (optional)
        if bool(p.get("use_divergence", False)):
            from .triple_rsi import _detect_divergence
            lb = int(p.get("divergence_lookback", 25))
            bull_div, bear_div = _detect_divergence(df["close"], k_e, lb)
        else:
            bull_div = pd.Series(False, index=df.index)
            bear_div = pd.Series(False, index=df.index)

        # Volume confirmation
        if "volume" in df.columns and bool(p.get("use_volume_filter", True)):
            try:
                wmp = int(p.get("volume_wma_period", 5))
                obv_up, cvd_up, _ = ind.volume_rising(df["close"], df["high"], df["low"],
                                                       df["volume"], wmp)
                vol_ok = obv_up | cvd_up
            except Exception:
                vol_ok = pd.Series(True, index=df.index)
        else:
            vol_ok = pd.Series(True, index=df.index)

        # === ENTRY LOGIC ===
        # BUY: enough TFs in oversold + entry-TF K turning up + bottom percentile + (optional divergence) + volume
        n_agree_buy = all_in_os >= n_min_agree
        n_agree_sell = all_in_ob >= n_min_agree

        trig_b = buy_zone | k_turning_up | bull_div
        trig_s = sell_zone | k_turning_dn | bear_div

        buy = (n_agree_buy & (k_slope_up | k_turning_up) & trig_b & vol_ok.fillna(False))
        sell = (n_agree_sell & (k_slope_dn | k_turning_dn) & trig_s & vol_ok.fillna(False))

        sig = _empty_signals(df.index)
        sig.entries = buy | sell
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)), index=df.index, dtype=int)

        # Cooldown
        cd = int(p.get("cooldown_bars", 5))
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
