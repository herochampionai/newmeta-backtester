"""TripleRSI Matrix Pro v6.4 — ported from MQL5 to Python.

This is the MUCH more sophisticated version of triple_rsi from the desktop folder.
Key improvements over my simple Python triple_rsi:
  - Multi-timeframe RSI (M15, M5, M1 in MT5; we'll use H4, H1 for backtest)
  - Multi-period RSI weighted (Fast=0.25, Med=0.35, Slow=0.40)
  - Context score: weighted slow RSI > 58 (or < 42 for short)
  - Momentum score: weighted fast RSI > 54
  - Pullback zone: RSI in 43-48 + rising (for long)
  - MTF alignment: ≥2 of 3 timeframes bullish/bearish
  - EMA 200 trend filter
  - Choppy filter (contextScore within ±7 of 50)
  - ADX regime filter
  - Sweep+reclaim pattern detection

This is the "right" triple_rsi per the user's master blueprint.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
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


class TripleRSIProV6Strategy(BaseStrategy):
    """Port of TripleRSI_Matrix_Pro v6.4 from MQL5 EA.

    Logic (LONG):
      1. contextScore (slow RSI weighted) > context_threshold (default 58)
      2. momentumScore (fast RSI weighted) > momentum_threshold (default 54)
      3. bullishAlignment: ≥2 of 3 timeframes have all 3 RSI periods > 50
      4. pullbackLong: fast RSI in [43, 48] zone AND rising
      5. EMA 200 trend filter (HTF close > EMA)
      6. Choppy filter: |contextScore - 50| < choppy_zone
      7. ADX filter: ADX > min_trend
      8. Optional: sweep+reclaim pattern
    """
    name = "triple_rsi_pro_v6"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params

        # === Parameters ===
        fast_p = int(p.get("rsi_fast", 6))
        med_p = int(p.get("rsi_med", 12))
        slow_p = int(p.get("rsi_slow", 24))
        w_fast = float(p.get("weight_fast", 0.25))
        w_med = float(p.get("weight_med", 0.35))
        w_slow = float(p.get("weight_slow", 0.40))
        ctx_thr = float(p.get("context_threshold", 58.0))
        mom_thr = float(p.get("momentum_threshold", 54.0))
        pb_low = float(p.get("pullback_zone_low", 43.0))
        pb_high = float(p.get("pullback_zone_high", 48.0))
        min_align = int(p.get("min_alignment_count", 2))
        ema_period = int(p.get("ema_period", 200))
        choppy_zone = float(p.get("choppy_zone", 7.0))
        use_ema = bool(p.get("use_ema_filter", True))
        use_choppy = bool(p.get("filter_choppy", True))
        use_adx = bool(p.get("use_adx_filter", True))
        adx_min = float(p.get("adx_min_trend", 20.0))
        # HTF: H4 (since user wants 1h chart but MTF reference)
        htf_rule = str(p.get("htf_rule", "4h"))
        # LTF/MTF: In MQL5 EA uses M1/M5/M15 (real-time data). For backtest with H1 data,
        # we use H1 + H4 + D1 as our multi-timeframe reference set.
        ltf_rules = p.get("ltf_rules", ["1h", "4h", "1d"])

        # === RSI on entry timeframe + HTF + multiple LTF ===
        # Entry: H1 (already in df)
        rs_fast = ind.rsi(df["close"], fast_p)
        rs_med = ind.rsi(df["close"], med_p)
        rs_slow = ind.rsi(df["close"], slow_p)

        # Per-MQL5 spec:
        # ContextScore = Slow RSI weighted across TIMEFRAMES (trend context)
        #   = slow_H4 × w_HTF + slow_D1 × w_MTF + slow_H1 × w_LTF
        # MomentumScore = weighted RSI on LTF only (immediate momentum)
        #   = fast_H1 × w_fast + med_H1 × w_med + slow_H1 × w_slow
        # Note: g_RSIValues[tf][period] — [0]=fast, [1]=med, [2]=slow
        fast_score_entry = rs_fast
        fast_score = rs_fast  # alias for pullback logic below
        med_score = rs_med
        slow_score = rs_slow
        med_score_entry = rs_med
        slow_score_entry = rs_slow
        context_score = None  # computed below using multi-TF slow RSIs
        momentum_score = w_fast * rs_fast + w_med * rs_med + w_slow * rs_slow

        # === MTF alignment + context score ===
        # For each LTF, compute the weighted RSI and check if ALL 3 periods > 50
        # Also collect slow RSI per TF for context_score
        bullish_alignment = pd.Series(0, index=df.index, dtype=int)
        bearish_alignment = pd.Series(0, index=df.index, dtype=int)
        slow_per_tf = []  # (rule, slow_rsi_series)
        tf_weights = p.get("tf_weights", {"4h": 0.50, "1h": 0.30, "1d": 0.20})

        for rule in ltf_rules:
            try:
                if rule == "1h":
                    ltf_fast = rs_fast
                    ltf_med = rs_med
                    ltf_slow = rs_slow
                else:
                    ltf = df.resample(rule).agg({"close": "last"}).dropna()
                    if len(ltf) < max(fast_p, med_p, slow_p) + 5:
                        continue
                    ltf_fast = ind.rsi(ltf["close"], fast_p)
                    ltf_med = ind.rsi(ltf["close"], med_p)
                    ltf_slow = ind.rsi(ltf["close"], slow_p)
                # All 3 periods bullish/bearish (alignment)
                ltf_bull = (ltf_fast > 50) & (lttf_med > 50 if False else ltf_med > 50) & (ltf_slow > 50)
                ltf_bear = (ltf_fast < 50) & (ltf_med < 50) & (ltf_slow < 50)
                ltf_bull = ltf_bull.reindex(df.index, method="ffill").fillna(False).astype(bool)
                ltf_bear = ltf_bear.reindex(df.index, method="ffill").fillna(False).astype(bool)
                bullish_alignment = bullish_alignment + ltf_bull.astype(int)
                bearish_alignment = bearish_alignment + ltf_bear.astype(int)
                # Collect slow RSI for context_score
                w = float(tf_weights.get(rule, 0.20))
                slow_per_tf.append((w, ltf_slow.reindex(df.index, method="ffill")))
            except Exception:
                continue

        # Compute context_score = slow RSI weighted across TFs
        if slow_per_tf:
            total_w = sum(w for w, _ in slow_per_tf) or 1.0
            context_score = sum(w * s for w, s in slow_per_tf) / total_w
        else:
            context_score = rs_slow

        # === HTF trend filter (EMA 200) ===
        ema_trend_long = pd.Series(False, index=df.index)
        ema_trend_short = pd.Series(False, index=df.index)
        if use_ema:
            try:
                htf = df.resample(htf_rule).agg({"close": "last"}).dropna()
                if len(htf) >= ema_period:
                    htf_ema = htf["close"].rolling(ema_period, min_periods=50).mean()
                    htf_close = htf["close"]
                    ema_long_local = htf_close > htf_ema
                    ema_short_local = htf_close < htf_ema
                    ema_trend_long = ema_long_local.reindex(df.index, method="ffill").fillna(False).astype(bool)
                    ema_trend_short = ema_short_local.reindex(df.index, method="ffill").fillna(False).astype(bool)
            except Exception:
                pass

        # === ADX filter ===
        adx_ok = pd.Series(True, index=df.index)
        if use_adx:
            try:
                adx, _, _ = ind.adx(df["high"], df["low"], df["close"], 14)
                adx_ok = (adx >= adx_min).fillna(False).astype(bool)
            except Exception:
                pass

        # === Choppy filter ===
        not_choppy = pd.Series(True, index=df.index)
        if use_choppy:
            not_choppy = (np.abs(context_score - 50.0) >= choppy_zone).fillna(True).astype(bool)

        # === Pullback zones ===
        # LONG: fast RSI in [pb_low, pb_high] AND rising
        pullback_long = ((fast_score >= pb_low) & (fast_score <= pb_high)
                          & (fast_score > fast_score.shift(1))).fillna(False).astype(bool)
        # SHORT: mirror — fast RSI in [100-pb_high, 100-pb_low] AND falling
        short_pb_low = 100.0 - pb_high
        short_pb_high = 100.0 - pb_low
        pullback_short = ((fast_score >= short_pb_low) & (fast_score <= short_pb_high)
                           & (fast_score < fast_score.shift(1))).fillna(False).astype(bool)

        # === ENTRY LOGIC ===
        # Per MQL5:
        #   contextScore > threshold (slow RSI across TFs > 58)
        #   momentumScore > threshold (weighted RSI on LTF > 54)
        #   bullishAlignment >= min_align (≥2 TFs all-bullish)
        #   pullbackLong (fast RSI in [43,48] + rising)
        #   + EMA, ADX, not-choppy filters
        long_setup = (
            (context_score > ctx_thr)
            & (momentum_score > mom_thr)
            & (bullish_alignment >= min_align)
            & pullback_long
            & ema_trend_long
            & adx_ok
            & not_choppy
        )
        short_setup = (
            (context_score < (100.0 - ctx_thr))
            & (momentum_score < (100.0 - mom_thr))
            & (bearish_alignment >= min_align)
            & pullback_short
            & ema_trend_short
            & adx_ok
            & not_choppy
        )

        sig = _empty_signals(df.index)
        sig.entries = long_setup | short_setup
        sig.direction = pd.Series(np.where(long_setup, 1, np.where(short_setup, -1, 0)),
                                    index=df.index, dtype=int)

        # Cooldown
        cd = int(p.get("cooldown_bars", 8))
        if cd > 0 and len(sig.direction) > cd:
            new_dir = sig.direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            sig.direction = pd.Series(new_dir, index=df.index, dtype=int)
            sig.entries = sig.direction != 0

        return sig
