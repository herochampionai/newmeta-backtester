"""Triple RSI V2 — REWRITTEN with adaptive thresholds + divergence + MTF confluence.

The original (V1) was too rigid: required oversold OR divergence, then MID<50, SLOW<55,
plus hook_up. On FX that's either too noisy (many false signals) or too restrictive (no trades).

V2 changes:
  1. ADAPTIVE thresholds: oversold/overbought scale with volatility (ATR%)
  2. MTF H4 confirmation: HTF RSI agrees with LTF signal
  3. STRICTER divergence: require both RSI divergence AND price pattern
  4. SLOPE filter: RSI must be turning (delta > 0 for buy, < 0 for sell)
  5. Multi-RSI confluence: require min 2 of 3 RSIs to agree on direction
  6. Volume confirmation: OBV rising OR CVD rising

Character: reversal with adaptive zones. Works on both FX and indices.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _detect_divergence_v2(price: pd.Series, osc: pd.Series, lookback: int = 20,
                          min_separation: int = 5):
    """Vectorized divergence: compare current bar to rolling pivot in window.
    Returns (bull_div, bear_div)."""
    n = len(price)
    bull = pd.Series(False, index=price.index)
    bear = pd.Series(False, index=price.index)
    if lookback < 2 or n < lookback + min_separation:
        return bull, bear
    # Rolling min/max of price over lookback window (excluding current via shift)
    rolling_min_p = price.shift(1).rolling(lookback, min_periods=min_separation).min()
    rolling_max_p = price.shift(1).rolling(lookback, min_periods=min_separation).max()
    # At the rolling-min bar, what was the osc value? Use merge_asof or shift
    # Approximation: compare current price to rolling min/max directly
    price_lower_low = price < rolling_min_p  # current is lower than any prev bar in window
    price_higher_high = price > rolling_max_p
    # At rolling min, osc was — approximate via current osc at the min bar
    # Use a simple proxy: compare current osc to osc at the bars in window
    # For vectorization, use rolling_min/max of OSC shifted similarly
    # This is an approximation but works for divergence detection
    rolling_min_o = osc.shift(1).rolling(lookback, min_periods=min_separation).min()
    rolling_max_o = osc.shift(1).rolling(lookback, min_periods=min_separation).max()
    # Bullish divergence: price lower low + osc higher low (osc not at its min)
    # i.e., osc_current > rolling_min_o (osc didn't make new low) but price did
    bull = price_lower_low & (osc > rolling_min_o + 1.0)
    bear = price_higher_high & (osc < rolling_max_o - 1.0)
    return bull.fillna(False), bear.fillna(False)


class TripleRSIV2Strategy(BaseStrategy):
    name = "triple_rsi_v2"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        pf = int(p.get("rsi_fast", 7))
        pm = int(p.get("rsi_mid", 14))
        ps = int(p.get("rsi_slow", 28))
        os_base = float(p.get("oversold", 30))
        ob_base = float(p.get("overbought", 70))
        # Adaptive thresholds: in high vol, loosen; in low vol, tighten
        atr_local = _atr(df, 14)
        atr_pct = atr_local / df["close"]
        vol_adj = (atr_pct - atr_pct.rolling(100, min_periods=20).mean()).fillna(0) * 100
        os_level = (os_base - vol_adj).clip(15, 45)
        ob_level = (ob_base + vol_adj).clip(55, 85)

        rf = ind.rsi(df["close"], pf)
        rm = ind.rsi(df["close"], pm)
        rs = ind.rsi(df["close"], ps)

        # Slope of fast RSI (is it turning?)
        rf_slope = rf - rf.shift(2)
        rm_slope = rm - rm.shift(2)

        # Multi-RSI confluence: how many agree on direction
        rf_oversold = rf < os_level
        rf_overbought = rf > ob_level
        rm_bull = rm < 50
        rm_bear = rm > 50
        rs_bull = rs < 55
        rs_bear = rs > 45
        # Buy signal: at least 2 of 3 RSIs in bullish zone, fast turning up
        buy_rsi_confluence = rf_oversold.astype(int) + rm_bull.astype(int) + rs_bull.astype(int) >= 2
        sell_rsi_confluence = rf_overbought.astype(int) + rm_bear.astype(int) + rs_bear.astype(int) >= 2

        # Slope turning
        buy_slope = rf_slope > 0
        sell_slope = rf_slope < 0

        # Divergence (optional, requires explicit divergence lookback)
        if bool(p.get("use_divergence", True)):
            lb = int(p.get("divergence_lookback", 25))
            bull_div, bear_div = _detect_divergence_v2(df["close"], rm, lb)
        else:
            bull_div = pd.Series(False, index=df.index)
            bear_div = pd.Series(False, index=df.index)

        # Volume confirmation: OBV rising OR CVD rising
        if "volume" in df.columns:
            try:
                wmp = int(p.get("volume_wma_period", 5))
                obv_up, cvd_up, _ = ind.volume_rising(df["close"], df["high"], df["low"],
                                                       df["volume"], wmp)
                vol_buy_ok = obv_up | cvd_up
                vol_sell_ok = obv_up | cvd_up
            except Exception:
                vol_buy_ok = pd.Series(True, index=df.index)
                vol_sell_ok = pd.Series(True, index=df.index)
        else:
            vol_buy_ok = pd.Series(True, index=df.index)
            vol_sell_ok = pd.Series(True, index=df.index)

        # Multi-TF RSI: H4 RSI agrees with H1 signal
        if bool(p.get("use_mtf_confirm", True)):
            try:
                h4 = df.resample("4h").agg({"close": "last"}).dropna()
                h4_rsi = ind.rsi(h4["close"], 14)
                h4_rsi = h4_rsi.reindex(df.index, method="ffill")
                mtf_buy_ok = h4_rsi < 60  # H4 not in overbought
                mtf_sell_ok = h4_rsi > 40
            except Exception:
                mtf_buy_ok = pd.Series(True, index=df.index)
                mtf_sell_ok = pd.Series(True, index=df.index)
        else:
            mtf_buy_ok = pd.Series(True, index=df.index)
            mtf_sell_ok = pd.Series(True, index=df.index)

        # BUY: confluence ≥2 + slope turning up + (oversold OR divergence) + volume + MTF
        require_div = bool(p.get("require_divergence", False))
        if require_div:
            trig_b = bull_div
            trig_s = bear_div
        else:
            trig_b = rf_oversold | bull_div
            trig_s = rf_overbought | bear_div

        buy = (buy_rsi_confluence & buy_slope & trig_b
               & vol_buy_ok.fillna(False) & mtf_buy_ok.fillna(False))
        sell = (sell_rsi_confluence & sell_slope & trig_s
                & vol_sell_ok.fillna(False) & mtf_sell_ok.fillna(False))

        sig = _empty_signals(df.index)
        sig.entries = buy | sell
        # Build direction as Series first
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)), index=df.index, dtype=int)

        # Cooldown to avoid signal spam
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
