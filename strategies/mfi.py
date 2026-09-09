"""MFI ENHANCED strategy. Mirror of MQL5 MFI_GetSignals (Gemini-fixed version).

Key facts:
  1. Case 3 is FIXED here (matches Multi Gemini.txt, line 5331-5333):
       Buy: MFI[0] > LevelOpenDn AND MFI[1] < LevelOpenDn  (UP-cross through lower)
       Sell: MFI[0] < LevelOpenUp AND MFI[1] > LevelOpenUp (DOWN-cross through upper)
     The original .mq5 has this backwards — it duplicates case 1.
  2. Slope is LINEAR REGRESSION slope over MFI_SlopeLookback bars (least squares).
  3. Slope filter logic:
       Buy passes if: StrongBullishSlope OR MFI_Slope > 0
       Sell passes if: StrongBearishSlope OR MFI_Slope < 0
     Slope filter alone kills if not in trend; divergence OVERRIDES it.
  4. 4 open cases + 4 close cases (close uses slope+divergence as enhancer).
  5. MFI divergence detection: 4 types (Bull, Bear, Hidden Bull, Hidden Bear).
     Vectorized port — checks if price makes new low but MFI doesn't (bullish div).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


class MFI_Strategy(BaseStrategy):
    name = "mfi"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        period = int(p.get("bars_calculate", 12))
        mfi_v = ind.mfi(df["high"], df["low"], df["close"], df["volume"], period)
        mfi_p = mfi_v.shift(1)

        # Slope (linear regression)
        use_slope = bool(p.get("use_slope_filter", False))
        slope_lb = int(p.get("slope_lookback", 5))
        min_slope = float(p.get("min_slope_strength", 3))
        if use_slope and slope_lb > 1:
            slope = ind.linear_regression_slope(mfi_v, slope_lb)
            slope_p = slope.shift(1)  # not used but available
            strong_bull_slope = slope > min_slope
            strong_bear_slope = slope < -min_slope
        else:
            slope = pd.Series(0.0, index=df.index)
            strong_bull_slope = pd.Series(True, index=df.index)
            strong_bear_slope = pd.Series(True, index=df.index)

        # Divergence (4 types)
        use_div = bool(p.get("use_divergence", False))
        use_hidden_div = bool(p.get("use_hidden_divergence", False))
        div_bars = int(p.get("divergence_bars", 10))
        if use_div or use_hidden_div:
            bull_div, bear_div, hidden_bull_div, hidden_bear_div = _detect_mfi_divergence(
                df["close"], mfi_v, lookback=div_bars, include_hidden=use_hidden_div)
        else:
            bull_div = pd.Series(False, index=df.index)
            bear_div = pd.Series(False, index=df.index)
            hidden_bull_div = pd.Series(False, index=df.index)
            hidden_bear_div = pd.Series(False, index=df.index)

        lev_open = float(p.get("level_open_orders", 90))
        lev_close = float(p.get("level_close_orders", 90))
        lev_up = lev_open
        lev_dn = 100 - lev_open
        lev_up_c = lev_close
        lev_dn_c = 100 - lev_close

        sig = _empty_signals(df.index)

        ot = int(p.get("open_orders_type", 3))
        if ot in (1, 2, 3, 4):
            buy, sell = _mfi_cases(ot, mfi_v, mfi_p, lev_up, lev_dn)
            # Apply slope filter
            if use_slope:
                buy = buy & (strong_bull_slope | (slope > 0))
                sell = sell & (strong_bear_slope | (slope < 0))
            # Divergence OVERRIDES slope filter
            if use_div:
                buy_override = bull_div | hidden_bull_div
                sell_override = bear_div | hidden_bear_div
                # Override means: even if slope killed, divergence re-enables
                buy = buy | (buy_override & ~buy & (buy | pd.Series(True, index=df.index)))
                # The above is awkward. Correct logic: if (slope killed) AND divergence → still allow
                # Simpler: re-evaluate from scratch with override
                raw_buy, raw_sell = _mfi_cases(ot, mfi_v, mfi_p, lev_up, lev_dn)
                if use_slope:
                    raw_buy_pass = raw_buy & (strong_bull_slope | (slope > 0))
                    raw_sell_pass = raw_sell & (strong_bear_slope | (slope < 0))
                else:
                    raw_buy_pass = raw_buy
                    raw_sell_pass = raw_sell
                # Override: if raw_signal AND divergence → force true
                buy = raw_buy_pass | (raw_buy & (bull_div | hidden_bull_div))
                sell = raw_sell_pass | (raw_sell & (bear_div | hidden_bear_div))
            sig.entries = buy | sell
            sig.direction = np.where(buy, 1, np.where(sell, -1, 0))

        ct = int(p.get("close_orders_type", 0))
        if ct in (1, 2, 3, 4):
            cb, cs = _mfi_cases(ct, mfi_v, mfi_p, lev_up_c, lev_dn_c)
            # Close enhancements (MQL5 lines 5397-5406)
            if use_slope:
                cb = cb | strong_bear_slope
                cs = cs | strong_bull_slope
            if use_div:
                cb = cb | bear_div
                cs = cs | bull_div
            sig.exits = cb | cs

        return sig


def _mfi_cases(open_type: int, mfi: pd.Series, mfi_p: pd.Series,
               lev_up: float, lev_dn: float) -> tuple[pd.Series, pd.Series]:
    """4 MFI cases. NOTE: case 3 is FIXED here (was duplicate of case 1 in original .mq5).
    Case 1: cross up through upper (entering OB) → buy / cross down through lower → sell
    Case 2: cross down through upper (exiting OB) → buy / cross up through lower → sell
    Case 3: trend-following — buy on UP-cross through lower, sell on DOWN-cross through upper
    Case 4: contrarian — buy on DOWN-cross through lower, sell on UP-cross through upper
    """
    z = pd.Series(False, index=mfi.index)
    if open_type == 1:
        # Contrarian entry: enter OB → buy (mean reversion), enter OS → sell
        buy = (mfi > lev_up) & (mfi_p < lev_up)
        sell = (mfi < lev_dn) & (mfi_p > lev_dn)
    elif open_type == 2:
        # Contrarian exit: exit OB → buy, exit OS → sell
        buy = (mfi < lev_up) & (mfi_p > lev_up)
        sell = (mfi > lev_dn) & (mfi_p < lev_dn)
    elif open_type == 3:
        # Trend-following (FIXED): buy on UP-cross through lower (exiting OS),
        # sell on DOWN-cross through upper (exiting OB)
        buy = (mfi > lev_dn) & (mfi_p < lev_dn)
        sell = (mfi < lev_up) & (mfi_p > lev_up)
    elif open_type == 4:
        # Contrarian fade: buy on DOWN-cross through lower (entering OS),
        # sell on UP-cross through upper (entering OB)
        buy = (mfi < lev_dn) & (mfi_p > lev_dn)
        sell = (mfi > lev_up) & (mfi_p < lev_up)
    else:
        buy = z.copy()
        sell = z.copy()
    return buy, sell


def _detect_mfi_divergence(close: pd.Series, mfi: pd.Series, lookback: int = 10,
                           include_hidden: bool = True) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Vectorized MFI divergence detection.

    Regular bullish: price makes lower low, MFI makes higher low → reversal up
    Regular bearish: price makes higher high, MFI makes lower high → reversal down
    Hidden bullish: price makes higher low, MFI makes lower low → trend continuation up
    Hidden bearish: price makes lower high, MFI makes higher high → trend continuation down

    Returns 4 bool Series aligned to df.index.
    """
    idx = close.index
    bull = pd.Series(False, index=idx)
    bear = pd.Series(False, index=idx)
    hbull = pd.Series(False, index=idx)
    hbear = pd.Series(False, index=idx)

    if lookback < 2:
        return bull, bear, hbull, hbear

    # Find local extrema in rolling windows
    close_arr = close.values
    mfi_arr = mfi.values
    n = len(close_arr)
    for i in range(lookback, n):
        win_close = close_arr[i - lookback:i + 1]
        win_mfi = mfi_arr[i - lookback:i + 1]
        # Use last point as potential divergence point
        last_close = close_arr[i]
        last_mfi = mfi_arr[i]
        # Find min/max in the window excluding last point
        prev_close_min = win_close[:-1].min()
        prev_close_max = win_close[:-1].max()
        prev_close_argmin = win_close[:-1].argmin()
        prev_close_argmax = win_close[:-1].argmax()
        prev_mfi_at_close_min = win_mfi[prev_close_argmin]
        prev_mfi_at_close_max = win_mfi[prev_close_argmax]

        # Regular bullish: price lower low, MFI higher low
        if last_close < prev_close_min and last_mfi > prev_mfi_at_close_min:
            bull.iloc[i] = True
        # Regular bearish: price higher high, MFI lower high
        if last_close > prev_close_max and last_mfi < prev_mfi_at_close_max:
            bear.iloc[i] = True
        # Hidden bullish: price higher low, MFI lower low
        if include_hidden:
            if last_close > prev_close_min and last_mfi < prev_mfi_at_close_min:
                hbull.iloc[i] = True
            # Hidden bearish: price lower high, MFI higher high
            if last_close < prev_close_max and last_mfi > prev_mfi_at_close_max:
                hbear.iloc[i] = True
    return bull, bear, hbull, hbear


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))