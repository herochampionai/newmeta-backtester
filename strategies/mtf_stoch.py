"""Quad Stochastic + Multi-Timeframe strategy.

Your spec:
  - Entry timeframe (sniper): 5/3/3 stochastic on 5-min bars
  - MTF context: 14/3/3, 40/4/6, 60/6/10 on higher TFs
  - BUY: all stochastics in oversold zone AND price made lower low BUT stochastic made
        higher low (bullish divergence) AND 5-min stochastic hooking up
  - SELL: mirror
  - Need divergence + meeting in/out of zone (k/d cross)

Implementation:
  - Use MTF context by resampling higher-TF bars to entry TF (forward-fill)
  - Divergence detection: scan lookback bars for price-vs-stoch turning points
  - Hooking up: k[0] > k[1]
  - Meeting: k crosses d (k[0] > d[0] AND k[1] <= d[1]) for "cross in zone"
            OR (k[0] < d[0] AND k[1] >= d[1]) for "cross out of zone"
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


class QuadStochStrategy(BaseStrategy):
    """Multi-timeframe Stochastic with quad stochastic entry sniper.

    Required inputs:
      - entry_timeframe: typically 5min; we assume data is at this TF
      - htf_timeframes: list of resample rules e.g. ['15min', '1h', '4h', '1d']
      - quad_params: list of (k, d, slowing) for each stochastic
      - htf_params: list of (k, d, slowing) corresponding to htf_timeframes
      - oversold, overbought levels
      - divergence_lookback: bars to look back for divergence

    NOTE: this strategy requires data resampled to entry TF. The pipeline
    handles resampling via the `resample_htf()` helper before signal generation.
    """
    name = "mtf_stoch"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        sig = _empty_signals(df.index)

        # 1. Entry TF: quad stochastic
        quad = p.get("quad_params", [(5, 3, 3), (14, 3, 3), (40, 4, 6), (60, 6, 10)])
        quad_k_list = []
        for k, d, sl in quad:
            k_s, d_s = ind.stochastic(df["high"], df["low"], df["close"], k, d, sl)
            quad_k_list.append(k_s)
        k5, k14, k40, k60 = quad_k_list

        # 2. HTF stochastics (assumed to be merged into df as columns already)
        #    We expect df to have `htf_<idx>_k` columns. If not, generate from df
        #    by resampling. For simplicity here we accept pre-resampled columns.
        htf_stochs = []
        for htf_idx in range(len(quad)):
            col_k = f"htf_{htf_idx}_k"
            if col_k in df.columns:
                htf_stochs.append(df[col_k])
            else:
                htf_stochs.append(quad_k_list[htf_idx])

        # 3. Threshold levels
        os = float(p.get("oversold", 20))
        ob = float(p.get("overbought", 80))

        # 4. Divergence on entry stochastic (k5 vs price)
        div_lb = int(p.get("divergence_lookback", 30))
        bull_div, bear_div = _detect_divergence(df["close"], k5, lookback=div_lb)

        # 5. Hook (current k > prev k) — momentum turning up
        k5_hook_up = k5 > k5.shift(1)
        k5_hook_dn = k5 < k5.shift(1)

        # 6. Meeting (k crosses d on entry TF only)
        k5_meet_up = (k5 > k14.shift(1)) & (k5.shift(1) <= k14.shift(1))  # 5m crossing 14-period
        # Simpler: k5 crosses its own signal line d
        # We re-compute stochastic with full output (k, d) for entry:
        _, d5_ = ind.stochastic(df["high"], df["low"], df["close"],
                                  quad[0][0], quad[0][1], quad[0][2])
        # Cross: k crosses d in zone (above is bullish when k<d, below when k>d)
        cross_in_zone = (k5 < d5_) & (k5.shift(1) >= d5_.shift(1))   # k dips into zone and crosses d
        cross_out_zone = (k5 > d5_) & (k5.shift(1) <= d5_.shift(1))  # k exits zone
        cross_up = (k5 > d5_) & (k5.shift(1) <= d5_.shift(1))
        cross_dn = (k5 < d5_) & (k5.shift(1) >= d5_.shift(1))

        # 7. HTF context: ALL HTF stochastics must be in OS zone (for buy)
        all_htf_os = pd.Series(True, index=df.index)
        all_htf_ob = pd.Series(True, index=df.index)
        for htf_k in htf_stochs:
            all_htf_os = all_htf_os & (htf_k < os)
            all_htf_ob = all_htf_ob & (htf_k > ob)

        # 8. Combined BUY signal
        buy = (
            all_htf_os                       # MTF all oversold
            & bull_div                       # bullish divergence on entry
            & k5_hook_up                     # hooking up
            & (cross_in_zone | cross_up)    # meeting/crossing in or out
            & (k5 < 50)                      # still below middle (room to run)
        )
        # 9. Combined SELL signal (mirror)
        sell = (
            all_htf_ob                       # MTF all overbought
            & bear_div                       # bearish divergence
            & k5_hook_dn
            & (cross_in_zone | cross_dn)
            & (k5 > 50)
        )

        sig.entries = buy | sell
        sig.direction = np.where(buy, 1, np.where(sell, -1, 0))
        return sig


def _detect_divergence(price: pd.Series, k: pd.Series, lookback: int = 30
                       ) -> tuple[pd.Series, pd.Series]:
    """Vectorized price/stoch divergence detection (regular bullish/bearish).

    Bullish: price made lower low in lookback, but k made higher low.
    Bearish: price made higher high, k made lower high.
    """
    idx = price.index
    bull = pd.Series(False, index=idx)
    bear = pd.Series(False, index=idx)
    if lookback < 2:
        return bull, bear
    p = price.values
    kk = k.values
    n = len(p)
    for i in range(lookback, n):
        win_p = p[i - lookback:i]
        win_k = kk[i - lookback:i]
        prev_min_p_idx = int(win_p[:-1].argmin())
        prev_max_p_idx = int(win_p[:-1].argmax())
        prev_min_p = win_p[prev_min_p_idx]
        prev_max_p = win_p[prev_max_p_idx]
        prev_min_k = win_k[prev_min_p_idx]
        prev_max_k = win_k[prev_max_p_idx]
        # Regular bullish: price lower low, k higher low
        if p[i] < prev_min_p and kk[i] > prev_min_k:
            bull.iloc[i] = True
        # Regular bearish: price higher high, k lower high
        if p[i] > prev_max_p and kk[i] < prev_max_k:
            bear.iloc[i] = True
    return bull, bear


def resample_htf(df: pd.DataFrame, htf_rules: list[str], quad_params: list
                 ) -> pd.DataFrame:
    """Add HTF stochastics as columns to df (forward-filled to entry TF).

    htf_rules: ['15min', '1h', '4h', '1d'] for each quad param level.
    quad_params: same length — list of (k, d, slowing).

    For each, resample to HTF, compute stochastic, then forward-fill to entry index.
    """
    df = df.copy()
    for i, rule in enumerate(htf_rules):
        k, d, sl = quad_params[i] if i < len(quad_params) else quad_params[-1]
        htf = df.resample(rule).agg({"open": "first", "high": "max",
                                       "low": "min", "close": "last"})
        htf = htf.dropna()
        if len(htf) < k + d + sl + 5:
            continue
        kk, dd = ind.stochastic(htf["high"], htf["low"], htf["close"], k, d, sl)
        # Forward-fill to entry TF
        df[f"htf_{i}_k"] = kk.reindex(df.index, method="ffill")
        df[f"htf_{i}_d"] = dd.reindex(df.index, method="ffill")
    return df


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))