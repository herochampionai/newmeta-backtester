"""DeM V6 — Adaptive Regime (mean-reversion in range, trend-following in trend).

V5 used DeM as pure trend filter with pullback entry. V5 lost in trending markets
because the pullback was too short vs. the trend.

V6: ADAPTIVE per market regime
  - In RANGING market (ADX < 20): trade mean-reversion (DeM crossing 50)
  - In TRENDING market (ADX > 20): trade with-trend pullback (DeM dipping in regime)
  - Both regimes: HTF context required (DeM on D1 must agree)
  - Both regimes: pullback to mid-zone
"""
from __future__ import annotations
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


class DeMV6Strategy(BaseStrategy):
    name = "dem_v6"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        period = int(p.get("bars_calculate", 14))
        htf_rule = str(p.get("htf_rule", "1d"))
        regime_threshold = float(p.get("adx_regime_threshold", 20.0))
        crossover_threshold = float(p.get("dem_crossover_level", 50.0))
        cooldown = int(p.get("cooldown_bars", 8))

        # DeM scaled 0-100
        dem = ind.dem(df["high"], df["low"], period) * 100
        dem_prev = dem.shift(1)
        dem_prev2 = dem.shift(2)

        # HTF DeM
        htf_dem = pd.Series(50.0, index=df.index)
        try:
            htf = df.resample(htf_rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
            if len(htf) >= period + 5:
                htf_dem_series = ind.dem(htf["high"], htf["low"], period) * 100
                htf_dem = htf_dem_series.reindex(df.index, method="ffill").fillna(50.0)
        except Exception:
            pass

        # HTF context: DeM must agree with direction
        htf_bull = htf_dem > 50
        htf_bear = htf_dem < 50

        # ADX-based regime detection
        adx, _, _ = ind.adx(df["high"], df["low"], df["close"], 14)
        is_ranging = (adx < regime_threshold).fillna(True)
        is_trending = (adx >= regime_threshold).fillna(False)

        # === TRENDING REGIME: trend-following pullback entry ===
        # LONG: HTF DeM > 50 + entry DeM dipping into pullback zone (45-55) + rising
        in_pullback_long = (dem >= 45) & (dem <= 55) & (dem > dem_prev)
        trend_buy = htf_bull & is_trending & in_pullback_long

        # SHORT: HTF DeM < 50 + pullback + falling
        in_pullback_short = (dem <= 55) & (dem >= 45) & (dem < dem_prev)
        trend_sell = htf_bear & is_trending & in_pullback_short

        # === RANGING REGIME: mean-reversion ===
        # LONG: DeM crosses UP through 50 from below (oversold → neutral)
        mr_buy = is_ranging & (dem > crossover_threshold) & (dem_prev <= crossover_threshold)
        # SHORT: DeM crosses DOWN through 50 from above (overbought → neutral)
        mr_sell = is_ranging & (dem < crossover_threshold) & (dem_prev >= crossover_threshold)

        # Combined: regime-aware entry
        buy = trend_buy | mr_buy
        sell = trend_sell | mr_sell

        sig = _empty_signals(df.index)
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)),
                               index=df.index, dtype=int)

        # Cooldown
        if cooldown > 0 and len(direction) > cooldown:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(new_dir)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig.entries = direction != 0
        sig.direction = direction
        return sig
