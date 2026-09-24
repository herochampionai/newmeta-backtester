"""Combined 9-strategy crypto/LRC algo ported from the supplied PineScript."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()


def _linreg_value(s: pd.Series, n: int) -> pd.Series:
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def calc(arr):
        if len(arr) < n or denom == 0:
            return np.nan
        y = np.asarray(arr, dtype=float)
        slope = ((x - x_mean) * (y - y.mean())).sum() / denom
        intercept = y.mean() - slope * x_mean
        return intercept + slope * (n - 1)

    return s.rolling(n, min_periods=n).apply(calc, raw=True).ffill().fillna(s)


def _crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def _apply_strategy_cooldown(raw: pd.Series, cooldown: int) -> pd.Series:
    raw = raw.fillna(False).astype(bool)
    if cooldown <= 0:
        return raw
    out = pd.Series(False, index=raw.index)
    last = -10**9
    for i, flag in enumerate(raw.to_numpy()):
        if flag and i - last > cooldown:
            out.iat[i] = True
            last = i
    return out


class CryptoNineStrategy(BaseStrategy):
    name = "crypto_9"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]
        volume = df.get("volume", pd.Series(1.0, index=df.index)).fillna(1.0)

        length = int(p.get("length", 21))
        deviations = float(p.get("deviations", 2.5))
        breakout_lookback = int(p.get("breakout_lookback", 20))
        breakout_confirm_bars = int(p.get("breakout_confirm_bars", 1))
        rsi_oversold = float(p.get("rsi_oversold", 30))
        rsi_overbought = float(p.get("rsi_overbought", 70))
        cooldown = int(p.get("cooldown_bars", 2))
        adx_period = int(p.get("adx_period", 14))
        trend_adx = float(p.get("trend_adx", 22))
        meanrev_adx = float(p.get("meanrev_adx", 20))
        volume_threshold = float(p.get("volume_threshold", 1.2))

        enable_macd_filter = bool(p.get("enable_macd_filter", False))
        enable_volume_filter = bool(p.get("enable_volume_filter", False))
        enable_trend_filter = bool(p.get("enable_trend_filter", False))

        use_breakout = bool(p.get("use_breakout", True))
        use_reversal = bool(p.get("use_reversal", True))
        use_ma_cross = bool(p.get("use_ma_cross", False))
        use_fake_breakout = bool(p.get("use_fake_breakout", True))
        use_bounce = bool(p.get("use_bounce", True))
        use_breakout_upper = bool(p.get("use_breakout_upper", False))
        use_rejection = bool(p.get("use_rejection", True))
        use_breakout_lower = bool(p.get("use_breakout_lower", False))
        use_midline_reversal = bool(p.get("use_midline_reversal", False))

        lr = _linreg_value(close, length)
        lr_stdev = close.rolling(length, min_periods=1).std(ddof=0)
        lr_upper = lr + deviations * lr_stdev
        lr_lower = lr - deviations * lr_stdev

        rsi = ind.rsi(close, length)
        macd_line, signal_line, _ = ind.macd(close, 12, 26, 9)
        macd_bullish = macd_line > signal_line
        macd_bearish = macd_line < signal_line

        avg_volume = _sma(volume, 20)
        volume_confirmed = (volume > avg_volume * volume_threshold) if enable_volume_filter else pd.Series(True, index=df.index)

        fast_ma = _sma(close, length)
        slow_ma = _sma(close, length * 2)
        ma50 = _sma(close, 50)
        ma200 = _sma(close, 200)

        bullish_trend = (close > ma50) & (ma50 > ma200)
        bearish_trend = (close < ma50) & (ma50 < ma200)
        trend_ok_long = bullish_trend if enable_trend_filter else pd.Series(True, index=df.index)
        trend_ok_short = bearish_trend if enable_trend_filter else pd.Series(True, index=df.index)
        macd_ok_long = macd_bullish if enable_macd_filter else pd.Series(True, index=df.index)
        macd_ok_short = macd_bearish if enable_macd_filter else pd.Series(True, index=df.index)

        adx, _, _ = ind.adx(high, low, close, adx_period)
        trend_regime = adx >= trend_adx
        meanrev_regime = adx < meanrev_adx

        breakout_high = high.rolling(breakout_lookback, min_periods=1).max()
        breakout_low = low.rolling(breakout_lookback, min_periods=1).min()
        price_above_high = close > breakout_high.shift(1)
        price_below_low = close < breakout_low.shift(1)
        breakout_long_count = price_above_high.astype(int).groupby((~price_above_high).cumsum()).cumsum()
        breakout_short_count = price_below_low.astype(int).groupby((~price_below_low).cumsum()).cumsum()

        breakout_long = (breakout_long_count >= breakout_confirm_bars) & volume_confirmed & trend_ok_long & macd_ok_long & trend_regime
        breakout_short = (breakout_short_count >= breakout_confirm_bars) & volume_confirmed & trend_ok_short & macd_ok_short & trend_regime

        reversal_long = _crossover(rsi, pd.Series(rsi_oversold, index=df.index)) & trend_ok_long & volume_confirmed & macd_ok_long & meanrev_regime
        reversal_short = _crossunder(rsi, pd.Series(rsi_overbought, index=df.index)) & trend_ok_short & volume_confirmed & macd_ok_short & meanrev_regime

        maco_long = _crossover(fast_ma, slow_ma) & volume_confirmed & macd_ok_long & trend_regime
        maco_short = _crossunder(fast_ma, slow_ma) & volume_confirmed & macd_ok_short & trend_regime

        failed_breakdown_long = (low.shift(1) < breakout_low.shift(2)) & (close > breakout_low.shift(2))
        failed_breakout_short = (high.shift(1) > breakout_high.shift(2)) & (close < breakout_high.shift(2))
        fake_breakout_long = failed_breakdown_long & volume_confirmed & trend_ok_long & macd_ok_long & meanrev_regime
        fake_breakout_short = failed_breakout_short & volume_confirmed & trend_ok_short & macd_ok_short & meanrev_regime

        bounce_long = (low <= lr_lower) & (close > lr_lower) & (close > open_) & volume_confirmed & trend_ok_long & macd_ok_long & meanrev_regime
        breakout_upper_long = (close > lr_upper) & (close > open_) & volume_confirmed & trend_ok_long & macd_ok_long & trend_regime
        rejection_short = (high >= lr_upper) & (close < lr_upper) & (close < open_) & volume_confirmed & trend_ok_short & macd_ok_short & meanrev_regime
        breakout_lower_short = (close < lr_lower) & (close < open_) & volume_confirmed & trend_ok_short & macd_ok_short & trend_regime
        midline_reversal_long = _crossover(close, lr) & volume_confirmed & trend_ok_long & macd_ok_long & meanrev_regime
        midline_reversal_short = _crossunder(close, lr) & volume_confirmed & trend_ok_short & macd_ok_short & meanrev_regime

        long_parts = [
            use_breakout and breakout_long,
            use_reversal and reversal_long,
            use_ma_cross and maco_long,
            use_fake_breakout and fake_breakout_long,
            use_bounce and bounce_long,
            use_breakout_upper and breakout_upper_long,
            use_midline_reversal and midline_reversal_long,
        ]
        short_parts = [
            use_breakout and breakout_short,
            use_reversal and reversal_short,
            use_ma_cross and maco_short,
            use_fake_breakout and fake_breakout_short,
            use_rejection and rejection_short,
            use_breakout_lower and breakout_lower_short,
            use_midline_reversal and midline_reversal_short,
        ]

        long_raw = pd.Series(False, index=df.index)
        short_raw = pd.Series(False, index=df.index)
        for part in long_parts:
            if isinstance(part, pd.Series):
                long_raw |= _apply_strategy_cooldown(part, cooldown)
        for part in short_parts:
            if isinstance(part, pd.Series):
                short_raw |= _apply_strategy_cooldown(part, cooldown)

        both = long_raw & short_raw
        long_raw = long_raw & ~both
        short_raw = short_raw & ~both

        sig = _empty_signals(df.index)
        sig.entries = long_raw | short_raw
        sig.direction = pd.Series(np.where(long_raw, 1, np.where(short_raw, -1, 0)), index=df.index, dtype=int)
        return sig


PATTERN_NAMES = [
    "breakout", "reversal", "ma_cross", "fake_breakout", "bounce",
    "breakout_upper", "rejection", "breakout_lower", "midline_reversal",
]

DEFAULT_PARAMS: dict = {
    "length": 21, "deviations": 2.5, "breakout_lookback": 20,
    "breakout_confirm_bars": 1, "rsi_oversold": 30, "rsi_overbought": 70,
    "cooldown_bars": 2, "adx_period": 14, "trend_adx": 22,
    "meanrev_adx": 20, "volume_threshold": 1.2,
    "enable_macd_filter": False, "enable_volume_filter": False,
    "enable_trend_filter": False,
    "use_breakout": True, "use_reversal": True, "use_ma_cross": False,
    "use_fake_breakout": True, "use_bounce": True,
    "use_breakout_upper": False, "use_rejection": True,
    "use_breakout_lower": False, "use_midline_reversal": False,
}

PATTERN_VARIANTS: dict[str, dict] = {}
_ALL_PATTERNS = [f"use_{p}" for p in PATTERN_NAMES]
for _pname in PATTERN_NAMES:
    _variant_name = f"crypto_9_{_pname}"
    _base = dict(DEFAULT_PARAMS)
    for _all_p in _ALL_PATTERNS:
        _base[_all_p] = False
    _base[f"use_{_pname}"] = True
    PATTERN_VARIANTS[_variant_name] = _base

