"""Indicators ported 1:1 from MQL5 / MT5 specs.
AC + AO are NOT in pandas-ta / ta — implemented manually per Bill Williams."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()


def ao(high: pd.Series, low: pd.Series, fast: int = 5, slow: int = 34) -> pd.Series:
    """Awesome Oscillator = SMA(median, fast) - SMA(median, slow).
    Median = (high + low) / 2."""
    med = (high + low) / 2
    return _sma(med, fast) - _sma(med, slow)


def ac(high: pd.Series, low: pd.Series, ao_period: int = 5) -> pd.Series:
    """Accelerator/Decelerator = AO - SMA(AO, ao_period)."""
    a = ao(high, low)
    return a - _sma(a, ao_period)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index. Returns (adx, plus_di, minus_di).
    Implementation matches MT5 iADX semantics: Wilder smoothing."""
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_s = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx_s, plus_di, minus_di


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index. Matches MT5 iMFI(VOLUME_TICK)."""
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff()
    pos_mf = np.where(delta > 0, mf, 0.0)
    neg_mf = np.where(delta < 0, mf, 0.0)
    pos_sum = pd.Series(pos_mf, index=high.index).rolling(period, min_periods=1).sum()
    neg_sum = pd.Series(neg_mf, index=high.index).rolling(period, min_periods=1).sum()
    mr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - 100 / (1 + mr)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD = EMA(fast) - EMA(slow), signal = EMA(macd, signal), hist = macd - signal."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    m = ema_fast - ema_slow
    s = m.ewm(span=signal, adjust=False).mean()
    h = m - s
    return m, s, h


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k: int = 5, d: int = 3, slowing: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator. MT5 STO_LOWHIGH mode."""
    ll = low.rolling(k, min_periods=1).min()
    hh = high.rolling(k, min_periods=1).max()
    raw_k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k_s = raw_k.rolling(slowing, min_periods=1).mean() if slowing > 1 else raw_k
    d_s = k_s.rolling(d, min_periods=1).mean()
    return k_s, d_s


def bollinger(close: pd.Series, period: int = 20, dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands. Returns (mid, upper, lower)."""
    mid = close.rolling(period, min_periods=1).mean()
    sd = close.rolling(period, min_periods=1).std(ddof=0)
    return mid, mid + dev * sd, mid - dev * sd


def dem(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    """DeMarker indicator. MT5 iDeMarker returns 0..1; EA multiplies by 100 internally."""
    up_max = (high - high.shift()).clip(lower=0)
    dn_max = (low.shift() - low).clip(lower=0)
    up_sum = up_max.rolling(period, min_periods=1).sum()
    dn_sum = dn_max.rolling(period, min_periods=1).sum()
    return up_sum / (up_sum + dn_sum).replace(0, np.nan)


def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Force Index = EMA((close - prev_close) * volume, period).
    MT5 iForce defaults to period=13, MODE_EMA. EA scales by ×20."""
    raw = (close - close.shift(1)) * volume
    return raw.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder). Matches MT5 iRSI.
    Returns 0..100 series."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def linear_regression_slope(series: pd.Series, lookback: int = 5) -> pd.Series:
    """Rolling linear regression slope (least squares fit). Matches MQL5 MFI slope calc.
    For each window of size `lookback`, computes slope of y = m*x + b over x=0..n-1."""
    n = lookback
    sum_x = (n - 1) * n / 2
    sum_x2 = sum(k * k for k in range(n))
    denominator = n * sum_x2 - sum_x * sum_x

    def _slope(arr):
        if len(arr) < n:
            return np.nan
        sum_y = arr.sum()
        sum_xy = (np.arange(n) * arr).sum()
        if denominator == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denominator

    # Use rolling apply on numpy arrays
    vals = series.values
    out = np.full(len(vals), np.nan)
    for i in range(n - 1, len(vals)):
        out[i] = _slope(vals[i - n + 1:i + 1])
    return pd.Series(out, index=series.index)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume. Matches MT5 iOBV."""
    direction = np.where(close > close.shift(1), 1.0,
                         np.where(close < close.shift(1), -1.0, 0.0))
    return pd.Series(direction * volume.fillna(0).values,
                     index=close.index).cumsum()


def cvd(close: pd.Series, high: pd.Series, low: pd.Series,
        volume: pd.Series) -> pd.Series:
    """Cumulative Volume Delta (proxy).

    MT5 has no native CVD; we approximate with close-location-value:
      clv = ((close-low) - (high-close)) / (high-low)
      flow = clv * volume
    Cumulative sum = CVD proxy. Rising CVD = buyers aggressive.
    """
    rng = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / rng
    clv = clv.fillna(0.0)
    flow = clv * volume.fillna(0)
    return flow.cumsum()


def wma(series: pd.Series, period: int = 5) -> pd.Series:
    """Weighted moving average (linear weights 1..n)."""
    w = np.arange(1, period + 1, dtype=float)
    return series.rolling(period, min_periods=1).apply(
        lambda x: float(np.dot(x, w[-len(x):]) / w[-len(x):].sum()),
        raw=True)


def volume_rising(close: pd.Series, high: pd.Series, low: pd.Series,
                  volume: pd.Series, wma_period: int = 5
                  ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Very permissive OBV/CVD rising filter.

    Returns (obv_rising, cvd_rising, pass_filter) where
      obv_rising = WMA5(OBV) rising
      cvd_rising = WMA5(CVD) rising
      pass_filter = obv_rising | cvd_rising
    Only blocks when BOTH are falling.
    """
    o = obv(close, volume)
    c = cvd(close, high, low, volume)
    ow = wma(o, wma_period)
    cw = wma(c, wma_period)
    obv_up = ow > ow.shift(1)
    cvd_up = cw > cw.shift(1)
    return obv_up.fillna(False), cvd_up.fillna(False), (obv_up | cvd_up).fillna(True)