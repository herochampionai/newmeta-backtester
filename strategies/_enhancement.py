"""Strategy enhancement framework — applies regime + session + candle + MTF + Optuna filters.

Wraps any base strategy with optional filters:
  1. Regime (ADX-based)        — disable when regime mismatches strategy character
  2. Session                   — only London/NY (FX) or US session (indices)
  3. Multi-timeframe trend     — HTF trend alignment
  4. Volatility (ATR)          — avoid low-vol noise + high-vol shock
  5. Candlestick pattern       — no entry on indecision/doji
  6. Volume                    — confirm with volume
  7. Day-of-week               — skip Mon/Fri gap risk
  8. Adaptive parameters       — per-ticker threshold tuning
  9. Open-type selection       — which of 8 AC cases (or equivalent) fits character
  10. Optuna search            — find best threshold combo per ticker

Advanced confluence filters:
  11. Fibonacci                — price near key fib level (38.2 / 50 / 61.8)
  12. FVG (Fair Value Gap)     — price entering unfilled gap
  13. IFVG (Inverse FVG)       — invalidated FVG acting as S/R
  14. MSS (Market Structure Shift) — recent break of swing structure
  15. Killzone                 — ICT killzone filter (London/NY open)
  16. Confluence score         — count active confluences, require min N
  17. Volume increase          — current vol > N * avg
  18. Momentum increase        — RSI/MACD acceleration
  19. Divergence               — price vs RSI/MACD divergence

Each filter returns (buy_keep, sell_keep) — bool Series that AND the base signals.
"""
from __future__ import annotations

import warnings
from typing import Callable

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from . import indicators as ind
from ._base import BaseStrategy, Signals

# ─── Filter primitives ────────────────────────────────────────────────────────

def regime_adx(df: pd.DataFrame, adx_period: int = 14,
               enable_above: float = 99.0, enable_below: float = -1.0,
               invert: bool = False) -> tuple[pd.Series, pd.Series]:
    """Returns (keep_all, keep_all) where keep_all is True only when ADX is in the configured band.
    invert=True means REVERSAL strategy: enable when ADX is LOW (ranging)."""
    adx, _, _ = ind.adx(df["high"], df["low"], df["close"], adx_period)
    if invert:
        # Reversal: enable when ranging (low ADX), disable when trending
        keep = adx <= enable_above
    else:
        # Trend: enable when trending (high ADX), disable when ranging
        keep = (adx >= enable_below) & (adx <= enable_above)
    return keep, keep


def session_filter(df: pd.DataFrame,
                   utc_hours: tuple[int, ...] = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
                   label: str = "London+NY") -> tuple[pd.Series, pd.Series]:
    """Only fire during configured UTC hours. London ~7-16, NY ~12-21, overlap ~12-16."""
    hour = pd.Series(df.index.hour, index=df.index)
    keep = hour.isin(utc_hours)
    return keep, keep


def volatility_atr(df: pd.DataFrame, atr_period: int = 14,
                   min_atr_pct: float = 0.0005,
                   max_atr_pct: float = 0.0050) -> tuple[pd.Series, pd.Series]:
    """Filter out dead markets (low ATR) and shock events (extreme ATR).
    ATR% = ATR / close."""
    atr = _atr(df, atr_period)
    atr_pct = atr / df["close"]
    keep = (atr_pct >= min_atr_pct) & (atr_pct <= max_atr_pct)
    return keep, keep


def candle_filter(df: pd.DataFrame, min_body_pct: float = 0.3,
                  max_upper_wick_pct: float = 0.5) -> tuple[pd.Series, pd.Series]:
    """Require a real candle — body > min_body_pct of total range.
    Optional: reject if upper wick > max_upper_wick_pct (for long entries)."""
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, 1e-9)
    body_pct = body / rng
    keep = body_pct >= min_body_pct
    return keep, keep


def mtf_trend(df: pd.DataFrame, sma_fast: int = 50, sma_slow: int = 200,
              atr_period: int = 14) -> tuple[pd.Series, pd.Series]:
    """Approximate HTF trend on H1: use a long SMA filter.
    For Buy: close > sma_slow (long-term uptrend).
    For Sell: close < sma_slow (long-term downtrend)."""
    sma_s = df["close"].rolling(sma_slow, min_periods=1).mean()
    keep_buy = df["close"] > sma_s
    keep_sell = df["close"] < sma_s
    return keep_buy, keep_sell


def volume_filter(df: pd.DataFrame, lookback: int = 20, mult: float = 0.8) -> tuple[pd.Series, pd.Series]:
    """Require volume > mult * avg_volume. Skip dead-volume bars."""
    if "volume" not in df.columns:
        return pd.Series(True, index=df.index), pd.Series(True, index=df.index)
    avg_vol = df["volume"].rolling(lookback, min_periods=1).mean()
    keep = df["volume"] >= avg_vol * mult
    return keep, keep


def day_filter(df: pd.DataFrame, blocked_days: tuple[int, ...] = (0, 4)) -> tuple[pd.Series, pd.Series]:
    """Skip Monday (0) and Friday (4) — gap risk days."""
    keep = ~df.index.dayofweek.isin(blocked_days)
    return keep, keep


def first_last_hour(df: pd.DataFrame, skip_first: int = 1, skip_last: int = 1) -> tuple[pd.Series, pd.Series]:
    """Skip first N hours and last N hour of each day. Avoid session boundaries."""
    keep = (~(df.index.hour < skip_first)) & (~(df.index.hour >= 24 - skip_last))
    return keep, keep


def trend_persistence(df: pd.DataFrame, sma_period: int = 20, bars_back: int = 5) -> tuple[pd.Series, pd.Series]:
    """Require trend to have persisted for bars_back — no flip-flopping."""
    sma = df["close"].rolling(sma_period, min_periods=1).mean()
    above = (df["close"] > sma).astype(int)
    below = (df["close"] < sma).astype(int)
    above_persisted = above.rolling(bars_back, min_periods=1).sum() >= bars_back
    below_persisted = below.rolling(bars_back, min_periods=1).sum() >= bars_back
    return above_persisted.astype(bool), below_persisted.astype(bool)


def rsi_filter(df: pd.DataFrame, period: int = 14,
               buy_below: float = 35.0, sell_above: float = 65.0) -> tuple[pd.Series, pd.Series]:
    """For REVERSAL: buy when RSI is oversold, sell when overbought."""
    rsi = ind.rsi(df["close"], period)
    keep_buy = rsi < buy_below
    keep_sell = rsi > sell_above
    return keep_buy, keep_sell


def macd_confirm(df: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9) -> tuple[pd.Series, pd.Series]:
    """MACD alignment: buy when MACD line > signal AND both rising; sell mirror."""
    ml, sl, _ = ind.macd(df["close"], fast, slow, sig)
    ml_up = ml > ml.shift(1)
    sl_up = sl > sl.shift(1)
    keep_buy = (ml > sl) & ml_up & sl_up
    keep_sell = (ml < sl) & ~ml_up & ~sl_up
    return keep_buy, keep_sell


def bollinger_filter(df: pd.DataFrame, period: int = 20, std: float = 2.0,
                     buy_below_pct: float = 0.05, sell_above_pct: float = 0.95) -> tuple[pd.Series, pd.Series]:
    """Buy when price is near lower band (mean-reversion), sell near upper band."""
    sma = df["close"].rolling(period, min_periods=1).mean()
    sd = df["close"].rolling(period, min_periods=1).std().fillna(0)
    upper = sma + std * sd
    lower = sma - std * sd
    rng = (upper - lower).replace(0, 1e-9)
    pct_b = (df["close"] - lower) / rng
    keep_buy = pct_b <= buy_below_pct
    keep_sell = pct_b >= sell_above_pct
    return keep_buy, keep_sell


# ─── Advanced confluence filters ─────────────────────────────────────────────

def fibonacci_levels(df: pd.DataFrame, lookback: int = 50,
                     tolerance_pct: float = 0.002,
                     levels: tuple[float, ...] = (0.382, 0.5, 0.618)) -> tuple[pd.Series, pd.Series]:
    """Buy near a fib support level (price retraced up into the level from below).
    Sell near a fib resistance level (price retraced down into the level from above).

    Uses rolling swing high/low over `lookback` bars."""
    swing_high = df["high"].rolling(lookback, min_periods=5).max()
    swing_low = df["low"].rolling(lookback, min_periods=5).min()
    rng = (swing_high - swing_low).replace(0, 1e-9)

    keep_buy = pd.Series(False, index=df.index)
    keep_sell = pd.Series(False, index=df.index)
    for lvl in levels:
        fib_price = swing_high - lvl * rng
        # Near = within tolerance_pct of fib price
        dist_pct = (df["close"] - fib_price).abs() / df["close"]
        near = dist_pct <= tolerance_pct
        # Buy: price coming up to a fib level (below fib, crossing up)
        # Sell: price coming down to a fib level (above fib, crossing down)
        below_fib = df["close"] < fib_price
        above_fib = df["close"] > fib_price
        crossed_up_from_below = below_fib.shift(1) & above_fib
        crossed_down_from_above = above_fib.shift(1) & below_fib
        keep_buy = keep_buy | (near & crossed_up_from_below)
        keep_sell = keep_sell | (near & crossed_down_from_above)
    # Fallback: also allow "near fib + in correct zone direction"
    # (without the cross requirement — softer)
    for lvl in levels:
        fib_price = swing_high - lvl * rng
        dist_pct = (df["close"] - fib_price).abs() / df["close"]
        near = dist_pct <= tolerance_pct
        keep_buy = keep_buy | (near & (df["close"] > fib_price.shift(1)))
        keep_sell = keep_sell | (near & (df["close"] < fib_price.shift(1)))
    return keep_buy, keep_sell


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Local ATR (true range rolling mean) — indicators module has no atr()."""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def fvg_bullish(df: pd.DataFrame, min_size_atr: float = 0.3,
                lookback: int = 20) -> tuple[pd.Series, pd.Series]:
    """Fair Value Gap (ICT): 3-bar pattern where bar1.high < bar3.low (gap up).
    Buy when price retraces into a bullish FVG (gap acts as support)."""
    atr = _atr(df, 14)
    h0, l0 = df["high"].shift(0), df["low"].shift(0)  # current
    h2, l2 = df["high"].shift(2), df["low"].shift(2)  # two bars back
    bull_fvg = (h2 < l0) & ((l0 - h2) >= atr * min_size_atr)
    # Bullish FVG zone = between h2 and l0 of the gap
    in_bull_fvg = bull_fvg.shift(0).fillna(False)  # currently inside an active gap
    # Also need price to be below the gap (retracing back into it) — bullish setup
    near_gap_low = (df["close"] >= h2) & (df["close"] <= l0)
    keep_buy = (bull_fvg.shift(1) | bull_fvg.shift(2)) & near_gap_low.shift(0)
    keep_buy = keep_buy.fillna(False)
    keep_sell = pd.Series(False, index=df.index)
    return keep_buy, keep_sell


def fvg_bearish(df: pd.DataFrame, min_size_atr: float = 0.3) -> tuple[pd.Series, pd.Series]:
    """Bearish FVG: 3-bar pattern where bar1.low > bar3.high (gap down).
    Sell when price retraces into the gap from below."""
    atr = _atr(df, 14)
    h0, l0 = df["high"], df["low"]
    h2, l2 = df["high"].shift(2), df["low"].shift(2)
    bear_fvg = (l2 > h0) & ((l2 - h0) >= atr * min_size_atr)
    near_gap_high = (df["close"] <= l2) & (df["close"] >= h0)
    keep_sell = (bear_fvg.shift(1) | bear_fvg.shift(2)) & near_gap_high.shift(0)
    keep_sell = keep_sell.fillna(False)
    keep_buy = pd.Series(False, index=df.index)
    return keep_buy, keep_sell


def ifvg_bullish(df: pd.DataFrame, min_size_atr: float = 0.3) -> tuple[pd.Series, pd.Series]:
    """Inverse FVG: a bullish FVG that was INVERTED (price closed through it from below).
    Acts as support on retest. Buy when price retraces back to the inverted zone."""
    atr = _atr(df, 14)
    h2, l2 = df["high"].shift(2), df["low"].shift(2)
    bull_fvg_zone = (l2 > h2.shift(0))  # zone range
    # Inversion: price eventually closes ABOVE the zone
    inverted = (df["close"] > l2) & (df["close"].shift(-5) <= l2)  # closed below 5 bars ago, now above
    near_zone = (df["close"] >= h2) & (df["close"] <= l2.shift(1))
    keep_buy = (inverted | pd.Series(inverted).shift(1).fillna(False)) & near_zone
    keep_buy = keep_buy.fillna(False)
    keep_sell = pd.Series(False, index=df.index)
    return keep_buy, keep_sell


def ifvg_bearish(df: pd.DataFrame, min_size_atr: float = 0.3) -> tuple[pd.Series, pd.Series]:
    """Inverse FVG bearish: inversion of a bearish FVG → resistance on retest."""
    atr = _atr(df, 14)
    h2, l2 = df["high"].shift(2), df["low"].shift(2)
    bear_fvg_zone = h2.shift(0) < l2
    inverted = (df["close"] < h2) & (df["close"].shift(-5) >= h2)
    near_zone = (df["close"] <= h2.shift(1)) & (df["close"] >= l2)
    keep_sell = (inverted | pd.Series(inverted).shift(1).fillna(False)) & near_zone
    keep_sell = keep_sell.fillna(False)
    keep_buy = pd.Series(False, index=df.index)
    return keep_buy, keep_sell


def mss_bullish(df: pd.DataFrame, swing_lookback: int = 10) -> tuple[pd.Series, pd.Series]:
    """Market Structure Shift: recent swing high broken → bullish MSS.
    Buy when current price breaks above the previous swing high after a downswing."""
    swing_high = df["high"].shift(1).rolling(swing_lookback, min_periods=2).max()
    prev_close_below = df["close"].shift(1) < swing_high.shift(1)
    break_above = df["close"] > swing_high
    keep_buy = prev_close_below & break_above
    keep_sell = pd.Series(False, index=df.index)
    return keep_buy.fillna(False), keep_sell


def mss_bearish(df: pd.DataFrame, swing_lookback: int = 10) -> tuple[pd.Series, pd.Series]:
    """Bearish MSS: break below previous swing low after an upswing."""
    swing_low = df["low"].shift(1).rolling(swing_lookback, min_periods=2).min()
    prev_close_above = df["close"].shift(1) > swing_low.shift(1)
    break_below = df["close"] < swing_low
    keep_sell = prev_close_above & break_below
    keep_sell = keep_sell.fillna(False)
    keep_buy = pd.Series(False, index=df.index)
    return keep_buy, keep_sell


def killzone_filter(df: pd.DataFrame,
                    london: tuple = (7, 11),
                    new_york: tuple = (12, 16),
                    asian: tuple = (0, 6),
                    enabled_sessions: tuple = ("london", "new_york")) -> tuple[pd.Series, pd.Series]:
    """ICT killzone filter: only fire in active killzones (default London + NY).
    london: 7-11 UTC, NY: 12-16 UTC, Asian (low vol): 0-6 UTC."""
    hour = pd.Series(df.index.hour, index=df.index)
    active = pd.Series(False, index=df.index)
    if "london" in enabled_sessions:
        active |= hour.between(london[0], london[1])
    if "new_york" in enabled_sessions:
        active |= hour.between(new_york[0], new_york[1])
    if "asian" in enabled_sessions:
        active |= hour.between(asian[0], asian[1])
    return active, active


def volume_increase(df: pd.DataFrame, lookback: int = 10,
                    mult: float = 1.5) -> tuple[pd.Series, pd.Series]:
    """Soft volume filter: current bar volume > mult * recent avg.
    Soft = both directions (no buy/sell split)."""
    if "volume" not in df.columns:
        return pd.Series(True, index=df.index), pd.Series(True, index=df.index)
    avg = df["volume"].rolling(lookback, min_periods=2).mean()
    keep = df["volume"] >= avg * mult
    return keep, keep


def momentum_increase(df: pd.DataFrame, period: int = 14,
                      lookback: int = 3) -> tuple[pd.Series, pd.Series]:
    """Momentum increase: RSI is RISING (not falling).
    Buy: RSI rising + RSI < 70 (not overbought).
    Sell: RSI falling + RSI > 30 (not oversold)."""
    rsi = ind.rsi(df["close"], period)
    rsi_diff = rsi.diff(lookback)
    keep_buy = (rsi_diff > 0) & (rsi < 70)
    keep_sell = (rsi_diff < 0) & (rsi > 30)
    return keep_buy.fillna(False), keep_sell.fillna(False)


def divergence_bullish(df: pd.DataFrame, rsi_period: int = 14,
                       pivot_lookback: int = 5) -> tuple[pd.Series, pd.Series]:
    """Bullish divergence: price makes lower low, RSI makes higher low.
    Reversal signal at swing low."""
    rsi = ind.rsi(df["close"], rsi_period)
    # Find local lows
    low_pivots = (df["low"] == df["low"].rolling(pivot_lookback * 2 + 1, center=True).min())
    # Two consecutive pivot lows
    last_two = low_pivots.shift(pivot_lookback)
    both_pivots = low_pivots & last_two
    price_lower_low = df["low"] < df["low"].shift(pivot_lookback)
    rsi_higher_low = rsi > rsi.shift(pivot_lookback)
    keep_buy = both_pivots & price_lower_low & rsi_higher_low
    keep_sell = pd.Series(False, index=df.index)
    return keep_buy.fillna(False), keep_sell


def divergence_bearish(df: pd.DataFrame, rsi_period: int = 14,
                       pivot_lookback: int = 5) -> tuple[pd.Series, pd.Series]:
    """Bearish divergence: price makes higher high, RSI makes lower high."""
    rsi = ind.rsi(df["close"], rsi_period)
    high_pivots = (df["high"] == df["high"].rolling(pivot_lookback * 2 + 1, center=True).max())
    last_two = high_pivots.shift(pivot_lookback)
    both_pivots = high_pivots & last_two
    price_higher_high = df["high"] > df["high"].shift(pivot_lookback)
    rsi_lower_high = rsi < rsi.shift(pivot_lookback)
    keep_sell = both_pivots & price_higher_high & rsi_lower_high
    keep_sell = keep_sell.fillna(False)
    keep_buy = pd.Series(False, index=df.index)
    return keep_buy, keep_sell


def confluence_score(df: pd.DataFrame, signal_direction: pd.Series,
                     filter_funcs: list, min_score: int = 2) -> tuple[pd.Series, pd.Series]:
    """SOFT filter: count how many of the supplied filter_funcs return True at this bar.
    Only allow signal if at least min_score confluences align in the SAME direction.

    This is softer than AND-combining because the base signal doesn't need ALL confluences
    to align — just a minimum number."""
    n = len(filter_funcs)
    if n == 0:
        return pd.Series(True, index=df.index), pd.Series(True, index=df.index)
    # Buy votes: each filter that would keep a buy signal
    # Sell votes: each filter that would keep a sell signal
    buy_score = pd.Series(0, index=df.index, dtype=int)
    sell_score = pd.Series(0, index=df.index, dtype=int)
    for f in filter_funcs:
        try:
            keep_buy, keep_sell = f(df)
            buy_score = buy_score + keep_buy.fillna(False).astype(int)
            sell_score = sell_score + keep_sell.fillna(False).astype(int)
        except Exception:
            pass
    # For Buy: need min_score buy-votes. For Sell: need min_score sell-votes.
    keep_buy = buy_score >= min_score
    keep_sell = sell_score >= min_score
    return keep_buy, keep_sell


# ─── Market context + Pullback + Pivots + ZigZag + Elliott ─────────────────

def market_context_pullback(df: pd.DataFrame,
                             htf_period: int = 100,
                             pullback_pct: float = 0.3,
                             pullback_lookback: int = 20) -> tuple[pd.Series, pd.Series]:
    """HTF market context + pullback entry.
    Detect HTF trend (HTF SMA slope). Force signals to align with HTF trend:
      - HTF uptrend → only allow BUY signals when price has pulled back ≥ pullback_pct from recent high
      - HTF downtrend → only allow SELL signals when price has rallied back ≥ pullback_pct from recent low
      - HTF ranging → allow both

    This is the "force the pullback" pattern: trade with the trend, enter on dips.
    """
    sma = df["close"].rolling(htf_period, min_periods=20).mean()
    sma_slope = sma.diff(5)  # 5-bar slope to detect trend direction
    trending_up = sma_slope > 0
    trending_down = sma_slope < 0

    # Recent high/low over pullback_lookback
    recent_high = df["high"].rolling(pullback_lookback, min_periods=2).max()
    recent_low = df["low"].rolling(pullback_lookback, min_periods=2).min()

    # Pullback from recent high = how much price has dropped from peak
    drop_from_high = 1 - (df["close"] / recent_high)
    rally_from_low = (df["close"] / recent_low) - 1
    has_pullback_in_uptrend = trending_up & (drop_from_high >= pullback_pct)
    has_pullback_in_downtrend = trending_down & (rally_from_low >= pullback_pct)

    # Buy: require uptrend + pullback OR ranging
    keep_buy = (~trending_down) & (has_pullback_in_uptrend | (~trending_up & ~trending_down))
    # Sell: require downtrend + pullback OR ranging
    keep_sell = (~trending_up) & (has_pullback_in_downtrend | (~trending_up & ~trending_down))
    return keep_buy.fillna(False), keep_sell.fillna(False)


def force_pullback(df: pd.DataFrame, pullback_pct: float = 0.005,
                   lookback: int = 10) -> tuple[pd.Series, pd.Series]:
    """Force pullback entry: require ≥ pullback_pct retracement from recent extreme before signal.
    Buy: only keep if price dropped ≥ pullback_pct from recent N-bar high.
    Sell: only keep if price rallied ≥ pullback_pct from recent N-bar low.
    This is a HARD filter — only allows signals that occur AFTER a pullback, not at extremes."""
    recent_high = df["high"].rolling(lookback, min_periods=2).max()
    recent_low = df["low"].rolling(lookback, min_periods=2).min()
    drop_pct = 1 - (df["close"] / recent_high)
    rally_pct = (df["close"] / recent_low) - 1
    keep_buy = drop_pct >= pullback_pct
    keep_sell = rally_pct >= pullback_pct
    return keep_buy.fillna(False), keep_sell.fillna(False)


def pivot_points(df: pd.DataFrame, lookback: int = 24,
                 tolerance_pct: float = 0.001) -> tuple[pd.Series, pd.Series]:
    """Classic floor pivots over lookback hours.
    PP = (H + L + C) / 3
    S1 = 2*PP - H, S2 = PP - (H - L), S3 = L - 2*(H - PP)
    R1 = 2*PP - L, R2 = PP + (H - L), R3 = H + 2*(PP - L)

    Buy near S1/S2/S3 (support); sell near R1/R2/R3 (resistance).
    For H1, lookback=24 = previous day. For H4, lookback=6."""
    h = df["high"].rolling(lookback, min_periods=2).max()
    l = df["low"].rolling(lookback, min_periods=2).min()
    c = df["close"].rolling(lookback, min_periods=2).mean()
    pp = (h + l + c) / 3
    s1 = 2 * pp - h
    s2 = pp - (h - l)
    s3 = l - 2 * (h - pp)
    r1 = 2 * pp - l
    r2 = pp + (h - l)
    r3 = h + 2 * (pp - l)

    keep_buy = pd.Series(False, index=df.index)
    keep_sell = pd.Series(False, index=df.index)
    for support in (s1, s2, s3):
        dist = (df["close"] - support).abs() / df["close"]
        near = dist <= tolerance_pct
        keep_buy = keep_buy | near
    for resistance in (r1, r2, r3):
        dist = (df["close"] - resistance).abs() / df["close"]
        near = dist <= tolerance_pct
        keep_sell = keep_sell | near
    return keep_buy.fillna(False), keep_sell.fillna(False)


def zigzag_swings(df: pd.DataFrame, threshold_pct: float = 0.02,
                  pullback_window: int = 5) -> tuple[pd.Series, pd.Series]:
    """ZigZag swing detector: mark swing highs/lows using rolling max/min pivots.
    Returns (buy_after_swing_low, sell_after_swing_high) — useful for reversal entries.

    A swing low = lowest low within a window. A swing high = highest high within a window.
    After a swing low, look for buy setups (reversal up).
    After a swing high, look for sell setups (reversal down)."""
    window = max(3, pullback_window * 2 + 1)
    roll_min = df["low"].rolling(window, center=True, min_periods=2).min()
    roll_max = df["high"].rolling(window, center=True, min_periods=2).max()
    swing_low = (df["low"] == roll_min)
    swing_high = (df["high"] == roll_max)
    # Forward-fill the swing flags for `pullback_window` bars
    recent_swing_low = swing_low.rolling(pullback_window, min_periods=1).max() > 0
    recent_swing_high = swing_high.rolling(pullback_window, min_periods=1).max() > 0
    # Only fire if price has retraced by threshold_pct from swing
    drop_from_high = 1 - (df["close"] / df["high"].rolling(pullback_window, min_periods=1).max())
    rally_from_low = (df["close"] / df["low"].rolling(pullback_window, min_periods=1).min()) - 1
    keep_buy = recent_swing_low & (drop_from_high <= threshold_pct)
    keep_sell = recent_swing_high & (rally_from_low <= threshold_pct)
    return keep_buy.fillna(False), keep_sell.fillna(False)


def elliott_wave_proxy(df: pd.DataFrame, swing_lookback: int = 20,
                       wave_threshold: float = 0.01) -> tuple[pd.Series, pd.Series]:
    """Simplified Elliott Wave proxy:
    - Detect swing structure: 5 consecutive higher highs/lows (impulse up) or lower (impulse down)
    - If in impulse up → allow buy on 2nd/4th wave pullback
    - If in impulse down → allow sell on 2nd/4th wave pullback
    - Otherwise: no signal (correction phase)

    This is a coarse proxy, not a real Elliott Wave count."""
    hh = df["high"] > df["high"].shift(1)
    hl = df["low"] > df["low"].shift(1)
    lh = df["high"] < df["high"].shift(1)
    ll = df["low"] < df["low"].shift(1)
    # Consecutive HH count
    hh_streak = hh.astype(int).groupby((~hh).cumsum()).cumsum()
    hl_streak = hl.astype(int).groupby((~hl).cumsum()).cumsum()
    ll_streak = ll.astype(int).groupby((~ll).cumsum()).cumsum()
    lh_streak = lh.astype(int).groupby((~lh).cumsum()).cumsum()

    # Impulse up: 5+ consecutive HH AND HL
    impulse_up = (hh_streak >= 5) & (hl_streak >= 5)
    impulse_down = (ll_streak >= 5) & (lh_streak >= 5)
    # Pullback in impulse = current close < recent high by wave_threshold
    pullback_up = impulse_up & (df["close"] < df["high"].rolling(swing_lookback, min_periods=1).max() * (1 - wave_threshold))
    pullback_down = impulse_down & (df["close"] > df["low"].rolling(swing_lookback, min_periods=1).min() * (1 + wave_threshold))

    keep_buy = pullback_up
    keep_sell = pullback_down
    return keep_buy.fillna(False), keep_sell.fillna(False)


def regime_filter_advanced(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    """Advanced regime detection:
    - Trending up: ADX > 25 + DI+ > DI-
    - Trending down: ADX > 25 + DI- > DI+
    - Ranging: ADX < 20
    - Volatile: ATR% > 1.5% (chaos)

    For trend-following: allow BUY only in trending up, SELL only in trending down.
    For mean-reversion: allow both in ranging, neither in trending.
    Returns BOTH (so user can pick what fits strategy character).
    """
    adx, pdi, mdi = ind.adx(df["high"], df["low"], df["close"], period)
    atr_local = _atr(df, period)
    atr_pct = atr_local / df["close"]
    trend_up = (adx > 25) & (pdi > mdi)
    trend_down = (adx > 25) & (mdi > pdi)
    ranging = adx < 20
    volatile = atr_pct > 0.015

    # Trend alignment: BUY only in trend_up, SELL only in trend_down
    trend_buy = trend_up & ~volatile
    trend_sell = trend_down & ~volatile
    # Mean reversion: BUY/SELL both in ranging
    mr_buy = ranging & ~volatile
    mr_sell = ranging & ~volatile
    return (trend_buy | mr_buy).fillna(False), (trend_sell | mr_sell).fillna(False)


def vwap_distance(df: pd.DataFrame, anchor_hour: int = 0,
                  min_dist_pct: float = 0.001,
                  max_dist_pct: float = 0.02) -> tuple[pd.Series, pd.Series]:
    """VWAP-like rolling mean (anchored to a daily reset).
    Buy: price below VWAP (mean-reversion up) by at least min_dist_pct.
    Sell: price above VWAP (mean-reversion down)."""
    if "volume" not in df.columns:
        return pd.Series(True, index=df.index), pd.Series(True, index=df.index)
    # Daily VWAP approximation: sum(price*vol) / sum(vol) since anchor_hour
    date = df.index.date
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    # Group by date — VWAP resets daily
    pv_cum = pv.groupby(date).cumsum()
    vol_cum = df["volume"].groupby(date).cumsum()
    vwap = pv_cum / vol_cum.replace(0, 1e-9)
    dist_pct = (df["close"] - vwap) / vwap
    keep_buy = (dist_pct <= -min_dist_pct) & (dist_pct >= -max_dist_pct)
    keep_sell = (dist_pct >= min_dist_pct) & (dist_pct <= max_dist_pct)
    return keep_buy.fillna(False), keep_sell.fillna(False)


def trend_strength(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                   very_strong_pct: float = 0.005) -> tuple[pd.Series, pd.Series]:
    """Trend strength filter — only allow signals aligned with strong trends.
    Strong uptrend: fast SMA > slow SMA AND gap > very_strong_pct.
    Strong downtrend: fast SMA < slow SMA AND gap > very_strong_pct.
    """
    sma_f = df["close"].rolling(fast, min_periods=5).mean()
    sma_s = df["close"].rolling(slow, min_periods=10).mean()
    gap_pct = (sma_f - sma_s) / sma_s
    strong_up = gap_pct > very_strong_pct
    strong_down = gap_pct < -very_strong_pct
    keep_buy = strong_up
    keep_sell = strong_down
    return keep_buy.fillna(False), keep_sell.fillna(False)


# ─── Wrapper ──────────────────────────────────────────────────────────────────

class EnhancedStrategy(BaseStrategy):
    """Wraps a base strategy with a list of filters."""
    name = "enhanced"

    def __init__(self, base: BaseStrategy, filters: list[Callable] | None = None,
                 params_override: dict | None = None):
        self.base = base
        self.filters = filters or []
        self.params = dict(params_override or {})
        self._name = f"enhanced_{base.name}"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, v):
        self._name = v

    def generate(self, df: pd.DataFrame) -> Signals:
        base_sig = self.base.generate(df)
        buy = base_sig.entries & (base_sig.direction == 1)
        sell = base_sig.entries & (base_sig.direction == -1)
        for f in self.filters:
            try:
                keep_buy, keep_sell = f(df)
                buy = buy & keep_buy.fillna(False).astype(bool)
                sell = sell & keep_sell.fillna(False).astype(bool)
            except Exception:
                pass  # If a filter fails, keep all signals (don't break the backtest)
        entries = buy | sell
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)),
                              index=df.index, dtype=int)
        return Signals(entries=entries, exits=base_sig.exits, direction=direction)
