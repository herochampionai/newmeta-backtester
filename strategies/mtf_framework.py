"""MTF framework — HTF referee + LTF sniper + creative structure/liquidity ideas.

Architecture:
  - HTF referee (H4 or D1): determines "allowed" direction (with strict gates)
  - LTF sniper (H1 or M15): generates the entry signal
  - Combined: HTF must agree with LTF, else no trade

Creative ideas (each a soft filter that scores +N):
  1. Swing structure: higher highs + higher lows (uptrend), lower highs + lower lows (downtrend)
  2. Close vs previous H/L: close > prev_high = breakout; close < prev_low = breakdown
  3. Candle type: bullish (close > open) vs bearish (close < open)
  4. 50% mid-range position: close > mid = bullish bias, close < mid = bearish bias
  5. Session liquidity: Asia low / London high / NY high (sessions where liquidity was taken)
  6. Daily target / gravity: previous day high/low, weekly pivot

Each strategy can be tuned to use any subset of these filters.

HTF referee configurations:
  - "h4_sma_trend": H4 close > H4 SMA = bullish
  - "h4_rsi_trend": H4 RSI > 50 = bullish
  - "d1_structure": D1 higher highs/lows = bullish
  - "h4_adx_filter": H4 ADX > 25 = trending (filters ranging markets)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from . import indicators as ind
from ._base import BaseStrategy, Signals


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


# ════════════════════════════════════════════════════════════════════════
#  HTF REFEREE
# ════════════════════════════════════════════════════════════════════════

def htf_referee(df: pd.DataFrame, rule: str = "4h",
                sma_period: int = 50,
                rsi_period: int = 14,
                adx_period: int = 14) -> tuple[pd.Series, pd.Series]:
    """Returns (htf_bull, htf_bear) — both bool Series on df.index.

    Modes:
      - 'sma_trend': close > htf_sma = bull, close < htf_sma = bear
      - 'rsi_trend': htf_rsi > 50 = bull, htf_rsi < 50 = bear
      - 'structure': higher highs + higher lows = bull (vs previous N bars)
      - 'adx_trending': htf_adx > 25 = trending (use as filter, not direction)
    """
    htf_bull = pd.Series(False, index=df.index)
    htf_bear = pd.Series(False, index=df.index)

    try:
        htf = df.resample(rule).agg({"close": "last", "high": "max", "low": "min"}).dropna()
        if len(htf) < max(sma_period, rsi_period, adx_period) + 5:
            return htf_bull, htf_bear

        if "sma" in rule or rule.endswith("h") or rule.endswith("d"):
            # SMA trend
            sma = htf["close"].rolling(sma_period, min_periods=20).mean()
            bull_local = htf["close"] > sma
            bear_local = htf["close"] < sma
        elif "rsi" in rule:
            rsi = ind.rsi(htf["close"], rsi_period)
            bull_local = rsi > 50
            bear_local = rsi < 50
        else:
            # Default: SMA
            sma = htf["close"].rolling(sma_period, min_periods=20).mean()
            bull_local = htf["close"] > sma
            bear_local = htf["close"] < sma

        # Forward-fill to H1 index
        htf_bull = bull_local.reindex(df.index, method="ffill").fillna(False).astype(bool)
        htf_bear = bear_local.reindex(df.index, method="ffill").fillna(False).astype(bool)
    except Exception:
        pass
    return htf_bull, htf_bear


def htf_structure_referee(df: pd.DataFrame, rule: str = "4h",
                          swing_lookback: int = 10,
                          n_struct_bars: int = 3) -> tuple[pd.Series, pd.Series]:
    """Detect swing structure on HTF: N consecutive higher highs/lows = bull."""
    htf_bull = pd.Series(False, index=df.index)
    htf_bear = pd.Series(False, index=df.index)
    try:
        htf = df.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(htf) < swing_lookback * (n_struct_bars + 2):
            return htf_bull, htf_bear

        hh = htf["high"] > htf["high"].shift(1)
        hl = htf["low"] > htf["low"].shift(1)
        lh = htf["high"] < htf["high"].shift(1)
        ll = htf["low"] < htf["low"].shift(1)

        hh_streak = hh.astype(int).groupby((~hh).cumsum()).cumsum()
        hl_streak = hl.astype(int).groupby((~hl).cumsum()).cumsum()
        ll_streak = ll.astype(int).groupby((~ll).cumsum()).cumsum()
        lh_streak = lh.astype(int).groupby((~lh).cumsum()).cumsum()

        bull = (hh_streak >= n_struct_bars) & (hl_streak >= n_struct_bars)
        bear = (ll_streak >= n_struct_bars) & (lh_streak >= n_struct_bars)

        htf_bull = bull.reindex(df.index, method="ffill").fillna(False).astype(bool)
        htf_bear = bear.reindex(df.index, method="ffill").fillna(False).astype(bool)
    except Exception:
        pass
    return htf_bull, htf_bear


def htf_adx_filter(df: pd.DataFrame, rule: str = "4h",
                   adx_period: int = 14,
                   threshold: float = 20.0) -> pd.Series:
    """HTF ADX filter — only allow signals when HTF is trending."""
    is_trending = pd.Series(True, index=df.index)
    try:
        htf = df.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(htf) >= adx_period + 5:
            adx, _, _ = ind.adx(htf["high"], htf["low"], htf["close"], adx_period)
            is_trending = (adx > threshold).reindex(df.index, method="ffill").fillna(True).astype(bool)
    except Exception:
        pass
    return is_trending


# ════════════════════════════════════════════════════════════════════════
#  LTF SNIPER (creative entry conditions)
# ════════════════════════════════════════════════════════════════════════

def swing_structure_sniper(df: pd.DataFrame, swing_lookback: int = 5,
                           min_separation: int = 3) -> tuple[pd.Series, pd.Series]:
    """LTF swing structure: HH+HL for buy, LH+LL for sell.

    Buy:  current bar makes a higher high AND higher low vs previous swing.
    Sell: current bar makes a lower high AND lower low.
    """
    roll_high = df["high"].rolling(swing_lookback, min_periods=2).max()
    roll_low = df["low"].rolling(swing_lookback, min_periods=2).min()
    hh = df["high"] >= roll_high
    hl = df["low"] >= roll_low
    lh = df["high"] <= roll_high
    ll = df["low"] <= roll_low
    buy_struct = hh & hl
    sell_struct = lh & ll
    return buy_struct.fillna(False), sell_struct.fillna(False)


def close_vs_prev_hl(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Buy: close > previous bar's high (breakout). Sell: close < previous bar's low."""
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    buy_breakout = df["close"] > prev_high
    sell_breakout = df["close"] < prev_low
    return buy_breakout.fillna(False), sell_breakout.fillna(False)


def candle_bias(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Buy: bullish candle (close > open). Sell: bearish candle (close < open)."""
    bull = df["close"] > df["open"]
    bear = df["close"] < df["open"]
    return bull.fillna(False), bear.fillna(False)


def mid_range_position(df: pd.DataFrame, lookback: int = 50,
                       bull_threshold: float = 0.5) -> tuple[pd.Series, pd.Series]:
    """Buy: close in upper half of recent range. Sell: close in lower half.
    Uses rolling percentile of close over rolling high/low range.
    """
    rh = df["high"].rolling(lookback, min_periods=10).max()
    rl = df["low"].rolling(lookback, min_periods=10).min()
    rng = (rh - rl).replace(0, 1e-9)
    pos = (df["close"] - rl) / rng
    buy_pos = pos >= bull_threshold
    sell_pos = pos <= (1 - bull_threshold)
    return buy_pos.fillna(False), sell_pos.fillna(False)


def session_liquidity_taken(df: pd.DataFrame,
                             asian_hours: tuple = (0, 1, 2, 3, 4, 5, 6),
                             london_hours: tuple = (7, 8, 9, 10, 11),
                             ny_hours: tuple = (12, 13, 14, 15, 16, 17, 18, 19),
                             lookback_bars: int = 24) -> tuple[pd.Series, pd.Series]:
    """Track which session's high/low was taken.
    Buy: previous Asian low was taken during London = bullish bias.
    Sell: previous Asian high was taken during London = bearish bias.
    """
    hour = pd.Series(df.index.hour, index=df.index)
    in_asian = hour.isin(asian_hours)
    in_london = hour.isin(london_hours)

    # Asia high/low
    asian_h = df["high"].where(in_asian).rolling(lookback_bars, min_periods=1).max()
    asian_l = df["low"].where(in_asian).rolling(lookback_bars, min_periods=1).min()

    # In London: did price break above Asian high or below Asian low?
    london_break_above = (df["close"] > asian_h.shift(1)) & in_london
    london_break_below = (df["close"] < asian_l.shift(1)) & in_london

    return london_break_above.fillna(False), london_break_below.fillna(False)


def daily_gravity(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Daily gravity: previous day's high/low act as targets.
    Buy: price approaching previous day's high (magnet up).
    Sell: price approaching previous day's low.
    """
    prev_high = df["high"].shift(1).rolling(24, min_periods=2).max()
    prev_low = df["low"].shift(1).rolling(24, min_periods=2).min()

    # Within 0.5 ATR of previous day's high → buy setup
    atr_local = _atr(df, 14)
    near_high = (df["close"] >= prev_high - 0.5 * atr_local) & (df["close"] <= prev_high + 0.2 * atr_local)
    near_low = (df["close"] <= prev_low + 0.5 * atr_local) & (df["close"] >= prev_low - 0.2 * atr_local)

    return near_high.fillna(False), near_low.fillna(False)


def weekly_pivot_bias(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Weekly pivot: PP = (H + L + C) / 3 over the past week (168 H1 bars).
    Buy: close > weekly PP. Sell: close < weekly PP.
    """
    h = df["high"].rolling(168, min_periods=20).max()
    l = df["low"].rolling(168, min_periods=20).min()
    c = df["close"].rolling(168, min_periods=20).mean()
    pp = (h + l + c) / 3
    above_pp = df["close"] > pp
    below_pp = df["close"] < pp
    return above_pp.fillna(False), below_pp.fillna(False)


# ════════════════════════════════════════════════════════════════════════
#  MTF WRAPPER
# ════════════════════════════════════════════════════════════════════════

class MTFStrategy(BaseStrategy):
    """Wraps a base LTF strategy with HTF referee + creative entry conditions.

    Usage:
        class MyStrat(MTFStrategy):
            ltf_strategy_cls = AC_AO_Strategy  # base signal generator on H1
            htf_rule = "4h"
            htf_mode = "sma_trend"
            structure_check = True
            use_candle_bias = True
            use_mid_range = True
            use_session_liquidity = False
            use_daily_gravity = False
            use_weekly_pivot = False
            require_confluence = 2  # min N creative filters must agree
    """
    ltf_strategy_cls = None
    name = "mtf_strategy"
    htf_rule = "4h"
    htf_mode = "sma_trend"
    htf_sma_period = 50
    htf_rsi_period = 14
    htf_adx_threshold = 20.0
    htf_struct_lookback = 10
    htf_struct_bars = 3
    require_htf_agreement = True  # hard gate: HTF must agree with LTF direction
    swing_lookback = 5
    use_swing_structure = True
    use_breakout = True
    use_candle_bias = True
    use_mid_range = True
    use_session_liquidity = False
    use_daily_gravity = False
    use_weekly_pivot = False
    require_confluence = 2  # min N creative filters agreeing (out of those enabled)
    cooldown_bars = 5

    def __init__(self, params: dict | None = None):
        self.params = dict(params or {})

    def _build_creative_filters(self, df: pd.DataFrame) -> list:
        """Return list of (buy_filter_fn, sell_filter_fn) pairs for creative conditions."""
        filters = []
        if self.use_swing_structure:
            bb, ss = swing_structure_sniper(df, self.swing_lookback)
            filters.append((bb, ss))
        if self.use_breakout:
            bb, ss = close_vs_prev_hl(df)
            filters.append((bb, ss))
        if self.use_candle_bias:
            bb, ss = candle_bias(df)
            filters.append((bb, ss))
        if self.use_mid_range:
            bb, ss = mid_range_position(df)
            filters.append((bb, ss))
        if self.use_session_liquidity:
            bb, ss = session_liquidity_taken(df)
            filters.append((bb, ss))
        if self.use_daily_gravity:
            bb, ss = daily_gravity(df)
            filters.append((bb, ss))
        if self.use_weekly_pivot:
            bb, ss = weekly_pivot_bias(df)
            filters.append((bb, ss))
        return filters

    def generate(self, df: pd.DataFrame) -> Signals:
        if self.ltf_strategy_cls is None:
            raise ValueError("Must set ltf_strategy_cls")

        # Apply params overrides
        for key, val in self.params.items():
            if hasattr(self, key):
                setattr(self, key, val)

        # 1. LTF base signal
        base = self.ltf_strategy_cls(params=self.params)
        sig = base.generate(df)
        buy = sig.entries.fillna(False) & (sig.direction == 1)
        sell = sig.entries.fillna(False) & (sig.direction == -1)

        # 2. HTF referee
        if self.require_htf_agreement:
            if self.htf_mode == "structure":
                htf_bull, htf_bear = htf_structure_referee(
                    df, self.htf_rule, self.htf_struct_lookback, self.htf_struct_bars)
            else:
                htf_bull, htf_bear = htf_referee(df, self.htf_rule,
                                                  self.htf_sma_period, self.htf_rsi_period)
            # HTF must agree with LTF signal direction
            # If HTF bullish, allow buys; if bearish, allow sells; if both/neither, block
            buy = buy & htf_bull
            sell = sell & htf_bear

        # 3. Optional ADX trending filter (HTF)
        if self.htf_adx_threshold > 0:
            trending = htf_adx_filter(df, self.htf_rule, 14, self.htf_adx_threshold)
            buy = buy & trending
            sell = sell & trending

        # 4. Creative entry filters (confluence scoring)
        creative_filters = self._build_creative_filters(df)
        if creative_filters and self.require_confluence > 0:
            buy_votes = pd.Series(0, index=df.index, dtype=int)
            sell_votes = pd.Series(0, index=df.index, dtype=int)
            for bb, ss in creative_filters:
                buy_votes = buy_votes + bb.fillna(False).astype(int)
                sell_votes = sell_votes + ss.fillna(False).astype(int)
            # Require at least N filters to agree in same direction as signal
            if self.require_confluence > 0:
                buy = buy & (buy_votes >= self.require_confluence)
                sell = sell & (sell_votes >= self.require_confluence)

        # 5. Cooldown
        direction = pd.Series(np.where(buy, 1, np.where(sell, -1, 0)),
                               index=df.index, dtype=int)
        if self.cooldown_bars > 0 and len(direction) > self.cooldown_bars:
            new_dir = direction.values.copy()
            last_signal_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_signal_idx < self.cooldown_bars:
                        new_dir[i] = 0
                    else:
                        last_signal_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        out = Signals(
            entries=direction != 0,
            exits=sig.exits,
            direction=direction,
        )
        return out


# ════════════════════════════════════════════════════════════════════════
#  MTF VERSIONS OF EACH V1 STRATEGY
# ════════════════════════════════════════════════════════════════════════

class MTFAC_AO(MTFStrategy):
    name = "mtf_ac_ao"
    ltf_strategy_cls = None  # set after class definition
    htf_rule = "4h"
    htf_mode = "sma_trend"
    htf_sma_period = 50
    use_swing_structure = True
    use_breakout = False
    use_candle_bias = True
    use_mid_range = True
    require_confluence = 2


class MTFADX(MTFStrategy):
    """ADX already proven profitable — keep its settings but add MTF safety net."""
    name = "mtf_adx"
    ltf_strategy_cls = None
    htf_rule = "4h"
    htf_mode = "structure"  # Use structure (HH+HL) for ADX — gives swing context
    htf_struct_lookback = 10
    htf_struct_bars = 3
    use_swing_structure = False  # ADX has its own structure logic
    use_breakout = False
    use_candle_bias = True
    use_mid_range = False
    require_confluence = 1  # Light — don't kill ADX trades
    cooldown_bars = 8


class MTFDeM(MTFStrategy):
    name = "mtf_dem"
    ltf_strategy_cls = None
    htf_rule = "4h"
    htf_mode = "sma_trend"
    htf_sma_period = 50
    use_swing_structure = True
    use_breakout = True
    use_candle_bias = True
    use_mid_range = True
    require_confluence = 2


class MTFFBB(MTFStrategy):
    name = "mtf_fbb"
    ltf_strategy_cls = None
    htf_rule = "4h"
    htf_mode = "sma_trend"
    htf_sma_period = 50
    use_swing_structure = True
    use_breakout = True
    use_candle_bias = True
    use_mid_range = False  # breakout indicators don't work well with mid-range filters
    require_confluence = 2


class MTFMFI(MTFStrategy):
    name = "mtf_mfi"
    ltf_strategy_cls = None
    htf_rule = "4h"
    htf_mode = "sma_trend"
    htf_sma_period = 50
    use_swing_structure = True
    use_breakout = False
    use_candle_bias = True
    use_mid_range = True
    require_confluence = 2


class MTFMS(MTFStrategy):
    name = "mtf_ms"
    ltf_strategy_cls = None
    htf_rule = "4h"
    htf_mode = "sma_trend"
    htf_sma_period = 50
    use_swing_structure = False  # MACD signal is already a trend signal
    use_breakout = True
    use_candle_bias = True
    use_mid_range = False
    require_confluence = 2


class MTFMTF_Stoch(MTFStrategy):
    name = "mtf_mtf_stoch"
    ltf_strategy_cls = None
    htf_rule = "1d"  # Daily for MTF
    htf_mode = "sma_trend"
    use_swing_structure = True
    use_breakout = False
    use_candle_bias = True
    use_mid_range = True
    require_confluence = 2


class MTFBB_RSI(MTFStrategy):
    name = "mtf_bb_rsi"
    ltf_strategy_cls = None
    htf_rule = "4h"
    htf_mode = "sma_trend"
    use_swing_structure = True
    use_breakout = False
    use_candle_bias = True
    use_mid_range = True
    require_confluence = 2


class MTFTriple_RSI(MTFStrategy):
    name = "mtf_triple_rsi"
    ltf_strategy_cls = None
    htf_rule = "4h"
    htf_mode = "sma_trend"
    use_swing_structure = True
    use_breakout = True
    use_candle_bias = True
    use_mid_range = True
    require_confluence = 2


class MTFQuad_Stoch(MTFStrategy):
    name = "mtf_quad_stoch"
    ltf_strategy_cls = None
    htf_rule = "1d"
    htf_mode = "sma_trend"
    use_swing_structure = True
    use_breakout = False
    use_candle_bias = True
    use_mid_range = True
    require_confluence = 2


class MTFStoch533(MTFStrategy):
    name = "mtf_stoch533"
    ltf_strategy_cls = None
    htf_rule = "1d"
    htf_mode = "sma_trend"
    use_swing_structure = True
    use_breakout = False
    use_candle_bias = True
    use_mid_range = True
    require_confluence = 2


class MTFMACD_Confluence(MTFStrategy):
    name = "mtf_macd_confluence"
    ltf_strategy_cls = None
    htf_rule = "4h"
    htf_mode = "sma_trend"
    use_swing_structure = True
    use_breakout = False
    use_candle_bias = True
    use_mid_range = False
    require_confluence = 2


# Bind ltf_strategy_cls at module level (after class definitions)
from . import (
    ac_ao as _ac_ao_mod,
)
from . import (
    adx as _adx_mod,
)
from . import (
    bb_rsi as _bb_rsi_mod,
)
from . import (
    dem as _dem_mod,
)
from . import (
    fbb as _fbb_mod,
)
from . import (
    macd_confluence as _macd_mod,
)
from . import (
    mfi as _mfi_mod,
)
from . import (
    ms as _ms_mod,
)
from . import (
    mtf_stoch as _mtf_stoch_mod,
)
from . import (
    quad_stoch as _quad_stoch_mod,
)
from . import (
    stoch533_mtf as _stoch533_mod,
)
from . import (
    triple_rsi as _triple_rsi_mod,
)

MTFAC_AO.ltf_strategy_cls = _ac_ao_mod.AC_AO_Strategy
MTFADX.ltf_strategy_cls = _adx_mod.ADX_Strategy
MTFDeM.ltf_strategy_cls = _dem_mod.DeM_Strategy
MTFFBB.ltf_strategy_cls = _fbb_mod.FBB_Strategy
MTFMFI.ltf_strategy_cls = _mfi_mod.MFI_Strategy
MTFMS.ltf_strategy_cls = _ms_mod.MS_Strategy
MTFMTF_Stoch.ltf_strategy_cls = _mtf_stoch_mod.QuadStochStrategy
MTFBB_RSI.ltf_strategy_cls = _bb_rsi_mod.BBRsiStrategy
MTFTriple_RSI.ltf_strategy_cls = _triple_rsi_mod.TripleRSIStrategy
MTFQuad_Stoch.ltf_strategy_cls = _quad_stoch_mod.QuadStochSameTF
MTFStoch533.ltf_strategy_cls = _stoch533_mod.Stoch533MTF
MTFMACD_Confluence.ltf_strategy_cls = _macd_mod.MACDConfluenceStrategy
