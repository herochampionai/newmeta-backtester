"""Build 6 new '2026 Edges' strategies from the user's research plan.

Edge #1: Liquidity Sweep + Momentum Reclaim (PDH/PDL sweep with reclaim)
Edge #2: Volatility Regime (BBW + ATR + RV, percentiles)
Edge #3: Session-Based NASDAQ (NY Open / Lunch / Power Hour)
Edge #4: Market Structure (HH+HL, Displacement, Retest, Continuation)
Edge #5: Ensemble (Trend30+Sweep20+VolExp25+Struct25, score>=75)
Edge #6: Synthetic Order Flow (Range+Volume+Close position, Aggression Score)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


def sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def _atr(df, period):
    """Standard ATR."""
    h = df['high']; l = df['low']; c = df['close']
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# =============================================================================
# EDGE #1: LIQUIDITY SWEEP + MOMENTUM RECLAIM
# =============================================================================
class LiquiditySweepStrategy(BaseStrategy):
    """Sweep previous day/period high/low with close back inside, then enter."""
    name = "liq_sweep"

    def generate(self, df):
        p = self.params
        lookback = int(p.get('lookback', 20))  # 20-bar PDH/PDL
        atr_period = int(p.get('atr_period', 14))
        sl_atr_mult = float(p.get('sl_atr', 1.0))
        tp_rr = float(p.get('tp_rr', 1.5))
        atr_avg_period = int(p.get('atr_avg_period', 20))
        vol_avg_period = int(p.get('vol_avg_period', 20))
        sma_period = int(p.get('sma_period', 200))
        cooldown = int(p.get('cooldown', 5))

        atr = _atr(df, atr_period)
        atr_avg = atr.rolling(atr_avg_period).mean()
        vol_avg = df['volume'].rolling(vol_avg_period).mean()
        sma200 = sma(df['close'], sma_period)
        pdh = df['high'].rolling(lookback).max().shift(1)
        pdl = df['low'].rolling(lookback).min().shift(1)

        # Sweep condition: wick beyond level but close back inside
        sweep_low = (df['low'] < pdl) & (df['close'] > pdl)
        sweep_high = (df['high'] > pdh) & (df['close'] < pdh)

        # Filters: ATR > avg, vol > avg, 200 SMA direction
        vol_atr_ok = (atr > atr_avg) & (df['volume'] > vol_avg)
        above_sma = df['close'] > sma200
        below_sma = df['close'] < sma200

        # Long: sweep low (price wicked below then closed back up)
        # Short: sweep high (price wicked above then closed back down)
        long_entry = sweep_low & vol_atr_ok & above_sma
        short_entry = sweep_high & vol_atr_ok & below_sma

        direction = pd.Series(np.where(long_entry.fillna(False), 1,
                                       np.where(short_entry.fillna(False), -1, 0)),
                               index=df.index, dtype=int)

        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


# =============================================================================
# EDGE #2: VOLATILITY REGIME
# =============================================================================
class VolatilityRegimeStrategy(BaseStrategy):
    """Detect regime: quiet / normal / explosive. Different logic per regime."""
    name = "vol_regime"

    def generate(self, df):
        p = self.params
        atr_period = int(p.get('atr_period', 14))
        bb_period = int(p.get('bb_period', 20))
        bb_mult = float(p.get('bb_mult', 2.0))
        rv_period = int(p.get('rv_period', 20))
        quiet_pct = float(p.get('quiet_pct', 25.0))
        explosive_pct = float(p.get('explosive_pct', 75.0))
        lookback_pct = int(p.get('lookback_pct', 200))
        ma_period = int(p.get('ma_period', 50))
        breakout_period = int(p.get('breakout_period', 20))
        cd = int(p.get('cooldown', 10))

        atr = _atr(df, atr_period)
        basis, upper, lower = ind.bollinger(df['close'], bb_period, bb_mult)
        bb_width = (upper - lower) / basis.replace(0, np.nan)
        returns = df['close'].pct_change()
        rv = returns.rolling(rv_period).std() * np.sqrt(252)
        ma = sma(df['close'], ma_period)

        # Percentile rank of BBWidth + RV + ATR
        bb_pct = bb_width.rolling(lookback_pct).rank(pct=True) * 100
        rv_pct = rv.rolling(lookback_pct).rank(pct=True) * 100
        atr_pct = atr.rolling(lookback_pct).rank(pct=True) * 100
        avg_vol_pct = (bb_pct + rv_pct + atr_pct) / 3

        is_quiet = avg_vol_pct < quiet_pct
        is_explosive = avg_vol_pct > explosive_pct
        is_normal = ~is_quiet & ~is_explosive

        # Quiet regime: fade range (mean reversion)
        # Buy when close < lower band, sell when close > upper band
        quiet_buy = is_quiet & (df['close'] < lower)
        quiet_sell = is_quiet & (df['close'] > upper)

        # Normal regime: trend following using MA crossover
        ma_x_up = (ma > ma.shift(1)) & (ma.shift(1) <= ma.shift(2))
        ma_x_down = (ma < ma.shift(1)) & (ma.shift(1) >= ma.shift(2))
        trend_buy = is_normal & ma_x_up
        trend_sell = is_normal & ma_x_down

        # Explosive regime: breakout
        high_n = df['high'].rolling(breakout_period).max().shift(1)
        low_n = df['low'].rolling(breakout_period).min().shift(1)
        exp_buy = is_explosive & (df['close'] > high_n)
        exp_sell = is_explosive & (df['close'] < low_n)

        long_entry = quiet_buy | trend_buy | exp_buy
        short_entry = quiet_sell | trend_sell | exp_sell

        direction = pd.Series(np.where(long_entry.fillna(False), 1,
                                       np.where(short_entry.fillna(False), -1, 0)),
                               index=df.index, dtype=int)

        if cd > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


# =============================================================================
# EDGE #3: SESSION-BASED (different per ticker)
# =============================================================================
class SessionBasedStrategy(BaseStrategy):
    """Different logic per session. Auto-detects ticker for NAS vs EUR sessions."""
    name = "session"

    def generate(self, df):
        p = self.params
        adx_period = int(p.get('adx_period', 14))
        bb_period = int(p.get('bb_period', 20))
        bb_mult = float(p.get('bb_mult', 2.0))
        atr_period = int(p.get('atr_period', 14))
        cd = int(p.get('cooldown', 10))

        # Detect ticker: NAS data is in UTC-4 (EDT) or UTC-5 (EST), EUR in UTC+0/+1
        # Check the time zone of the data
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            # NAS data is at -04:00, EUR at +01:00
            tz_offset = df.index[0].utcoffset().total_seconds() / 3600
        else:
            tz_offset = 0

        is_nas = tz_offset < -2

        # Hour in local time
        hour = pd.Series(df.index.hour, index=df.index)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            # Convert to local time (US Eastern for NAS, UTC for EUR)
            if is_nas:
                local_idx = df.index.tz_convert('America/New_York')
            else:
                local_idx = df.index  # EUR data already at UTC/GMT
            hour = pd.Series(local_idx.hour, index=df.index)

        adx_val, di_plus, di_minus = ind.adx(df['high'], df['low'], df['close'], adx_period)
        basis, upper, lower = ind.bollinger(df['close'], bb_period, bb_mult)
        atr = _atr(df, atr_period)

        if is_nas:
            # NASDAQ sessions (US Eastern)
            # Asia: 18:00-02:00 (prev night)
            # London: 03:00-08:00
            # NYSE Open: 09:30-11:00 (volatility)
            # Lunch: 12:00-14:00 (quiet, mean revert)
            # Power Hour: 15:00-16:00 (momentum)
            nyse_open = hour.isin([9, 10])
            lunch = hour.isin([12, 13])
            power = hour.isin([15, 16])
        else:
            # EURUSD sessions (UTC)
            # Asia: 22:00-06:00
            # London: 07:00-12:00 (range expansion)
            # NYSE: 13:00-17:00 (overlap, momentum)
            # Lunch: 16:00-19:00
            london = hour.isin([7, 8, 9, 10, 11, 12])
            nyse = hour.isin([13, 14, 15, 16, 17])
            lunch = hour.isin([18, 19, 20])

        if is_nas:
            # NYSE Open: ADX>25 breakout
            nyse_open_long = nyse_open & (di_plus > di_minus) & (adx_val > 25) & (df['close'] > upper)
            nyse_open_short = nyse_open & (di_minus > di_plus) & (adx_val > 25) & (df['close'] < lower)
            # Lunch: BB mean revert
            lunch_long = lunch & (df['close'] < lower)
            lunch_short = lunch & (df['close'] > upper)
            # Power Hour: momentum (close > open by ATR)
            body = abs(df['close'] - df['open'])
            power_long = power & (df['close'] > df['open']) & (body > atr * 0.5)
            power_short = power & (df['close'] < df['open']) & (body > atr * 0.5)
            long_entry = nyse_open_long | lunch_long | power_long
            short_entry = nyse_open_short | lunch_short | power_short
        else:
            # EURUSD: London breakout, NYSE momentum, lunch range
            london_long = london & (df['close'] > upper)
            london_short = london & (df['close'] < lower)
            body = abs(df['close'] - df['open'])
            nyse_long = nyse & (df['close'] > df['open']) & (body > atr * 0.5)
            nyse_short = nyse & (df['close'] < df['open']) & (body > atr * 0.5)
            lunch_long = lunch & (df['close'] < lower)
            lunch_short = lunch & (df['close'] > upper)
            long_entry = london_long | nyse_long | lunch_long
            short_entry = london_short | nyse_short | lunch_short

        direction = pd.Series(np.where(long_entry.fillna(False), 1,
                                       np.where(short_entry.fillna(False), -1, 0)),
                               index=df.index, dtype=int)

        if cd > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


# =============================================================================
# EDGE #4: MARKET STRUCTURE ENGINE
# =============================================================================
class MarketStructureStrategy(BaseStrategy):
    """Higher High + Higher Low + Displacement + Retest + Continuation."""
    name = "structure"

    def generate(self, df):
        p = self.params
        swing_n = int(p.get('swing_n', 3))
        disp_atr_mult = float(p.get('disp_atr_mult', 1.2))
        atr_period = int(p.get('atr_period', 14))
        retest_window = int(p.get('retest_window', 5))
        cd = int(p.get('cooldown', 10))

        atr = _atr(df, atr_period)

        # Swing highs and lows
        def is_swing_high(series, n):
            return (series == series.rolling(2*n+1, center=True).max())

        def is_swing_low(series, n):
            return (series == series.rolling(2*n+1, center=True).min())

        swing_high = is_swing_high(df['high'], swing_n)
        swing_low = is_swing_low(df['low'], swing_n)

        # Higher highs and higher lows (trend structure)
        last_high = swing_high.shift(1).ffill()
        prev_high = swing_high.shift(2).ffill()
        hh = (df['high'] > last_high) & (last_high > prev_high)

        last_low = swing_low.shift(1).ffill()
        prev_low = swing_low.shift(2).ffill()
        hl = (df['low'] > last_low) & (last_low > prev_low)

        # Displacement: large body candle > 1.2 ATR
        body = abs(df['close'] - df['open'])
        disp_up = hh & (body > disp_atr_mult * atr) & (df['close'] > df['open'])
        disp_dn = ~hl & (body > disp_atr_mult * atr) & (df['close'] < df['open'])

        # Retest: price comes back to test the breakout level within N bars
        # For long: pullback to swing low zone after displacement up
        breakout_high = df['high'].rolling(swing_n).max().shift(1)
        retest_long = disp_up.shift(retest_window).fillna(False) & (df['low'] <= breakout_high.shift(retest_window).fillna(df['close'])) & (df['close'] > df['open'])

        breakout_low = df['low'].rolling(swing_n).min().shift(1)
        retest_short = disp_dn.shift(retest_window).fillna(False) & (df['high'] >= breakout_low.shift(retest_window).fillna(df['close'])) & (df['close'] < df['open'])

        long_entry = disp_up | retest_long.fillna(False)
        short_entry = disp_dn | retest_short.fillna(False)

        direction = pd.Series(np.where(long_entry.fillna(False), 1,
                                       np.where(short_entry.fillna(False), -1, 0)),
                               index=df.index, dtype=int)

        if cd > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


# =============================================================================
# EDGE #6: SYNTHETIC ORDER FLOW
# =============================================================================
class SynthOrderFlowStrategy(BaseStrategy):
    """Aggression score from candle spread + ATR + Volume + Close position."""
    name = "synth_of"

    def generate(self, df):
        p = self.params
        atr_period = int(p.get('atr_period', 14))
        range_atr_mult = float(p.get('range_atr_mult', 1.5))
        vol_atr_mult = float(p.get('vol_atr_mult', 1.5))
        close_pct_top = float(p.get('close_pct_top', 0.75))
        close_pct_bot = float(p.get('close_pct_bot', 0.25))
        atr_avg_period = int(p.get('atr_avg_period', 20))
        vol_avg_period = int(p.get('vol_avg_period', 20))
        sma_period = int(p.get('sma_period', 200))
        cd = int(p.get('cooldown', 5))

        atr = _atr(df, atr_period)
        atr_avg = atr.rolling(atr_avg_period).mean()
        vol_avg = df['volume'].rolling(vol_avg_period).mean()
        sma200 = sma(df['close'], sma_period)

        # Range expansion
        candle_range = df['high'] - df['low']
        range_exp = candle_range > range_atr_mult * atr
        # Volume expansion
        vol_exp = df['volume'] > vol_atr_mult * vol_avg
        # Close position within candle range (0 = bottom, 1 = top)
        close_pos = (df['close'] - df['low']) / candle_range.replace(0, np.nan)
        close_top = close_pos > close_pct_top  # closed in top 25%
        close_bot = close_pos < close_pct_bot  # closed in bottom 25%

        # Bullish aggression: range expansion + volume expansion + close at top
        bull_aggression = range_exp & vol_exp & close_top
        # Bearish aggression: range expansion + volume expansion + close at bottom
        bear_aggression = range_exp & vol_exp & close_bot

        # 200 SMA direction filter
        long_entry = bull_aggression & (df['close'] > sma200)
        short_entry = bear_aggression & (df['close'] < sma200)

        direction = pd.Series(np.where(long_entry.fillna(False), 1,
                                       np.where(short_entry.fillna(False), -1, 0)),
                               index=df.index, dtype=int)

        if cd > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


# =============================================================================
# EDGE #5: ENSEMBLE (Trend30 + Sweep20 + VolExp25 + Struct25)
# =============================================================================
class EnsembleStrategy(BaseStrategy):
    """Combine 4 modules: Trend + Sweep + Vol Expansion + Structure.
    Each module scores up to 25 points. Total = 100. Trade if >= 75."""
    name = "ensemble"

    def generate(self, df):
        p = self.params
        score_threshold = float(p.get('score_threshold', 75))
        cd = int(p.get('cooldown', 15))

        # Use the other strategies internally with simple params
        # Trend: SMA crossover (using ADX filter)
        sma_fast = sma(df['close'], 9)
        sma_slow = sma(df['close'], 21)
        trend_up = (sma_fast > sma_slow).astype(int)
        trend_down = (sma_fast < sma_slow).astype(int)

        # Sweep: PDH/PDL sweep
        pdh = df['high'].rolling(20).max().shift(1)
        pdl = df['low'].rolling(20).min().shift(1)
        sweep_long = (df['low'] < pdl) & (df['close'] > pdl)
        sweep_short = (df['high'] > pdh) & (df['close'] < pdh)

        # Vol Expansion: candle range > ATR
        atr = _atr(df, 14)
        candle_range = df['high'] - df['low']
        vol_exp = candle_range > atr

        # Structure: HH/HL
        hh = df['high'] > df['high'].shift(1).rolling(5).max()
        hl = df['low'] > df['low'].shift(1).rolling(5).min()
        lh = df['high'] < df['high'].shift(1).rolling(5).max()
        ll = df['low'] < df['low'].shift(1).rolling(5).min()

        # Score calculation
        # Bullish:
        # Trend (30): trend_up = 30, else 0
        # Sweep (20): sweep_long = 20
        # Vol (25): vol_exp = 25
        # Structure (25): hh & hl = 25, hh alone = 12, hl alone = 12
        bull_score = (
            trend_up * 30 +
            sweep_long.astype(int) * 20 +
            vol_exp.astype(int) * 25 +
            ((hh & hl).astype(int) * 25 + hh.astype(int) * 12 + hl.astype(int) * 12).clip(upper=25)
        )
        bear_score = (
            trend_down * 30 +
            sweep_short.astype(int) * 20 +
            vol_exp.astype(int) * 25 +
            ((lh & ll).astype(int) * 25 + lh.astype(int) * 12 + ll.astype(int) * 12).clip(upper=25)
        )

        long_entry = bull_score >= score_threshold
        short_entry = bear_score >= score_threshold

        direction = pd.Series(np.where(long_entry.fillna(False), 1,
                                       np.where(short_entry.fillna(False), -1, 0)),
                               index=df.index, dtype=int)

        if cd > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cd:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig
