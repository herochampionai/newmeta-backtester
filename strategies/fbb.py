"""FBB strategy — BREAKOUT ONLY + Force Index.

Per user spec: FBB plays breakout/breakdown only, NOT reversals.
  Type_1: BB breakout/breakdown (case 2)
  Type_2: Force Index oscillator (case 8)

Mean-reversion (close-back-inside + RSI divergence) lives in bb_rsi.py.

Default config (breakout):
  open_orders_type_1 = 2 (breakout)
  open_orders_type_2 = 8 (Force Index oscillator)
  level_open_orders_1 = 0.0  (always pass distance check)
  level_open_orders_2 = 50.0
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


class FBB_Strategy(BaseStrategy):
    name = "fbb"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        period = int(p.get("bars_calculate", 20))
        dev = float(p.get("deviation", 1.8))

        # Bollinger Bands
        mid, up, lo = ind.bollinger(df["close"], period, dev)
        # Force Index scaled by ×20 (MQL5 convention)
        force = ind.force_index(df["close"], df["volume"], period) * 20
        force_prev = force.shift(1)

        sig = _empty_signals(df.index)
        ot1 = int(p.get("open_orders_type_1", 2))
        ot2 = int(p.get("open_orders_type_2", 8))
        lev1 = float(p.get("level_open_orders_1", 0.0))   # distance band check (pips)
        lev2 = float(p.get("level_open_orders_2", 50.0))  # oscillator level

        # Previous bar values for the "close back inside" pattern
        prev_close = df["close"].shift(1)
        prev_low = df["low"].shift(1)
        prev_high = df["high"].shift(1)
        prev_upper = up.shift(1)
        prev_lower = lo.shift(1)
        prev_mid = mid.shift(1)

        # === Type_1: BB breakout / breakdown ONLY ===
        buy = pd.Series(False, index=df.index)
        sell = pd.Series(False, index=df.index)
        if ot1 == 1:
            # LEGACY reversal (close-back-inside) — kept for compat, not recommended.
            # Use bb_rsi for reversals instead.
            buy = (prev_low < prev_lower) & (df["close"] > lo)
            sell = (prev_high > prev_upper) & (df["close"] < up)
        elif ot1 == 2:
            # Breakout: prev bar above upper → buy, below lower → sell
            buy = prev_low > prev_upper
            sell = prev_high < prev_lower

        # === Type_2: Force Index oscillator with level ===
        buy2 = pd.Series(False, index=df.index)
        sell2 = pd.Series(False, index=df.index)
        if ot2 > 0:
            # Pattern 8 (default): Force crossed above +level from below → buy,
            # crossed below -level from above → sell
            if ot2 == 8:
                cross_up = (force > lev2) & (force_prev <= lev2)
                cross_dn = (force < -lev2) & (force_prev >= -lev2)
                buy2 = cross_up
                sell2 = cross_dn
            else:
                # Generic: level-based oscillator
                # Buy: force > +level and rising
                buy2 = (force > lev2) & (force > force_prev)
                # Sell: force < -level and falling
                sell2 = (force < -lev2) & (force < force_prev)

        # Combine: Type_1 OR Type_2 (per EA MQL5: OpenBuy_1 || OpenBuy_2)
        final_buy = buy | buy2
        final_sell = sell | sell2

        # Distance band gate: if lev1 != 0, gate by band width
        if lev1 != 0 and (ot1 > 0 or ot2 > 0):
            dist_pips = (up - lo) / 0.0001
            if lev1 > 0:
                dist_ok = dist_pips > lev1
            else:
                dist_ok = dist_pips < abs(lev1)
            final_buy = final_buy & dist_ok
            final_sell = final_sell & dist_ok

        sig.entries = final_buy | final_sell
        sig.direction = np.where(final_buy, 1, np.where(final_sell, -1, 0))
        return sig