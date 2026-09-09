"""FBB (Bollinger + Force Index) strategy. Mirror of MQL5 FBB_GetSignals.

Key facts:
  1. Type_1 uses BOLLINGER bands on PREVIOUS bar (bar 1):
       Case 1 (mean reversion): prev bar poked outside band, current closed inside.
         Buy:  prev.Low < prev.Lower AND curr.Close > Lower
         Sell: prev.High > prev.Upper AND curr.Close < Upper
       Case 2 (breakout):
         Buy:  prev.Low > Upper (close above upper band prev bar)
         Sell: prev.High < Lower
  2. Type_2 uses FORCE INDEX scaled by ×20:
       8 cases mirror MS pattern (continuation, cross, fade, deep).
  3. DistanceBands gate: ((UB-LB)/SymbolPoints) vs FBB_LevelOpenOrders_1.
       level > 0: distance > level required
       level < 0: distance < |level| required
       level == 0: always pass
  4. Final signal: BOTH types must agree (FBB_OpenBuy_1 && FBB_OpenBuy_2).
  5. Special: Type_1 == 0 OR Type_2 == 0 → that type is always true (BOTH signal set).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


class FBB_Strategy(BaseStrategy):
    name = "fbb"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        period = int(p.get("bars_calculate", 20))
        dev = float(p.get("deviation", 1.8))
        mid, up, lo = ind.bollinger(df["close"], period, dev)
        force_raw = ind.force_index(df["close"], df["volume"], 13)
        # Scale Force ×20 (MQL5 line 5007)
        force = force_raw * 20
        force_p = force.shift(1)  # Force_Value_1 = bar 1
        force_pp = force.shift(2)  # Force_Value_2 = bar 2
        # Distance bands in points (MQL5 uses SymbolPoints which is 10*point for 5-digit)
        # For 5-digit brokers: (UB - LB) / (10 * point). For our backtest use simple /pip.
        pip_size = 0.0001
        symbol_points_factor = 10  # 5-digit brokers
        dist_pips = (up - lo) / (pip_size * symbol_points_factor)
        # Previous bar values
        prev_low = df["low"].shift(1)
        prev_high = df["high"].shift(1)
        prev_upper = up.shift(1)
        prev_lower = lo.shift(1)
        curr_close = df["close"]

        sig = _empty_signals(df.index)

        lev_o1 = float(p.get("level_open_orders_1", 0))
        lev_o2 = float(p.get("level_open_orders_2", 50))
        lev_c1 = float(p.get("level_close_orders_1", 40))
        lev_c2 = float(p.get("level_close_orders_2", 40))

        lev_up_o = lev_o2
        lev_dn_o = -lev_o2
        lev_up_c = lev_c2
        lev_dn_c = -lev_c2

        # Distance bands gate (only check if level != 0)
        dist_ok_open = pd.Series(True, index=df.index)
        if lev_o1 > 0:
            dist_ok_open = dist_pips > lev_o1
        elif lev_o1 < 0:
            dist_ok_open = dist_pips < abs(lev_o1)

        # Type_1: Bollinger-based
        ot1 = int(p.get("open_orders_type_1", 1))
        if ot1 == 0:
            buy1 = pd.Series(True, index=df.index)
            sell1 = pd.Series(True, index=df.index)
        elif ot1 in (1, 2) and dist_ok_open.any():
            buy1 = pd.Series(False, index=df.index)
            sell1 = pd.Series(False, index=df.index)
            if ot1 == 1:
                # Mean reversion: prev bar poked outside, current closed inside
                buy1 = (prev_low < prev_lower) & (curr_close > prev_lower)
                sell1 = (prev_high > prev_upper) & (curr_close < prev_upper)
            elif ot1 == 2:
                # Breakout: prev bar fully above upper → buy; below lower → sell
                buy1 = prev_low > prev_upper
                sell1 = prev_high < prev_lower
            buy1 = buy1 & dist_ok_open
            sell1 = sell1 & dist_ok_open
        else:
            buy1 = pd.Series(False, index=df.index)
            sell1 = pd.Series(False, index=df.index)

        # Type_2: Force Index (8 cases)
        ot2 = int(p.get("open_orders_type_2", 0))
        if ot2 == 0:
            buy2 = pd.Series(True, index=df.index)
            sell2 = pd.Series(True, index=df.index)
        elif 1 <= ot2 <= 8:
            buy2, sell2 = _force_cases(ot2, force, force_p, force_pp, lev_up_o, lev_dn_o)
        else:
            buy2 = pd.Series(False, index=df.index)
            sell2 = pd.Series(False, index=df.index)

        # Combined: BOTH types must agree (MQL5 line 5202)
        buy = buy1 & buy2
        sell = sell1 & sell2

        sig.entries = buy | sell
        sig.direction = np.where(buy, 1, np.where(sell, -1, 0))

        # Close (Type_1 + Type_2 both contribute)
        dist_ok_close = pd.Series(True, index=df.index)
        if lev_c1 > 0:
            dist_ok_close = dist_pips > lev_c1
        elif lev_c1 < 0:
            dist_ok_close = dist_pips < abs(lev_c1)

        ct1 = int(p.get("close_orders_type_1", 1))
        cb1 = pd.Series(False, index=df.index)
        cs1 = pd.Series(False, index=df.index)
        if ct1 in (1, 2) and dist_ok_close.any():
            if ct1 == 1:
                # Close mirror of open case 1
                cb1 = (prev_high > prev_upper) & (curr_close < prev_upper)
                cs1 = (prev_low < prev_lower) & (curr_close > prev_lower)
            elif ct1 == 2:
                cb1 = prev_low > prev_upper
                cs1 = prev_high < prev_lower
            cb1 = cb1 & dist_ok_close
            cs1 = cs1 & dist_ok_close

        ct2 = int(p.get("close_orders_type_2", 4))
        cb2 = pd.Series(False, index=df.index)
        cs2 = pd.Series(False, index=df.index)
        if 1 <= ct2 <= 8:
            cb2, cs2 = _force_cases(ct2, force, force_p, force_pp, lev_up_c, lev_dn_c)

        # Close signal: EITHER close type can fire
        sig.exits = cb1 | cs1 | cb2 | cs2

        return sig


def _force_cases(open_type: int, force: pd.Series, force_p: pd.Series, force_pp: pd.Series,
                 lev_up: float, lev_dn: float) -> tuple[pd.Series, pd.Series]:
    """8 Force-Index cases (mirrors MS cases 1-8 but on Force values).
    force_p = Force_Value_1 (bar 1), force_pp = Force_Value_2 (bar 2).
    """
    z = pd.Series(False, index=force.index)
    if open_type == 1:
        buy = (force_p > lev_up) & (force_pp > lev_up) & (force_p > force_pp)
        sell = (force_p < lev_dn) & (force_pp < lev_dn) & (force_p < force_pp)
    elif open_type == 2:
        buy = (force_p > lev_up) & (force_pp < lev_up) & (force_p > force_pp)
        sell = (force_p < lev_dn) & (force_pp > lev_dn) & (force_p < force_pp)
    elif open_type == 3:
        buy = (force_p > lev_up) & (force_pp > lev_up) & (force_p < force_pp)
        sell = (force_p < lev_dn) & (force_pp < lev_dn) & (force_p > force_pp)
    elif open_type == 4:
        buy = (force_p < lev_up) & (force_pp > lev_up) & (force_p < force_pp)
        sell = (force_p > lev_dn) & (force_pp < lev_dn) & (force_p > force_pp)
    elif open_type == 5:
        buy = (force_p < lev_dn) & (force_pp < lev_dn) & (force_p < force_pp)
        sell = (force_p > lev_up) & (force_pp > lev_up) & (force_p > force_pp)
    elif open_type == 6:
        buy = (force_p < lev_dn) & (force_pp > lev_dn) & (force_p < force_pp)
        sell = (force_p > lev_up) & (force_pp < lev_up) & (force_p > force_pp)
    elif open_type == 7:
        buy = (force_p < lev_dn) & (force_pp < lev_dn) & (force_p > force_pp)
        sell = (force_p > lev_up) & (force_pp > lev_up) & (force_p < force_pp)
    elif open_type == 8:
        buy = (force_p > lev_dn) & (force_pp < lev_dn) & (force_p > force_pp)
        sell = (force_p < lev_up) & (force_pp > lev_up) & (force_p < force_pp)
    else:
        buy = z.copy()
        sell = z.copy()
    return buy, sell


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))