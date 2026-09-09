"""AC + AO COMBO strategy. Mirror of MQL5 AC_GetSignals.

Key facts I got wrong before:
  1. MQL5 scales AC by ×40000 then compares with LevelOpenUp directly.
     Raw AC on EURUSD H1 is ~[-0.05, 0.05]. Scaled = ~[-2000, 2000].
     So a level of 80 means raw AC > 0.002.
  2. AO is used at RAW scale for the sync filter (not scaled).
  3. There are 8 distinct open cases, not 1 default.
  4. Close cases use the same 8-case switch with potentially different level.

AC_OpenOrdersType semantics (Buy condition first, then mirror for Sell):
  1: AC[0]>LevelUp AND AC[1]>LevelUp AND AC[0]>AC[1]   (continuation)
  2: AC[0]>LevelUp AND AC[1]<LevelUp AND AC[0]>AC[1]   (cross up)
  3: AC[0]>LevelUp AND AC[1]>LevelUp AND AC[0]<AC[1]   (fade top)
  4: AC[0]<LevelUp AND AC[1]>LevelUp AND AC[0]<AC[1]   (cross down)
  5: AC[0]<LevelDn AND AC[1]<LevelDn AND AC[0]<AC[1]   (continuation down)
  6: AC[0]<LevelDn AND AC[1]>LevelDn AND AC[0]<AC[1]   (cross down from -)
  7: AC[0]<LevelDn AND AC[1]<LevelDn AND AC[0]>AC[1]   (fade bottom)
  8: AC[0]>LevelDn AND AC[1]<LevelDn AND AC[0]>AC[1]   (cross up from -)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


class AC_AO_Strategy(BaseStrategy):
    name = "ac_ao"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        # Raw AC, AO (unscaled for sync filters)
        ac_raw = ind.ac(df["high"], df["low"], ao_period=5)
        ao_raw = ind.ao(df["high"], df["low"], 5, 34)
        # Scaled AC for level comparisons (×40000, like MQL5)
        ac = ac_raw * 40000
        ac_prev = ac.shift(1)
        # Acceleration filter: requires momentum to be accelerating
        # acceleration = (AC[0]-AC[1]) - (AC[1]-AC[2])
        m1 = ac_raw - ac_raw.shift(1)
        m2 = ac_raw.shift(1) - ac_raw.shift(2)
        acceleration = m1 - m2
        use_accel = bool(p.get("use_acceleration_filter", False))
        min_accel = float(p.get("min_acceleration", 0.0005))
        strong_bull_accel = acceleration > min_accel if use_accel else pd.Series(True, index=df.index)
        strong_bear_accel = acceleration < -min_accel if use_accel else pd.Series(True, index=df.index)
        # AO synchronization filter: AO > 0 AND AO rising (raw scale)
        ao_mom = ao_raw - ao_raw.shift(1)
        use_ao_sync = bool(p.get("use_ao_synchronization", False))
        min_ao_sync = float(p.get("min_ao_synchronization", 0.0003))
        if use_ao_sync:
            ao_bull_sync = (ao_raw > 0) & (ao_mom > min_ao_sync)
            ao_bear_sync = (ao_raw < 0) & (ao_mom < -min_ao_sync)
        else:
            ao_bull_sync = pd.Series(True, index=df.index)
            ao_bear_sync = pd.Series(True, index=df.index)

        # Levels (scaled)
        open_type = int(p.get("open_orders_type", 1))
        close_type = int(p.get("close_orders_type", 0))
        lev_open = float(p.get("level_open_orders", 80))
        lev_close = float(p.get("level_close_orders", 70))
        lev_up = lev_open
        lev_dn = -lev_open

        buy, sell = _ac_cases(open_type, ac, ac_prev, lev_up, lev_dn)

        # Apply ENHANCEMENTS
        buy = buy & strong_bull_accel & ao_bull_sync
        sell = sell & strong_bear_accel & ao_bear_sync

        sig = _empty_signals(df.index)
        sig.entries = buy | sell
        sig.direction = np.where(buy, 1, np.where(sell, -1, 0))

        # Close signals (also scaled, uses separate level_close_orders)
        if close_type > 0:
            lev_up_c = lev_close
            lev_dn_c = -lev_close
            cb, cs = _ac_cases(close_type, ac, ac_prev, lev_up_c, lev_dn_c)
            sig.exits = cb | cs

        return sig


def _ac_cases(open_type: int, ac: pd.Series, ac_prev: pd.Series,
              lev_up: float, lev_dn: float) -> tuple[pd.Series, pd.Series]:
    """Vectorized port of all 8 AC cases. Returns (buy, sell) bool Series."""
    z = pd.Series(False, index=ac.index)
    if open_type == 1:
        # Both bars above LevelUp AND current above prev → buy continuation
        buy = (ac > lev_up) & (ac_prev > lev_up) & (ac > ac_prev)
        sell = (ac < lev_dn) & (ac_prev < lev_dn) & (ac < ac_prev)
    elif open_type == 2:
        # Crossed up: prev below, current above, rising
        buy = (ac > lev_up) & (ac_prev < lev_up) & (ac > ac_prev)
        sell = (ac < lev_dn) & (ac_prev > lev_dn) & (ac < ac_prev)
    elif open_type == 3:
        # Above level both bars, but fading (current below prev)
        buy = (ac > lev_up) & (ac_prev > lev_up) & (ac < ac_prev)
        sell = (ac < lev_dn) & (ac_prev < lev_dn) & (ac > ac_prev)
    elif open_type == 4:
        # Crossed down from above (current below LevelUp, prev above)
        buy = (ac < lev_up) & (ac_prev > lev_up) & (ac < ac_prev)
        sell = (ac > lev_dn) & (ac_prev < lev_dn) & (ac > ac_prev)
    elif open_type == 5:
        # Both bars below LevelDn AND current below prev → sell continuation
        buy = (ac < lev_dn) & (ac_prev < lev_dn) & (ac < ac_prev)
        sell = (ac > lev_up) & (ac_prev > lev_up) & (ac > ac_prev)
    elif open_type == 6:
        # Crossed down from above (current below LevelDn, prev above)
        buy = (ac < lev_dn) & (ac_prev > lev_dn) & (ac < ac_prev)
        sell = (ac > lev_up) & (ac_prev < lev_up) & (ac > ac_prev)
    elif open_type == 7:
        # Below level both bars, but reversing (current above prev)
        buy = (ac < lev_dn) & (ac_prev < lev_dn) & (ac > ac_prev)
        sell = (ac > lev_up) & (ac_prev > lev_up) & (ac < ac_prev)
    elif open_type == 8:
        # Crossed up from below (current above LevelDn, prev below)
        buy = (ac > lev_dn) & (ac_prev < lev_dn) & (ac > ac_prev)
        sell = (ac < lev_up) & (ac_prev > lev_up) & (ac < ac_prev)
    else:
        buy = z.copy()
        sell = z.copy()
    return buy, sell


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))