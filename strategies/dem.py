"""DeM (DeMarker) ORIGINAL. Mirror of MQL5 DeM_GetSignals.

Key facts:
  1. DeMarker is scaled by ×100 (MQL5 line 4858): raw 0..1 becomes 0..100.
  2. Levels: LevelOpenUp = DeM_LevelOpenOrders, LevelOpenDn = 100 - DeM_LevelOpenOrders.
     Same for close.
  3. Case 3 has PRICE ACTION confirmation:
       Buy: MFI crosses up through LevelDn AND bar_close > bar_open (bullish bar)
       Sell: MFI crosses down through LevelUp AND bar_close < bar_open (bearish bar)
  4. 4 open cases + 4 close cases.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


class DeM_Strategy(BaseStrategy):
    name = "dem"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        period = int(p.get("bars_calculate", 20))
        # Scale DeMarker ×100 like MQL5
        dem = ind.dem(df["high"], df["low"], period) * 100
        dem_p = dem.shift(1)
        # Price action confirmation (close vs open)
        bullish_bar = df["close"] > df["open"]
        bearish_bar = df["close"] < df["open"]

        sig = _empty_signals(df.index)

        lev_open = float(p.get("level_open_orders", 75))
        lev_close = float(p.get("level_close_orders", 70))
        lev_up = lev_open
        lev_dn = 100 - lev_open
        lev_up_c = lev_close
        lev_dn_c = 100 - lev_close

        ot = int(p.get("open_orders_type", 3))
        if ot > 0:
            buy, sell = _dem_cases(ot, dem, dem_p, lev_up, lev_dn,
                                   bullish_bar, bearish_bar)
            sig.entries = buy | sell
            sig.direction = np.where(buy, 1, np.where(sell, -1, 0))

        ct = int(p.get("close_orders_type", 0))
        if ct > 0:
            cb, cs = _dem_cases(ct, dem, dem_p, lev_up_c, lev_dn_c,
                                bullish_bar, bearish_bar)
            sig.exits = cb | cs

        return sig


def _dem_cases(open_type: int, dem: pd.Series, dem_p: pd.Series,
               lev_up: float, lev_dn: float,
               bullish_bar: pd.Series, bearish_bar: pd.Series
               ) -> tuple[pd.Series, pd.Series]:
    """All 4 DeM cases. Cases 3 require price action confirmation."""
    z = pd.Series(False, index=dem.index)
    if open_type == 1:
        # Cross up through upper → buy; cross down through lower → sell
        buy = (dem > lev_up) & (dem_p < lev_up)
        sell = (dem < lev_dn) & (dem_p > lev_dn)
    elif open_type == 2:
        # Cross down through upper → buy (fade); cross up through lower → sell
        buy = (dem < lev_up) & (dem_p > lev_up)
        sell = (dem > lev_dn) & (dem_p < lev_dn)
    elif open_type == 3:
        # Cross up through lower with bullish bar → buy
        # Cross down through upper with bearish bar → sell
        buy = (dem > lev_dn) & (dem_p < lev_dn) & bullish_bar
        sell = (dem < lev_up) & (dem_p > lev_up) & bearish_bar
    elif open_type == 4:
        # Cross up through lower (no bar confirm) → buy; mirror for sell
        buy = (dem > lev_dn) & (dem_p < lev_dn)
        sell = (dem < lev_up) & (dem_p > lev_up)
    else:
        buy = z.copy()
        sell = z.copy()
    return buy, sell


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))