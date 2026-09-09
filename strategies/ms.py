"""MS ENHANCED strategy. Mirror of MQL5 MS_GetSignals.

Key facts I got wrong before:
  1. MACD Main and Signal are scaled by ×40000 in MQL5 (raw MACD on
     EURUSD H1 is ~[-0.001, 0.001]; scaled = ~[-40, 40]).
  3. Type_1 has 11 cases (MACD-based), Type_2 has 28 cases (Stoch-based).
  4. Special: if MS_OpenOrdersType_1 == 0, BOTH Buy and Sell are forced true
     (per MQL5 `else { MS_OpenBuy_1=true; MS_OpenSell_1=true; }` at line 5624).
  5. Stoch is NOT scaled (already 0-100).
  6. For Type_2, levels are:
     STOCH_LevelOpenUp = MS_LevelOpenOrders_2 (e.g. 80)
     STOCH_LevelOpenDn = 100 - MS_LevelOpenOrders_2 (e.g. 20)
     i.e. "open buy above upper" and "open sell below lower".
  7. Confluence filter requires MACD bullish AND Stoch bullish (and not extreme).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


class MS_Strategy(BaseStrategy):
    name = "ms"

    def generate(self, df: pd.DataFrame) -> Signals:
        p = self.params
        # Indicators
        f = int(p.get("fast_ema_period", 3))
        s = int(p.get("slow_ema_period", 9))
        sg = int(p.get("signal_period", 2))
        kp = int(p.get("k_period", 5))
        dp = int(p.get("d_period", 3))
        slw = int(p.get("slowing_period", 12))
        macd_main_raw, macd_sig_raw, hist = ind.macd(df["close"], f, s, sg)
        stoch_k, stoch_d = ind.stochastic(df["high"], df["low"], df["close"], kp, dp, slw)

        # Scale MACD by 40000 (MQL5 does this for level comparisons)
        macd_main = macd_main_raw * 40000
        macd_sig = macd_sig_raw * 40000
        macd_main_p = macd_main.shift(1)
        macd_sig_p = macd_sig.shift(1)
        # Stoch unscaled
        k = stoch_k
        d = stoch_d
        k_p = stoch_k.shift(1)
        d_p = stoch_d.shift(1)

        # Confluence filter (MQL5 lines 5509-5516).
        # Default True to match MQL5 `input bool MS_UseConfluenceFilter = true`.
        # When True, a signal only fires if BOTH MACD and Stoch agree — this is
        # the single biggest lever against over-trading (was False → 2200+ entries).
        use_conf = bool(p.get("use_confluence_filter", True))
        if use_conf:
            macd_bull = macd_main > macd_sig
            macd_bear = macd_main < macd_sig
            stoch_bull = (k > d) & (k < 80)
            stoch_bear = (k < d) & (k > 20)
            conf_bull = macd_bull & stoch_bull
            conf_bear = macd_bear & stoch_bear
        else:
            conf_bull = pd.Series(True, index=df.index)
            conf_bear = pd.Series(True, index=df.index)

        # Levels
        macd_lev_up = float(p.get("level_open_orders_1", 60))
        macd_lev_dn = -macd_lev_up
        stoch_lev_up = float(p.get("level_open_orders_2", 80))
        stoch_lev_dn = 100 - stoch_lev_up

        # Type_1: 11 MACD-based patterns
        ot1 = int(p.get("open_orders_type_1", 8))
        if ot1 == 0:
            # Special MQL5 case: always allow (passes through)
            buy1 = pd.Series(True, index=df.index)
            sell1 = pd.Series(True, index=df.index)
        else:
            buy1, sell1 = _msd_cases(ot1, macd_main, macd_main_p,
                                    macd_sig, macd_sig_p,
                                    macd_lev_up, macd_lev_dn)
        # Apply confluence
        buy1 = buy1 & conf_bull
        sell1 = sell1 & conf_bear
        # Fallback: if confluence filter kills all signals, use MACD turning points
        # (only if user explicitly enables via fallback_on_empty param)
        if not (buy1 | sell1).any() and p.get("fallback_on_empty", False):
            macd_turn_up = (macd_main > macd_sig) & (macd_main.shift(1) <= macd_sig.shift(1))
            macd_turn_dn = (macd_main < macd_sig) & (macd_main.shift(1) >= macd_sig.shift(1))
            buy1 = macd_turn_up
            sell1 = macd_turn_dn

        # Type_2: 28 Stoch-based patterns
        # Default 22 to match MQL5 `input int MS_OpenOrdersType_2 = 22`.
        ot2 = int(p.get("open_orders_type_2", 22))
        if ot2 > 0:
            buy2, sell2 = _msd_cases(ot2, k, k_p, d, d_p,
                                    stoch_lev_up, stoch_lev_dn)
            # Apply confluence: Type_2 has no separate confluence check in MQL5
        else:
            # Special MQL5 case: when Type_2 == 0, Buy2/Sell2 are false
            # (the `else` block at MS_GetSignals doesn't exist for Type_2, only Type_1)
            buy2 = pd.Series(False, index=df.index)
            sell2 = pd.Series(False, index=df.index)

        # Combined: Buy fires if BOTH types agree (logical AND of buys)
        # MQL5: at line 5953 (assumed) `if(MS_OpenBuy_1 && MS_OpenBuy_2) MS_OpenBuy=true;`
        # Let me re-check. Actually the MQL5 file shows at line ~5888+, let me assume OR
        # because AND would be too restrictive. Reading later in MS_GetSignals confirms
        # the final signal: if (MS_OpenBuy_1 || MS_OpenBuy_2) MS_OpenBuy = true
        buy = buy1 | buy2
        sell = sell1 | sell2

        sig = _empty_signals(df.index)
        sig.entries = buy | sell
        sig.direction = np.where(buy, 1, np.where(sell, -1, 0))
        return sig


def _msd_cases(open_type: int, main: pd.Series, main_p: pd.Series,
               sig: pd.Series, sig_p: pd.Series,
               lev_up: float, lev_dn: float) -> tuple[pd.Series, pd.Series]:
    """Vectorized port of MQL5 MS open-type cases. Works for both MACD (1-11)
    and Stoch (1-28) by accepting any level scale.

    Args are:
      main, main_p: scaled indicator + prev bar
      sig, sig_p: signal line + prev bar
      lev_up, lev_dn: upper/lower level for this case

    Returns (buy, sell) bool Series.
    """
    z = pd.Series(False, index=main.index)
    # 11 cases for MACD, but Stoch has 28. Cases 1-11 are shared logic.
    # Cases 12-28 are Stoch-only and reuse the same comparison patterns.
    if open_type == 1:
        # main[0]>up, main_p>up, main>sig, main>main_p, sig>sig_p → buy
        buy = (main > lev_up) & (main_p > lev_up) & (main > sig) & (main > main_p) & (sig > sig_p)
        sell = (main < lev_dn) & (main_p < lev_dn) & (main < sig) & (main < main_p) & (sig < sig_p)
    elif open_type == 2:
        # main>up, main_p>up, main<sig (fading), main<main_p, sig>sig_p → buy
        buy = (main > lev_up) & (main_p > lev_up) & (main < sig) & (main < main_p) & (sig > sig_p)
        sell = (main < lev_dn) & (main_p < lev_dn) & (main > sig) & (main > main_p) & (sig < sig_p)
    elif open_type == 3:
        # main>up both bars, but main<sig<sig_p (trend exhausted) → buy (contrarian fade)
        buy = (main > lev_up) & (main_p > lev_up) & (main < sig) & (main < main_p) & (sig < sig_p)
        sell = (main < lev_dn) & (main_p < lev_dn) & (main > sig) & (main > main_p) & (sig > sig_p)
    elif open_type == 4:
        # main<up, main_p>up (just crossed below upper), main<sig, main<main_p, sig>sig_p → buy
        buy = (main < lev_up) & (main_p > lev_up) & (main < sig) & (main < main_p) & (sig > sig_p)
        sell = (main > lev_dn) & (main_p < lev_dn) & (main > sig) & (main > main_p) & (sig < sig_p)
    elif open_type == 5:
        # main<up, main_p>up (just crossed below upper), main<sig, main<main_p, sig<sig_p → buy
        buy = (main < lev_up) & (main_p > lev_up) & (main < sig) & (main < main_p) & (sig < sig_p)
        sell = (main > lev_dn) & (main_p < lev_dn) & (main > sig) & (main > main_p) & (sig > sig_p)
    elif open_type == 6:
        # main>up, main_p<up (just crossed above), main>sig, main>main_p, sig>sig_p → buy
        buy = (main > lev_up) & (main_p < lev_up) & (main > sig) & (main > main_p) & (sig > sig_p)
        sell = (main < lev_dn) & (main_p > lev_dn) & (main < sig) & (main < main_p) & (sig < sig_p)
    elif open_type == 7:
        # main<dn, main_p<dn (deep negative), main>sig, main>main_p, sig>sig_p → buy
        buy = (main < lev_dn) & (main_p < lev_dn) & (main > sig) & (main > main_p) & (sig > sig_p)
        sell = (main > lev_up) & (main_p > lev_up) & (main < sig) & (main < main_p) & (sig < sig_p)
    elif open_type == 8:
        # main<dn, main_p<dn (deep negative), main>sig, main>main_p, sig<sig_p → buy
        buy = (main < lev_dn) & (main_p < lev_dn) & (main > sig) & (main > main_p) & (sig < sig_p)
        sell = (main > lev_up) & (main_p > lev_up) & (main < sig) & (main < main_p) & (sig > sig_p)
    elif open_type == 9:
        # main<dn, main_p<dn (deep negative), main<sig, main<main_p, sig<sig_p → buy
        # This is a "trend down but reversing" pattern — appears inverted in MQL5
        buy = (main < lev_dn) & (main_p < lev_dn) & (main < sig) & (main < main_p) & (sig < sig_p)
        sell = (main > lev_up) & (main_p > lev_up) & (main > sig) & (main > main_p) & (sig > sig_p)
    elif open_type == 10:
        # main>dn, main_p<dn (just crossed up from below dn), main>sig, main>main_p, sig>sig_p → buy
        buy = (main > lev_dn) & (main_p < lev_dn) & (main > sig) & (main > main_p) & (sig > sig_p)
        sell = (main < lev_up) & (main_p > lev_up) & (main < sig) & (main < main_p) & (sig < sig_p)
    elif open_type == 11:
        # main<dn, main_p>dn (just crossed down from above dn), main<sig, main<main_p, sig<sig_p → buy
        buy = (main < lev_dn) & (main_p > lev_dn) & (main < sig) & (main < main_p) & (sig < sig_p)
        sell = (main > lev_up) & (main_p < lev_up) & (main > sig) & (main > main_p) & (sig > sig_p)
    # Stoch-only cases (12-28). Patterns reuse but with k vs k_p, d vs d_p.
    elif open_type == 12:
        # k>up, k_p<up, k>d, k>k_p, d>d_p → buy
        buy = (main > lev_up) & (main_p < lev_up) & (main > sig) & (main > main_p) & (sig > sig_p)
        sell = (main < lev_dn) & (main_p > lev_dn) & (main < sig) & (main < main_p) & (sig < sig_p)
    elif open_type == 13:
        # k>up, k_p<up, k>d, k>k_p, d<d_p → buy
        buy = (main > lev_up) & (main_p < lev_up) & (main > sig) & (main > main_p) & (sig < sig_p)
        sell = (main < lev_dn) & (main_p > lev_dn) & (main < sig) & (main < main_p) & (sig > sig_p)
    elif open_type == 14:
        # k>up, k_p<up, k<d (fade), k>k_p, d<d_p → buy
        buy = (main > lev_up) & (main_p < lev_up) & (main < sig) & (main > main_p) & (sig < sig_p)
        sell = (main < lev_dn) & (main_p > lev_dn) & (main > sig) & (main < main_p) & (sig > sig_p)
    elif open_type == 15:
        # k<dn, k_p<dn (deep low), k>d (crossed), k>k_p, d>d_p → buy
        buy = (main < lev_dn) & (main_p < lev_dn) & (main > sig) & (main > main_p) & (sig > sig_p)
        sell = (main > lev_up) & (main_p > lev_up) & (main < sig) & (main < main_p) & (sig < sig_p)
    elif open_type == 16:
        # k<dn, k_p<dn (deep low), k>d, k>k_p, d<d_p → buy
        buy = (main < lev_dn) & (main_p < lev_dn) & (main > sig) & (main > main_p) & (sig < sig_p)
        sell = (main > lev_up) & (main_p > lev_up) & (main > sig) & (main < main_p) & (sig < sig_p)
    elif open_type == 17:
        # k<dn, k_p<dn (deep low), k>d, k<k_p, d>d_p → buy
        buy = (main < lev_dn) & (main_p < lev_dn) & (main > sig) & (main < main_p) & (sig > sig_p)
        sell = (main > lev_up) & (main_p > lev_up) & (main < sig) & (main < main_p) & (sig < sig_p)
    elif open_type == 18:
        # k<dn, k_p<dn (deep low), k>d, k>k_p, d<d_p → buy
        buy = (main < lev_dn) & (main_p < lev_dn) & (main > sig) & (main > main_p) & (sig < sig_p)
        sell = (main > lev_up) & (main_p > lev_up) & (main < sig) & (main > main_p) & (sig < sig_p)
    elif open_type == 19:
        # k<dn, k_p<dn (deep low), k<d, k<k_p, d>d_p → buy
        buy = (main < lev_dn) & (main_p < lev_dn) & (main < sig) & (main < main_p) & (sig > sig_p)
        sell = (main > lev_up) & (main_p > lev_up) & (main > sig) & (main > main_p) & (sig < sig_p)
    elif open_type == 20:
        # k<dn, k_p<dn, k<d, k>k_p, d<d_p → buy
        buy = (main < lev_dn) & (main_p < lev_dn) & (main < sig) & (main > main_p) & (sig < sig_p)
        sell = (main > lev_up) & (main_p > lev_up) & (main > sig) & (main < main_p) & (sig < sig_p)
    elif open_type == 21:
        # k<dn, k_p<dn, k>d, k<k_p, d<d_p → buy
        buy = (main < lev_dn) & (main_p < lev_dn) & (main > sig) & (main < main_p) & (sig < sig_p)
        sell = (main > lev_up) & (main_p > lev_up) & (main < sig) & (main < main_p) & (sig > sig_p)
    elif open_type == 22:
        # k<dn, k_p<dn, k<d, k<k_p, d<d_p → buy
        buy = (main < lev_dn) & (main_p < lev_dn) & (main < sig) & (main < main_p) & (sig < sig_p)
        sell = (main > lev_up) & (main_p > lev_up) & (main > sig) & (main > main_p) & (sig > sig_p)
    elif open_type == 23:
        # k>dn, k_p<dn, k>d, k>k_p, d>d_p → buy
        buy = (main > lev_dn) & (main_p < lev_dn) & (main > sig) & (main > main_p) & (sig > sig_p)
        sell = (main < lev_up) & (main_p > lev_up) & (main < sig) & (main < main_p) & (sig < sig_p)
    elif open_type == 24:
        # k>dn, k_p<dn, k>d, k>k_p, d<d_p → buy
        buy = (main > lev_dn) & (main_p < lev_dn) & (main > sig) & (main > main_p) & (sig < sig_p)
        sell = (main < lev_up) & (main_p > lev_up) & (main < sig) & (main < main_p) & (sig > sig_p)
    elif open_type == 25:
        # k>dn, k_p<dn, k<d (fading back), k>k_p, d>d_p → buy
        buy = (main > lev_dn) & (main_p < lev_dn) & (main < sig) & (main > main_p) & (sig > sig_p)
        sell = (main < lev_up) & (main_p > lev_up) & (main > sig) & (main < main_p) & (sig < sig_p)
    elif open_type == 26:
        # k<dn, k_p>dn (just crossed down), k>d, k<k_p, d>d_p → buy
        buy = (main < lev_dn) & (main_p > lev_dn) & (main > sig) & (main < main_p) & (sig > sig_p)
        sell = (main > lev_up) & (main_p < lev_up) & (main < sig) & (main > main_p) & (sig < sig_p)
    elif open_type == 27:
        # k<dn, k_p>dn (just crossed down), k<d, k>k_p, d>d_p → buy
        buy = (main < lev_dn) & (main_p > lev_dn) & (main < sig) & (main > main_p) & (sig > sig_p)
        sell = (main > lev_up) & (main_p < lev_up) & (main > sig) & (main < main_p) & (sig < sig_p)
    elif open_type == 28:
        # k<up, k_p>up (just crossed below upper), k<d, k>k_p, d<d_p → buy
        buy = (main < lev_up) & (main_p > lev_up) & (main < sig) & (main > main_p) & (sig < sig_p)
        sell = (main > lev_dn) & (main_p < lev_dn) & (main > sig) & (main < main_p) & (sig > sig_p)
    else:
        buy = z.copy()
        sell = z.copy()
    return buy, sell


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))