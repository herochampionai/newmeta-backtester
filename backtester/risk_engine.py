"""Adaptive risk engine — multi-layer SL/TP, smart sizing, FVG protection, scale-outs.

Layers per position:
  SL: hard = max(fixed_pips, atr_mult * ATR) → breakeven at +1R → ATR trailing
  TP: TP1 (xR, take half) + TP2 (weighted blend of fixed-pips and ATR targets,
      remainder rides trailing). If FVG protection holds, TP1 is DEFERRED —
      why take profit when a bullish FVG is pushing you up.
  Scale-out: close half of remainder when RSI is OB/OS AND opposing
      divergence prints against the running position.
  Sizing: risk a fixed % of equity on the hard-SL distance, scaled by vote
      confidence (majority agreement) and the AdaptiveSizer streak state.

One position at a time; new entries while open are ignored.
Fills at stop/target levels with slippage; commission per close.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from . import adaptive as _ad
from strategies import indicators as ind


@dataclass
class RiskConfig:
    risk_pct: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 5.0
    base_lot: float = 0.1          # fallback when SL distance is degenerate
    sl_fixed_pips: float = 50.0
    sl_atr_mult: float = 1.5
    atr_period: int = 14
    breakeven_at_R: float = 1.0
    trailing_start_R: float = 1.5
    trailing_atr_mult: float = 1.0
    tp1_R: float = 1.5
    tp2_fixed_pips: float = 100.0
    tp2_atr_mult: float = 3.0
    tp2_w_fixed: float = 0.5       # weight of fixed target in TP2 blend
    fvg_lookback: int = 20         # bars to look back for fresh FVG
    fvg_fresh_bars: int = 5        # gap must have formed within these bars
    fvg_hold: bool = True          # defer TP1 while protected by FVG
    rsi_period: int = 14
    rsi_ob: float = 70.0
    rsi_os: float = 30.0
    div_lookback: int = 30
    commission_pips: float = 0.7
    slippage_pips: float = 0.3
    pip_size: float = 0.0001
    contract_size: float = 100_000.0
    init_cash: float = 10_000.0

    @classmethod
    def profile(cls, name: str, **overrides) -> "RiskConfig":
        """Profit-mode presets.

        snitch: fast money — TP1 @1.0R, trail starts @1.0R tight (0.7 ATR),
            no FVG hold, eager scale-out (RSI 65/35).
        balanced: TP1 @1.5R, trail @1.5R / 1.0 ATR, FVG hold on.
        wide: runners — TP1 @2.0R, trail @2.0R / 1.5 ATR, FVG hold on,
            TP2 pushed (4 ATR leg), patient scale-out (RSI 75/25).
        """
        presets = {
            "snitch": dict(tp1_R=1.0, trailing_start_R=1.0, trailing_atr_mult=0.7,
                           fvg_hold=False, rsi_ob=65.0, rsi_os=35.0,
                           tp2_fixed_pips=60.0, tp2_atr_mult=2.0),
            "balanced": dict(),
            "wide": dict(tp1_R=2.0, trailing_start_R=2.0, trailing_atr_mult=1.5,
                         fvg_hold=True, rsi_ob=75.0, rsi_os=25.0,
                         tp2_fixed_pips=150.0, tp2_atr_mult=4.0),
        }
        if name not in presets:
            raise ValueError(f"unknown profit profile: {name}")
        cfg = cls(**{**presets[name], **overrides})
        return cfg


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _fresh_fvg(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               i: int, lookback: int, fresh_bars: int = 5):
    """Most recent unmitigated FVG strictly before bar i.

    The gap must have FORMED within the last `fresh_bars` (fresh push),
    and still hold (unmitigated) — stale zones don't count as protection.
    Returns ('bull', top) / ('bear', bottom) / (None, nan).
    """
    start = max(2, i - lookback)
    for j in range(i - 1, start - 1, -1):
        if i - 1 - j > fresh_bars:
            break
        if low[j] > high[j - 2]:  # bullish gap
            if close[i - 1] > low[j] and low[i - 1] > high[j - 2]:
                return "bull", low[j]
        if high[j] < low[j - 2]:  # bearish gap
            if close[i - 1] < high[j] and high[i - 1] < low[j - 2]:
                return "bear", high[j]
    return None, np.nan


def run_risk(df: pd.DataFrame, entries: pd.Series, direction: pd.Series,
             confidence: pd.Series | None = None,
             cfg: RiskConfig | None = None,
             sizer: _ad.AdaptiveSizer | None = None) -> dict:
    """Event-driven backtest with adaptive risk layers.

    entries/direction: voted signals (direction ∈ {-1,0,+1}).
    confidence: 0..1 vote strength per bar (optional; default 0.75).
    Returns dict(equity, trades, stats).
    """
    cfg = cfg or RiskConfig()
    sizer = sizer or _ad.AdaptiveSizer()
    n = len(df)
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    ent = entries.values.astype(bool) if isinstance(entries, pd.Series) else np.asarray(entries, bool)
    drc = direction.values.astype(int) if isinstance(direction, pd.Series) else np.asarray(direction, int)
    conf = (confidence.values if isinstance(confidence, pd.Series)
            else np.full(n, 0.75) if confidence is None else np.asarray(confidence, float))

    atr = _atr(df["high"], df["low"], df["close"], cfg.atr_period).values
    from strategies.triple_rsi import _detect_divergence as _div
    rsi = ind.rsi(df["close"], cfg.rsi_period)
    bull_div, bear_div = _div(df["close"], rsi, cfg.div_lookback)
    bull_div_v, bear_div_v = bull_div.values, bear_div.values

    pip = cfg.pip_size
    cost_per_close = (cfg.commission_pips + cfg.slippage_pips) * pip * cfg.contract_size  # $ per lot

    equity = cfg.init_cash
    eq_curve = np.full(n, equity)
    trades = []
    pos = None  # dict while open

    def open_pos(i, d):
        nonlocal equity
        a = atr[i] if atr[i] > 0 else cfg.sl_fixed_pips * pip
        sl_dist = max(cfg.sl_fixed_pips * pip, cfg.sl_atr_mult * a)
        risk_dollars = equity * cfg.risk_pct
        lots = risk_dollars / (sl_dist * cfg.contract_size) if sl_dist > 0 else cfg.base_lot
        lots *= (0.5 + 0.5 * float(np.clip(conf[i], 0, 1)))  # vote confidence 0.5x..1x
        lots *= (sizer.get_lot(cfg.base_lot) / cfg.base_lot)  # streak state
        lots = float(np.clip(lots, cfg.min_lot, cfg.max_lot))
        entry = cl[i]
        tp1d = cfg.tp1_R * sl_dist
        blend = (cfg.tp2_w_fixed * cfg.tp2_fixed_pips * pip
                 + (1 - cfg.tp2_w_fixed) * cfg.tp2_atr_mult * a)
        tp2d = max(blend, 1.25 * tp1d)  # TP2 must sit beyond TP1 by construction
        return {"dir": d, "entry": entry, "lots": lots, "rem": lots, "bar": i,
                "sl_dist": sl_dist,
                "sl": entry - d * sl_dist,
                "tp1": entry + d * tp1d,
                "tp2": entry + d * tp2d, "tp1_done": False, "be": False, "trail": None}

    def close_lots(p, lots, price, i, reason):
        nonlocal equity
        pnl = p["dir"] * (price - p["entry"]) * lots * cfg.contract_size - cost_per_close * lots
        equity += pnl
        trades.append({"entry_bar": p["bar"], "exit_bar": i, "dir": p["dir"],
                       "lots": lots, "pnl": pnl, "reason": reason})
        sizer.on_trade_close(pnl)
        p["rem"] -= lots

    for i in range(1, n):
        if pos is None:
            if ent[i] and drc[i] != 0:
                pos = open_pos(i, int(drc[i]))
            eq_curve[i] = equity
            continue
        d = pos["dir"]
        # --- trailing ratchet ---
        if d == 1:
            fav = hi[i]
            if fav >= pos["entry"] + cfg.trailing_start_R * pos["sl_dist"]:
                t = fav - cfg.trailing_atr_mult * atr[i]
                pos["trail"] = t if pos["trail"] is None else max(pos["trail"], t)
        else:
            fav = lo[i]
            if fav <= pos["entry"] - cfg.trailing_start_R * pos["sl_dist"]:
                t = fav + cfg.trailing_atr_mult * atr[i]
                pos["trail"] = t if pos["trail"] is None else min(pos["trail"], t)
        eff_sl = pos["trail"] if pos["trail"] is not None else pos["sl"]
        # --- hard SL / trailing stop ---
        hit_sl = (lo[i] <= eff_sl) if d == 1 else (hi[i] >= eff_sl)
        if hit_sl:
            close_lots(pos, pos["rem"], eff_sl, i, "trail" if pos["trail"] is not None else "sl")
            pos = None
            eq_curve[i] = equity
            continue
        # --- breakeven at +1R ---
        if not pos["be"]:
            be_trig = (hi[i] >= pos["entry"] + cfg.breakeven_at_R * pos["sl_dist"]) if d == 1 else \
                      (lo[i] <= pos["entry"] - cfg.breakeven_at_R * pos["sl_dist"])
            if be_trig:
                pos["sl"] = pos["entry"]
                pos["be"] = True
        # --- FVG protection (hold, don't take profit into strength) ---
        fvg_side, _ = _fresh_fvg(hi, lo, cl, i, cfg.fvg_lookback, cfg.fvg_fresh_bars)
        protected = cfg.fvg_hold and ((d == 1 and fvg_side == "bull") or (d == -1 and fvg_side == "bear"))
        # --- TP1: take half unless protected ---
        if not pos["tp1_done"] and not protected:
            hit_tp1 = (hi[i] >= pos["tp1"]) if d == 1 else (lo[i] <= pos["tp1"])
            if hit_tp1:
                close_lots(pos, pos["rem"] / 2, pos["tp1"], i, "tp1")
                pos["tp1_done"] = True
        # --- TP2: remainder at blended target ---
        if pos["rem"] > 0:
            hit_tp2 = (hi[i] >= pos["tp2"]) if d == 1 else (lo[i] <= pos["tp2"])
            if hit_tp2:
                close_lots(pos, pos["rem"], pos["tp2"], i, "tp2")
                pos = None
                eq_curve[i] = equity
                continue
        # --- scale-out: OB/OS + opposing divergence → close half of remainder ---
        if pos is not None and pos["rem"] > 0:
            if d == 1 and rsi.iloc[i] > cfg.rsi_ob and bear_div_v[i]:
                close_lots(pos, pos["rem"] / 2, cl[i], i, "scaleout_div")
            elif d == -1 and rsi.iloc[i] < cfg.rsi_os and bull_div_v[i]:
                close_lots(pos, pos["rem"] / 2, cl[i], i, "scaleout_div")
        eq_curve[i] = equity
    # flatten any open position at last close
    if pos is not None and pos["rem"] > 0:
        close_lots(pos, pos["rem"], cl[-1], n - 1, "eod")
    eq_curve[-1] = equity

    tdf = pd.DataFrame(trades)
    wins = (tdf["pnl"] > 0).sum() if not tdf.empty else 0
    stats = {
        "n_trades": len(tdf),
        "win_rate": float(wins / len(tdf)) if len(tdf) else 0.0,
        "net_pnl": float(tdf["pnl"].sum()) if len(tdf) else 0.0,
        "final_equity": float(equity),
        "by_reason": tdf.groupby("reason")["pnl"].agg(["count", "sum"]).to_dict("index") if len(tdf) else {},
        "sizer_state": sizer.get_state(),
    }
    return {"equity": pd.Series(eq_curve, index=df.index), "trades": tdf, "stats": stats}
