#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_grid.py  (Phase 2 - "true hybrid" neutral grid core engine)

WHAT "TRUE HYBRID" MEANS HERE
-----------------------------
Most grid bots are one of two things:
  * pure grid      - static range, dumb inventory, blows up in trends
  * pure trend     - low hit-rate directional betting

This engine is a HYBRID of a neutral grid base plus continuous risk/adaptive
overlays, designed the way we agreed after the calibration work:

  1. CONTINUOUS STATE, NO REGIME CLIFF.
     Instead of classifying "BULL/BEAR/CHOP" with ADX thresholds, an
     AdaptiveState keeps a soft trend score tau in (-1, 1) (tanh of
     log(price/EMA_fast) normalised by ATR%) and a smoothed daily ATR%.
     Grid asymmetry is a SMOOTH function of tau, so one wick cannot flip
     the whole deployment.

  2. NEUTRAL GRID CORE (mean-reversion engine).
     Flat start. Buy-ladder below price, sell-ladder above. A fill opens a
     cycle whose paired opposite order recycles inventory back to flat.
     Profit = spread - 2 x fee. This is the edge (fees + turnover).

  3. FUNDING-AWARE INVENTORY MANAGEMENT.
     In a trend the grid necessarily parks inventory on the counter-trend
     side (buys in a sell-off = long bags). Funding is settled per 8h and
     paid/received on net position. Inventory caps are widened on the side
     that EARNs funding and tightened on the side that PAYS.

  4. RISK OVERLAYS (survivability first - the "Simons" layer).
     - per-side notional inventory caps
     - margin-ratio kill switch (stop opening new cycles)
     - re-anchor only on STRUCTURAL events (trend flip, vol change >30%,
       price leaving the grid), never on daily PnL noise
     - compounding happens naturally: every redeploy recomputes level size
       from current cash equity

  5. SESSION PROFIT-TAKING (your operational model).
     The ladder is ~300 levels of pure LIMIT orders (buys below, sells
     above) with the step clamped to your verified 0.4%..1.7% window. As
     soon as the session is +0.1% of equity the residual inventory is
     flattened AT MARKET (taker), profit banked, and the bot re-opens
     COMPOUNDED. A market flatten only ever happens to bank or to cut a
     drawdown bag - never to enter or exit a grid trade.

ARCHITECTURE (layers)
---------------------
  AdaptiveState     trend score + ATR smoothing (pure)
  GridEngine        orders, cycles, fills, funding, fees, caps, sessions
                    (pure; exchange-agnostic)
  Simulator         deterministic candle-path replay (demo / backtest)
  BybitAdapter      (Phase 3) maps engine orders <-> Bybit v5 REST/WS

Example runs
------------
    python hybrid_grid.py --interval 4h --years 2                 # BTC, 300 lvl
    python hybrid_grid.py --symbol ETHUSDT --interval 4h --years 2
    python hybrid_grid.py --symbol SOLUSDT --interval 4h --years 1
    python hybrid_grid.py --alloc 0.2 --target 0.002 --quiet
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import itertools
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Small value objects
# ---------------------------------------------------------------------------
OID = itertools.count(1)

# side:  +1 = LONG (bought first, sells to close) | -1 = SHORT (sold first)
# kind:  "OPEN"  = seed order that opens a new cycle
#        "CLOSE" = paired order that closes an open cycle


class Order:
    __slots__ = ("oid", "side", "kind", "px", "qty", "cycle_id")

    def __init__(self, side, kind, px, qty, cycle_id=None):
        self.oid = next(OID)
        self.side = side          # +1 buy-side opens long / -1 sell-side
        self.kind = kind
        self.px = float(px)
        self.qty = float(qty)
        self.cycle_id = cycle_id

    @property
    def is_buy(self):
        # OPEN buy (long entry) or CLOSE buy (covers short)
        return self.kind == "OPEN" and self.side == 1 or \
               self.kind == "CLOSE" and self.side == -1

    @property
    def key(self):
        return (self.kind, self.side, round(self.px, 8))

    def __repr__(self):
        return "<O%s %s/%s px=%.4f qty=%.6f>" % (
            self.oid, "BUY" if self.is_buy else "SELL", self.kind,
            self.px, self.qty)


GLOBAL_CID = itertools.count(1)


class Cycle:
    """One open grid round-trip. side=+1 long (buy low, sell high)."""
    __slots__ = ("cid", "side", "open_px", "qty", "open_ts")

    def __init__(self, side, open_px, qty, open_ts):
        self.cid = next(GLOBAL_CID)
        self.side = side
        self.open_px = float(open_px)
        self.qty = float(qty)
        self.open_ts = open_ts

    def unrealized(self, px):
        return self.side * self.qty * (px - self.open_px)

    def realized(self, close_px):
        return self.side * self.qty * (close_px - self.open_px)


CYCLES = []  # registry so cid are unique per process
# ---------------------------------------------------------------------------
# AdaptiveState - continuous soft trend/volatility state (no regime cliff)
# ---------------------------------------------------------------------------
class AdaptiveState:
    """
    Maintains EMA-fast / ATR (Wilder-ish) on bar closes and produces:
      tau      trend score in (-1, 1),  >0 = bullish drift
      atr_pct  smoothed daily ATR as a fraction of price
    Everything continuous - no ADX thresholds, no discrete regime labels.
    """

    def __init__(self, ema_period=10, atr_period=14, score_scale=2.0):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.score_scale = score_scale
        self.ema = None
        self.atr = None
        self.tr_hist = []
        self.prev_close = None
        self.n = 0

    def update(self, close, high, low):
        self.n += 1
        a = 2.0 / (self.ema_period + 1.0)
        self.ema = close if self.ema is None \
            else close * a + self.ema * (1.0 - a)

        if self.prev_close is not None:
            tr = max(high - low, abs(high - self.prev_close),
                     abs(low - self.prev_close))
            self.tr_hist.append(tr)
            if len(self.tr_hist) >= self.atr_period:
                if self.atr is None:               # Wilder seed
                    self.atr = sum(self.tr_hist[-self.atr_period:]) \
                        / self.atr_period
                else:
                    self.atr = (self.atr * (self.atr_period - 1) + tr) \
                        / self.atr_period
        self.prev_close = close

    @property
    def ready(self):
        return self.atr is not None and self.ema is not None and self.ema > 0

    @property
    def atr_pct(self):
        return self.atr / self.ema if self.ready else 0.02

    @property
    def tau(self):
        """Soft trend score in (-1,1). Positive = upward drift."""
        if not self.ready:
            return 0.0
        drift = math.log(max(self.prev_close, 1e-12) / self.ema)
        z = drift / max(self.atr_pct * self.score_scale, 1e-4)
        return math.tanh(z)

    def vol_change_pct(self, ref_atr):
        if ref_atr is None or ref_atr <= 0 or not self.ready:
            return 0.0
        return abs(self.atr - ref_atr) / ref_atr

    def est_funding_rate(self):
        """Signed funding guess per settlement: longs pay when tau>0."""
        return math.tanh(self.tau * 0.8) * 0.0001
# ---------------------------------------------------------------------------
# GridEngine - neutral ladder + overlays
# ---------------------------------------------------------------------------
# Surgical feature gates — every smart-grid enhancement can be enabled or
# disabled independently (UI toggles mirror these keys 1:1).
FEATURES = {
    "session_banking": True,   # flatten & bank at +target, re-open compounded
    "compounding": True,       # redeploy level size from grown equity
    "funding_tilt": True,      # widen caps on the side that earns funding
    "asym_tilt": True,         # soft trend tilt on channel/fence split
    "drawdown_shed": True,     # market-flatten cut on structural break
    "reanchor_trend": True,    # structural re-anchor on trend flip
    "reanchor_vol": True,      # structural re-anchor on vol shift >30%
    "reanchor_exit": True,     # structural re-anchor when price leaves fences
    "reanchor_schedule": True, # scheduled refresh (step_days) if set
}


class GridEngine:
    """
    Session-based neutral grid (your operational model):

      * ~n_levels (300) LIMIT orders: buys below / sells above price, spaced
        adaptively inside [spacing_min, spacing_max] of price (defaults = your
        verified 0.4%..1.7% window). Everything is maker limit; nothing chases.
      * At every +target_pct of equity (0.1% default) the residual inventory
        is flattened AT MARKET, profit banked, redeploy COMPOUNDED (level size
        re-derived from grown cash equity).
      * Drawdown shed guards trends that never reach +0.1%.
      * Funding-aware inventory caps + structural re-anchors unchanged.
    """

    def __init__(self, adaptive, equity=10000.0, n_levels=300,
                 spacing_min=0.004, spacing_max=0.017, spacing_k=0.2,
                 asym=0.0, alloc=0.35, target_pct=0.001,
                 kill_ratio=0.8, step_days=0.0, shed_frac=0.12,
                 maker_bps=2.0, taker_bps=5.5,
                 span_mode="corridor", m_base=6.0, features=None):
        self.adaptive = adaptive
        self.equity0 = float(equity)
        self.n_levels = n_levels
        self.spacing_min = spacing_min
        self.spacing_max = spacing_max
        self.spacing_k = spacing_k
        self.asym = asym
        self.alloc = alloc
        self.target_pct = target_pct
        self.kill_ratio = kill_ratio
        self.step_days = step_days      # >0 => scheduled re-anchor refresh
        self.shed_frac = shed_frac
        self.maker_bps = maker_bps
        self.taker_bps = taker_bps
        self.span_mode = span_mode      # "corridor" | "channel"
        self.m_base = m_base            # channel mode: ATR multiplier
        self.features = dict(FEATURES)
        if features:
            self.features.update({k: bool(v) for k, v in features.items()
                                  if k in self.features})
        self._killed = False
        self.min_equity = float(equity)
        self.peak_equity = float(equity)
        self.session_start_eq = float(equity)
        self.sessions = 0
        self.shed_events = 0
        self.flatten_counts = {}

        self.orders = {}                # oid -> Order
        self.cycles = {}                # cid -> Cycle
        self.close_map = {}             # close-order oid -> cycle cid

        self.center = None
        self.f_up = self.f_dn = 0.0
        self.step_px = 0.0
        self.spacing_frac = 0.0
        self.anchor_atr = None
        self.anchor_tau = 0.0
        self.anchor_day = 0

        self.grid_pnl = 0.0
        self.funding_pnl = 0.0
        self.fees = 0.0
        self.fills = 0
        self.cycles_closed = 0
        self.killed_days = 0
        self.reanchors = 0
        self.max_abs_notional = 0.0
        self.events = []

    # ---- helpers -----------------------------------------------------------
    @property
    def fee_rate(self):
        return self.maker_bps / 10000.0

    @property
    def taker_rate(self):
        return self.taker_bps / 10000.0

    def cash_equity(self):
        return self.equity0 + self.grid_pnl + self.funding_pnl - self.fees

    def net_qty(self):
        return sum(c.side * c.qty for c in self.cycles.values())

    def side_notional(self, side):
        return sum(c.qty * c.open_px for c in self.cycles.values()
                   if c.side == side)

    def unrealized(self, px):
        return sum(c.unrealized(px) for c in self.cycles.values())

    def total_equity(self, px):
        return self.cash_equity() + self.unrealized(px)

    def _log(self, kind, msg):
        self.events.append((kind, msg))

    def _fees_on(self, qty, px):
        fee = qty * px * self.fee_rate
        self.fees += fee
        return fee

    # ---- range & sizing ----------------------------------------------------
    def _caps(self):
        """Funding-aware per-side notional caps vs CURRENT equity."""
        f = self.adaptive.est_funding_rate()     # >0 means longs pay
        if not self.features.get("funding_tilt", True):
            f = 0.0
        tilt = max(-0.5, min(0.5, f / 0.0002))
        eq = self.cash_equity()
        cap_long = eq * self.alloc * (1.0 - tilt)
        cap_short = eq * self.alloc * (1.0 + tilt)
        return cap_long, cap_short

    def _can_open(self, side):
        if self.side_notional(side) >= self._caps()[0 if side == 1 else 1]:
            return False
        return True

    def notional_risk(self):
        return max(self.side_notional(1), self.side_notional(-1))

    def _place(self, side, kind, px, qty, cycle_id=None):
        px = float(px)
        if kind == "OPEN":
            if any(o.kind == "OPEN" and o.side == side
                   and abs(o.px - px) < 1e-9 for o in self.orders.values()):
                return None
        o = Order(side, kind, px, qty, cycle_id)
        self.orders[o.oid] = o
        return o

    # ---- deployment --------------------------------------------------------
    def deploy(self, price, ts, day_idx):
        """
        Seed a fresh neutral ladder around `price`.

        Two ladder geometries (switch with span_mode):

        * "corridor" - your literal spec: 300 levels whose step is chosen
          from ATR inside the verified 0.4%..1.7% window; the span therefore
          becomes (n-1)*step (wide - covers multi-month trends).
        * "channel"  - Phase-1 calibration geometry: fences are m*ATR% with
          the soft trend tilt (m_up/m_dn), i.e. a CONTAINED channel; the step
          is then derived = clamp(channel/(n-1), 0.4%, 1.7%) and, if that
          floor forces it, the channel widens just enough to respect the
          minimum step. Keeps every level inside your verified spacing window
          while the CHANNEL itself stays where Phase-1 said ranges live.
        """
        self.center = price
        atr_p = self.adaptive.atr_pct
        tau = self.adaptive.tau
        eff_asym = self.asym if self.features.get("asym_tilt", True) else 0.0

        if self.span_mode == "channel":
            # 1) Phase-1 calibrated, continuously-asymmetric fences
            m_up = max(self.m_base * (1.0 + eff_asym * tau), 0.5)
            m_dn = max(self.m_base * (1.0 - eff_asym * tau), 0.5)
            w = m_up + m_dn
            f_up = m_up * atr_p
            f_dn = m_dn * atr_p
            # 2) step must live inside the verified window; if the channel is
            #    too narrow for (n-1) steps at >=min spacing, widen the
            #    channel (keep ratio) until it fits; if fewer levels fit
            #    inside the max spacing, shrink effective ladder length.
            total = f_up + f_dn
            target = total / max(self.n_levels - 1, 1)
            if target < self.spacing_min:
                total = self.spacing_min * (self.n_levels - 1)
            spacing = min(total / max(self.n_levels - 1, 1),
                          self.spacing_max)
            n_eff = max(2, int(round(total / spacing)) + 1)
            if n_eff > self.n_levels:          # defensive clamp
                spacing = total / max(self.n_levels - 1, 1)
                n_eff = self.n_levels
            scale = total / max(f_up + f_dn, 1e-12)
            f_up, f_dn = f_up * scale, f_dn * scale
        else:                                    # corridor mode
            spacing = max(self.spacing_min,
                          min(self.spacing_max, self.spacing_k * atr_p))
            w_up = 0.5 + 0.5 * eff_asym * tau
            w_up = min(0.62, max(0.38, w_up))
            total = spacing * (self.n_levels - 1)
            f_up, f_dn = total * w_up, total * (1.0 - w_up)
            n_eff = self.n_levels

        self.spacing_frac = spacing
        self.step_px = spacing * price
        self.f_up = f_up
        self.f_dn = f_dn
        p_hi = price * (1.0 + f_up)
        p_lo = price * (1.0 - f_dn)

        # cancel only pure seeds; never touch CLOSE orders of live cycles
        for oid in [oid for oid, o in self.orders.items() if o.kind == "OPEN"]:
            del self.orders[oid]

        half = max((n_eff - 1) / 2.0, 1.0)
        basis = self.cash_equity() if self.features.get("compounding", True) \
            else self.equity0
        qty_usd = basis * self.alloc / half
        n_seed = 0
        for j in range(n_eff):
            px = p_lo + j * (p_hi - p_lo) / max(n_eff - 1, 1)
            if px < price - self.step_px * 0.5:
                if self._place(1, "OPEN", px, qty_usd / px):
                    n_seed += 1
            elif px > price + self.step_px * 0.5:
                if self._place(-1, "OPEN", px, qty_usd / px):
                    n_seed += 1
        self.anchor_atr = self.adaptive.atr
        self.anchor_tau = tau
        self.anchor_day = day_idx
        self.reanchors += 1
        return n_seed
        # ---- structural re-anchor decisions (never PnL-based) ------------------
    def should_reanchor(self, price, day_idx):
        reasons = []
        tau = self.adaptive.tau
        if self.features.get("reanchor_trend", True) and \
                tau * self.anchor_tau < 0.0 and abs(tau) > 0.3:
            reasons.append("trend-flip")
        elif self.features.get("reanchor_vol", True) and self.anchor_atr and \
                self.adaptive.vol_change_pct(self.anchor_atr) > 0.30:
            reasons.append("vol-shift")
        if self.features.get("reanchor_exit", True) and self.center is not None:
            p_hi = self.center * (1.0 + self.f_up)
            p_lo = self.center * (1.0 - self.f_dn)
            if price > p_hi or price < p_lo:
                reasons.append("price-exit")
        if self.features.get("reanchor_schedule", True) and self.step_days > 0 \
                and day_idx - self.anchor_day >= self.step_days:
            reasons.append("schedule")
        return reasons

    # ---- order matching -----------------------------------------------------
    def _fill(self, order, ts):
        px = order.px
        qty = order.qty
        self.orders.pop(order.oid, None)
        self.fills += 1
        self._fees_on(qty, px)
        is_buy = order.is_buy
        long_cap, short_cap = self._caps()

        if order.kind == "OPEN":
            side = 1 if is_buy else -1
            cap = long_cap if side == 1 else short_cap
            if self._killed or self.side_notional(side) + qty * px > cap:
                self._log("blocked", "open %s at %.1f (inventory cap/kill)"
                          % ("LONG" if side == 1 else "SHORT", px))
                return
            cyc = Cycle(side, px, qty, ts)
            self.cycles[cyc.cid] = cyc
            target = px + self.step_px if side == 1 else px - self.step_px
            close_o = self._place(side, "CLOSE", target, qty, cycle_id=cyc.cid)
            self.close_map[close_o.oid] = cyc.cid
            self._log("open", "%s %.4f -> close %.4f qty %.5f"
                      % ("LONG" if side == 1 else "SHORT", px, target, qty))
        else:
            cid = self.close_map.pop(order.oid, None)
            cyc = self.cycles.pop(cid, None) if cid else None
            if cyc is None:
                return
            self.grid_pnl += cyc.realized(px)
            self.cycles_closed += 1
            self._log("close", "%s opened %.4f closed %.4f pnl %.4f"
                      % ("LONG" if cyc.side == 1 else "SHORT",
                         cyc.open_px, px, cyc.realized(px)))
            # re-arm the seed so this ladder level keeps recycling
            if not self._killed:
                self._place(cyc.side, "OPEN", cyc.open_px, qty)
        self.max_abs_notional = max(self.max_abs_notional,
                                    self.notional_risk())

    def _segment(self, pa, pb, ts):
        """Match limit orders along a monotonic move pa -> pb."""
        if pb == pa:
            return
        if pb < pa:                      # falling: buys fill
            cand = [o for o in self.orders.values()
                    if o.is_buy and pb <= o.px < pa]
            cand.sort(key=lambda o: -o.px)
        else:                            # rising: sells fill
            cand = [o for o in self.orders.values()
                    if not o.is_buy and pa < o.px <= pb]
            cand.sort(key=lambda o: o.px)
        for o in cand:
            if o.oid in self.orders:
                self._fill(o, ts)

    def apply_funding(self, rate):
        """Settlement: signed rate; longs pay when rate>0."""
        for cyc in self.cycles.values():
            self.funding_pnl -= rate * cyc.side * cyc.qty * cyc.open_px

    def flatten(self, px, ts, reason="manual"):
        """
        Market-flatten EVERYTHING (one taker fill per open cycle) and return
        to cash. Used both to bank a +target_pct session and to cut a
        drawdown bag. Charges taker fees, resets baselines so the next deploy
        starts a fresh session, COMPOUNDED on whatever equity remains.
        Returns net realized PnL of the flatten (after taker fees).
        """
        realized = 0.0
        taker_fees = 0.0
        for cyc in list(self.cycles.values()):
            pnl = cyc.realized(px)
            self.grid_pnl += pnl
            realized += pnl
            fee = cyc.qty * px * self.taker_rate
            self.fees += fee
            taker_fees += fee
            self.cycles_closed += 1
        self.cycles.clear()
        self.close_map.clear()
        for oid in [oid for oid, o in self.orders.items()]:   # all LIMIT rest
            del self.orders[oid]                              # is cancelled
        self.center = None
        self._killed = False
        self.peak_equity = self.cash_equity()       # re-baseline -> no loop
        self.session_start_eq = self.cash_equity()  # next session starts here
        self.flatten_counts[reason] = self.flatten_counts.get(reason, 0) + 1
        self._log(reason, "flatten @ %.1f realized %.2f" % (px, realized))
        return realized - taker_fees

    def flatten_reason(self, eq, px):
        """
        Decide whether to bank/cut at a day boundary. Returns one of
        "session" (target reached), "drawdown" (structural break), or None.
        Session checks first - profit first, survival second.
        """
        if not self.cycles:
            return None
        # cost to flatten at market with current inventory (BOTH sides - the
        # whole book is emptied, so a long AND short bag both pay taker)
        est_taker = (self.side_notional(1) + self.side_notional(-1)) \
            * self.taker_rate
        if self.target_pct > 0 and \
                self.features.get("session_banking", True):
            target = self.session_start_eq * (1.0 + self.target_pct)
            if eq - est_taker >= target:
                return "session"
        if self.shed_frac <= 0 or \
                not self.features.get("drawdown_shed", True):
            return None
        dd = 1.0 - eq / max(self.peak_equity, 1e-9)
        if dd <= self.shed_frac:
            return None
        tau = self.adaptive.tau
        long_side = self.side_notional(1)
        short_side = self.side_notional(-1)
        adverse = False
        if long_side > 0 or short_side > 0:
            u_long = sum(c.unrealized(px) for c in self.cycles.values()
                         if c.side == 1)
            u_short = sum(c.unrealized(px) for c in self.cycles.values()
                          if c.side == -1)
            adverse = ((long_side > 0 and
                        (-u_long / long_side > 0.35 or tau < -0.45)) or
                       (short_side > 0 and
                        (-u_short / short_side > 0.35 or tau > 0.45)))
        return "drawdown" if adverse else None

    def snapshot(self, px, ts, day_idx):
        return {
            "ts": ts, "day": day_idx, "px": px,
            "tau": self.adaptive.tau, "atr_pct": self.adaptive.atr_pct,
            "f_up": self.f_up, "f_dn": self.f_dn,
            "center": self.center, "n_orders": len(self.orders),
            "n_cycles": len(self.cycles), "net_qty": self.net_qty(),
            "long_notional": self.side_notional(1),
            "short_notional": self.side_notional(-1),
            "cash_equity": self.cash_equity(),
            "unrealized": self.unrealized(px),
            "grid_pnl": self.grid_pnl, "funding_pnl": self.funding_pnl,
            "fees": self.fees, "fills": self.fills,
            "cycles_closed": self.cycles_closed,
        }
# ---------------------------------------------------------------------------
# Simulator - deterministic bar-path replay (demo / future backtester)
# ---------------------------------------------------------------------------
def load_history(args):
    """Load daily candles via the bybit_backfill CSV store (download if missing)."""
    import bybit_backfill as bb
    interval = bb.normalize_interval(args.interval)
    path = bb.file_path(args.data_dir, args.category, args.symbol, interval)
    candles = bb.read_csv(path) if os.path.exists(path) else {}
    if not candles:
        end = int(__import__("time").time() * 1000)
        start = end - (int(args.years * 365) + 60) * 86_400_000
        print("downloading %s %s %s ..." % (args.symbol, interval,
                                            args.category), flush=True)
        rows = bb.fetch_kline_range(args.category, args.symbol, interval,
                                    start, end, 0.2)
        if not rows:
            raise SystemExit("no candles returned")
        bb.write_csv(path, rows)
        candles = bb.read_csv(path)
    tss = sorted(candles)
    bars = []
    for t in tss:
        c = candles[t]
        bars.append((t, float(c[0]), float(c[1]), float(c[2]), float(c[3])))
    return bars


def _date_of(ms):
    return dt.datetime.utcfromtimestamp(ms / 1000.0).date()


def simulate(bars, cfg):
    """
    Deterministic replay with decoupled granularities:

      * fills are matched on EVERY candle's path  open->low->high->close,
        so a 1h/4h feed gives intraday fill resolution;
      * the adaptive state (ATR%, trend score) is updated once per UTC day on
        the day's aggregate bar - same cadence as the live re-anchor loop.

    Re-anchor / funding / kill / shed all happen once per day at the day
    boundary, exactly as the production engine would schedule them.
    """
    adaptive = AdaptiveState()
    eng = GridEngine(adaptive,
                     equity=cfg["equity"], n_levels=cfg["n_levels"],
                     spacing_min=cfg["spacing_min"],
                     spacing_max=cfg["spacing_max"],
                     spacing_k=cfg["spacing_k"], asym=cfg["asym"],
                     span_mode=cfg["span_mode"], m_base=cfg["m_base"],
                     alloc=cfg["alloc"], target_pct=cfg["target_pct"],
                     kill_ratio=cfg["kill_ratio"], step_days=cfg["step_days"],
                     shed_frac=cfg["shed_frac"], maker_bps=cfg["maker_bps"],
                     taker_bps=cfg["taker_bps"],
                     features=cfg.get("features"))

    warmup_days = max(adaptive.atr_period + 2, 16)
    start_day = _date_of(bars[0][0])
    prev = None                 # last traded price (continuous across candles)
    cur_day = None
    day_agg = None              # (high, low, close) accumulated so far
    active = False
    hist = []
    last_snap = -1
    curve = []
    markers = []
    n_candles = len(bars)
    traded_days = 0

    def record(px, day_no, ts):
        s = eng.snapshot(px, ts, day_no)
        s["m2m"] = s["cash_equity"] + s["unrealized"]
        s["sessions"] = eng.sessions
        s["sheds"] = eng.shed_events
        s["spacing_frac"] = eng.spacing_frac
        s["span_mode"] = eng.span_mode
        curve.append(s)

    def decide(deploy_price, day_no, ts, is_reseed):
        """Day-boundary scheduling: flatten? -> kill -> deploy -> funding."""
        if not active:
            return
        # 1. day just ended: settle funding, then check profit/risk targets
        eng.apply_funding(adaptive.est_funding_rate())
        eq = eng.cash_equity() + eng.unrealized(deploy_price)
        eng.peak_equity = max(eng.peak_equity, eq)
        eng.min_equity = min(eng.min_equity, eq)
        reason = eng.flatten_reason(eq, deploy_price)
        if reason:
            base = eng.session_start_eq
            eng.flatten(deploy_price, ts, reason)
            if reason == "session":
                eng.sessions += 1
            else:
                eng.shed_events += 1
            session_gain = eng.cash_equity() - base   # true banked result
            markers.append({"day": day_no, "px": deploy_price, "kind": reason,
                            "gain": session_gain,
                            "cash": eng.cash_equity(),
                            "sessions": eng.sessions,
                            "sheds": eng.shed_events})
            if not cfg.get("quiet"):
                print("  %s #%d @ %.0f  session result %+.2f USDT  "
                      "(%+.3f%% of session eq)"
                      % ("BANK" if reason == "session" else "SHED",
                         eng.sessions if reason == "session"
                         else eng.shed_events, deploy_price, session_gain,
                         session_gain / max(base, 1.0) * 100.0))
        risk = eng.notional_risk() / max(eng.cash_equity(), 1.0)
        eng._killed = risk > cfg["kill_ratio"]
        if eng._killed:
            eng.killed_days += 1

        # 2. structural re-anchor / fresh (compounded) reseed after flatten
        reasons = eng.should_reanchor(deploy_price, day_no)
        if eng.center is None or reasons:
            n_seed = eng.deploy(deploy_price, ts, day_no)
            if not cfg.get("quiet"):
                if is_reseed:
                    print("  seed #%d @ %.0f  tau=%+.2f  step %.2f%%  "
                          "span +%.0f%%/-%.0f%%  orders=%d"
                          % (eng.reanchors, deploy_price, adaptive.tau,
                             eng.spacing_frac * 100, eng.f_up * 100,
                             eng.f_dn * 100, n_seed))
                else:
                    print("  re-anchor #%d @ %.0f  tau=%+.2f  step %.2f%%  "
                          "reasons=%s"
                          % (eng.reanchors, deploy_price, adaptive.tau,
                             eng.spacing_frac * 100, ",".join(reasons)))
        record(deploy_price, day_no, ts)

    for bar in bars:
        ts, o, h, l, c = bar
        d = _date_of(ts)
        day_no = (d - start_day).days

        if cur_day is None:
            # first candle of the dataset: just start accumulating
            cur_day = d
            day_agg = [h, l, c]
            if prev is None:
                prev = c
            if day_no >= warmup_days:
                active = True
                decide(prev, day_no, ts, is_reseed=True)
            continue

        if d != cur_day:
            # --- new UTC day: close the books on the day that just ended -----
            if day_agg is not None:
                adaptive.update(day_agg[2], day_agg[0], day_agg[1])
            if active:
                decide(prev, day_no, ts, is_reseed=False)
            cur_day = d
            day_agg = [h, l, c]
            if not active and day_no >= warmup_days:
                active = True
                decide(prev, day_no, ts, is_reseed=True)
        else:
            day_agg[0] = max(day_agg[0], h)
            day_agg[1] = min(day_agg[1], l)
            day_agg[2] = c

        if not active:
            prev = c
            continue

        # --- intraday fill precision: walk this candle's path -----------------
        for pa, pb in ((prev, o), (o, l), (l, h), (h, c)):
            eng._segment(pa, pb, ts)
        prev = c
        traded_days = max(traded_days, day_no)

        if day_no % 60 == 0 and day_no != last_snap and day_no != 0:
            hist.append(eng.snapshot(c, ts, day_no))
            last_snap = day_no

    residual = eng.unrealized(prev) if prev is not None else 0.0
    if active and prev is not None:
        record(prev, day_no, ts)
    eng.curve = curve
    eng.markers = markers
    return eng, hist, residual, warmup_days, n_candles, traded_days


# ---------------------------------------------------------------------------
# Reporting & CLI
# ---------------------------------------------------------------------------
def report(eng, hist, residual, warmup_days, n_candles, traded_days,
           symbol, interval):
    cash = eng.cash_equity()
    net = cash + residual - eng.equity0
    lines = []
    lines.append("=" * 76)
    lines.append("TRUE HYBRID NEUTRAL GRID - result  (%s %s)" % (symbol,
                                                                 interval))
    lines.append("=" * 76)
    lines.append("candles          : %d   trading days: %d"
                 % (n_candles, traded_days + 1 - warmup_days))
    lines.append("sessions closed  : %d  (banked at +%.1f%% of equity)"
                 % (eng.sessions, eng.target_pct * 100))
    lines.append("drawdown sheds   : %d   re-anchors: %d"
                 % (eng.shed_events, eng.reanchors))
    lines.append("min equity (m2m): %10.2f USDT  (%.1f%% from start)"
                 % (eng.min_equity,
                    (1.0 - eng.min_equity / eng.equity0) * 100))
    lines.append("grid step used   : %.2f%%  (clamped %.2f%%..%.2f%% of price)"
                 % (eng.spacing_frac * 100, eng.spacing_min * 100,
                    eng.spacing_max * 100))
    lines.append("")
    lines.append("fills            : %d   cycles closed: %d"
                 % (eng.fills, eng.cycles_closed))
    lines.append("grid PnL         : %10.2f USDT" % eng.grid_pnl)
    lines.append("funding PnL      : %10.2f USDT" % eng.funding_pnl)
    lines.append("fees paid        : %10.2f USDT" % eng.fees)
    lines.append("unrealized (res) : %10.2f USDT" % residual)
    lines.append("-------------------------------------------")
    lines.append("NET (cash+unrl)  : %10.2f USDT  (%+.2f%% of equity)"
                 % (net, net / eng.equity0 * 100))
    lines.append("max one-side inv.: %10.2f USDT" % eng.max_abs_notional)
    if eng.cycles_closed:
        avg = eng.grid_pnl / eng.cycles_closed
        lines.append("avg profit/cycle : %10.4f USDT" % avg)
    lines.append("")
    if hist:
        lines.append("  sampled state every ~60 bars (tau / fences / inv):")
        for s in hist[-6:]:
            lines.append(
                "   day %5d  px %9.0f  tau %+.2f  f +%.1f%%/-%.1f%%  "
                "long $%6.0f  short $%6.0f  eq %.0f"
                % (s["day"], s["px"], s["tau"], s["f_up"] * 100,
                   s["f_dn"] * 100, s["long_notional"], s["short_notional"],
                   s["cash_equity"] + s["unrealized"]))
    lines.append("=" * 76)
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(
        prog="hybrid_grid",
        description="True hybrid neutral grid core engine (deterministic "
                    "demo on Bybit history).")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--category", default="linear")
    p.add_argument("--interval", default="1d")
    p.add_argument("--data-dir", default="bybit_data")
    p.add_argument("--years", type=float, default=2.0,
                   help="bars to replay (last N years)")
    p.add_argument("--equity", type=float, default=10000.0)
    p.add_argument("--levels", type=int, default=300, dest="n_levels",
                   help="total ladder levels (both sides)")
    p.add_argument("--spacing-min", type=float, default=0.004,
                   dest="spacing_min",
                   help="smallest allowed grid step as price fraction "
                        "(your verified 0.4%%)")
    p.add_argument("--spacing-max", type=float, default=0.017,
                   dest="spacing_max",
                   help="largest allowed grid step as price fraction "
                        "(your verified 1.7%%)")
    p.add_argument("--spacing-k", type=float, default=0.2, dest="spacing_k",
                   help="maps daily ATR%% to step: step=clamp(ATR%%*k,..)")
    p.add_argument("--asym", type=float, default=0.0,
                   help="soft trend pull on up/down split (0 = symmetric)")
    p.add_argument("--span-mode", default="corridor", dest="span_mode",
                   choices=["corridor", "channel"],
                   help="corridor = 300 lvl * 0.4-1.7%% step (wide); "
                        "channel = Phase-1 m*ATR fences, step clamped inside "
                        "0.4-1.7%%")
    p.add_argument("--m-base", type=float, default=6.0, dest="m_base",
                   help="channel-mode ATR multiplier for the fences")
    p.add_argument("--target", type=float, default=0.001, dest="target_pct",
                   help="session profit target as equity fraction "
                        "(0.001 = +0.1%%); flatten & compound at target")
    p.add_argument("--alloc", type=float, default=0.35,
                   help="worst-case one-side notional as fraction of equity")
    p.add_argument("--maker-bps", type=float, default=2.0, dest="maker_bps",
                   help="maker fee in bps (limit fills)")
    p.add_argument("--taker-bps", type=float, default=5.5, dest="taker_bps",
                   help="taker fee in bps (market flattens)")
    p.add_argument("--kill-ratio", type=float, default=0.8,
                   help="notional/equity that disables new cycles")
    p.add_argument("--shed-frac", type=float, default=0.12, dest="shed_frac",
                   help="cut via market flatten when a session is this far "
                        "below its running peak (0 disables)")
    p.add_argument("--step-days", type=float, default=0.0,
                   help=">0 => scheduled re-anchor refresh every N days")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-flatten/re-anchor lines")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    bars = load_history(args)
    if len(bars) < 120:
        raise SystemExit("need more history (%d bars)." % len(bars))
    # restrict replay to the last N years of available bars
    cutoff = bars[-1][0] - int(args.years * 365) * 86_400_000
    bars = [b for b in bars if b[0] >= cutoff]

    cfg = dict(equity=args.equity, n_levels=args.n_levels,
               spacing_min=args.spacing_min, spacing_max=args.spacing_max,
               spacing_k=args.spacing_k, asym=args.asym,
               span_mode=args.span_mode, m_base=args.m_base,
               target_pct=args.target_pct, alloc=args.alloc,
               maker_bps=args.maker_bps, taker_bps=args.taker_bps,
               kill_ratio=args.kill_ratio, step_days=args.step_days,
               shed_frac=args.shed_frac, quiet=args.quiet)
    eng, hist, residual, warmup_days, n_candles, traded_days = \
        simulate(bars, cfg)
    print(report(eng, hist, residual, warmup_days, n_candles, traded_days,
                 args.symbol, bb_interval_label(args)))
    return 0


def bb_interval_label(args):
    import bybit_backfill as bb
    return bb.normalize_interval(args.interval)


if __name__ == "__main__":
    sys.exit(main())





