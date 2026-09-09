"""Grid + Recovery simulator for the multi-strategy EA.
Models the EA's behavior when:
  - MakeOrdersGrid > 0  (grid in loss / profit / both)
  - Per-strategy recovery lot multiplier after losses

State machine per strategy:
  - Each strategy has its own grid of layers
  - Initial trade on signal (size = base_lot × recovery_multiplier_if_applicable)
  - Grid mode 1 (in_loss): open new layer when price moves AGAINST by pips_between_orders
  - Grid mode 2 (in_profit): open new layer when price moves WITH profit by pips_between_orders
  - Grid mode 3 (both): either direction
  - Each grid layer uses lot × multiplier^(layer_index) when grid multiplier > 0
  - TP/SL per strategy grid: when total PnL of all layers hits grid_take_profit or grid_stop_loss,
    close ALL layers of that strategy (realizes combined PnL)

Recovery mode (per-strategy):
  - 0: no recovery multiplier
  - 1: after close-at-loss, next trade uses lot × recovery_multiplier
  - 2: based on highest profit reached (more complex)

This is per-strategy: each strategy has its own grid + recovery state.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


GRID_NONE = 0
GRID_LOSS = 1
GRID_PROFIT = 2
GRID_LOSS_AND_PROFIT = 3

RECOVERY_NONE = 0
RECOVERY_LAST_CLOSING = 1
RECOVERY_HIGHER_PROFITS = 2


@dataclass
class GridLayer:
    strategy: str
    direction: int          # +1 long, -1 short
    entry_price: float
    lot: float
    bar_index: int
    layer_index: int        # 0 for initial, 1+ for grid layers


@dataclass
class GridClosedTrade:
    strategy: str
    direction: int
    entry_price: float      # avg entry price of all layers
    exit_price: float
    lots: list              # list of lots per layer
    layers: int
    pnl: float              # realized total PnL in $
    exit_bar: int
    reason: str             # "tp_grid", "sl_grid", "reverse_signal"


@dataclass
class GridState:
    """Per-strategy grid state."""
    open_layers: list[GridLayer] = field(default_factory=list)
    closed_trades: list[GridClosedTrade] = field(default_factory=list)
    last_close_was_loss: bool = False
    last_close_pnl: float = 0.0
    last_close_lots: float = 0.0  # for recovery sizing


class GridRecoveryManager:
    """Simulate the EA's grid + recovery behavior per strategy.

    Usage:
        mgr = GridRecoveryManager(grid_mode=GRID_LOSS_AND_PROFIT,
                                   pips_between_orders=30,
                                   grid_lot_multiplier=1.5,
                                   grid_take_profit=50.0,
                                   grid_stop_loss=200.0,
                                   recovery_mode=RECOVERY_LAST_CLOSING,
                                   recovery_lot_multiplier=2.0,
                                   base_lot=0.1,
                                   max_layers=8,
                                   pip_size=0.0001,
                                   contract_size=100_000)
        for each bar:
            trades = mgr.on_bar_close(strategy_name, signal_direction, bar_high, bar_low,
                                      bar_close, bar_index, commission_pips, slippage_pips)
            # trades is list of GridClosedTrade (one per grid close)
    """
    def __init__(self, grid_mode: int = GRID_LOSS_AND_PROFIT,
                 pips_between_orders: float = 30.0,
                 grid_lot_multiplier: float = 1.5,
                 grid_take_profit: float = 50.0,
                 grid_stop_loss: float = 200.0,
                 max_layers: int = 8,
                 recovery_mode: int = RECOVERY_NONE,
                 recovery_lot_multiplier: float = 2.0,
                 base_lot: float = 0.1,
                 pip_size: float = 0.0001,
                 contract_size: float = 100_000,
                 per_strategies: dict | None = None):
        self.grid_mode = grid_mode
        self.pips_between_orders = pips_between_orders
        self.grid_lot_multiplier = grid_lot_multiplier
        self.grid_take_profit = grid_take_profit
        self.grid_stop_loss = grid_stop_loss
        self.max_layers = max_layers
        self.recovery_mode = recovery_mode
        self.recovery_lot_multiplier = recovery_lot_multiplier
        self.base_lot = base_lot
        self.pip_size = pip_size
        self.contract_size = contract_size
        self.states: dict[str, GridState] = {}
        self.equity_curve: list[tuple[int, float, str]] = []  # (bar, equity_delta, strategy)

    def _state(self, strategy: str) -> GridState:
        if strategy not in self.states:
            self.states[strategy] = GridState()
        return self.states[strategy]

    def _lot_for_new_layer(self, strategy: str, layer_index: int) -> float:
        """Lot for the (layer_index)th grid layer, accounting for recovery multiplier."""
        st = self._state(strategy)
        base = self.base_lot * (self.grid_lot_multiplier ** layer_index)
        # Recovery: if last close was loss AND recovery mode enabled
        if (self.recovery_mode == RECOVERY_LAST_CLOSING and
                st.last_close_was_loss and layer_index == 0):
            base *= self.recovery_lot_multiplier
        return base

    def on_bar_close(self, strategy: str, signal_direction: int,
                     bar_high: float, bar_low: float, bar_close: float,
                     bar_index: int, commission_pips: float = 0.7,
                     slippage_pips: float = 0.3) -> list[GridClosedTrade]:
        """Process one bar: handle grid layers + signal.
        signal_direction ∈ {-1, 0, +1}; 0 = no signal this bar.

        Returns list of closed grid trades (may be empty)."""
        st = self._state(strategy)
        closed: list[GridClosedTrade] = []

        # 1. Check if existing grid should be closed by TP/SL or reversed signal
        if st.open_layers:
            # Compute current PnL of all layers
            pnl = self._current_grid_pnl(st.open_layers, bar_close)
            should_close = False
            reason = ""
            if self.grid_take_profit > 0 and pnl >= self.grid_take_profit:
                should_close = True
                reason = "tp_grid"
            elif self.grid_stop_loss > 0 and pnl <= -self.grid_stop_loss:
                should_close = True
                reason = "sl_grid"
            elif (signal_direction != 0 and
                  signal_direction != st.open_layers[0].direction and
                  self.grid_mode == GRID_NONE):
                # Reverse signal — close existing in GRID_NONE mode
                should_close = True
                reason = "reverse_signal"
            if should_close:
                tr = self._close_grid(st, bar_close, bar_index, reason,
                                       commission_pips, slippage_pips)
                closed.append(tr)
                self.equity_curve.append((bar_index, tr.pnl, strategy))

        # 2. If signal fires
        if signal_direction != 0:
            if not st.open_layers:
                # Open initial layer (always allowed, regardless of grid_mode)
                lot = self._lot_for_new_layer(strategy, 0)
                layer = GridLayer(strategy=strategy, direction=signal_direction,
                                   entry_price=bar_close, lot=lot, bar_index=bar_index,
                                   layer_index=0)
                st.open_layers.append(layer)
                self._apply_entry_costs(st, commission_pips, slippage_pips)
            elif self.grid_mode != GRID_NONE:
                # Grid existing layers — only when grid mode is enabled
                new_layer = self._maybe_open_grid_layer(
                    st, signal_direction, bar_high, bar_low, bar_close, bar_index)
                if new_layer:
                    st.open_layers.append(new_layer)
                    self._apply_entry_costs(st, commission_pips, slippage_pips)

        return closed

    def _maybe_open_grid_layer(self, st: GridState, signal_dir: int,
                                 high: float, low: float, close: float,
                                 bar_index: int) -> GridLayer | None:
        """Check if a grid layer should open based on distance from existing layers."""
        if len(st.open_layers) >= self.max_layers:
            return None
        layers = st.open_layers
        # All layers should be same direction (grid mode keeps direction)
        if any(l.direction != layers[0].direction for l in layers):
            return None
        # Compute thresholds
        pip_dist = self.pips_between_orders * self.pip_size
        layer_idx = len(layers)
        # Higher/Lower open prices
        entries = np.array([l.entry_price for l in layers])
        highest = entries.max()
        lowest = entries.min()
        direction = layers[0].direction
        open_layer = False
        if direction == 1:  # Long grid
            # Grid in loss (price moved DOWN by pips from lowest open)
            if self.grid_mode in (GRID_LOSS, GRID_LOSS_AND_PROFIT):
                if (lowest - low) >= pip_dist:
                    open_layer = True
            # Grid in profit (price moved UP by pips from highest open)
            if self.grid_mode in (GRID_PROFIT, GRID_LOSS_AND_PROFIT):
                if (high - highest) >= pip_dist:
                    open_layer = True
        else:  # Short grid
            if self.grid_mode in (GRID_LOSS, GRID_LOSS_AND_PROFIT):
                if (high - highest) >= pip_dist:
                    open_layer = True
            if self.grid_mode in (GRID_PROFIT, GRID_LOSS_AND_PROFIT):
                if (lowest - low) >= pip_dist:
                    open_layer = True
        if open_layer:
            lot = self._lot_for_new_layer(st.open_layers[0].strategy, layer_idx)
            return GridLayer(strategy=st.open_layers[0].strategy, direction=direction,
                              entry_price=close, lot=lot, bar_index=bar_index,
                              layer_index=layer_idx)
        return None

    def _current_grid_pnl(self, layers: list[GridLayer], current_price: float) -> float:
        total = 0.0
        for l in layers:
            pips = (current_price - l.entry_price) / self.pip_size * l.direction
            total += pips * self.pip_size * l.lot * self.contract_size
        return total

    def _apply_entry_costs(self, st: GridState, commission_pips: float, slippage_pips: float) -> None:
        """Subtract entry costs from the most recent layer's PnL."""
        if not st.open_layers:
            return
        cost_per_lot = (commission_pips + slippage_pips) * self.pip_size * self.contract_size
        st.open_layers[-1]._entry_cost = cost_per_lot * st.open_layers[-1].lot

    def _close_grid(self, st: GridState, exit_price: float, bar_index: int,
                     reason: str, commission_pips: float, slippage_pips: float) -> GridClosedTrade:
        """Close all open layers of the strategy grid."""
        layers = list(st.open_layers)
        lots = [l.lot for l in layers]
        entries = np.array([l.entry_price for l in layers])
        avg_entry = float(np.average(entries, weights=lots))
        # PnL from price movement
        direction = layers[0].direction
        pips_moved = (exit_price - avg_entry) / self.pip_size * direction
        gross_pnl = pips_moved * self.pip_size * sum(lots) * self.contract_size
        # Commission + slippage on entry and exit
        total_lots = sum(lots)
        cost_per_lot = (commission_pips + slippage_pips) * self.pip_size * self.contract_size
        total_costs = cost_per_lot * total_lots * 2  # entry + exit
        net_pnl = gross_pnl - total_costs
        tr = GridClosedTrade(
            strategy=layers[0].strategy, direction=direction,
            entry_price=avg_entry, exit_price=exit_price, lots=lots,
            layers=len(layers), pnl=net_pnl, exit_bar=bar_index, reason=reason,
        )
        st.closed_trades.append(tr)
        st.last_close_was_loss = net_pnl < 0
        st.last_close_pnl = net_pnl
        st.last_close_lots = total_lots
        st.open_layers.clear()
        return tr

    def total_closed_trades(self) -> list[GridClosedTrade]:
        out = []
        for st in self.states.values():
            out.extend(st.closed_trades)
        return sorted(out, key=lambda t: t.exit_bar)

    def summary(self) -> dict:
        """Return aggregate metrics over all closed grid trades."""
        trades = self.total_closed_trades()
        if not trades:
            return {"n_grid_trades": 0}
        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        n = len(pnls)
        return {
            "n_grid_trades": n,
            "grid_total_pnl": float(pnls.sum()),
            "grid_win_rate": float((pnls > 0).mean()),
            "grid_avg_pnl": float(pnls.mean()),
            "grid_avg_win": float(wins.mean()) if len(wins) else 0.0,
            "grid_avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "grid_largest_win": float(pnls.max()),
            "grid_largest_loss": float(pnls.min()),
            "grid_max_layers": max((t.layers for t in trades), default=0),
        }

    def to_trades_df(self) -> pd.DataFrame:
        trades = self.total_closed_trades()
        if not trades:
            return pd.DataFrame()
        rows = []
        for t in trades:
            rows.append({
                "strategy": t.strategy, "direction": t.direction,
                "entry": t.entry_price, "exit": t.exit_price,
                "lots": t.lots, "n_layers": t.layers,
                "pnl": t.pnl, "exit_bar": t.exit_bar, "reason": t.reason,
            })
        return pd.DataFrame(rows)


# Patch: add _entry_cost attribute to GridLayer for accounting
GridLayer._entry_cost = 0.0