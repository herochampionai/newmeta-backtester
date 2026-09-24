"""Unit tests for backtester/grid_recovery.py — GridRecoveryManager +
surgical feature integration (B4: recovery_restart, basket_money_tp,
profit_lock_trail, carry_adjusted_tp).

Covers: basic grid opening/closing, TP/SL, recovery_restart cooldown +
size reduction, basket_money_tp close, profit_lock_trail floor,
carry_adjusted_tp rollover extension, summary close-reason breakdown.
"""
import numpy as np
import pandas as pd
import pytest

from backtester.grid_recovery import (
    GridRecoveryManager,
    GridLayer,
    GridClosedTrade,
    GridState,
    GRID_NONE,
    GRID_LOSS,
    GRID_PROFIT,
    GRID_LOSS_AND_PROFIT,
    RECOVERY_NONE,
    RECOVERY_LAST_CLOSING,
    RECOVERY_HIGHER_PROFITS,
)


def make_ts(hour: int, minute: int = 0, day: int = 1) -> pd.Timestamp:
    return pd.Timestamp(f"2024-01-{day:02d} {hour:02d}:{minute:02d}:00")


# PnL = (price - entry) * direction * lot * contract_size
# With pip_size=0.0001, lot=0.1, contract_size=10000:
#   50 pips (0.0050) → 0.005 * 0.1 * 10000 = $5.0
#   500 pips (0.0500) → 0.050 * 0.1 * 10000 = $50.0
LOT = 0.1
PIP = 0.0001
CONTRACT = 10_000


class TestGridConstants:
    def test_grid_constants(self) -> None:
        assert GRID_NONE == 0
        assert GRID_LOSS == 1
        assert GRID_PROFIT == 2
        assert GRID_LOSS_AND_PROFIT == 3

    def test_recovery_constants(self) -> None:
        assert RECOVERY_NONE == 0
        assert RECOVERY_LAST_CLOSING == 1
        assert RECOVERY_HIGHER_PROFITS == 2


class TestGridLayer:
    def test_defaults(self) -> None:
        l = GridLayer(strategy="fbb", direction=1, entry_price=1.1, lot=0.1, bar_index=0, layer_index=0)
        assert l.strategy == "fbb"
        assert l.direction == 1
        assert hasattr(l, "_entry_cost")


class TestGridRecoveryManagerInit:
    def test_defaults(self) -> None:
        mgr = GridRecoveryManager()
        assert mgr.grid_mode == GRID_LOSS_AND_PROFIT
        assert mgr.base_lot == 0.1
        assert mgr.max_layers == 8
        assert mgr.recovery_restart_enabled is False
        assert mgr.basket_money_tp_enabled is False
        assert mgr.profit_lock_trail_enabled is False
        assert mgr.carry_adjusted_tp_enabled is False

    def test_states_empty(self) -> None:
        mgr = GridRecoveryManager()
        assert mgr.states == {}


class TestBasicGrid:
    def test_open_layer(self) -> None:
        mgr = GridRecoveryManager(grid_mode=GRID_NONE, base_lot=0.1)
        closed = mgr.on_bar_close("fbb", signal_direction=1, bar_high=1.1050,
                                   bar_low=1.0950, bar_close=1.1000, bar_index=0)
        assert len(closed) == 0
        st = mgr._state("fbb")
        assert len(st.open_layers) == 1
        assert st.open_layers[0].direction == 1

    def test_no_signal_no_open(self) -> None:
        mgr = GridRecoveryManager(grid_mode=GRID_NONE)
        closed = mgr.on_bar_close("fbb", signal_direction=0, bar_high=1.1050,
                                   bar_low=1.0950, bar_close=1.1000, bar_index=0)
        assert len(closed) == 0
        assert len(mgr._state("fbb").open_layers) == 0

    def test_grid_loss_layer(self) -> None:
        """Grid layer opens on a SIGNAL bar when price has moved far enough."""
        mgr = GridRecoveryManager(grid_mode=GRID_LOSS, pip_size=PIP,
                                    pips_between_orders=30, max_layers=5, base_lot=LOT,
                                    contract_size=CONTRACT)
        # First open at 1.1000
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        # Price drops 100 pips (0.0100), send signal → grid layer opens
        mgr.on_bar_close("fbb", 1, 1.0980, 1.0900, 1.0950, 1)
        st = mgr._state("fbb")
        assert len(st.open_layers) == 2  # initial + grid layer

    def test_grid_loss_layer_no_signal_no_open(self) -> None:
        """Without a signal, grid layers are not opened (only on signal bars)."""
        mgr = GridRecoveryManager(grid_mode=GRID_LOSS, pip_size=PIP,
                                    pips_between_orders=30, max_layers=5, base_lot=LOT,
                                    contract_size=CONTRACT)
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        # Price drops 100 pips, but NO signal
        mgr.on_bar_close("fbb", 0, 1.0980, 1.0900, 1.0950, 1)
        st = mgr._state("fbb")
        assert len(st.open_layers) == 1  # only initial layer

    def test_grid_stop_loss(self) -> None:
        """SL triggers when PnL drops to -grid_stop_loss dollars.

        Floating point note: (1.0950 - 1.1000) / 0.0001 = -49.999... not exactly -50.
        So PnL ≈ -$5.0.  Use SL=4.0 to stay clearly above the actual loss.
        """
        mgr = GridRecoveryManager(grid_mode=GRID_NONE, grid_stop_loss=4.0,
                                    pip_size=PIP, contract_size=CONTRACT, base_lot=LOT)
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        # Move down 50 pips → PnL ≈ -$5.0 (below SL of $4)
        closed = mgr.on_bar_close("fbb", 0, 1.0950, 0.9950, 1.0950, 1)
        assert len(closed) == 1
        assert closed[0].reason == "sl_grid"

    def test_grid_take_profit(self) -> None:
        """TP triggers when PnL reaches grid_take_profit dollars.

        Floating point: (1.1050 - 1.1000) / 0.0001 = 49.999... → PnL ≈ $4.9999.
        Use TP=4.0 to stay clearly below the actual PnL.
        """
        mgr = GridRecoveryManager(grid_mode=GRID_NONE, grid_take_profit=4.0,
                                    pip_size=PIP, contract_size=CONTRACT, base_lot=LOT)
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        # Move up 50 pips → PnL ≈ $5.0 (above TP of $4)
        closed = mgr.on_bar_close("fbb", 0, 1.1080, 1.0990, 1.1050, 1)
        assert len(closed) == 1
        assert closed[0].reason == "tp_grid"


class TestRecoveryRestart:
    def test_cooldown_after_shed(self) -> None:
        """After a losing close, recovery_restart sets cooldown."""
        mgr = GridRecoveryManager(
            grid_mode=GRID_NONE, grid_stop_loss=5.0,
            pip_size=PIP, contract_size=CONTRACT, base_lot=LOT,
            recovery_restart_enabled=True, recovery_restart_cooldown_bars=5,
        )
        # Open position at 1.1000
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        # Move down 50 pips → SL triggers (loss of $5), sets cooldown
        closed = mgr.on_bar_close("fbb", 0, 1.0950, 0.9950, 1.0950, 1)
        assert len(closed) == 1
        assert closed[0].reason == "sl_grid"
        st = mgr._state("fbb")
        # Cooldown is set to 5, then immediately decremented by 1 in same call
        assert st.recovery_cooldown_remaining == 4

    def test_size_reduction_during_cooldown(self) -> None:
        mgr = GridRecoveryManager(
            grid_mode=GRID_NONE, grid_stop_loss=5.0,
            pip_size=PIP, contract_size=CONTRACT, base_lot=0.1,
            recovery_restart_enabled=True, recovery_restart_size_mult=0.5,
            recovery_restart_cooldown_bars=5,
        )
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        mgr.on_bar_close("fbb", 0, 1.0950, 0.9950, 1.0950, 1)
        st = mgr._state("fbb")
        assert st.recovery_cooldown_remaining == 4  # 5 set, 1 decremented
        # Re-enter during cooldown → should be half size
        mgr.on_bar_close("fbb", 1, 1.0980, 1.0920, 1.0950, 2)
        st = mgr._state("fbb")
        assert len(st.open_layers) == 1
        assert st.open_layers[0].lot == pytest.approx(0.05)  # 0.1 * 0.5
        # Cooldown should be at 3 (was 4, decremented again)
        assert st.recovery_cooldown_remaining == 3

    def test_cooldown_decrements_each_bar(self) -> None:
        mgr = GridRecoveryManager(
            grid_mode=GRID_NONE, grid_stop_loss=5.0,
            pip_size=PIP, contract_size=CONTRACT, base_lot=LOT,
            recovery_restart_enabled=True, recovery_restart_cooldown_bars=3,
        )
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        mgr.on_bar_close("fbb", 0, 1.0950, 0.9950, 1.0950, 1)
        # Cooldown set to 3, then immediately decremented → 2
        assert mgr._state("fbb").recovery_cooldown_remaining == 2

    def test_disabled_doesnt_set_cooldown(self) -> None:
        mgr = GridRecoveryManager(grid_mode=GRID_NONE, recovery_restart_enabled=False,
                                    grid_stop_loss=5.0, pip_size=PIP,
                                    contract_size=CONTRACT, base_lot=LOT)
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        mgr.on_bar_close("fbb", 0, 1.0950, 0.9950, 1.0950, 1)
        assert mgr._state("fbb").recovery_cooldown_remaining == 0


class TestBasketMoneyTP:
    def test_closes_on_basket_tp(self) -> None:
        """When basket PnL hits the $ threshold, all layers close."""
        mgr = GridRecoveryManager(
            grid_mode=GRID_NONE, pip_size=PIP, contract_size=CONTRACT, base_lot=LOT,
            basket_money_tp_enabled=True, basket_take_profit_usd=4.0,
            grid_take_profit=1_000_000,  # disable regular TP
        )
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        # Move up 50 pips → PnL = 0.005 * 0.1 * 10000 = $5.0 = basket_take_profit_usd
        closed = mgr.on_bar_close("fbb", 0, 1.1080, 1.0990, 1.1050, 1)
        assert len(closed) == 1
        assert closed[0].reason == "basket_money_tp"

    def test_no_close_below_threshold(self) -> None:
        mgr = GridRecoveryManager(
            grid_mode=GRID_NONE, pip_size=PIP, contract_size=CONTRACT, base_lot=LOT,
            basket_money_tp_enabled=True, basket_take_profit_usd=1000.0,
        )
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        # Only 5 pips up → PnL = $0.5, below $1000 threshold
        closed = mgr.on_bar_close("fbb", 0, 1.1005, 1.0990, 1.1005, 1)
        assert len(closed) == 0


class TestProfitLockTrail:
    def test_closes_when_pnl_drops_after_peak(self) -> None:
        """profit_lock_trail tracks peak PnL of current basket; if PnL falls
        by lock_pct below the peak, close.

        Floating point: 50-pip move gives PnL ≈ $4.9999, not exactly $5.
        With lock_pct=50%, floor = 4.9999 * 0.5 ≈ $2.50.
        Dropping to 1.0950 gives PnL ≈ -$5 (well below floor → close triggers).
        """
        mgr = GridRecoveryManager(
            grid_mode=GRID_NONE, pip_size=PIP, contract_size=CONTRACT, base_lot=LOT,
            profit_lock_trail_enabled=True, profit_lock_pct=50.0,
            grid_take_profit=1_000_000, grid_stop_loss=1_000_000,  # disable regular
        )
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        # Move up 50 pips → PnL ≈ $5.0, peak set
        mgr.on_bar_close("fbb", 0, 1.1100, 1.0990, 1.1050, 1)
        # No close yet (5 < 1M TP)
        # Move back down 50 pips → PnL ≈ -$5 (well below floor of ~$2.50)
        closed = mgr.on_bar_close("fbb", 0, 1.0980, 1.0920, 1.0950, 2)
        assert len(closed) == 1
        assert closed[0].reason == "profit_lock_trail"

    def test_disabled_doesnt_lock(self) -> None:
        mgr = GridRecoveryManager(
            grid_mode=GRID_NONE, pip_size=PIP, contract_size=CONTRACT, base_lot=LOT,
            profit_lock_trail_enabled=False, grid_take_profit=1_000_000,
            grid_stop_loss=1_000_000,
        )
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        # Move far down → no lock, no TP/SL → no close
        closed = mgr.on_bar_close("fbb", 0, 1.0980, 1.0920, 1.0950, 1)
        assert len(closed) == 0


class TestCarryAdjustedTP:
    def test_rollover_extension(self) -> None:
        mgr = GridRecoveryManager(
            carry_adjusted_tp_enabled=True, rollover_window_hours=8,
        )
        # 7:00 → next 8:00 boundary, 1h before → in window
        ts = make_ts(hour=7, minute=0, day=1)
        assert mgr._in_rollover_window(ts)

    def test_not_in_rollover(self) -> None:
        mgr = GridRecoveryManager(
            carry_adjusted_tp_enabled=True, rollover_window_hours=8,
        )
        ts = make_ts(hour=3, minute=0, day=1)
        assert not mgr._in_rollover_window(ts)

    def test_rollover_at_boundary(self) -> None:
        mgr = GridRecoveryManager(
            carry_adjusted_tp_enabled=True, rollover_window_hours=8,
        )
        # Exactly at 8:00 boundary → rollover is settling
        ts = make_ts(hour=8, minute=0, day=1)
        assert mgr._in_rollover_window(ts)

    def test_tp_extended_near_rollover(self) -> None:
        """Near rollover, TP is extended by tp_extension_pct, so 40 pips
        does NOT close (40 < 62.5 effective TP)."""
        mgr = GridRecoveryManager(
            grid_mode=GRID_NONE, pip_size=PIP, contract_size=CONTRACT, base_lot=LOT,
            carry_adjusted_tp_enabled=True, rollover_window_hours=8,
            tp_extension_pct=25.0, grid_take_profit=5.0,
            grid_stop_loss=1_000_000,
        )
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0,
                         bar_timestamp=make_ts(hour=7, minute=30, day=1))
        # Move up 40 pips → PnL = $4.0 < $5 × 1.25 = $6.25 extended TP
        closed = mgr.on_bar_close("fbb", 0, 1.1080, 1.0990, 1.1040, 1,
                                  bar_timestamp=make_ts(hour=7, minute=0, day=1))
        assert len(closed) == 0

    def test_disabled_rollover(self) -> None:
        mgr = GridRecoveryManager(
            carry_adjusted_tp_enabled=False, rollover_window_hours=8,
        )
        ts = make_ts(hour=7, minute=30, day=1)
        # _in_rollover_window still works for inspection
        assert mgr._in_rollover_window(ts)
        # But carry_adjusted_tp_enabled=False means no TP extension in on_bar_close


class TestSummary:
    def test_empty_summary(self) -> None:
        mgr = GridRecoveryManager()
        s = mgr.summary()
        assert s == {"n_grid_trades": 0}

    def test_summary_with_trades(self) -> None:
        mgr = GridRecoveryManager(
            grid_mode=GRID_NONE, pip_size=PIP, contract_size=CONTRACT, base_lot=LOT,
            grid_take_profit=4.0, grid_stop_loss=100.0,
        )
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        # Close at profit: 50 pips → $5.0
        mgr.on_bar_close("fbb", 0, 1.1080, 1.0990, 1.1050, 1)
        s = mgr.summary()
        assert s["n_grid_trades"] == 1
        assert "close_reasons" in s
        assert s["grid_total_pnl"] != 0
        assert "grid_win_rate" in s

    def test_summary_close_reasons(self) -> None:
        mgr = GridRecoveryManager(
            grid_mode=GRID_NONE, pip_size=PIP, contract_size=CONTRACT, base_lot=LOT,
            grid_take_profit=4.0, grid_stop_loss=4.0,
            recovery_restart_enabled=True,
        )
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        # TP close: 50 pips up → $5
        mgr.on_bar_close("fbb", 0, 1.1080, 1.0990, 1.1050, 1)
        s = mgr.summary()
        assert "close_reasons" in s
        assert "tp_grid" in s["close_reasons"]

    def test_summary_with_surgical_features(self) -> None:
        mgr = GridRecoveryManager(
            recovery_restart_enabled=True,
            basket_money_tp_enabled=True,
        )
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        s = mgr.summary()
        assert s == {"n_grid_trades": 0}


class TestToTradesDF:
    def test_empty(self) -> None:
        mgr = GridRecoveryManager()
        df = mgr.to_trades_df()
        assert df.empty

    def test_with_trades(self) -> None:
        mgr = GridRecoveryManager(
            grid_mode=GRID_NONE, pip_size=PIP, contract_size=CONTRACT, base_lot=LOT,
            grid_take_profit=4.0, grid_stop_loss=200.0,
        )
        mgr.on_bar_close("fbb", 1, 1.1050, 1.0950, 1.1000, 0)
        # TP close: 50 pips up → PnL ≈ $5
        mgr.on_bar_close("fbb", 0, 1.1080, 1.0990, 1.1050, 1)
        df = mgr.to_trades_df()
        assert len(df) == 1
        assert "strategy" in df.columns
        assert "direction" in df.columns
        assert "pnl" in df.columns
        assert "reason" in df.columns

    def test_total_closed_trades_sorted(self) -> None:
        mgr = GridRecoveryManager(
            grid_mode=GRID_NONE, pip_size=PIP, contract_size=CONTRACT, base_lot=LOT,
            grid_take_profit=4.0, grid_stop_loss=200.0,
        )
        # Strategy A: one trade, closes on TP at bar 5
        mgr.on_bar_close("A", 1, 1.1050, 1.0950, 1.1000, 0)
        mgr.on_bar_close("A", 0, 1.1080, 1.0990, 1.1050, 5)
        # Strategy B: one trade, closes on TP at bar 10
        mgr.on_bar_close("B", 1, 1.1050, 1.0950, 1.1000, 0)
        mgr.on_bar_close("B", 0, 1.1080, 1.0990, 1.1050, 10)
        trades = mgr.total_closed_trades()
        assert len(trades) == 2
        exit_bars = [t.exit_bar for t in trades]
        assert exit_bars == sorted(exit_bars)
