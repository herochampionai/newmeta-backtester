"""Deep backtest mode — tick-level execution for high accuracy.

Regular backtest uses OHLC bars (next-bar fill). Deep backtest uses actual
tick data (or synthetic tick-from-bar) and simulates:
  - Variable spread (real or synthetic)
  - Slippage per fill (path-dependent)
  - Order queue position (we hit the bid/ask we see)
  - Stop/take-profit triggered intra-bar

For now, deep backtest = synthetic tick-from-bar (zig-zag) + variable spread.
Real tick data requires MT5 + recent history.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from backtester.metrics_v2 import compute_all
from backtester.analytics import streak_stats, annual_trade_count
from backtester.grid_recovery import GridRecoveryManager, GRID_NONE
from data.tick_data import synthesize_ticks_from_bars, fetch_ticks_with_priority
from data.spread_spec import default_spread_series, get_spreads_mt5


def deep_backtest(df: pd.DataFrame,
                   signals_by_strategy: dict,
                   ticks_per_bar: int = 20,
                   use_real_ticks: bool = False,
                   use_real_spreads: bool = False,
                   symbol: str = "EURUSD",
                   tick_start: str = "2024-01-01",
                   tick_end: str = "2024-12-31",
                   grid_mode: int = GRID_NONE,
                   pips_between_orders: float = 30.0,
                   grid_lot_multiplier: float = 1.5,
                   grid_take_profit: float = 50.0,
                   grid_stop_loss: float = 200.0,
                   max_grid_layers: int = 8,
                   recovery_mode: int = 0,
                   recovery_lot_multiplier: float = 2.0,
                   base_lot: float = 0.1,
                   contract_size: float = 100_000,
                   pip_size: float = 0.0001,
                   max_spread_pips: float | None = 5.0,
                   slippage_pips: float = 0.2,
                   tp_pips: float | None = None,
                   sl_pips: float | None = None,
                   partial_frac: float = 0.0,
                   leverage: float = 30.0,
                   borrow_pct_per_day: float = 0.0,
                   stopout_level_pct: float = 50.0,
                   progress_every: int = 5000) -> dict:
    """Run backtest on tick data for higher accuracy.

    Args:
        df: OHLCV DataFrame (entry-level granularity)
        signals_by_strategy: {name: (entries, direction)}
        ticks_per_bar: how many synthetic ticks per OHLC bar (if use_real_ticks=False)
        use_real_ticks: try to fetch real ticks from MT5 (slow)
        use_real_spreads: try to fetch real broker spread history
        symbol: instrument symbol (for real tick / spread fetch)
        tick_start: start date for real tick fetch
        tick_end: end date for real tick fetch
        grid_mode: GRID_NONE for pure, or grid mode
        pips_between_orders: distance between grid layers (pips)
        grid_lot_multiplier: lot multiplier per grid layer
        grid_take_profit: grid basket TP in USD
        grid_stop_loss: grid basket SL in USD
        max_grid_layers: cap on number of grid layers
        recovery_mode: 0=none, 1=last close martingale
        recovery_lot_multiplier: lot multiplier for recovery trades
        base_lot: initial lot size
        contract_size: units per lot
        pip_size: price precision for pip conversion

    Returns dict with: equity, trades, metrics, tick_stats, etc.
    """
    # 1. Get ticks
    if use_real_ticks:
        # fetch_ticks_with_priority returns (df, source, info) — 3-tuple
        ticks, _tick_source_str, info = fetch_ticks_with_priority(
            symbol, tick_start, tick_end,
            prefer_ticks=True, ticks_per_bar=ticks_per_bar,
        )
        if ticks is not None:
            tick_source = info.get("source", "real_ticks")
        else:
            ticks = synthesize_ticks_from_bars(df, ticks_per_bar=ticks_per_bar)
            tick_source = "synthetic_ticks_fallback"
    else:
        ticks = synthesize_ticks_from_bars(df, ticks_per_bar=ticks_per_bar)
        tick_source = "synthetic_ticks"

    # 2. Get spreads
    if use_real_spreads:
        real_spreads = get_spreads_mt5(symbol, n_days=7)
        if real_spreads is not None and len(real_spreads) > 0:
            spreads = real_spreads.reindex(df.index, method="ffill").fillna(method="bfill")
            if spreads.isna().all():
                spreads = default_spread_series(df.index)
        else:
            spreads = default_spread_series(df.index)
    else:
        spreads = default_spread_series(df.index)

    # 3. Build grid manager (use forwarded params so tick sim matches OHLC config)
    mgr = GridRecoveryManager(
        grid_mode=grid_mode,
        pips_between_orders=pips_between_orders,
        grid_lot_multiplier=grid_lot_multiplier,
        grid_take_profit=grid_take_profit,
        grid_stop_loss=grid_stop_loss,
        max_layers=max_grid_layers,
        recovery_mode=recovery_mode,
        recovery_lot_multiplier=recovery_lot_multiplier,
        base_lot=base_lot,
        pip_size=pip_size,
        contract_size=contract_size,
    )

    # 4. Tick-by-tick loop with intra-bar order triggers (MT5 every-tick style)
    #  Mid ticks are direction-aware (bull: O->L->H->C, bear: O->H->L->C) so
    #  SL-before-TP ambiguity resolves like MT5. Fills use bid/ask, not close.
    from backtester.engine_pure import infer_periods_per_year
    n_bars = len(df)
    sig_arrays = {}
    for name, sig in signals_by_strategy.items():
        if len(sig) == 3:
            e, ex, d = sig
            if isinstance(e, pd.Series):
                e = e.values
            if isinstance(ex, pd.Series):
                ex = ex.values
            if isinstance(d, pd.Series):
                d = d.values
            sig_arrays[name] = (e.astype(bool), ex.astype(bool), d.astype(int))
        else:
            entries, direction = sig
            if isinstance(entries, pd.Series):
                entries = entries.values
            if isinstance(direction, pd.Series):
                direction = direction.values
            sig_arrays[name] = (entries.astype(bool), np.zeros(n_bars, dtype=bool), direction.astype(int))

    spread_arr = spreads.values
    avg_spread = float(np.mean(spread_arr)) if len(spread_arr) else 1.0
    # Synthesize mids once with realistic spread, then re-derive bid/ask per bar
    # so per-bar session spread (night widening) is honoured per tick.
    ticks = synthesize_ticks_from_bars(df, ticks_per_bar=ticks_per_bar, seed=42,
                                       spread_pips=avg_spread, pip_size=pip_size)
    tick_mids = ticks["last"].values if len(ticks) else np.array([])
    ticks_per = ticks_per_bar

    fills_count = 0
    intra_bar_fills = 0
    next_bar_fills = 0
    slippage_frac = 0.00002  # ~0.2 pip adverse MT5-style slippage baseline

    if grid_mode == GRID_NONE:
        # --- Pure tick-accurate single-position sim per strategy (order-ticket fills) ---
        import math
        from backtester.order_engine import market_fill, check_exits_tick, Ticket
        all_trades: list[dict] = []
        equity = pd.Series(10000.0, index=df.index, dtype=float)
        cash = 10000.0
        rejected_entries = 0
        atr_pips = float((df["high"] - df["low"]).abs().mean() / pip_size) if pip_size else 20.0
        for strat_name, (entries_arr, exits_arr, direction_arr) in sig_arrays.items():
            position = 0
            entry_price = 0.0
            entry_idx = 0
            entry_lot = base_lot
            ticket: Ticket | None = None
            for bar_idx in range(n_bars):
                s0 = bar_idx * ticks_per
                s1 = s0 + ticks_per
                bar_ticks = tick_mids[s0:s1] if s0 < len(tick_mids) else np.array([float(df["close"].iloc[bar_idx])])
                bar_spread_pips = float(spread_arr[bar_idx])
                bar_spread_price = bar_spread_pips * pip_size
                # Intrabar stop/limit walk (SL-before-TP) across all ticks in bar
                if position != 0 and ticket is not None:
                    hit = None
                    for tm in bar_ticks:
                        bid_t, ask_t = tm - bar_spread_price / 2, tm + bar_spread_price / 2
                        hit = check_exits_tick(position, bid_t, ask_t, ticket.sl_price, ticket.tp_price)
                        if hit:
                            fill = bid_t if position > 0 else ask_t
                            intra_bar_fills += 1
                            if partial_frac > 0 and hit == "tp":
                                pnl = ticket.partial(float(partial_frac), fill, contract_size)
                                cash += pnl
                                if ticket.lots <= 1e-9:
                                    all_trades.append({"strategy": strat_name, "entry_price": entry_price,
                                                       "exit_price": fill, "direction": position, "pnl": pnl,
                                                       "Entry Timestamp": df.index[entry_idx], "Exit Timestamp": df.index[bar_idx],
                                                       "entry_bar": entry_idx, "exit_bar": bar_idx, "reason": "tp_partial"})
                                    position = 0; ticket = None
                            else:
                                pnl = (fill - entry_price) * position * contract_size * entry_lot
                                cash += pnl
                                all_trades.append({"strategy": strat_name, "entry_price": entry_price,
                                                   "exit_price": fill, "direction": position, "pnl": pnl,
                                                   "Entry Timestamp": df.index[entry_idx], "Exit Timestamp": df.index[bar_idx],
                                                   "entry_bar": entry_idx, "exit_bar": bar_idx, "reason": hit})
                                fills_count += 1
                                position = 0; ticket = None
                            break
                # Signal handling: entries/exits evaluated at bar open (tick 0),
                # fill via order-ticket at ask/bid with spread-reject — NOT bar close.
                if bool(exits_arr[bar_idx]) and position != 0:
                    px = float(bar_ticks[0])
                    fill = market_fill(position, px, bar_spread_price, bar_spread_pips,
                                       max_spread_pips, slippage_pips, pip_size, atr_pips)
                    if fill is None:
                        rejected_entries += 1
                    else:
                        pnl = (fill - entry_price) * position * contract_size * entry_lot
                        cash += pnl
                        all_trades.append({"strategy": strat_name, "entry_price": entry_price,
                                           "exit_price": fill, "direction": position, "pnl": pnl,
                                           "Entry Timestamp": df.index[entry_idx], "Exit Timestamp": df.index[bar_idx],
                                           "entry_bar": entry_idx, "exit_bar": bar_idx, "reason": "signal_exit"})
                        fills_count += 1
                        position = 0; ticket = None
                if bool(entries_arr[bar_idx]) and int(direction_arr[bar_idx]) != 0:
                    nd = int(direction_arr[bar_idx])
                    if position != 0 and nd != position:
                        px = float(bar_ticks[0])
                        fill = market_fill(position, px, bar_spread_price, bar_spread_pips,
                                           None, slippage_pips, pip_size, atr_pips)
                        if fill is not None:
                            pnl = (fill - entry_price) * position * contract_size * entry_lot
                            cash += pnl
                            all_trades.append({"strategy": strat_name, "entry_price": entry_price,
                                               "exit_price": fill, "direction": position, "pnl": pnl,
                                               "Entry Timestamp": df.index[entry_idx], "Exit Timestamp": df.index[bar_idx],
                                               "entry_bar": entry_idx, "exit_bar": bar_idx, "reason": "flip"})
                            fills_count += 1
                        position = 0; ticket = None
                    if position == 0:
                        px = float(bar_ticks[0])
                        fill = market_fill(nd, px, bar_spread_price, bar_spread_pips,
                                           max_spread_pips, slippage_pips, pip_size, atr_pips)
                        if fill is None:
                            rejected_entries += 1
                        else:
                            position = nd
                            entry_price = fill
                            entry_idx = bar_idx
                            entry_lot = base_lot
                            sl = (fill - sl_pips * pip_size) if (sl_pips and nd > 0) else ((fill + sl_pips * pip_size) if (sl_pips and nd < 0) else None)
                            tp = (fill + tp_pips * pip_size) if (tp_pips and nd > 0) else ((fill - tp_pips * pip_size) if (tp_pips and nd < 0) else None)
                            ticket = Ticket(direction=nd, lots=base_lot, entry_price=fill, entry_bar=bar_idx, sl_price=sl, tp_price=tp)
                            fills_count += 1
                # Mark-to-market at bar close bid/mid + borrow fee + margin stop-out
                mtm_px = float(df["close"].iloc[bar_idx])
                if position != 0:
                    # Short borrow fee (annual %/365 per day appx per bar)
                    if position < 0 and borrow_pct_per_day > 0:
                        bars_per_day = 24.0 if n_bars > 500 else 1.0
                        try:
                            _med = float(pd.Series(df.index).diff().dropna().dt.total_seconds().median())
                            bars_per_day = max(86400.0 / _med, 1.0)
                        except Exception:
                            pass
                        cash -= abs(entry_lot * contract_size * mtm_px) * float(borrow_pct_per_day) / 100.0 / bars_per_day
                    mtm = (mtm_px - entry_price) * position * contract_size * entry_lot
                    cur_eq = cash + mtm
                    # Margin stop-out (MT5/LEAN rule): margin = notional/leverage
                    _marg_used = abs(entry_lot * contract_size * entry_price) / max(float(leverage), 1.0)
                    _lvl = (cur_eq / _marg_used * 100.0) if _marg_used > 0 else 9999.0
                    if "_min_lvl" not in locals():
                        _min_lvl = 9999.0
                    _min_lvl = min(_min_lvl, _lvl)
                    if _lvl < float(stopout_level_pct):
                        fill = mtm_px - (bar_spread_price / 2 if position > 0 else -bar_spread_price / 2)
                        pnl = (fill - entry_price) * position * contract_size * entry_lot
                        cash += pnl
                        all_trades.append({"strategy": strat_name, "entry_price": entry_price,
                                           "exit_price": fill, "direction": position, "pnl": pnl,
                                           "Entry Timestamp": df.index[entry_idx], "Exit Timestamp": df.index[bar_idx],
                                           "entry_bar": entry_idx, "exit_bar": bar_idx, "reason": "stopout"})
                        fills_count += 1
                        position = 0; ticket = None
                        equity.iloc[bar_idx:] = cash
                    else:
                        equity.iloc[bar_idx:] = cur_eq
                else:
                    equity.iloc[bar_idx:] = cash
            # close open at end
            if position != 0:
                px = float(df["close"].iloc[-1])
                fill = (px - math.copysign(bar_spread_price / 2, position))
                pnl = (fill - entry_price) * position * contract_size * entry_lot
                cash += pnl
                all_trades.append({"strategy": strat_name, "entry_price": entry_price,
                                   "exit_price": fill, "direction": position, "pnl": pnl,
                                   "Entry Timestamp": df.index[entry_idx], "Exit Timestamp": df.index[-1],
                                   "entry_bar": entry_idx, "exit_bar": n_bars - 1, "reason": "eod"})
        trades = pd.DataFrame(all_trades)
        eq_df = pd.DataFrame()
    else:
        for bar_idx in range(n_bars):
            bar = df.iloc[bar_idx]
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])
            spread_pips = float(spread_arr[bar_idx])
            # Intrabar detection: bar range crossing a grid step means MT5 would
            # trigger layers mid-bar while OHLC path only sees bar-close.
            bar_range_pips = (bar_high - bar_low) / pip_size if pip_size else 0.0
            if bar_range_pips >= float(pips_between_orders):
                intra_bar_fills += 1
            else:
                next_bar_fills += 1
            # Process each strategy's signal
            for strat_name, sig in sig_arrays.items():
                entries_arr, _, direction_arr = sig
                sig_dir = 0
                if entries_arr[bar_idx]:
                    sig_dir = int(direction_arr[bar_idx])
                closed = mgr.on_bar_close(
                    strategy=strat_name, signal_direction=sig_dir,
                    bar_high=bar_high, bar_low=bar_low,
                    bar_close=bar_close, bar_index=bar_idx,
                    commission_pips=spread_pips / 2,  # use spread as commission
                    slippage_pips=spread_pips / 4,
                )
                fills_count += 1
            if progress_every and (bar_idx + 1) % progress_every == 0:
                print(f"  processed {bar_idx + 1}/{n_bars} bars...")

        # 5. Build result
        trades = mgr.to_trades_df()
        eq_df = pd.DataFrame(mgr.equity_curve, columns=["bar_index", "pnl", "strategy"]) \
            if mgr.equity_curve else pd.DataFrame()
        if not eq_df.empty:
            equity = pd.Series(10000.0, index=df.index)
            for _, row in eq_df.iterrows():
                equity.iloc[int(row["bar_index"]):] += float(row["pnl"])
        else:
            equity = pd.Series(10000.0, index=df.index)

    returns = equity.pct_change().fillna(0)
    metrics = compute_all(returns, trades, equity, periods_per_year=infer_periods_per_year(df))
    return {
        "equity": equity,
        "trades": trades,
        "metrics": metrics,
        "tick_source": tick_source,
        "spread_source": "real_broker" if use_real_spreads else "synthetic_session_model",
        "n_bars": n_bars,
        "n_ticks_synthesized": len(ticks),
        "avg_spread_pips": float(spread_arr.mean()),
        "max_spread_pips": float(spread_arr.max()),
        "fills_count": fills_count,
        "intra_bar_fills_estimate": intra_bar_fills,
        "rejected_entries": int(locals().get("rejected_entries", 0)),
        "min_margin_level": float(locals().get("_min_lvl", 9999.0)) if "grid_mode" in locals() and grid_mode == GRID_NONE else None,
        "leverage": float(leverage),
        "borrow_pct_per_day": float(borrow_pct_per_day),
        "max_spread_pips_setting": max_spread_pips,
        "deep_mode": True,
    }