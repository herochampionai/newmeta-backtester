"""Backtest engine FULL — thin orchestrator that composes pure + grid + adaptive + swap.

Pipeline:
  1. If grid_mode != GRID_NONE → run_grid()  (uses GridRecoveryManager)
     Else                   → run_pure()   (uses vectorbt)
  2. If adaptive_enabled    → apply_adaptive()
  3. If swap_enabled        → apply_swap()

Each overlay is independently optional. The flag name means what it says.
"""
from __future__ import annotations

import pandas as pd

from backtester.adaptive import AdaptiveConfig
from backtester.engine_adaptive import apply_adaptive
from backtester.engine_grid import run_grid
from backtester.engine_pure import run_pure
from backtester.engine_swap import apply_swap
from backtester.grid_recovery import (
    GRID_LOSS,
    GRID_LOSS_AND_PROFIT,
    GRID_NONE,
    GRID_PROFIT,
)
from core.surgical_features import is_enabled

GRID_LABELS = {GRID_NONE: "None", GRID_LOSS: "Loss", GRID_PROFIT: "Profit",
               GRID_LOSS_AND_PROFIT: "Loss+Profit"}


def _generate_tp_sl_exits(df: pd.DataFrame, signals_by_strategy: dict[str, tuple],
                           pip_size: float, tp_pips: float = 50.0, sl_pips: float = 30.0) -> dict[str, tuple]:
    """Augment 2-tuple strategies (entries, direction) with synthetic TP/SL exits.

    For strategies that already have 3-tuple (entries, exits, direction), pass through unchanged.
    For 2-tuple strategies, generate exit signals when price hits TP or SL level from entry.
    """
    augmented = {}
    for name, sig in signals_by_strategy.items():
        if len(sig) == 3:
            augmented[name] = sig
            continue

        entries, direction = sig
        if hasattr(entries, 'fillna'):
            entries = entries.fillna(False).astype(bool)
        else:
            entries = pd.Series(entries, index=df.index).fillna(False).astype(bool)

        if hasattr(direction, 'fillna'):
            direction = direction.fillna(0).astype(int)
        else:
            direction = pd.Series(direction, index=df.index).fillna(0).astype(int)

        # Generate synthetic exits: TP/SL based on entry price
        exits = pd.Series(False, index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        position = 0
        entry_price = 0.0

        for i in range(len(df)):
            if entries.iloc[i] and direction.iloc[i] != 0:
                # Close existing position if any (flip)
                if position != 0:
                    exits.iloc[i] = True
                # Open new position
                position = direction.iloc[i]
                entry_price = close.iloc[i]
            elif position != 0:
                # Check TP/SL
                if position > 0:  # Long
                    tp_level = entry_price + tp_pips * pip_size
                    sl_level = entry_price - sl_pips * pip_size
                    # Check if high hit TP or low hit SL within this bar
                    if high.iloc[i] >= tp_level:
                        exits.iloc[i] = True
                        position = 0
                    elif low.iloc[i] <= sl_level:
                        exits.iloc[i] = True
                        position = 0
                else:  # Short
                    tp_level = entry_price - tp_pips * pip_size
                    sl_level = entry_price + sl_pips * pip_size
                    if low.iloc[i] <= tp_level:
                        exits.iloc[i] = True
                        position = 0
                    elif high.iloc[i] >= sl_level:
                        exits.iloc[i] = True
                        position = 0

        augmented[name] = (entries, exits, direction)
    return augmented


def _apply_trailing_stops(df: pd.DataFrame, signals_by_strategy: dict[str, tuple],
                          params: dict | None, pip_size: float) -> dict[str, tuple]:
    """Augment 3-tuple strategies with TP/SL/trailing/breakeven exits from params.

    Preserves existing exits (e.g. session closes) and adds stop-based exits.
    Intrabar checks via high/low. Conservative ordering: SL before TP when both hit.
    Entry fills at bar close, so stops are only checked from the bar AFTER entry.

    Reads pct/trail keys (tp_pct, sl_pct, trail_pips, breakeven_*) — disjoint from
    _generate_tp_sl_exits' pips keys (default_tp_pips/default_sl_pips), so the two
    augmentations never double-apply.
    """
    if not params:
        return signals_by_strategy
    tp_pct = params.get("tp_pct") or params.get("TP_Percent")
    sl_pct = params.get("sl_pct") or params.get("SL_Percent")
    tp_pips = params.get("tp_pips") or params.get("default_tp_pips")
    sl_pips = params.get("sl_pips") or params.get("default_sl_pips")
    trail_pips = params.get("trail_pips")
    be_act = params.get("breakeven_activation_pips")
    be_buf = params.get("breakeven_buffer_pips")
    if not any([tp_pct, sl_pct, tp_pips, sl_pips, trail_pips]):
        return signals_by_strategy

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    def _series(v, default):
        if hasattr(v, "fillna"):
            return v.fillna(default)
        return pd.Series(v, index=df.index).fillna(default)

    augmented = {}
    for name, sig in signals_by_strategy.items():
        if len(sig) != 3:
            augmented[name] = sig
            continue
        entries, exits, direction = sig
        entries = _series(entries, False).astype(bool)
        exits = _series(exits, False).astype(bool)
        direction = _series(direction, 0).astype(int)

        new_exits = exits.copy()
        position = 0
        entry_price = 0.0
        peak = 0.0
        sl_level = 0.0

        for i in range(n):
            sig_dir = int(direction.iloc[i])
            if bool(entries.iloc[i]) and sig_dir != 0:
                if position != 0:
                    new_exits.iloc[i] = True  # flip: close old at this bar
                position = sig_dir
                entry_price = float(close[i])
                peak = entry_price
                if sl_pct:
                    sl_level = (entry_price * (1 - float(sl_pct) / 100.0) if position > 0
                                else entry_price * (1 + float(sl_pct) / 100.0))
                elif sl_pips:
                    sl_level = (entry_price - float(sl_pips) * pip_size if position > 0
                                else entry_price + float(sl_pips) * pip_size)
                else:
                    sl_level = 0.0
                continue
            if position == 0:
                continue
            # Existing exit (session close etc.) closes the position
            if bool(exits.iloc[i]):
                position = 0
                continue
            # Track favourable extreme
            if position > 0:
                peak = max(peak, float(high[i]))
            else:
                peak = min(peak, float(low[i]))
            # Breakeven: ratchet SL to entry +/- buffer once price moves be_act in favour
            if be_act and be_buf:
                if position > 0 and (peak - entry_price) >= float(be_act) * pip_size:
                    sl_level = max(sl_level, entry_price + float(be_buf) * pip_size)
                elif position < 0 and (entry_price - peak) >= float(be_act) * pip_size:
                    sl_level = min(sl_level, entry_price - float(be_buf) * pip_size)
            # Trailing stop ratchet
            if trail_pips:
                if position > 0:
                    sl_level = max(sl_level, peak - float(trail_pips) * pip_size)
                else:
                    sl_level = min(sl_level, peak + float(trail_pips) * pip_size)
            # Stop checks (conservative: SL before TP)
            hit = False
            if position > 0:
                if sl_level > 0 and float(low[i]) <= sl_level:
                    hit = True
                elif tp_pct and float(high[i]) >= entry_price * (1 + float(tp_pct) / 100.0):
                    hit = True
                elif tp_pips and float(high[i]) >= entry_price + float(tp_pips) * pip_size:
                    hit = True
            else:
                if sl_level > 0 and float(high[i]) >= sl_level:
                    hit = True
                elif tp_pct and float(low[i]) <= entry_price * (1 - float(tp_pct) / 100.0):
                    hit = True
                elif tp_pips and float(low[i]) <= entry_price - float(tp_pips) * pip_size:
                    hit = True
            if hit:
                new_exits.iloc[i] = True
                position = 0

        augmented[name] = (entries, new_exits, direction)
    return augmented


def run_full(df: pd.DataFrame,
              signals_by_strategy: dict[str, tuple],
              init_cash: float = 10_000.0,
              commission_pips: float = 0.7,
              slippage_pips: float = 0.3,
              spread_pips: float = 0.0,
              commission_pct: float = 0.0,
              pip_size: float = 0.0001,
              contract_size: float = 100_000,
              symbol: str = "EURUSD",
              # Grid overlay
              grid_mode: int = GRID_NONE,
              pips_between_orders: float = 30.0,
              grid_lot_multiplier: float = 1.5,
              grid_take_profit: float = 50.0,
              grid_stop_loss: float = 200.0,
              max_grid_layers: int = 4,
              recovery_mode: int = 0,
              recovery_lot_multiplier: float = 2.0,
              base_lot: float = 0.1,
              # Adaptive overlay
              adaptive_enabled: bool = False,
              adaptive_config: AdaptiveConfig | None = None,
              # Swap overlay
              swap_enabled: bool = False,
              long_swap_pips: float = -0.5,
              short_swap_pips: float = 0.2,
               # Surgical features (B4) — passed from strategy params
               params: dict | None = None,
               # Tick-aware simulation
               tick_resolution: str | None = None,
               # Deep tick-level backtest (intra-bar TP/SL detection)
               # off      = OHLC bar-level path (default, fastest)
               # synthetic = synthesize ticks from OHLC bars + variable spread
               # real     = try to fetch real ticks from MT5, fall back to synthetic
               tick_mode: str = "off",
               ticks_per_bar: int = 20,
               use_real_spreads: bool = False,
               max_spread_pips: float | None = 5.0,
               slippage_pips_tick: float = 0.2,
               spec=None,
               leverage: float = 30.0,
               borrow_pct_per_day: float = 0.0,
               stopout_level_pct: float = 50.0,
               strict_data: bool = False,
               require_real_ticks: bool = False,
               ) -> dict:
    """Run backtest with optional overlays.

    Each overlay (grid / adaptive / swap) runs ONLY when its flag is True.
    The flag name means what it says — no hidden behavior.

    Tick-level execution:
      tick_mode="off"       → standard OHLC bar-level path (default)
      tick_mode="synthetic" → tick-by-tick simulation via engine_deep.deep_backtest()
                              with synthesized intra-bar ticks + variable spread.
                              Detects intra-bar TP/SL triggers (e.g. wicks).
      tick_mode="real"      → same as synthetic but tries real MT5 ticks first.

    Note: When tick_mode is on, adaptive and swap overlays are skipped
    (engine_deep owns the full equity curve). Grid is supported via deep_backtest
    internally.

    Returns dict with all the result keys.
    """
    # === Strict data gate (kill silent synthetic + stale + critical issues) ===
    if strict_data:
        from backtester.data_quality import analyze_data_quality, gate_check
        from backtester.pro_suite import grade_data
        # Determine expected source based on tick_mode
        expected_source = "synthetic_ticks" if tick_mode in ("synthetic", "real") else "ohlc_bars"
        _g = grade_data(df, {"source": expected_source})
        if _g["grade"] in ("F", "D") or "synthetic" in _g["source"].lower():
            raise ValueError(f"Strict data gate: {_g['loud_banner']} — connect MT5 or load cached real data.")
        # Also run comprehensive data quality check
        _dq = analyze_data_quality(df, symbol=symbol or "UNKNOWN", timeframe="?", source=expected_source, pip_size=pip_size)
        _gate = gate_check(_dq, min_grade="C")
        if not _gate["passed"]:
            raise ValueError(f"Strict data gate: {_dq.loud_banner} — min grade C required, got {_gate['grade']}. Blocking: {_gate['blocking_flags']}")
        # Reuse _dq for result enrichment
        data_quality = _dq.to_dict()
        data_quality_banner = _dq.loud_banner
    else:
        # === Data Quality Dashboard (always run, enriches result) ===
        from backtester.data_quality import analyze_data_quality
        expected_source = "synthetic_ticks" if tick_mode in ("synthetic", "real") else "ohlc_bars"
        _dq = analyze_data_quality(df, symbol=symbol or "UNKNOWN", timeframe="?", source=expected_source, pip_size=pip_size)
        data_quality = _dq.to_dict()
        data_quality_banner = _dq.loud_banner

    # === Tick-level dispatch (short-circuits the OHLC pipeline) ===
    if tick_mode != "off":
        from backtester.engine_deep import deep_backtest
        use_real = (tick_mode == "real")
        result = deep_backtest(
            df, signals_by_strategy,
            ticks_per_bar=ticks_per_bar,
            use_real_ticks=use_real,
            use_real_spreads=use_real_spreads,
            symbol=symbol or "EURUSD",
            tick_start=str(df.index[0]) if len(df) > 0 else "2024-01-01",
            tick_end=str(df.index[-1]) if len(df) > 0 else "2024-12-31",
            # Forward ALL grid/recovery params so deep_backtest uses the same
            # configuration as the OHLC path (fixes the silent config-drop bug
            # where pips_between_orders / grid_lot_multiplier / max_layers /
            # grid TP+SL / recovery settings were ignored).
            grid_mode=grid_mode,
            pips_between_orders=pips_between_orders,
            grid_lot_multiplier=grid_lot_multiplier,
            grid_take_profit=grid_take_profit,
            grid_stop_loss=grid_stop_loss,
            max_grid_layers=max_grid_layers,
            recovery_mode=recovery_mode,
            recovery_lot_multiplier=recovery_lot_multiplier,
            base_lot=base_lot,
            contract_size=contract_size,
            pip_size=pip_size,
            max_spread_pips=max_spread_pips,
            slippage_pips=slippage_pips_tick,
            tp_pips=(params.get("default_tp_pips") if params else None),
            sl_pips=(params.get("default_sl_pips") if params else None),
            leverage=leverage,
            borrow_pct_per_day=borrow_pct_per_day,
            stopout_level_pct=stopout_level_pct,
            require_real_ticks=require_real_ticks,
        )
        # Deep_backtest hardcodes equity start at 10000. Rescale to user's init_cash
        # so the comparison chart and metrics reflect the requested capital.
        if init_cash != 10_000.0 and "equity" in result:
            eq = result["equity"]
            final_pnl = float(eq.iloc[-1]) - 10_000.0
            result["equity"] = pd.Series(init_cash + (eq - 10_000.0), index=eq.index)
            # Re-mark metrics that depend on equity curve
            if "metrics" in result and isinstance(result["metrics"], dict):
                result["metrics"]["net_pnl"] = result["metrics"].get("net_pnl", 0) + (init_cash - 10_000.0)
        # Enrich with execution metadata so the UI can show it consistently
        result["execution"] = {
            "commission_pips": float(commission_pips),
            "slippage_pips": float(slippage_pips),
            "spread_pips": float(spread_pips),
            "commission_pct": float(commission_pct),
            "pip_size": float(pip_size),
            "contract_size": float(contract_size),
            "base_lot": float(base_lot),
            "tick_mode": tick_mode,
            "ticks_per_bar": ticks_per_bar,
            "use_real_spreads": use_real_spreads,
            "grid_mode": int(grid_mode),
            "pips_between_orders": float(pips_between_orders),
            "grid_lot_multiplier": float(grid_lot_multiplier),
            "grid_take_profit": float(grid_take_profit),
            "grid_stop_loss": float(grid_stop_loss),
            "max_grid_layers": int(max_grid_layers),
            "recovery_mode": int(recovery_mode),
            "recovery_lot_multiplier": float(recovery_lot_multiplier),
        }
        result["active_overlays"] = ["deep_tick"]
        result["tick_resolution"] = tick_resolution  # preserve for compat
        # Attach data quality report
        result["data_quality"] = data_quality
        result["data_quality_banner"] = data_quality_banner
        # Surgical features info (all off when tick mode owns the path)
        result["surgical_features"] = {
            name: is_enabled(params, name)
            for name in ("anomaly_gate", "event_blackout", "recovery_restart",
                         "basket_money_tp", "profit_lock_trail", "carry_adjusted_tp")
        }
        return result

    # Stage 0: augment 2-tuple strategies with synthetic TP/SL exits
    signals_by_strategy = _generate_tp_sl_exits(
        df, signals_by_strategy, pip_size,
        tp_pips=float(params.get("default_tp_pips", 50.0)) if params else 50.0,
        sl_pips=float(params.get("default_sl_pips", 30.0)) if params else 30.0,
    )
    # Stage 0b: apply pct/trailing/breakeven stops from params (e.g. Light9 profiles)
    signals_by_strategy = _apply_trailing_stops(df, signals_by_strategy, params, pip_size)

    # Stage 1: base engine (pure OR grid)
    if grid_mode != GRID_NONE:
        # Grid has its own basket TP/SL — strip strategy-level exits before passing
        grid_signals = {}
        for _name, _sig in signals_by_strategy.items():
            if len(_sig) == 3:
                grid_signals[_name] = (_sig[0], _sig[2])  # (entries, direction)
            else:
                grid_signals[_name] = _sig
        result = run_grid(
            df, grid_signals,
            init_cash=init_cash,
            commission_pips=commission_pips, slippage_pips=slippage_pips,
            spread_pips=spread_pips, commission_pct=commission_pct,
            pip_size=pip_size, contract_size=contract_size,
            grid_mode=grid_mode,
            pips_between_orders=pips_between_orders,
            grid_lot_multiplier=grid_lot_multiplier,
            grid_take_profit=grid_take_profit,
            grid_stop_loss=grid_stop_loss,
            max_grid_layers=max_grid_layers,
            recovery_mode=recovery_mode,
            recovery_lot_multiplier=recovery_lot_multiplier,
            base_lot=base_lot,
            params=params,
        )
        result["active_overlays"] = ["grid"]
    else:
        result = run_pure(
            df, signals_by_strategy,
            init_cash=init_cash,
            commission_pips=commission_pips, slippage_pips=slippage_pips,
            spread_pips=spread_pips, commission_pct=commission_pct,
            pip_size=pip_size,
            contract_size=contract_size,
            tick_resolution=tick_resolution,
        )
        result["active_overlays"] = []

    result["execution"] = {
        "commission_pips": float(commission_pips),
        "slippage_pips": float(slippage_pips),
        "spread_pips": float(spread_pips),
        "commission_pct": float(commission_pct),
        "pip_size": float(pip_size),
        "contract_size": float(contract_size),
        "base_lot": float(base_lot),
        "leverage": float(leverage),
        "borrow_pct_per_day": float(borrow_pct_per_day),
        "stopout_level_pct": float(stopout_level_pct),
        "strict_data": bool(strict_data),
        "spec_fallback": bool(locals().get("_spec_fallback", False)),
    }

    # Attach data quality report
    result["data_quality"] = data_quality
    result["data_quality_banner"] = data_quality_banner

    # Stage 2: adaptive overlay (opt-in)
    if adaptive_enabled:
        result = apply_adaptive(result, adaptive_config=adaptive_config, base_lot=base_lot)

    # Stage 3: swap overlay (opt-in) — broker-true spec wins over manual pips
    _spec_fallback = False
    if swap_enabled:
        _spec = spec
        if _spec is None:
            try:
                from backtester.symbol_spec import get_spec
                _spec = get_spec(symbol or "EURUSD")
                pip_size = float(getattr(_spec, "pip_size", pip_size))
                contract_size = float(getattr(_spec, "contract_size", contract_size))
            except Exception:
                # Spec DB missing/unreadable: manual pips stand in, flagged.
                _spec = None
                _spec_fallback = True
        result = apply_swap(result, df,
                            long_swap_pips=long_swap_pips,
                            short_swap_pips=short_swap_pips,
                            pip_size=pip_size,
                            contract_size=contract_size,
                            triple_day=int(getattr(_spec, "triple_day", 2)) if _spec else 2,
                            spec=_spec)

    # Surgical features info for the result
    result["surgical_features"] = {
        name: is_enabled(params, name)
        for name in ("anomaly_gate", "event_blackout", "recovery_restart",
                     "basket_money_tp", "profit_lock_trail", "carry_adjusted_tp")
    }
    return result
