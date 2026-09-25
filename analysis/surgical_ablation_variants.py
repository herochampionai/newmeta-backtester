"""Try alternative configs — different strategy, different params, different asset.

Verifies whether the FAIL verdict is intrinsic to the features, or specific
to fbb/EURUSD/defaults.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_LOSS_AND_PROFIT, RECOVERY_HIGHER_PROFITS
from backtester.metrics_v2 import compute_all
from core.surgical_features import SURGICAL_FEATURES, enable, get_feature_defaults
from data.cache import load as load_cache
from strategies import STRATEGY_REGISTRY


def run_single(df, strategy_name, params, **grid_overrides):
    cls = STRATEGY_REGISTRY[strategy_name]
    sig = cls(params=params).generate(df)
    signals = {strategy_name: (sig.entries.fillna(False).astype(bool),
                                pd.Series(sig.direction, index=df.index).fillna(0).astype(int))}
    grid = dict(
        grid_mode=GRID_LOSS_AND_PROFIT, base_lot=0.1,
        grid_take_profit=50, grid_stop_loss=200,
        max_grid_layers=4, pips_between_orders=30, grid_lot_multiplier=1.5,
        recovery_mode=RECOVERY_HIGHER_PROFITS, recovery_lot_multiplier=2.0,
        init_cash=10_000.0, commission_pips=0.7, slippage_pips=0.3,
        spread_pips=1.0, pip_size=0.0001, contract_size=100_000,
    )
    grid.update(grid_overrides)
    r = run_full(df, signals, params=params, **grid)
    m = compute_all(r["equity"].pct_change().fillna(0), r.get("trades"), r["equity"])
    return {"net_pnl": m.get("net_pnl", 0.0), "sharpe": m.get("sharpe", 0.0),
            "win_rate": m.get("win_rate", 0.0), "profit_factor": m.get("profit_factor", 0.0)}


def build_params(feat, overrides=None):
    base = get_feature_defaults()
    if feat is None:
        return base
    if feat == "all":
        for n in SURGICAL_FEATURES:
            base = enable(base, n, overrides.get(n) if overrides else None)
        return base
    return enable(base, feat, overrides.get(feat) if overrides else None)


def test_variant(label, strategy_name, df, params_overrides=None, grid_overrides=None):
    print(f"\n=== {label} ({strategy_name}) ===")
    grid_overrides = grid_overrides or {}
    params_overrides = params_overrides or {}

    base_m = run_single(df, strategy_name, build_params(None), **grid_overrides)
    print(f"  BASELINE:                 PnL ${base_m['net_pnl']:+,.0f} | "
          f"Sharpe {base_m['sharpe']:+.2f} | PF {base_m['profit_factor']:.2f} | WR {base_m['win_rate']*100:.1f}%")

    for feat in SURGICAL_FEATURES:
        m = run_single(df, strategy_name, build_params(feat, params_overrides), **grid_overrides)
        d_pnl = m["net_pnl"] - base_m["net_pnl"]
        marker = "✓" if d_pnl > 0 and m["sharpe"] > base_m["sharpe"] else ("=" if abs(d_pnl) < 1 else "✗")
        print(f"  {marker} {feat:<22} PnL ${m['net_pnl']:+,.0f} (Δ${d_pnl:+,.0f}) | "
              f"Sharpe {m['sharpe']:+.2f} (Δ{m['sharpe']-base_m['sharpe']:+.2f}) | "
              f"PF {m['profit_factor']:.2f}")


def main():
    df, meta = load_cache("EURUSD", "H1")
    df_oos = df[df.index >= pd.Timestamp("2024-07-01", tz="UTC")].copy()

    # Variant 1: Different strategy (ac_ao)
    test_variant("EURUSD H1 + ac_ao strategy", "ac_ao", df_oos)

    # Variant 2: fbb with TIGHTER grid (faster closes, more basket_money_tp firings)
    test_variant("EURUSD H1 + fbb + tight grid (TP=$20, max=8)",
                 "fbb", df_oos,
                 grid_overrides=dict(grid_take_profit=20, grid_stop_loss=100, max_grid_layers=8))

    # Variant 3: fbb with bigger basket_money_tp threshold (so it differs from grid_tp)
    test_variant("EURUSD H1 + fbb + basket_tp=$200 (above grid_tp)",
                 "fbb", df_oos,
                 params_overrides=dict(basket_money_tp={"enabled": True, "basket_take_profit_usd": 200}))

    # Variant 4: fbb with wider profit_lock_trail (less aggressive)
    test_variant("EURUSD H1 + fbb + profit_lock_pct=85 (looser)",
                 "fbb", df_oos,
                 params_overrides=dict(profit_lock_trail={"enabled": True, "profit_lock_pct": 85}))

    # Variant 5: fbb with basket_money_tp=$20 (below grid_tp, fires earlier)
    test_variant("EURUSD H1 + fbb + basket_tp=$20 (fires earlier than grid_tp)",
                 "fbb", df_oos,
                 params_overrides=dict(basket_money_tp={"enabled": True, "basket_take_profit_usd": 20}))


if __name__ == "__main__":
    main()
