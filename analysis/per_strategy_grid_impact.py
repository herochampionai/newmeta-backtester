"""Per-strategy baseline vs grid-overlay comparison.

User hypothesis:
  Each strategy runs WITHOUT grid first (pure signal)
  Then WITH grid (GRID_LOSS_AND_PROFIT) which should
  'recover, heal and boost performance' on losing trades.

Output:
  Per strategy: pure PnL/Sharpe/WR/PF/DD → grid PnL/Sharpe/WR/PF/DD → Δ
  Ranked by absolute PnL improvement with grid.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_LOSS_AND_PROFIT, GRID_NONE, RECOVERY_HIGHER_PROFITS
from backtester.metrics_v2 import compute_all
from data.cache import load as load_cache
from strategies import CRYPTO_STRAT_EA_REGISTRY, MULTI_STRAT_EA_REGISTRY

SYMBOL = "EURUSD"
TIMEFRAME = "H1"
OOS_START = pd.Timestamp("2024-07-01", tz="UTC")  # 6-month OOS
INIT_CASH = 10_000.0
# Consistent grid config across all strategies (apples-to-apples)
GRID_KW = dict(
    base_lot=0.1,
    grid_take_profit=50,
    grid_stop_loss=200,
    max_grid_layers=4,
    pips_between_orders=30,
    grid_lot_multiplier=1.5,
    recovery_mode=RECOVERY_HIGHER_PROFITS,
    recovery_lot_multiplier=2.0,
)
COMMON_KW = dict(
    init_cash=INIT_CASH,
    commission_pips=0.7,
    slippage_pips=0.3,
    spread_pips=1.0,
    pip_size=0.0001,
    contract_size=100_000,
)


def run_strategy(df, strategy_name, grid_mode):
    cls = MULTI_STRAT_EA_REGISTRY.get(strategy_name) or CRYPTO_STRAT_EA_REGISTRY.get(strategy_name)
    if cls is None:
        return None
    inst = cls()
    try:
        sig = inst.generate(df)
    except Exception as e:
        return {"error": f"generate failed: {e}"}
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    n_entries = int(entries.sum())
    if n_entries == 0:
        return {"error": "no entries", "n_entries": 0}

    signals = {strategy_name: (entries, direction)}
    kwargs = dict(COMMON_KW)
    if grid_mode != GRID_NONE:
        kwargs.update(GRID_KW)
        kwargs["grid_mode"] = grid_mode

    try:
        r = run_full(df, signals, **kwargs)
    except Exception as e:
        return {"error": f"run_full failed: {e}"}
    m = compute_all(r["equity"].pct_change().fillna(0), r.get("trades"), r["equity"])
    return {
        "n_entries": n_entries,
        "net_pnl": m.get("net_pnl", 0.0),
        "total_return": m.get("total_return", 0.0),
        "sharpe": m.get("sharpe", 0.0),
        "sortino": m.get("sortino", 0.0),
        "calmar": m.get("calmar", 0.0),
        "max_dd": m.get("max_drawdown", 0.0),
        "win_rate": m.get("win_rate", 0.0),
        "profit_factor": m.get("profit_factor", 0.0),
        "n_trades": m.get("n_trades", 0),
        "expectancy": m.get("expectancy", 0.0),
    }


def main():
    df, meta = load_cache(SYMBOL, TIMEFRAME)
    df_oos = df[df.index >= OOS_START].copy()
    print("=" * 100)
    print("PER-STRATEGY: PURE vs GRID OVERLAY")
    print(f"Asset: {SYMBOL} {TIMEFRAME} | OOS: {df_oos.index[0].date()} → {df_oos.index[-1].date()} "
          f"({len(df_oos):,} bars)")
    print(f"Grid: GRID_LOSS_AND_PROFIT, TP=${GRID_KW['grid_take_profit']}, SL=${GRID_KW['grid_stop_loss']}, "
          f"max_layers={GRID_KW['max_grid_layers']}, multiplier={GRID_KW['grid_lot_multiplier']}, "
          f"recovery=last_close")
    print("=" * 100)

    # Strategies: FX multi (13) + crypto (1)
    fx_strats = list(MULTI_STRAT_EA_REGISTRY.keys())
    crypto_strats = list(CRYPTO_STRAT_EA_REGISTRY.keys())
    all_strats = fx_strats + crypto_strats
    print(f"\nStrategies: {len(fx_strats)} FX + {len(crypto_strats)} crypto = {len(all_strats)} total\n")

    results = {}
    for s in all_strats:
        print(f"  [{s:<22}]", end=" ", flush=True)
        pure = run_strategy(df_oos, s, GRID_NONE)
        grid = run_strategy(df_oos, s, GRID_LOSS_AND_PROFIT)
        results[s] = {"pure": pure, "grid": grid}
        if pure and "error" not in pure and grid and "error" not in grid:
            print(f"pure=${pure['net_pnl']:+,.0f} | grid=${grid['net_pnl']:+,.0f} | "
                  f"Δ=${grid['net_pnl']-pure['net_pnl']:+,.0f}")
        else:
            err = (pure or {}).get("error", "?") if isinstance(pure, dict) else "?"
            err2 = (grid or {}).get("error", "?") if isinstance(grid, dict) else "?"
            print(f"ERR pure={err} grid={err2}")

    # Results table
    print("\n" + "=" * 100)
    print("RESULTS — PURE (no grid) vs GRID OVERLAY")
    print("=" * 100)
    print(f"{'Strategy':<24} {'─ PURE ─':>32} {'─ GRID ─':>32}")
    print(f"{'':<24} {'PnL':>10} {'Sharpe':>7} {'WR':>6} {'PF':>5} {'DD':>7}    "
          f"{'PnL':>10} {'Sharpe':>7} {'WR':>6} {'PF':>5} {'DD':>7}")
    print("-" * 110)

    summary = []
    for s, d in results.items():
        p = d["pure"]; g = d["grid"]
        if not isinstance(p, dict) or "error" in p or not isinstance(g, dict) or "error" in g:
            print(f"{s:<24} {'ERROR':>10} {'—':>7} {'—':>6} {'—':>5} {'—':>7}    "
                  f"{'ERROR':>10} {'—':>7} {'—':>6} {'—':>5} {'—':>7}")
            continue
        print(f"{s:<24} "
              f"${p['net_pnl']:>+9,.0f} {p['sharpe']:>+6.2f} {p['win_rate']*100:>5.1f}% "
              f"{p['profit_factor']:>4.2f} {p['max_dd']*100:>+6.2f}%    "
              f"${g['net_pnl']:>+9,.0f} {g['sharpe']:>+6.2f} {g['win_rate']*100:>5.1f}% "
              f"{g['profit_factor']:>4.2f} {g['max_dd']*100:>+6.2f}%")
        summary.append({
            "strategy": s,
            "pure_pnl": p["net_pnl"], "pure_sharpe": p["sharpe"],
            "pure_wr": p["win_rate"], "pure_pf": p["profit_factor"], "pure_dd": p["max_dd"],
            "pure_n_trades": p["n_trades"],
            "grid_pnl": g["net_pnl"], "grid_sharpe": g["sharpe"],
            "grid_wr": g["win_rate"], "grid_pf": g["profit_factor"], "grid_dd": g["max_dd"],
            "grid_n_trades": g["n_trades"],
            "delta_pnl": g["net_pnl"] - p["net_pnl"],
            "delta_sharpe": g["sharpe"] - p["sharpe"],
            "delta_wr": (g["win_rate"] - p["win_rate"]) * 100,
            "delta_pf": g["profit_factor"] - p["profit_factor"],
            "delta_dd": (g["max_dd"] - p["max_dd"]) * 100,
        })

    # Ranked by absolute ΔPnL
    print("\n" + "=" * 100)
    print("RANKED BY GRID IMPACT (Δ PnL — grid should 'recover, heal, boost')")
    print("=" * 100)
    summary.sort(key=lambda r: r["delta_pnl"], reverse=True)
    print(f"{'Rank':<5} {'Strategy':<24} {'Pure PnL':>10} {'Grid PnL':>10} {'Δ PnL':>10} "
          f"{'Δ Sharpe':>9} {'Δ WR %':>8} {'Δ PF':>7} {'Δ DD %':>9} {'Verdict':<12}")
    print("-" * 110)
    for i, r in enumerate(summary, 1):
        d = r["delta_pnl"]
        if d > 50 and r["delta_sharpe"] > 0:
            verdict = "✓ BOOST"
        elif d > 0 and r["delta_sharpe"] <= 0:
            verdict = "~ pnl only"
        elif abs(d) < 50:
            verdict = "= neutral"
        else:
            verdict = "✗ HURT"
        print(f"{i:<5} {r['strategy']:<24} ${r['pure_pnl']:>+9,.0f} ${r['grid_pnl']:>+9,.0f} "
              f"${d:>+9,.0f} {r['delta_sharpe']:>+8.2f} {r['delta_wr']:>+7.1f}% "
              f"{r['delta_pf']:>+6.2f} {r['delta_dd']:>+8.2f}% {verdict:<12}")

    # Save CSV
    out = Path("output/per_strategy_grid_impact.csv")
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
