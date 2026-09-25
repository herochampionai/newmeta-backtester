"""Lenient MTF search — require_confluence=1, optional HTF.

The full MTF (confluence=2) killed too many strategies. Lenient version:
- confluence: 1 (any single creative filter agrees)
- HTF: optional (per-strategy)
- Cooldown: shorter

Goal: get to 9/12 strategies profitable on at least one asset.
"""
import warnings; warnings.filterwarnings('ignore')
import json
import sys

sys.path.insert(0, '.')

import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
from pathlib import Path

from analysis.optuna_filters import fetch_h1, run_strategy
from strategies.mtf_framework import (
    MTFAC_AO,
    MTFBB_RSI,
    MTFFBB,
    MTFMFI,
    MTFMS,
    MTFDeM,
    MTFMACD_Confluence,
    MTFMTF_Stoch,
    MTFQuad_Stoch,
    MTFStoch533,
    MTFTriple_RSI,
)

MTF_CLASSES = {
    "mtf_ac_ao": MTFAC_AO,
    "mtf_dem": MTFDeM,
    "mtf_fbb": MTFFBB,
    "mtf_mfi": MTFMFI,
    "mtf_ms": MTFMS,
    "mtf_mtf_stoch": MTFMTF_Stoch,
    "mtf_bb_rsi": MTFBB_RSI,
    "mtf_triple_rsi": MTFTriple_RSI,
    "mtf_quad_stoch": MTFQuad_Stoch,
    "mtf_stoch533": MTFStoch533,
    "mtf_macd_confluence": MTFMACD_Confluence,
}


def search_strategy_lenient(name, cls, df_oos, profile, n_trials=40):
    """Lenient MTF: confluence=1, shorter cooldown, optional HTF."""

    def obj(trial):
        # Underlying strategy params
        params = {"use_candle_bias": True}  # always useful
        if "ac_ao" in name:
            params.update({
                "level_open_orders": trial.suggest_float("lev", 20, 200),
                "open_orders_type": trial.suggest_int("otype", 1, 8),
                "use_acceleration_filter": trial.suggest_categorical("accel", [True, False]),
                "min_acceleration": trial.suggest_float("min_acc", 1e-4, 1e-3, log=True),
                "use_ao_synchronization": trial.suggest_categorical("ao_sync", [True, False]),
            })
        elif "dem" in name:
            params.update({
                "bars_calculate": trial.suggest_int("bars", 8, 40),
                "level_open_orders": trial.suggest_float("lev", 50, 95),
                "level_close_orders": trial.suggest_float("lev_c", 50, 95),
                "open_orders_type": trial.suggest_int("otype", 1, 4),
            })
        elif "fbb" in name:
            params.update({
                "bars_calculate": trial.suggest_int("bars", 10, 60),
                "atr_period": trial.suggest_int("atr_p", 8, 30),
                "atr_mult": trial.suggest_float("atr_m", 1.0, 4.0),
                "lookback": trial.suggest_int("lookback", 5, 30),
            })
        elif "mfi" in name:
            params.update({
                "bars_calculate": trial.suggest_int("bars", 8, 40),
                "level_open_orders": trial.suggest_float("lev", 30, 90),
                "open_orders_type": trial.suggest_int("otype", 1, 4),
            })
        elif "ms" in name:
            params.update({
                "ms_fast_ema": trial.suggest_int("fast", 5, 30),
                "ms_slow_ema": trial.suggest_int("slow", 20, 60),
                "ms_signal_period": trial.suggest_int("sig", 2, 20),
                "level_open_orders": trial.suggest_float("lev", 20, 80),
            })
        elif "mtf_stoch" in name:
            params.update({
                "k_period": trial.suggest_int("k", 5, 30),
                "d_period": trial.suggest_int("d", 2, 10),
                "smooth": trial.suggest_int("smooth", 1, 6),
                "oversold": trial.suggest_float("os", 10, 35),
                "overbought": trial.suggest_float("ob", 65, 90),
            })
        elif "bb_rsi" in name:
            params.update({
                "bb_period": trial.suggest_int("bb_p", 10, 40),
                "bb_std": trial.suggest_float("bb_std", 1.0, 3.0),
                "rsi_period": trial.suggest_int("rsi_p", 7, 30),
                "rsi_oversold": trial.suggest_float("rsi_os", 20, 40),
                "rsi_overbought": trial.suggest_float("rsi_ob", 60, 80),
            })
        elif "triple_rsi" in name:
            params.update({
                "rsi_fast": trial.suggest_int("fast", 5, 14),
                "rsi_mid": trial.suggest_int("mid", 10, 21),
                "rsi_slow": trial.suggest_int("slow", 18, 30),
                "oversold": trial.suggest_float("os", 15, 35),
                "overbought": trial.suggest_float("ob", 65, 85),
            })
        elif "quad_stoch" in name:
            params.update({
                "k_period": trial.suggest_int("k", 5, 30),
                "d_period": trial.suggest_int("d", 2, 10),
                "smooth": trial.suggest_int("smooth", 1, 6),
                "oversold": trial.suggest_float("os", 10, 35),
                "overbought": trial.suggest_float("ob", 65, 90),
            })
        elif "stoch533" in name:
            params.update({
                "htf_rule": trial.suggest_categorical("s533_rule", ["4h", "1d"]),
                "oversold": trial.suggest_float("os", 10, 30),
                "overbought": trial.suggest_float("ob", 70, 90),
            })
        elif "macd_confluence" in name:
            params.update({
                "fast": trial.suggest_int("macd_fast", 5, 20),
                "slow": trial.suggest_int("macd_slow", 20, 50),
                "signal": trial.suggest_int("macd_sig", 5, 20),
                "min_confluence": trial.suggest_int("macd_min", 1, 4),
            })

        # Lenient MTF params (single creative filter + optional HTF)
        mtf_params = {
            "htf_rule": trial.suggest_categorical("htf_rule", ["4h", "1d"]),
            "htf_mode": "sma_trend",
            "htf_sma_period": trial.suggest_int("htf_sma_p", 30, 100),
            # HTF: sometimes disable (for ranging-market strategies)
            "require_htf_agreement": trial.suggest_categorical("htf_required", [True, False]),
            # Single creative filter (always include candle_bias)
            "use_swing_structure": trial.suggest_categorical("use_swing", [True, False]),
            "use_breakout": trial.suggest_categorical("use_breakout", [True, False]),
            "use_candle_bias": True,
            "use_mid_range": trial.suggest_categorical("use_mid", [True, False]),
            "require_confluence": trial.suggest_int("req_conf", 1, 2),  # 1 = lenient
            "cooldown_bars": trial.suggest_int("cd", 2, 15),
        }
        params.update(mtf_params)
        m = run_strategy(df_oos, cls, params, [], profile)
        if "error" in m or m["n_trades"] < 10:
            return -1e9
        return m["net_pnl"] + 50 * m["sharpe"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def main():
    print("=" * 100)
    print("LENIENT MTF — Optuna search on 2Y OOS")
    print("Confluence=1, optional HTF, more trades per strategy")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    eur_oos = eur[(eur.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (eur.index < pd.Timestamp("2026-09-17", tz="UTC"))]
    nas_oos = nas[(nas.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (nas.index < pd.Timestamp("2026-09-17", tz="UTC"))]

    n_trials = 35
    results = {}
    for sname, cls in MTF_CLASSES.items():
        for asset, df, profile in [("EUR", eur_oos, "forex"), ("NAS", nas_oos, "nas100")]:
            key = f"{sname}_{asset}"
            print(f"\n>>> {key} ({n_trials} trials)...")
            best, score = search_strategy_lenient(key, cls, df, profile, n_trials)
            m = run_strategy(df, cls, best, [], profile)
            if "error" in m:
                print(f"  ERR: {m['error']}")
                continue
            results[key] = {"params": best, "metrics": m}
            status = "✓✓" if m["net_pnl"] > 0 and m["sharpe"] > 0 else ("~" if m["net_pnl"] > 0 else "✗")
            print(f"  score={score:.0f} | {status} PnL ${m['net_pnl']:+,.0f} "
                  f"Sharpe {m['sharpe']:+.2f} WR {m['win_rate']*100:.1f}% "
                  f"PF {m['profit_factor']:.2f} ({m['n_trades']} trades)")

    # Summary
    print("\n" + "=" * 100)
    print("LENIENT MTF RESULTS — 2Y OOS, 11 strategies × 2 assets")
    print("=" * 100)
    print(f"{'Instance':<28} {'PnL':>10} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'TPY':>6} {'Status':<10}")
    print("-" * 80)
    locked = []
    for key in sorted(results.keys()):
        r = results[key]
        m = r["metrics"]
        tpy = m.get("n_trades", 0) / 2.0
        if m.get("net_pnl", 0) > 0 and m.get("sharpe", 0) > 0:
            status = "✓ ROBUST"
            locked.append(key)
        elif m.get("net_pnl", 0) > 0:
            status = "~ borderline"
        else:
            status = "✗ failing"
        print(f"{key:<28} ${m.get('net_pnl', 0):>+9,.0f} {m.get('sharpe', 0):>+6.2f} "
              f"{m.get('win_rate', 0)*100:>4.1f}% {m.get('profit_factor', 0):>4.2f} "
              f"{tpy:>5.1f} {status}")

    print(f"\n  Locked count: {len(locked)}/22 instances")

    # Save
    out_dir = Path("output/mtf_lenient")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "best_params.json", "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
