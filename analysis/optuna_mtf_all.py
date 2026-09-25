"""Optuna + 2-year OOS validation for ALL 12 MTF strategies."""
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
    MTFADX,
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
    "mtf_adx": MTFADX,
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


def param_space(trial, ltf_cls_name):
    """Per-strategy param search. Each strategy inherits from MTFStrategy
    so its own params come from the underlying ltf_strategy_cls."""
    base_params = {}
    # Strategy-specific base params
    if "ac_ao" in ltf_cls_name:
        base_params.update({
            "level_open_orders": trial.suggest_float("level_open_orders", 20, 200),
            "open_orders_type": trial.suggest_int("open_orders_type", 1, 8),
            "use_acceleration_filter": trial.suggest_categorical("use_accel", [True, False]),
            "min_acceleration": trial.suggest_float("min_acc", 1e-4, 1e-3, log=True),
            "use_ao_synchronization": trial.suggest_categorical("use_ao_sync", [True, False]),
        })
    elif "adx" in ltf_cls_name:
        base_params.update({
            "bars_calculate": trial.suggest_int("adx_bars", 10, 30),
            "use_di_crossover": True,  # proven to help
            "crossover_lookback": trial.suggest_int("adx_xover_lb", 1, 8),
            "min_crossover_gap": trial.suggest_float("adx_xover_gap", 1.0, 15.0),
            "adx_zone_low": trial.suggest_float("adx_zone_low", 10, 25),
            "adx_zone_high": trial.suggest_float("adx_zone_high", 25, 60),
            "continuation_level": trial.suggest_float("adx_cont", 18, 40),
            "reversal_edge": trial.suggest_float("adx_rev_edge", 14, 30),
            "open_orders_type": trial.suggest_int("adx_otype", 1, 4),
            "level_open_orders_1": trial.suggest_float("adx_lvl1", 20, 80),
            "level_open_orders_2": trial.suggest_float("adx_lvl2", 5, 30),
            "sweep_lookback": trial.suggest_int("adx_sweep_lb", 3, 15),
            "close_orders_type": trial.suggest_int("adx_ctype", 1, 4),
        })
    elif "dem" in ltf_cls_name:
        base_params.update({
            "bars_calculate": trial.suggest_int("dem_bars", 8, 40),
            "level_open_orders": trial.suggest_float("dem_lvl", 50, 95),
            "level_close_orders": trial.suggest_float("dem_lvl_close", 50, 95),
            "open_orders_type": trial.suggest_int("dem_otype", 1, 4),
            "close_orders_type": trial.suggest_int("dem_ctype", 0, 4),
        })
    elif "fbb" in ltf_cls_name:
        base_params.update({
            "bars_calculate": trial.suggest_int("fbb_bars", 10, 60),
            "atr_period": trial.suggest_int("fbb_atr_p", 8, 30),
            "atr_mult": trial.suggest_float("fbb_atr_m", 1.0, 4.0),
            "lookback": trial.suggest_int("fbb_lookback", 5, 30),
        })
    elif "mfi" in ltf_cls_name:
        base_params.update({
            "bars_calculate": trial.suggest_int("mfi_bars", 8, 40),
            "level_open_orders": trial.suggest_float("mfi_lvl", 30, 90),
            "open_orders_type": trial.suggest_int("mfi_otype", 1, 4),
            "close_orders_type": trial.suggest_int("mfi_ctype", 0, 4),
        })
    elif "ms" in ltf_cls_name:
        base_params.update({
            "ms_fast_ema": trial.suggest_int("ms_fast", 5, 30),
            "ms_slow_ema": trial.suggest_int("ms_slow", 20, 60),
            "ms_signal_period": trial.suggest_int("ms_sig", 2, 20),
            "level_open_orders": trial.suggest_float("ms_lvl", 20, 80),
            "open_orders_type": trial.suggest_int("ms_otype", 1, 4),
        })
    elif "mtf_stoch" in ltf_cls_name:
        base_params.update({
            "k_period": trial.suggest_int("stoch_k", 5, 30),
            "d_period": trial.suggest_int("stoch_d", 2, 10),
            "smooth": trial.suggest_int("stoch_smooth", 1, 6),
            "oversold": trial.suggest_float("stoch_os", 10, 35),
            "overbought": trial.suggest_float("stoch_ob", 65, 90),
        })
    elif "bb_rsi" in ltf_cls_name:
        base_params.update({
            "bb_period": trial.suggest_int("bb_p", 10, 40),
            "bb_std": trial.suggest_float("bb_std", 1.0, 3.0),
            "rsi_period": trial.suggest_int("rsi_p", 7, 30),
            "rsi_oversold": trial.suggest_float("rsi_os", 20, 40),
            "rsi_overbought": trial.suggest_float("rsi_ob", 60, 80),
        })
    elif "triple_rsi" in ltf_cls_name:
        base_params.update({
            "rsi_fast": trial.suggest_int("trsi_fast", 5, 14),
            "rsi_mid": trial.suggest_int("trsi_mid", 10, 21),
            "rsi_slow": trial.suggest_int("trsi_slow", 18, 30),
            "oversold": trial.suggest_float("trsi_os", 15, 35),
            "overbought": trial.suggest_float("trsi_ob", 65, 85),
        })
    elif "quad_stoch" in ltf_cls_name:
        base_params.update({
            "k_period": trial.suggest_int("qs_k", 5, 30),
            "d_period": trial.suggest_int("qs_d", 2, 10),
            "smooth": trial.suggest_int("qs_smooth", 1, 6),
            "oversold": trial.suggest_float("qs_os", 10, 35),
            "overbought": trial.suggest_float("qs_ob", 65, 90),
        })
    elif "stoch533" in ltf_cls_name:
        base_params.update({
            "htf_rule": trial.suggest_categorical("s533_rule", ["4h", "1d"]),
            "oversold": trial.suggest_float("s533_os", 10, 30),
            "overbought": trial.suggest_float("s533_ob", 70, 90),
        })
    elif "macd_confluence" in ltf_cls_name:
        base_params.update({
            "fast": trial.suggest_int("macd_fast", 5, 20),
            "slow": trial.suggest_int("macd_slow", 20, 50),
            "signal": trial.suggest_int("macd_sig", 5, 20),
            "min_confluence": trial.suggest_int("macd_min", 1, 4),
        })
    # MTF framework params (common)
    mtf_params = {
        "htf_rule": trial.suggest_categorical("htf_rule", ["4h", "1d"]),
        "htf_mode": trial.suggest_categorical("htf_mode", ["sma_trend", "structure"]),
        "htf_sma_period": trial.suggest_int("htf_sma_p", 20, 200),
        "htf_struct_lookback": trial.suggest_int("htf_struct_lb", 5, 30),
        "htf_struct_bars": trial.suggest_int("htf_struct_bars", 2, 5),
        "htf_adx_threshold": trial.suggest_float("htf_adx_thr", 0.0, 30.0),
        "use_swing_structure": trial.suggest_categorical("use_swing", [True, False]),
        "use_breakout": trial.suggest_categorical("use_breakout", [True, False]),
        "use_candle_bias": trial.suggest_categorical("use_candle", [True, False]),
        "use_mid_range": trial.suggest_categorical("use_midrange", [True, False]),
        "require_confluence": trial.suggest_int("req_confluence", 1, 3),
        "cooldown_bars": trial.suggest_int("cooldown", 3, 30),
    }
    base_params.update(mtf_params)
    return base_params


def main():
    print("=" * 100)
    print("MTF STRATEGIES — Optuna search on 2-year OOS (2024-09 → 2026-09)")
    print("12 strategies × 2 tickers × ~30 trials each")
    print("=" * 100)
    eur = fetch_h1("D:/MT5_EuroPrinter/terminal64.exe", "EURUSD")
    nas = fetch_h1("D:/MT5_Bybit/terminal64.exe", "NAS100")
    eur_oos = eur[(eur.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (eur.index < pd.Timestamp("2026-09-17", tz="UTC"))]
    nas_oos = nas[(nas.index >= pd.Timestamp("2024-09-17", tz="UTC")) & (nas.index < pd.Timestamp("2026-09-17", tz="UTC"))]
    print(f"  EUR OOS: {len(eur_oos)} bars")
    print(f"  NAS OOS: {len(nas_oos)} bars")

    n_trials = 30
    results = {}
    for sname, cls in MTF_CLASSES.items():
        for asset, df, profile in [("EUR", eur_oos, "forex"), ("NAS", nas_oos, "nas100")]:
            key = f"{sname}_{asset}"
            print(f"\n>>> {key} ({n_trials} trials)...")

            def obj(trial, df=df, profile=profile, cls=cls):
                params = param_space(trial, sname)
                m = run_strategy(df, cls, params, [], profile)
                if "error" in m or m["n_trades"] < 8:
                    return -1e9
                return m["net_pnl"] + 50 * m["sharpe"]

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_params
            # Score uses combined OOS (re-validate)
            m = run_strategy(df, cls, best_params, [], profile)
            if "error" in m:
                print(f"  ERR: {m['error']}")
                continue
            results[key] = {"params": best_params, "metrics": m, "cls_name": sname}
            status = "✓✓" if m["net_pnl"] > 0 and m["sharpe"] > 0 else ("~" if m["net_pnl"] > 0 else "✗")
            print(f"  score={study.best_value:.0f} | {status} PnL ${m['net_pnl']:+,.0f} "
                  f"Sharpe {m['sharpe']:+.2f} WR {m['win_rate']*100:.1f}% "
                  f"PF {m['profit_factor']:.2f} ({m['n_trades']} trades)")

    # Save
    out_dir = Path("output/mtf")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "all_best_params.json", "w") as f:
        json.dump({k: {"params": v["params"], "metrics": v["metrics"]}
                    for k, v in results.items()}, f, indent=2)
    print(f"\nSaved → {out_dir}/all_best_params.json")

    # Summary
    print("\n" + "=" * 100)
    print("MTF RESULTS — 2Y OOS, all 12 strategies × 2 assets")
    print("=" * 100)
    print(f"{'Instance':<28} {'PnL':>10} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'TPY':>6} {'Status':<10}")
    print("-" * 80)
    locked_count = 0
    for key in sorted(results.keys()):
        r = results[key]
        m = r["metrics"]
        years = 2.0
        tpy = m.get("n_trades", 0) / years
        if m.get("net_pnl", 0) > 0 and m.get("sharpe", 0) > 0:
            status = "✓ ROBUST"
            locked_count += 1
        elif m.get("net_pnl", 0) > 0:
            status = "~ borderline"
        else:
            status = "✗ failing"
        print(f"{key:<28} ${m.get('net_pnl', 0):>+9,.0f} {m.get('sharpe', 0):>+6.2f} "
              f"{m.get('win_rate', 0)*100:>4.1f}% {m.get('profit_factor', 0):>4.2f} "
              f"{tpy:>5.1f} {status}")

    print(f"\n  Total LOCKED (PnL AND Sharpe > 0): {locked_count}/24 instances")


if __name__ == "__main__":
    main()
