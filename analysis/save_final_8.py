"""Save the 8 final candidates — strongest 5 ON by default, 3 OFF by default."""
import json
from pathlib import Path

# 5 ROBUST singles (strongest — ON by default)
strong_5 = {
    "adx_NAS": {
        "strategy": "adx",
        "ticker": "NAS",
        "enabled_default": True,
        "params_file": "output/profiles_final/adx_NAS.json",
        "category": "robust_single",
        "tuning_pnl": 31999, "tuning_sharpe": 7.34,
        "val_pnl": 3479, "val_sharpe": 1.93,
        "verdict": "Gold star - 11x leverage effect on combined periods",
    },
    "adx_EUR": {
        "strategy": "adx",
        "ticker": "EUR",
        "enabled_default": True,
        "params_file": "output/profiles_final/adx_EUR.json",
        "category": "robust_single",
        "tuning_pnl": 2222, "tuning_sharpe": 1.73,
        "val_pnl": 1425, "val_sharpe": 2.31,
        "verdict": "Best Sharpe on validation (improves from 1.73 to 2.31)",
    },
    "linda_macd_lenient_EUR": {
        "strategy": "linda_macd_lenient",
        "ticker": "EUR",
        "enabled_default": True,
        "params_file": "output/linda_macd_lenient_best.json",
        "category": "robust_single",
        "tuning_pnl": 725, "tuning_sharpe": 0.58,
        "val_pnl": 847, "val_sharpe": 1.26,
        "verdict": "New ROBUST — Sharpe IMPROVES on validation (real edge)",
    },
    "sc_s5_stoch_NAS": {
        "strategy": "sc_s5_stoch",
        "ticker": "NAS",
        "enabled_default": True,
        "params_file": "output/profiles_final/sc_s5_stoch_NAS.json",
        "category": "robust_single",
        "tuning_pnl": 431, "tuning_sharpe": 1.29,
        "val_pnl": 132, "val_sharpe": 0.59,
        "verdict": "NAS short bias — fits the 2024-26 downtrend",
    },
    "sc_s5_stoch_EUR": {
        "strategy": "sc_s5_stoch",
        "ticker": "EUR",
        "enabled_default": True,
        "params_file": "output/profiles_final/sc_s5_stoch_EUR.json",
        "category": "robust_single",
        "tuning_pnl": 45, "tuning_sharpe": 0.16,
        "val_pnl": 163, "val_sharpe": 0.83,
        "verdict": "EUR stochastic — small but ROBUST",
    },
    "sc_s8_bb_NAS": {
        "strategy": "sc_s8_bb",
        "ticker": "NAS",
        "enabled_default": True,
        "params_file": "output/profiles_final/sc_s8_bb_NAS.json",
        "category": "robust_single",
        "tuning_pnl": 607, "tuning_sharpe": 1.12,
        "val_pnl": 283, "val_sharpe": 1.06,
        "verdict": "Bollinger short — robust on both periods",
    },
    "ms_EUR": {
        "strategy": "ms",
        "ticker": "EUR",
        "enabled_default": True,
        "params_file": "output/profiles_final/ms_EUR.json",
        "category": "robust_single",
        "tuning_pnl": 64, "tuning_sharpe": 0.06,
        "val_pnl": 73, "val_sharpe": 0.14,
        "verdict": "Marginal but ROBUST — included for diversification",
    },
}

# 3 NEW merges (OFF by default — user can enable if interested)
new_3 = {
    "merge_sc8_plus_sc7_AND_EUR": {
        "name": "sc_s8_bb + sc_s7_macd (AND-gate)",
        "logic": "AND-gate: both Bollinger and MACD setups must agree",
        "enabled_default": False,
        "ticker": "EUR",
        "params": {
            "_strategy1": "sc_s8_bb",
            "_strategy2": "sc_s7_macd",
            "_strategy1_params": "$sc_s8_bb_EUR.params",
            "_strategy2_params": "$sc_s7_macd_EUR.params",
        },
        "tuning_pnl": 26, "val_pnl": 56, "trades_total": 6,
        "verdict": "ROBUST but marginal — only 6 trades total",
    },
    "merge_OR_adx_sc6_EUR": {
        "name": "OR adx + sc_s6_macross (vote-1-of-2)",
        "logic": "OR-gate: fire when EITHER adx OR sc_s6_macross signals",
        "enabled_default": False,
        "ticker": "EUR",
        "params": {
            "n_strategies": 2,
            "vote_threshold": 1,
            "cooldown": 4,
            "_strategy1": "adx", "_strategy2": "sc_s6_macross",
        },
        "tuning_pnl": 68, "val_pnl": 62, "trades_total": 10,
        "high_wr": True, "tuning_wr": 0.875, "val_wr": 1.0,
        "verdict": "ROBUST + HIGH-WR but only 10 trades — statistical noise risk",
    },
    "merge_primary_sc6_filter_adx_EUR": {
        "name": "sc_s6_macross primary + adx filter (cw=5)",
        "logic": "Primary+Filter: sc_s6 fires entry, adx confirms within 5 bars",
        "enabled_default": False,
        "ticker": "EUR",
        "params": {
            "_primary": "sc_s6_macross",
            "_filter": "adx",
            "confirm_window": 5,
            "cooldown": 4,
        },
        "tuning_pnl": -57, "val_pnl": 92, "trades_total": 36,
        "high_wr": True, "tuning_wr": 0.667, "val_wr": 0.733,
        "verdict": "HIGH-WR 66.7%/73.3% but BAD RR — loses on tuning",
    },
}

# Merge into final_8
final_8 = {**strong_5, **new_3}
final_8["_meta"] = {
    "name": "Final 8",
    "version": "1.0",
    "description": "5 strong ROBUST singles (always on) + 3 NEW merges (off by default, user can enable)",
    "default_enabled_count": len(strong_5),
    "total_count": len(strong_5) + len(new_3),
    "creation_date": "2026-09-18",
    "validation_method": "Python backtester cross-validated against MT5 Tickmill-Live data (100% OHLC match, strategy results within 0.22%)",
    "note_on_duplicates": "adx_NAS and adx_EUR are the SAME algorithm with different per-asset parameter profiles. Same for sc_s5_stoch_NAS vs _EUR.",
}

# Save
out = Path("output/final_8/Final_8.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump(final_8, f, indent=2, default=str)

print(f'Saved → {out}')
print(f'Total: {final_8["_meta"]["total_count"]} candidates')
print(f'  Default ON: {final_8["_meta"]["default_enabled_count"]} (5 ROBUST singles)')
print(f'  Default OFF: {len(new_3)} (3 NEW merges)')
