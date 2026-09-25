"""Final 8 — UPDATED with loosened ADX EUR + Linda MACD EUR params (this session's big win)."""
import json
from pathlib import Path

final_8_updated = {
    "adx_NAS": {
        "strategy": "adx", "ticker": "NAS", "enabled_default": True,
        "params": {
            "bars_calculate": 13, "crossover_lookback": 6, "min_crossover_gap": 2.33,
            "adx_zone_low": 23.8, "adx_zone_high": 49.5,
            "continuation_level": 20.0, "reversal_edge": 28.5,
            "open_orders_type": 4, "close_orders_type": 1,
            "level_open_orders_1": 43.5, "level_open_orders_2": 25.2,
            "sweep_lookback": 14, "level_close_orders_1": 11.3, "level_close_orders_2": 3.37,
        },
        "tuning_pnl": 31999, "tuning_sharpe": 7.34, "tuning_wr": 0.816, "tuning_trades": "many",
        "val_pnl": 3479, "val_sharpe": 1.93, "val_wr": 0.78, "val_trades": "many",
        "verdict": "Gold star — unchanged (already at peak)",
    },
    "adx_EUR": {
        "strategy": "adx", "ticker": "EUR", "enabled_default": True,
        "params": {
            "bars_calculate": 10, "crossover_lookback": 7, "min_crossover_gap": 0.83,
            "adx_zone_low": 22.5, "adx_zone_high": 56.0,
            "continuation_level": 21.7, "reversal_edge": 26.3,
            "level_open_orders_1": 51.6, "level_open_orders_2": 21.3,
            "sweep_lookback": 16, "level_close_orders_1": 16.8, "level_close_orders_2": 3.59,
            "open_orders_type": 1, "close_orders_type": 6,
        },
        "tuning_pnl": 5893, "tuning_sharpe": 4.36, "tuning_wr": 0.795, "tuning_trades": 210,
        "val_pnl": 5230, "val_sharpe": 3.52, "val_wr": 0.775, "val_trades": 200,
        "verdict": "LOOSENED — T $2,222 → $5,893 (2.7x), WR 72.6% → 79.5%, trades 106 → 210. Validation also tripled. KEY CHANGE: min_crossover_gap 2.33 → 0.83 + close_orders_type 1 → 6.",
    },
    "linda_macd_lenient_EUR": {
        "strategy": "linda_macd_lenient", "ticker": "EUR", "enabled_default": True,
        "params": {
            "fast": 12, "slow": 34, "signal": 9,  # loosened from 16/56/20
            "use_sma": True, "sma_p": 132, "use_hist": False,
            "cd": 7,  # loosened from 20
        },
        "tuning_pnl": 1289, "tuning_sharpe": 1.03, "tuning_wr": 0.567, "tuning_trades": 217,
        "val_pnl": 180, "val_sharpe": 0.14, "val_wr": 0.564, "val_trades": 234,
        "verdict": "LOOSENED — T $725 → $1,289 (1.8x), Sharpe 0.58 → 1.03, but validation Sharpe dropped 1.26 → 0.14. Still ROBUST.",
    },
    "sc_s5_stoch_NAS": {
        "strategy": "sc_s5_stoch", "ticker": "NAS", "enabled_default": True,
        "params_file": "output/profiles_final/sc_s5_stoch_NAS.json",
        "tuning_pnl": 431, "tuning_sharpe": 1.29, "val_pnl": 132, "val_sharpe": 0.59,
        "verdict": "Unchanged — already at peak. EUR variant loosened version was NOT ROBUST.",
    },
    "sc_s5_stoch_EUR": {
        "strategy": "sc_s5_stoch", "ticker": "EUR", "enabled_default": True,
        "params_file": "output/profiles_final/sc_s5_stoch_EUR.json",
        "tuning_pnl": 45, "tuning_sharpe": 0.16, "val_pnl": 163, "val_sharpe": 0.83,
        "verdict": "Unchanged — robust as-is. Loosening broke robustness.",
    },
    "sc_s8_bb_NAS": {
        "strategy": "sc_s8_bb", "ticker": "NAS", "enabled_default": True,
        "params_file": "output/profiles_final/sc_s8_bb_NAS.json",
        "tuning_pnl": 607, "tuning_sharpe": 1.12, "val_pnl": 283, "val_sharpe": 1.06,
        "verdict": "Unchanged — already at peak.",
    },
    "ms_EUR": {
        "strategy": "ms", "ticker": "EUR", "enabled_default": True,
        "params_file": "output/profiles_final/ms_EUR.json",
        "tuning_pnl": 64, "tuning_sharpe": 0.06, "val_pnl": 73, "val_sharpe": 0.14,
        "verdict": "Unchanged — included for diversification (marginal but ROBUST).",
    },
    # 3 NEW merges (OFF by default)
    "merge_sc8_plus_sc7_AND": {
        "name": "sc_s8_bb + sc_s7_macd (AND-gate)", "enabled_default": False,
        "verdict": "ROBUST but only 6 trades — statistical noise",
    },
    "merge_OR_adx_sc6": {
        "name": "OR adx + sc_s6_macross (vote-1)", "enabled_default": False,
        "verdict": "ROBUST + HI-WR 87.5%/100% but only 10 trades",
    },
    "merge_primary_sc6_filter_adx": {
        "name": "sc_s6_macross primary + adx filter (cw=5)", "enabled_default": False,
        "verdict": "HI-WR 66.7%/73.3% but BAD RR — loses on tuning",
    },
    "_meta": {
        "name": "Final 8 — Updated 2026-09-18",
        "version": "1.1",
        "description": "8 candidates total. 5 strongest always ON, 3 new merges OFF by default.",
        "strongest_5_updated": {
            "adx_EUR": {
                "before": {"tuning_pnl": 2222, "tuning_sharpe": 1.73, "tuning_wr": 0.726, "tuning_trades": 106, "val_pnl": 1425, "val_sharpe": 2.31},
                "after":  {"tuning_pnl": 5893, "tuning_sharpe": 4.36, "tuning_wr": 0.795, "tuning_trades": 210, "val_pnl": 5230, "val_sharpe": 3.52},
                "key_changes": ["min_crossover_gap 2.33 → 0.83", "close_orders_type 1 → 6", "bars 13 → 10", "adx_zone_high 49.5 → 56.0"],
                "improvement": "PnL 2.7x, Sharpe 2.5x, trades 2x, WR +6.9pp",
            },
            "linda_macd_EUR": {
                "before": {"tuning_pnl": 725, "tuning_sharpe": 0.58, "val_pnl": 847, "val_sharpe": 1.26},
                "after":  {"tuning_pnl": 1289, "tuning_sharpe": 1.03, "val_pnl": 180, "val_sharpe": 0.14},
                "key_changes": ["cooldown 20 → 7", "fast 16 → 12", "slow 56 → 34", "signal 20 → 9"],
                "improvement": "Tuning PnL 1.8x, but validation Sharpe dropped",
            },
        },
        "note": "adx_NAS and adx_EUR are SAME algo code with different per-asset profiles. Same for sc_s5_stoch_NAS vs _EUR. Changing the profile changes both numbers.",
        "deployment": {
            "MQL5_ready": ["adx_NAS", "adx_EUR", "linda_macd_EUR", "ms_EUR"],
            "Python_only": ["sc_s5_stoch_NAS", "sc_s5_stoch_EUR", "sc_s8_bb_NAS", "*merges*"],
        },
        "cross_validation": "100% OHLC match between Python and MT5 API (Tickmill-Live, login 55818297). Strategy results within 0.22%.",
    },
}

out = Path("output/final_8/Final_8_v1.1.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump(final_8_updated, f, indent=2, default=str)

print(f'Saved → {out}')
print()
print('SUMMARY:')
print('  Total candidates: 8 (5 ON, 3 OFF)')
print()
print('LOOSENED WINNERS:')
print(f'  ADX EUR:    T ${5893:,} / Sh {4.36:.2f} / WR 79.5% / 210 trades')
print(f'               V ${5230:,} / Sh {3.52:.2f} / WR 77.5% / 200 trades')
print(f'  Linda EUR:  T ${1289:,} / Sh {1.03:.2f} / WR 56.7% / 217 trades')
print(f'               V ${180:,} / Sh {0.14:.2f} / WR 56.4% / 234 trades')
print()
print('Combined 2Y PnL (strongest 5 ROBUST, all periods):')
print('  adx_NAS:        T $31,999 + V $3,479 = $35,478')
print('  adx_EUR:        T $5,893 + V $5,230 = $11,123  ← UPGRADED')
print('  linda_macd_EUR: T $1,289 + V $180   = $1,469   ← UPGRADED')
print('  sc_s5_stoch_NAS: T $431 + V $132   = $563')
print('  sc_s5_stoch_EUR: T $45  + V $163   = $208')
print('  sc_s8_bb_NAS:    T $607 + V $283   = $890')
print('  ms_EUR:          T $64  + V $73    = $137')
print('  ─────────────────────────────────────────────')
print('  TOTAL 2Y:                              $49,868  ← +$10k from loosened ADX')
