"""Generate MT5 .set files for locked V1 winners + production multi-instance runner.

For each locked V1 strategy profile, generate a .set file that can be loaded
directly into MT5 Strategy Tester. .set file format:
  Variable=Value
  // comment

For multi-instance deployment, the user attaches TwelveStrategies.mq5 on
multiple charts (e.g., EURUSD H1 + NAS100 H1), each loading its own .set file.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '.')
import json
from pathlib import Path

# Parameter name mapping: Python Optuna -> MQL5 input
# Each strategy has its own MQL5 input params
AC_MAPPING = {
    "bars_calculate": "AC_BarsCalculate",
    "open_orders_type": "AC_OpenOrdersType",
    "level_open_orders": "AC_LevelOpenOrders",
    "close_orders_type": "AC_CloseOrdersType",
    "level_close_orders": "AC_LevelCloseOrders",
    "use_acceleration_filter": "AC_UseAccelerationFilter",
    "min_acceleration": "AC_MinAcceleration",
    "use_ao_synchronization": "AC_UseAOSynchronization",
    "min_ao_synchronization": "AC_MinAOSynchronization",
    "acceleration_bars": "AC_AccelerationBars",
}

ADX_MAPPING = {
    "bars_calculate": "ADX_BarsCalculate",
    "use_di_crossover": "ADX_UseDICrossover",
    "crossover_lookback": "ADX_CrossoverLookback",
    "min_crossover_gap": "ADX_MinCrossoverGap",
    "open_orders_type": "ADX_OpenOrdersType",
    "level_open_orders_1": "ADX_LevelOpenOrders_1",
    "level_open_orders_2": "ADX_LevelOpenOrders_2",
    "close_orders_type": "ADX_CloseOrdersType",
    "level_close_orders_1": "ADX_LevelCloseOrders_1",
    "level_close_orders_2": "ADX_LevelCloseOrders_2",
    "use_zone_logic": "ADX_UseZoneLogic",
    "adx_zone_low": "ADX_ZoneLow",
    "adx_zone_high": "ADX_ZoneHigh",
    "continuation_level": "ADX_ContinuationLevel",
    "reversal_edge": "ADX_ReversalEdge",
    "sweep_lookback": "ADX_SweepLookback",
}

MS_MAPPING = {
    "ms_fast_ema": "MS_Fast_EMA_Period",
    "ms_slow_ema": "MS_Slow_EMA_Period",
    "ms_signal_period": "MS_Signal_Period",
    "open_orders_type": "MS_OpenOrdersType",
    "level_open_orders": "MS_LevelOpenOrders",
    "close_orders_type": "MS_CloseOrdersType",
    "level_close_orders": "MS_LevelCloseOrders",
}

MAPPING = {
    "ac_ao": AC_MAPPING,
    "adx": ADX_MAPPING,
    "ms": MS_MAPPING,
}


def to_mt5_set(strategy_name: str, params: dict, profile_name: str,
                output_path: Path, profile_meta: dict):
    """Convert Python params to MT5 .set file format."""
    mapping = MAPPING.get(strategy_name)
    if mapping is None:
        print(f"  Skipping {strategy_name}: no MQL5 mapping (V4 strategy, needs porting)")
        return None
    lines = []
    lines.append(f"; {profile_name}")
    lines.append(f"; Strategy: {strategy_name}")
    lines.append(f"; Ticker: {profile_meta.get('ticker', '?')}")
    lines.append(f"; Timeframe: {profile_meta.get('timeframe', 'H1')}")
    lines.append(f"; OOS window: {profile_meta.get('oos_window', {}).get('start', '?')} → "
                  f"{profile_meta.get('oos_window', {}).get('end', '?')}")
    lines.append(f"; Net PnL: ${profile_meta.get('metrics', {}).get('net_pnl', 0):+,.0f}")
    lines.append(f"; Sharpe: {profile_meta.get('metrics', {}).get('sharpe', 0):+.2f}")
    lines.append(f"; WR: {profile_meta.get('metrics', {}).get('win_rate', 0)*100:.1f}%")
    lines.append(f"; PF: {profile_meta.get('metrics', {}).get('profit_factor', 0):.2f}")
    lines.append("")
    # Symbol selector (TRADE_EURUSD_4 = 4, TRADE_NAS100 not in list — use CHART_PAIR for NAS)
    if profile_meta.get("ticker") == "EUR":
        lines.append("PairToTrade=4")  # TRADE_EURUSD_4
    else:
        # NAS not in the MQL5 enum — user must attach on NAS chart directly
        lines.append("PairToTrade=0")  # CHART_PAIR — use whatever chart the EA is on
    lines.append("TimeFrame=PERIOD_H1")
    lines.append("")
    # Map params
    for py_key, val in params.items():
        mql_key = mapping.get(py_key)
        if mql_key is None:
            continue
        # Handle bool
        if isinstance(val, bool):
            lines.append(f"{mql_key}={'true' if val else 'false'}")
        elif isinstance(val, float):
            lines.append(f"{mql_key}={val:.6f}")
        elif isinstance(val, int):
            lines.append(f"{mql_key}={val}")
        else:
            lines.append(f"{mql_key}={val}")
    # Enable ONLY this strategy (disable others)
    lines.append("")
    lines.append("; === Enable ONLY this strategy ===")
    # All strategy runs map to StrategyRun = true/false. By default all are true.
    # Set this strategy's run = true, others = false.
    enabled_strats = {"ac_ao", "adx", "ms"}
    all_strats = ["ac_ao", "adx", "dem", "fbb", "mfi", "ms",
                  "mtf_stoch", "bb_rsi", "triple_rsi", "quad_stoch",
                  "stoch533_mtf", "macd_confluence"]
    for s in all_strats:
        run_var = f"{s.upper()}_StrategyRun"
        if s == strategy_name:
            lines.append(f"{run_var}=true")
        elif s in enabled_strats:
            lines.append(f"{run_var}=false")
    # Other strats (mfi, dem, fbb, etc.) — leave at default
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def main():
    print("=" * 100)
    print("MT5 .set FILES + PRODUCTION DEPLOYMENT")
    print("=" * 100)
    profiles_dir = Path("output/profiles")
    set_dir = Path("output/set_files")
    set_dir.mkdir(parents=True, exist_ok=True)

    # Process each locked profile
    generated = []
    for profile_path in sorted(profiles_dir.glob("*.json")):
        with open(profile_path) as f:
            profile = json.load(f)
        strategy = profile["strategy"]
        # Skip V4 strategies (not in MQL5 yet)
        if "_v4" in strategy:
            print(f"  [skip] {profile_path.name}: V4 strategy, needs MQL5 porting")
            continue
        params = profile["params"]
        # Special handling for AC+AO strategy (Python: "ac_ao", MQL5: same)
        mql_name = strategy
        set_path = set_dir / f"{strategy}_{profile['ticker']}.set"
        result = to_mt5_set(mql_name, params, profile_path.stem, set_path, profile)
        if result:
            print(f"  [ok] {profile_path.name} → {set_path}")
            generated.append(set_path)
        else:
            print(f"  [skip] {profile_path.name}")

    print(f"\n  Generated {len(generated)} .set files in {set_dir}")

    # Build production multi-instance launcher
    print("\n" + "=" * 100)
    print("PRODUCTION MULTI-INSTANCE LAUNCHER")
    print("=" * 100)
    print("""
DEPLOYMENT GUIDE — single chart per EA instance, multiple instances run independently

1. ATTACH EA: TwelveStrategies.mq5 on each chart (one chart per instance)
2. LOAD .set FILE for that instance:
   - MT5 → Tools → Strategy Tester
   - Select "TwelveStrategies" EA
   - Click "Inputs" → "Load" → choose the .set file
   - Set Symbol/TimeFrame to match the .set file
3. ENABLE AUTO-TRADING on each chart

RECOMMENDED INSTANCES (independent, each runs alone):

""")
    for profile_path in sorted(profiles_dir.glob("*.json")):
        with open(profile_path) as f:
            profile = json.load(f)
        strategy = profile["strategy"]
        m = profile.get("metrics", {})
        note = ""
        if "_v4" in strategy:
            note = " (V4 — Python only, needs MQL5 porting)"
        print(f"  [{profile['ticker']:<6}] {strategy:<25} PnL ${m.get('net_pnl', 0):+8,.0f} "
              f"Sharpe {m.get('sharpe', 0):+.2f} WR {m.get('win_rate', 0)*100:.1f}% "
              f"PF {m.get('profit_factor', 0):.2f}{note}")

    # Save deployment manifest
    manifest = {
        "deployment_date": "2026-09-17",
        "oos_window": "2024-09-17 → 2026-09-17 (24 months)",
        "instances": [],
    }
    for profile_path in sorted(profiles_dir.glob("*.json")):
        with open(profile_path) as f:
            profile = json.load(f)
        set_file = set_dir / f"{profile['strategy']}_{profile['ticker']}.set"
        if not set_file.exists():
            set_file = None
        manifest["instances"].append({
            "strategy": profile["strategy"],
            "ticker": profile["ticker"],
            "timeframe": profile["timeframe"],
            "version": profile["version"],
            "metrics": profile["metrics"],
            "set_file": str(set_file) if set_file else None,
            "ready_for_mt5": set_file is not None,
            "needs_mql5_port": "_v4" in profile["strategy"],
        })
    with open("output/DEPLOYMENT_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Saved → output/DEPLOYMENT_MANIFEST.json")


if __name__ == "__main__":
    main()
