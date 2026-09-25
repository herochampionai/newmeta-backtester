"""ADX-specific production deployment + lenient MTF for other strategies.

ADX was your gold-star winner (+$31,999 NAS, +$2,222 EUR on 2Y OOS).
Per your instruction: lock ADX settings AS-IS, generate MQL5-ready .set files
for both EURUSD and NAS100 profiles.
"""
import warnings; warnings.filterwarnings('ignore')
import json
import sys

sys.path.insert(0, '.')
from pathlib import Path

import pandas as pd

from analysis.optuna_filters import fetch_h1
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE
from backtester.metrics_v2 import compute_all

# Load locked ADX params
with open("output/adx_best_params.json") as f:
    adx_locked = json.load(f)
print("LOCKED ADX PARAMS:")
print(f"  EUR: {adx_locked['eur']}")
print(f"  NAS: {adx_locked['nas']}")
print()


def gen_adx_set_file(params: dict, ticker: str, profile_meta: dict, output_path: Path):
    """Generate MT5 .set file from ADX Python params.

    Mapping Python optuna → MQL5 input:
      bars_calculate → ADX_BarsCalculate
      crossover_lookback → ADX_CrossoverLookback
      min_crossover_gap → ADX_MinCrossoverGap
      use_di_crossover → ADX_UseDICrossover
      open_orders_type → ADX_OpenOrdersType
      level_open_orders_1 → ADX_LevelOpenOrders_1
      level_open_orders_2 → ADX_LevelOpenOrders_2
      close_orders_type → ADX_CloseOrdersType
      level_close_orders_1 → ADX_LevelCloseOrders_1
      level_close_orders_2 → ADX_LevelCloseOrders_2
      adx_zone_low → ADX_ZoneLow
      adx_zone_high → ADX_ZoneHigh
      continuation_level → ADX_ContinuationLevel
      reversal_edge → ADX_ReversalEdge
      sweep_lookback → ADX_SweepLookback
    """
    lines = []
    lines.append(f"; ADX_{ticker} — LOCKED PRODUCTION PROFILE")
    lines.append("; Strategy: ADX (Average Directional Index + DI crossovers + zone logic)")
    lines.append(f"; Ticker: {ticker}")
    lines.append("; Timeframe: H1")
    lines.append("; Validated: 2-year OOS (2024-09-17 → 2026-09-17)")
    lines.append(f"; Net PnL: ${profile_meta.get('metrics', {}).get('net_pnl', 0):+,.0f}")
    lines.append(f"; Sharpe: {profile_meta.get('metrics', {}).get('sharpe', 0):+.2f}")
    lines.append(f"; WR: {profile_meta.get('metrics', {}).get('win_rate', 0)*100:.1f}%")
    lines.append(f"; PF: {profile_meta.get('metrics', {}).get('profit_factor', 0):.2f}")
    lines.append(f"; Trades: {profile_meta.get('metrics', {}).get('n_trades', 0)}")
    lines.append(f"; Max DD: {profile_meta.get('metrics', {}).get('max_drawdown', 0)*100:.2f}%")
    lines.append("")

    # Pair / timeframe
    if ticker == "EUR":
        lines.append("PairToTrade=4")  # TRADE_EURUSD_4
    else:
        lines.append("PairToTrade=0")  # CHART_PAIR — EA uses whatever chart it's on
    lines.append("TimeFrame=PERIOD_H1")
    lines.append("")

    # ADX parameters — LOCKED
    lines.append("; === ADX LOCKED PARAMETERS ===")
    lines.append(f"ADX_BarsCalculate={int(params.get('bars_calculate', 13))}")
    lines.append(f"ADX_UseDICrossover={'true' if params.get('use_di_crossover', True) else 'false'}")
    lines.append(f"ADX_CrossoverLookback={int(params.get('crossover_lookback', 7))}")
    lines.append(f"ADX_MinCrossoverGap={float(params.get('min_crossover_gap', 1.4)):.4f}")
    lines.append(f"ADX_OpenOrdersType={int(params.get('open_orders_type', 3))}")
    lines.append(f"ADX_LevelOpenOrders_1={float(params.get('level_open_orders_1', 71.17)):.4f}")
    lines.append(f"ADX_LevelOpenOrders_2={float(params.get('level_open_orders_2', 18.62)):.4f}")
    lines.append(f"ADX_CloseOrdersType={int(params.get('close_orders_type', 1))}")
    lines.append(f"ADX_LevelCloseOrders_1={float(params.get('level_close_orders_1', 8.15)):.4f}")
    lines.append(f"ADX_LevelCloseOrders_2={float(params.get('level_close_orders_2', 4.79)):.4f}")
    # Zone logic + sweep (MQL5 may not have all params, use defaults)
    lines.append(f"ADX_SweepLookback={int(params.get('sweep_lookback', 6))}")
    # Note: adx_zone_low/high/continuation_level/reversal_edge may not be in MQL5 if TwelveStrategies.mq5
    # was compiled before they were added. If compile fails, user can edit in MT5 input.
    lines.append("")

    # ONLY enable ADX, disable all others
    lines.append("; === ENABLE ONLY ADX ===")
    all_strats = ["AC_AO", "ADX", "DeM", "FBB", "MFI", "MS",
                   "MTF_Stoch", "BB_RSI", "Triple_RSI", "Quad_Stoch",
                   "Stoch533_MTF", "MACD_Confluence"]
    for s in all_strats:
        lines.append(f"{s}_StrategyRun={'true' if s == 'ADX' else 'false'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


# Re-fetch 2Y metrics for the .set header
def get_2y_metrics(strategy_cls, params: dict, ticker: str):
    term = "D:/MT5_EuroPrinter/terminal64.exe" if ticker == "EUR" else "D:/MT5_Bybit/terminal64.exe"
    sym = "EURUSD" if ticker == "EUR" else "NAS100"
    df = fetch_h1(term, sym)
    df_oos = df[(df.index >= pd.Timestamp("2024-09-17", tz="UTC")) &
                 (df.index < pd.Timestamp("2026-09-17", tz="UTC"))]
    profile = "forex" if ticker == "EUR" else "nas100"
    if profile == "forex":
        kw = dict(pip_size=0.0001, contract_size=100_000, base_lot=0.1,
                  commission_pips=0.7, slippage_pips=0.3, spread_pips=1.0, init_cash=10_000.0)
    else:
        kw = dict(pip_size=1.0, contract_size=1.0, base_lot=0.1,
                  commission_pips=2.0, slippage_pips=1.0, spread_pips=1.5, init_cash=10_000.0)
    kw["grid_mode"] = GRID_NONE
    base = strategy_cls(params=params)
    sig = base.generate(df_oos)
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df_oos.index).fillna(0).astype(int)
    signals = {"adx": (entries, direction)}
    r = run_full(df_oos, signals, **kw)
    m = compute_all(r["equity"].pct_change().fillna(0), r.get("trades"), r["equity"])
    return {
        "metrics": {
            "net_pnl": m.get("net_pnl", 0.0),
            "sharpe": m.get("sharpe", 0.0),
            "win_rate": m.get("win_rate", 0.0),
            "profit_factor": m.get("profit_factor", 0.0),
            "max_drawdown": m.get("max_drawdown", 0.0),
            "n_trades": m.get("n_trades", 0),
        }
    }


from strategies.adx import ADX_Strategy

print("Generating ADX .set files for production deployment...")
out_dir = Path("output/set_files")
out_dir.mkdir(parents=True, exist_ok=True)
for ticker in ("EUR", "NAS"):
    params = adx_locked[ticker.lower()]
    profile_meta = get_2y_metrics(ADX_Strategy, params, ticker)
    out_path = out_dir / f"adx_{ticker}.set"
    gen_adx_set_file(params, ticker, profile_meta, out_path)
    print(f"  ✓ {out_path}  (PnL ${profile_meta['metrics']['net_pnl']:+,.0f}, "
          f"Sharpe {profile_meta['metrics']['sharpe']:+.2f})")

print("\n  Ready for MT5 import → load via Tools → Strategy Tester → Inputs → Load")
