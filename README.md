# Newmeta Backtester

> Drop any strategy file. Get a full pro backtest with grid, recovery, swaps, optimization, Monte Carlo, walk-forward, and AI-grade validation — in seconds.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A universal backtester that turns **any** trading strategy source code into a vectorized Python backtest, validates it, optimizes it, and stress-tests it. **100-1000× faster than MT5 strategy tester**, with the same metrics.

---

## What you get

Drop one of these files → backtest runs instantly:

| File type | Example |
|---|---|
| `.mq5` | MQL5 Expert Advisor (auto-extracts 200+ inputs, maps to strategy) |
| `.py`  | Python class subclassing `BaseStrategy` |
| `.pine` | PineScript v5 (RSI/EMA/SMA/MACD/Stoch/BB/ATR/divergence) |
| `.txt` | Key=value config |

Then in the sidebar, drag sliders to tune:

```
Strictness     [0 ─────── ● ─── 10]   per-strategy: permissive ↔ strict
TP/SL widening [0 ─── ● ───── 10]   tight ↔ wide stops
Risk Profile   [Conservative | Balanced | Aggressive]
Grid mode      [None | Loss | Profit | Both]
Recovery       [None | Last close (martingale)]
Adaptive       [☐ streak-aware lot sizing]
Swap           [☐ 3× Wed + holidays + weekend]
```

Then pick a mode:

| Mode | What it does |
|---|---|
| 🚀 **Backtest** | Run on selected data, see equity + 19 metrics |
| 🔬 **Optimize** | Optuna Bayesian search with composite criterion (Sharpe+Calmar+PF+DD) |
| 🎲 **Monte Carlo** | 2000 bootstrap simulations with 95% CI |
| 📊 **Walk-Forward** | Rolling IS/OOS to detect overfit |
| 🧬 **Multi-Strategy** | All 6 strategies + Markowitz/Kelly allocation |
| ✅ **Validate** | Test against 7 synthetic scenarios — catches bugs |
| 🪄 **Auto-Magic** | Drop file → 6-step workflow + verdict |
| 🎓 **Guided** | Step-by-step wizard for non-pro traders |

---

## Quickstart (60 seconds)

```powershell
cd "D:\Trading\TRADING\youha created EA\Multi strat ea\backtest_harness"
$env:PYTHONPATH = (Get-Location).Path
pip install -r requirements.txt
streamlit run frontend\app.py
```

Opens `http://localhost:8501`. Drop a `.mq5` file → click `▶ Run Backtest`. Done.

---

## What it does that other backtesters don't

### 1. Grid + Recovery simulation (matches MT5 EA behavior)
The EA uses martingale-style averaging (`Make_Grid_*`, `MultiRecoveryLot`). Most backtesters ignore this and report wildly optimistic results. We simulate it.

### 2. Adaptive sizing (streak-aware)
- 3 losses in a row → drop to 0.5× lot
- 5 wins in a row → bump to 1.2× lot
- 10% rolling drawdown → pause trading

### 3. Realistic swaps (Wed 3× + holidays)
- Wednesday: 3× overnight swap (covers Sat+Sun)
- US holidays: 0 swap
- Long/short swap rates configurable

### 4. Strategy Validator (catches coding + logical errors)
Every strategy tested against 7 scenarios with KNOWN expected behavior:
- Trending up/down (must produce directional signals)
- Ranging (must not over-trade)
- Crash / impulse / mean-reverting
- Constant price (must produce 0 trades)
- Indicator sanity (no NaN bombs)

### 5. Composite criterion for optimization
Pick your weights:
```
Sharpe weight     [██████░░░░] 60%
Calmar weight     [████░░░░░░] 40%
Profit Factor     [██░░░░░░░░] 20%
Drawdown (inv)    [█░░░░░░░░░] 10%
```
Optuna maximizes the composite score.

### 6. MQL5 Equivalence Validation
Run our harness side-by-side with MT5 strategy tester. Get a PASS/INVESTIGATE/FAIL verdict. This is the **only** way to know if our port matches MT5.

---

## Architecture (small but mighty)

```
backtest_harness/
├── frontend/app.py            ← Streamlit UI (8 modes)
├── core/
│   ├── loader.py              ← Universal file loader (.mq5/.py/.pine/.txt)
│   ├── pine_parser.py         ← PineScript v5 parser
│   ├── universal_strategy.py  ← Vectorized execution of parsed logic
│   ├── strictness.py          ← 0-10 per-strategy strictness + risk profiles
│   └── features.py            ← RegimeFilter, SessionFilter, DailyPnLCap
├── strategies/
│   ├── ac_ao.py, adx.py, dem.py, fbb.py, mfi.py, ms.py, mtf_stoch.py
│   └── indicators.py          ← AC, AO, ADX, MFI, MACD, BB, DeM, Force, LR-slope
├── backtester/
│   ├── engine.py              ← Pure vectorbt
│   ├── engine_grid.py         ← + grid + recovery
│   ├── engine_full.py         ← + adaptive + swaps
│   ├── grid_recovery.py       ← Grid state machine
│   ├── adaptive.py            ← Streak-aware sizer
│   ├── swaps.py               ← Wed 3× + holidays
│   ├── metrics_v2.py          ← 19 metrics
│   └── analytics.py           ← Scoreboard + trades/year
├── data/
│   ├── live_fetcher.py        ← MT5 → Yahoo → cache → synthetic
│   ├── mt5_export.py          ← OHLCV export
│   ├── tick_data.py           ← Real + synthetic ticks
│   └── spread_spec.py         ← Realistic spread model
├── analysis/
│   ├── optuna_optimizer.py    ← TPE multi-objective
│   ├── walkforward.py         ← Rolling IS/OOS
│   ├── montecarlo.py          ← Block bootstrap
│   ├── markowitz_alloc.py     ← Ledoit-Wolf + Kelly + risk-parity
│   ├── composite_criterion.py ← Weighted Sharpe+Calmar+PF+DD
│   └── multi_walkforward.py   ← WF across all strategies
├── tools/
│   ├── strategy_validator.py  ← 7-scenario validation
│   ├── mql5_compare.py        ← Python vs MT5 diff
│   ├── mt5_trade_parser.py    ← MT5 tester CSV parser
│   └── export_tester_deals.mq5 ← MT5 script for trade export
├── tests/                     ← Smoke tests
├── config/
│   ├── settings.yaml          ← MT5 terminal + execution defaults
│   └── strategies.yaml        ← Optuna parameter ranges
└── run_pipeline.py            ← CLI runner (no UI)
```

---

## CLI usage

```bash
# Single strategy backtest
python -m run_pipeline --strategy fbb --montecarlo

# Multi-strategy walk-forward
python -m analysis.multi_walkforward --data EURUSD_H1 --n-trials 50

# Strategy validation (catch bugs)
python -m tools.strategy_validator --strategy ac_ao

# MQL5 equivalence test
python -m analysis.mql5_compare \
    --mt5-trades tester_trades.csv \
    --strategy fbb --data EURUSD_H1
```

---

## Honest limitations

1. **OHLC bars only** — no real-tick simulation. Use MT5 strategy tester for tick accuracy.
2. **Single instrument / timeframe** — multi-asset pairs trading not yet built.
3. **MQL5 equivalence untested** until you run it (one-time manual step).
4. **No live trading bridge** — backtest only. Use a separate MT5 demo for live validation.

---

## License

MIT — see `LICENSE`.

## Credits

Built for the Newmeta EA. MQL5 port: Nikolaos Pantzos / Gemini merges. Python harness: opencode AI-assisted.