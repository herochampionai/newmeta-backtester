# Newmeta Research Lab - Backtester

This backtester is the mother-company research tool. It is separate from the MQL5 EA scripts.

The EA scripts are outputs after research is complete:

```text
Multi Strat EA = 12-strategy MQL5 EA family
9 Crypto Strategy EA = separate MQL5 EA family
```

The backtester exists to decide what is worth converting, optimizing, and running in MQL5.

## Target Specialty

The backtester should compete around these capabilities:

```text
Most realistic simulations
Institutional-grade research
Crypto + Gold / XAUUSD support
Advanced optimization
Walk-forward analysis
Slippage / commission / spread / swap modeling
```

## Product Positioning

### QuantConnect Standard

What to learn from it:

```text
multi-asset research
clean parameter studies
walk-forward validation
portfolio-level thinking
robust reporting
```

Newmeta target:

```text
local and fast like Python, but organized like an institutional research terminal
```

### MT5 Strategy Tester Standard

What to learn from it:

```text
tick-style realism
broker settings
spread and commission impact
MQL5 optimization workflow
simple path from backtest to live EA
```

Newmeta target:

```text
make every Python result explainable enough to compare with MT5 and prepare MQL5 scripts
```

### TradingView Standard

What to learn from it:

```text
fast idea validation
visual strategy inspection
Pine-to-research workflow
quick crypto iteration
```

Newmeta target:

```text
accept Pine/Python/MQL5 ideas, then test them with stronger execution assumptions
```

## Required Backtester Modules

### 1. Data Layer

Must support:

```text
MT5 OHLC/tick export
local parquet/cache data
CSV import
crypto symbols
XAUUSD/gold symbols
multiple timeframes
repeatable datasets for optimization
```

Missing or needs hardening:

```text
symbol metadata per market: pip size, contract size, tick value, quote currency
spread history or realistic spread presets
crypto exchange fee presets
XAUUSD broker presets
```

### 2. Execution Simulation

Must model:

```text
commission
slippage
spread
swap/overnight cost
position sizing
grid/recovery overlays
long/short direction
multi-strategy baskets
```

Already present:

```text
commission_pips
slippage_pips
grid/recovery engine
adaptive sizing overlay
swap overlay
```

Missing or needs hardening:

```text
spread model separate from slippage
market-specific contract/tick value handling
partial fills or fill failure assumptions
session filters
news/holiday filters
per-symbol execution presets
```

### 3. Strategy Research

Must support:

```text
single-strategy backtest
multi-strategy portfolio backtest
strategy validation
signal count inspection
entry/exit diagnostics
MQL5 equivalence comparison
```

Known gap:

```text
FBB/MS close-order logic still needs MQL5-equivalent exits
```

### 4. Optimization

Primary criterion:

```text
profit factor high
max drawdown low
```

Required scoring logic:

```text
reject tiny-trade results
penalize large drawdown
penalize unstable equity
prefer profit factor above 1.3-1.5
prefer enough trades to trust the result
```

Missing or needs hardening:

```text
save best parameter profiles from the app
compare top trials side by side
stress test best trial with worse slippage/spread
export optimization report
```

### 5. Walk-Forward Analysis

Must support:

```text
train window
out-of-sample test window
rolling windows
parameter stability report
in-sample vs out-of-sample score comparison
```

Definition of good:

```text
OOS remains positive
OOS drawdown remains acceptable
best parameters do not change wildly between windows
profit factor survives outside training
```

### 6. Reporting

Must show:

```text
final equity
net profit
profit factor
max drawdown
win rate
trade count
expectancy
Sharpe/Sortino/Calmar
trades per year
equity curve
drawdown curve
trade log
per-strategy scoreboard
```

Missing or needs hardening:

```text
one-page final verdict
red/yellow/green readiness status
why a strategy passed or failed
export PDF/HTML report
save research runs with timestamp and config
```

## Finalization Checklist

### Must Finish Before Calling It Production Research

```text
1. Add trader-facing help text for every Grid/Data/Risk field.
2. Add market presets: Forex, XAUUSD, Crypto.
3. Split spread from slippage.
4. Add symbol metadata: pip size, contract size, tick value.
5. Add drawdown + profit-factor optimization preset.
6. Add minimum-trade penalty to optimization.
7. Add walk-forward pass/fail verdict.
8. Add stress test: worse spread/slippage/swap.
9. Implement FBB/MS exit logic if those strategies are part of final EA research.
10. Add MQL5 comparison report before exporting any script.
```

### After Research Passes

```text
1. Freeze best parameters.
2. Generate or update the matching MQL5 EA script.
3. Compile in MetaEditor.
4. Run MT5 Strategy Tester.
5. Compare MT5 vs Python results.
6. Only then mark the strategy ready for demo/live testing.
```

## Operating Rule

Do not mix product names.

```text
Backtester = Newmeta Research Lab - Backtester
Multi Strat EA = MQL5 12-strategy EA
9 Crypto Strategy EA = separate MQL5 EA
```

The backtester is the research engine. The EAs are execution products.

## Market Benchmark Target

The goal is not to clone one platform. The goal is to combine the best parts that matter for Newmeta Research Lab - Backtester.

### QuantConnect Target: Overall Research King

Bring into Newmeta:

```text
multi-asset research structure
parameter optimization
walk-forward validation
portfolio and multi-strategy analysis
large dataset discipline
fee/slippage realism
institutional-style reports
```

Newmeta implementation priority:

```text
1. Research run database
2. Parameter sweep/Optuna studies
3. Walk-forward scorecards
4. Multi-symbol/multi-asset support
5. Portfolio comparison reports
```

### MT5 Strategy Tester Target: Gold / Forex King

Bring into Newmeta:

```text
MQL5-first workflow
XAUUSD and forex realism
spread/commission/swap settings
broker-like contract sizes
EA comparison against Strategy Tester
fast transition from research to MQL5 script
```

Newmeta implementation priority:

```text
1. Symbol profiles for EURUSD, GBPUSD, USDJPY, XAUUSD
2. Tick value / pip size / contract size model
3. Spread separated from slippage
4. MT5 export and comparison report
5. MQL5 compile/test checklist
```

### NinjaTrader Target: Futures Professional Layer

Bring into Newmeta later:

```text
GC gold futures support
order-flow style assumptions
session replay mindset
futures contract metadata
```

Newmeta implementation priority:

```text
Later phase, after spot gold/XAUUSD and crypto are strong.
```

### TradingView Target: Fast Visual Strategy Lab

Bring into Newmeta:

```text
fast idea validation
Pine strategy import path
clear visual result inspection
crypto and gold charting workflow
```

Newmeta implementation priority:

```text
1. Pine/Python/MQL5 upload stays supported
2. Strategy detection stays simple
3. Results shown quickly before deeper research
4. Export clean reports after validation
```

## Feature Scorecard

### Most Realistic Simulations

Current:

```text
grid/recovery engine
commission pips
slippage pips
swap overlay
adaptive sizing overlay
```

Needed:

```text
spread model
market presets
symbol metadata
session/holiday filters
stress test execution model
MT5-vs-Python comparison
```

### Institutional-Grade Research

Current:

```text
single backtest
multi-strategy mode
Monte Carlo mode
walk-forward mode
validation mode
optimization mode
```

Needed:

```text
saved research runs
experiment comparison table
pass/fail scorecards
parameter stability report
HTML/PDF report export
```

### Crypto + Gold / XAUUSD Support

Current:

```text
symbol input is generic
cached data path supports parquet/CSV
strategy layer can test crypto_9 and MQL5-style strategies
```

Needed:

```text
Crypto market preset
XAUUSD market preset
contract/tick/pip rules per symbol
crypto fee model percent-based
gold spread/commission presets
```

### Advanced Optimization

Current:

```text
Optuna optimization exists
composite criteria exists
crypto_9 drawdown/profit-factor script exists
```

Needed:

```text
built-in PF/DD criterion in app
minimum trade-count penalty
overfit penalty
top-N trials table
save best params to profile
stress test best params automatically
```

### Walk-Forward Analysis

Current:

```text
walk-forward mode exists
train/test windows exist
summary exists
```

Needed:

```text
OOS pass/fail verdict
OOS profit factor
OOS max drawdown
parameter drift report
walk-forward heatmap/table
```

### Slippage / Commission Modeling

Current:

```text
commission_pips
slippage_pips
swap pips/day
```

Needed:

```text
spread separate from slippage
fixed commission and percent commission modes
crypto taker/maker fees
XAUUSD broker settings
worse-case stress presets
```

## Build Order

### Phase 1: Execution Realism

```text
market presets
symbol metadata
spread model
commission model
slippage model
swap model cleanup
```

### Phase 2: Research Quality

```text
PF/DD optimizer preset
minimum-trade filter
top trial comparison
walk-forward verdict
stress test panel
```

### Phase 3: Production Bridge

```text
MQL5 comparison reports
MQL5 export/compile workflow
EA parameter profile files
final readiness report
```

### Phase 4: Newmeta Hub Readiness

```text
saved research runs
searchable experiment history
report export
multi-user/project structure
hub integration API
```
