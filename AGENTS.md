# AGENTS.md — Backtest Harness Quality Standards

## Project Overview
QuantConnect-grade research lab for the Multi-strategy MT5 EA. Python harness
drives optimization, walk-forward analysis, and surgical feature validation.
MT5 is the live-execution skin; Python is the research engine.

**PYTHONPATH**: Must be set to the backtest harness root for all test/lint runs.

```powershell
$env:PYTHONPATH = "D:\Trading\TRADING\youha created EA\Multi strat ea\backtest_harness"
python -m pytest tests/ -v
```

## Surgical Feature Contract (Roadmap B4)
- All 6 features (`anomaly_gate`, `event_blackout`, `recovery_restart`,
  `basket_money_tp`, `profit_lock_trail`, `carry_adjusted_tp`) are
  **default-OFF**.
- Acceptance barrier: must beat baseline OOS on the walk-forward harness
  before any toggle is flipped to enabled by default.
- Feature toggles are injected into strategy `params` as dicts:
  `{"feature_name": {"enabled": True, ...params}}`.
- Use `core/surgical_features.py` helpers: `is_enabled()`,
  `get_feature_params()`, `enable()`, `disable()`, `get_feature_defaults()`.

## Code Quality Standards

### Python
- **Target**: Python 3.11+
- **Type hints**: Required on all public functions and class methods.
- **Imports**: stdlib → third-party → local (3 groups, blank line between).
- **No wildcard imports** (`from x import *`).
- **f-strings** preferred over `.format()` or `%`.
- **Docstrings**: One-line summary + optional extended description for
  classes and public functions. Include Parameters/Returns sections.
- **Comments**: Only for non-obvious logic. No commented-out code.

### Testing
- **Pytest** with class-based test organization: `TestClassName::test_method_name`.
- **Target coverage**: 90%+ for `core/` and `backtester/` modules.
- **Target files**: `tests/test_*.py` for pytest-style; `tests/smoke_*.py`
  for integration smoke tests.
- Smoke tests use print-based output (legacy convention) — do not modify
  unless adding new smoke coverage.
- **Floating-point comparisons**: Use pytest.approx or tolerance-aware
  assertions — grid PnL is computed as `(price - entry) * direction * lot *
  contract_size` and may not be exactly equal to expectations due to FP.

### Linting & Types
- **Linter**: `py_compile` for syntax (must pass on all changed files).
- **Run before any commit**: `python -m py_compile <changed_files>`.
- **Target**: `ruff` if available in environment; `py_compile` is the
  minimum bar.

### File Conventions
- `core/` — Strategy-agnostic modules: regime detection, surgical features,
  macro events, strictness profiles, strategy profiling.
- `backtester/` — Simulation engines: pure (vectorbt), grid recovery,
  adaptive, swaps, full orchestrator.
- `strategies/` — Individual strategy implementations (fbb, adx, ms, etc.).
  Each has a `BaseStrategy` subclass with a `generate(df) -> Signals` method.
  - `analysis/` — Walk-forward, optimization, Monte Carlo, scanner, research
     exports. Key scripts:
     - `walkforward.py` — basic WF loop (single strategy, `run_direction`)
     - `wf_surgical_validation.py` — paired WF: grid-only vs all-surgical-ON
     - `wf_grid_vs_pure.py` — paired WF: grid OFF vs grid ON (overlay validation)
     - `genetic_optimizer.py` — NSGA-II multi-objective optimizer (Sharpe/Calmar/DD)
     - `data_loader.py` — multi-asset cache → Yahoo → synthetic fallback
      - `multi_walkforward.py` — multi-strategy WF across all 12 strategies
      - `wf_parallel.py` — explicit strategy×symbol WF (manual mode, --strategies required). 21 strategies: 12 core + 9 crypto_9 pattern variants
      - `portfolio_wf.py` — portfolio-level WF with risk_parity/equal allocation, per-strategy contribution, bootstrap Sharpe significance
- `config/` — YAML parameter ranges, JSON macro calendar.
- `frontend/` — Streamlit UI.
- `tests/` — pytest unit tests + smoke tests.

## Key Files
- `core/surgical_features.py:51` — `SURGICAL_FEATURES` registry (ALL default-FALSE)
- `core/anomaly_gate.py:71` — `AnomalyGate` class (no-op when feature disabled)
- `core/macro_events.py:58` — `MacroEventCalendar` (config/events.json → DEFAULT_EVENTS)
- `backtester/grid_recovery.py:80` — `GridRecoveryManager` (grid TP/SL in $)
- `backtester/engine_grid.py:19` — `run_grid()` (resolves surgical params from dict)
- `backtester/engine_full.py:27` — `run_full()` (orchestrates overlays, returns surgical_features status)
- `analysis/walkforward.py:14` — `WFWindow` (tracks idle_day_count for anomaly rule)
  - `frontend/app.py` — Streamlit UI with NinjaTrader/TradingView-style viz
  - `launch_api.py:20` — Starts FastAPI HTTP API on port 8765 (serves harness to widget)

## Acceptance Bar
Any new feature toggles default-OFF until it beats baseline OOS on the walk-forward
harness. Run `wf_grid_vs_pure.py` to validate grid overlay; `wf_surgical_validation.py`
to validate all 6 surgical features simultaneously.
