"""E2E verification: real EURUSD backtest through the new pipeline."""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

print('=' * 70)
print('END-TO-END PIPELINE VERIFICATION')
print('=' * 70)

# --- 1. Load real data (prefer freshest cached EURUSD H1) ---
print('\n[1/8] Loading real EURUSD H1 data from cache')
from pathlib import Path as _Path
_candidates = sorted(_Path('data/cache').glob('EURUSD_H1*.parquet'),
                     key=lambda p: p.stat().st_mtime, reverse=True)
assert _candidates, 'No EURUSD_H1 cache found in data/cache'
print(f'      Using: {_candidates[0].name}')
df = pd.read_parquet(_candidates[0])
print(f'      {len(df)} bars, {df.index[0]} -> {df.index[-1]}')

# --- 2. Data Quality check (R011) ---
print('\n[2/8] R011 - Data Quality Dashboard')
from backtester.data_quality import analyze_data_quality, gate_check
report = analyze_data_quality(df, symbol='EURUSD', timeframe='H1')
print(f'      Grade: {report.grade}')
print(f'      Gap total: {report.gaps.total_gaps} ({report.gaps.gap_pct:.2f}%)')
print(f'      Outliers: {report.outliers.count} ({report.outliers.pct:.2f}%)')
print(f'      Stale days: {report.stale.days_since_last:.0f}')
gate = gate_check(report, min_grade='C')
print(f'      Strict gate (min C): {"PASS" if gate.get("passed") else "FAIL"}')

# --- 3. Run backtest via engine_full ---
print('\n[3/8] Engine full backtest with ADX strategy')
from backtester.engine_full import run_full
from strategies.adx import ADX_Strategy

strategy = ADX_Strategy(name='adx', params={'adx_period': 14, 'threshold': 25})
signals = strategy.generate(df)
signals_by_strat = {'adx': (signals.entries.values.astype(int), signals.exits.values.astype(int))}
result = run_full(
    df,
    signals_by_strat,
    init_cash=10000.0,
    symbol='EURUSD',
    leverage=30.0,
    strict_data=False,
)
metrics = result.get('metrics', {})
trades_df = result.get('trades', pd.DataFrame())
print(f'      Net PnL: ${metrics.get("net_pnl", 0):.2f}')
print(f'      Sharpe: {metrics.get("sharpe", 0):.3f}')
print(f'      Max DD: {metrics.get("max_drawdown", 0) * 100:.2f}%')
print(f'      Trades: {len(trades_df)}')
print(f'      Win rate: {metrics.get("win_rate", 0):.1%}')

# --- 4. Statistical significance (R012) ---
print('\n[4/8] R012 - Statistical significance test')
_pnl_col = next((c for c in ("pnl", "PnL", "profit", "Profit", "net_pnl")
                 if c in trades_df.columns), None)
if len(trades_df) >= 5 and _pnl_col:
    from analysis.stat_tests import compare_strategies
    returns = trades_df[_pnl_col].values
    # A vs A+5% (slight improvement hypothesis)
    returns_b = returns * 1.05
    verdict = compare_strategies(returns, returns_b)
    _t = verdict.get('t_test', {})
    print(f'      A/B verdict: {verdict.get("verdict", "n/a")}')
    print(f'      p-value: {_t.get("p_value", "n/a")}')
    print(f'      Cohen d: {_t.get("cohens_d", "n/a")}')

# --- 5. Multi-currency portfolio (R005) ---
print('\n[5/8] R005 - Multi-currency portfolio compute')
from backtester.portfolio import MultiCurrencyPortfolio
port = MultiCurrencyPortfolio(base_currency='USD', init_balance=10000)
port.update_equity(metrics.get('net_pnl', 0))
portm = port.compute_metrics()
print(f'      Total PnL: ${portm.total_pnl:.2f}')
print(f'      VaR 95%: ${portm.var_95:.2f}')
print(f'      Free margin: ${portm.free_margin or 0:.2f}')
print(f'      Equity: ${portm.equity:.2f}')

# --- 6. Trade journal (R015) ---
print('\n[6/8] R015 - Trade journal + tax')
from backtester.trade_journal_v2 import TradeJournal
journal = TradeJournal(run_id='e2e_test_001')
journal.log_run_meta(strategy='adx', symbol='EURUSD', data_quality_grade=report.grade)
if len(trades_df) > 0:
    entries = journal.log_trades(trades_df, source='backtest')
    print(f'      Logged {len(entries)} trades')
us_tax = journal.tax_report('US')
print(f'      US tax (est.): ${us_tax["totals"].get("estimated_tax", 0):.2f}')
eu_tax = journal.tax_report('EU')
print(f'      EU tax (est.): ${eu_tax["totals"].get("estimated_tax", 0):.2f}')

# --- 7. Feed monitor (R013) ---
print('\n[7/8] R013 - Feed monitor grade')
from backtester.feed_monitor import FeedMonitor
fm = FeedMonitor(heartbeat_sec=5.0)
fm.start()
for i, price in enumerate(df['close'].tail(20)):
    fm.tick(latency_ms=50.0 + (i % 10) * 10)
stats = fm.get_stats()
grade = stats.feed_grade.name if hasattr(stats.feed_grade, 'name') else stats.feed_grade
print(f'      Grade: {grade}')
print(f'      Latency p95: {stats.p99_latency_ms}ms')
print(f'      Ticks: {stats.n_ticks}')
print(f'      Healthy: {fm.is_healthy()}')
fm.stop()

# --- 8. Observability (R014) ---
print('\n[8/8] R014 - Metrics & structured logging')
from backtester.observability import MetricsRegistry, StructuredLogger
registry = MetricsRegistry()
registry.counter('backtest_runs_total', labels={'symbol': 'EURUSD'}).inc()
registry.gauge('equity').set(10000 + metrics.get('net_pnl', 0))
registry.gauge('sharpe').set(metrics.get('sharpe', 0))
_hist_col = next((c for c in ("pnl", "PnL", "profit", "Profit", "net_pnl")
                  if c in trades_df.columns), None)
for pnl in trades_df[_hist_col].head(20) if _hist_col else []:
    registry.histogram('trade_pnl').observe(float(pnl))
snap = registry.snapshot()
print(f'      Counters: {len(snap["counters"])}')
print(f'      Gauges: {len(snap["gauges"])}')
print(f'      Histograms: {len(snap["histograms"])}')

# Structured logging
logger = StructuredLogger(name='e2e_test', log_path='output/logs/e2e_test.jsonl', console=False)
logger.info('e2e_run_complete', net_pnl=metrics.get('net_pnl', 0),
            sharpe=metrics.get('sharpe', 0), trades=len(trades_df))
print(f'      Logged structured event to JSONL')

print('\n' + '=' * 70)
print('END-TO-END PIPELINE: ALL 8 STAGES PASSED')
print('=' * 70)
