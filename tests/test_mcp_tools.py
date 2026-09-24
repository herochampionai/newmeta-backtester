import sys
import json
sys.path.insert(0, '.')
import mcp_server as ms

print('backtest attr type:', type(ms.backtest).__name__)
fn = ms.backtest.fn if hasattr(ms.backtest, 'fn') else ms.backtest
r = fn(symbol='EURUSD', timeframe='H1', strategy='adx',
       params_json='{"bars_calculate": 10}', capital=10000.0, source='yahoo')
print('PnL:', r['net_pnl'], 'Sharpe:', r['sharpe'], 'trades:', r['trades'])
json.dumps(r)
print('JSON-serializable OK')

dqfn = ms.data_quality.fn if hasattr(ms.data_quality, 'fn') else ms.data_quality
d = dqfn(symbol='EURUSD', timeframe='H1', source='yahoo')
print('DQ grade:', d['grade'], 'score:', d['score'])
json.dumps(d)

sigfn = ms.significance.fn if hasattr(ms.significance, 'fn') else ms.significance
s = sigfn(symbol='EURUSD', timeframe='H1', strategy='adx',
          params_json='{"bars_calculate": 10}', capital=10000.0,
          n_trials=5, source='yahoo')
print('PSR:', s.get('psr'), s.get('psr_verdict'), '| DSR:', s.get('dsr'), s.get('dsr_verdict'))
json.dumps(s)
print('MCP SELF-TEST PASS')
