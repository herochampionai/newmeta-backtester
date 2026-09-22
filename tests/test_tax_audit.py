"""Tax calculation audit: covers positive/negative PnL, mixed, long/short term, all jurisdictions."""
import sys
sys.path.insert(0, '.')

import pandas as pd
from backtester.trade_journal_v2 import TradeJournal

print('=' * 75)
print('TAX CALCULATION AUDIT')
print('=' * 75)


def make_trades(pnls, hold_bars_list=None):
    """Build a TradeJournal with synthetic trades."""
    if hold_bars_list is None:
        hold_bars_list = [10] * len(pnls)  # default: all short-term
    trades_df = pd.DataFrame({
        'symbol': ['EURUSD'] * len(pnls),
        'direction': [1] * len(pnls),
        'entry_price': [1.1] * len(pnls),
        'exit_price': [1.1001] * len(pnls),
        'lots': [0.1] * len(pnls),
        'pnl': pnls,
        'pnl_pct': [0.001] * len(pnls),
        'commission': [0.7] * len(pnls),
        'swap': [0] * len(pnls),
        'entry_bar': list(range(len(pnls))),
        'exit_bar': [b + 10 for b in range(len(pnls))],
        'timestamp': [f'2026-01-{i+1:02d}' for i in range(len(pnls))],
    })
    hold_bars_per_entry = hold_bars_list
    j = TradeJournal(run_id='audit')
    j.log_trades(trades_df)
    return j, hold_bars_per_entry


# ---------- TEST 1: All wins (positive PnL) ----------
print('\n[T1] All wins: $100 gain × 10 trades')
j, holds = make_trades([100] * 10)
us = j.tax_report('US')
print(f'  gross_pnl={us["totals"]["gross_pnl"]}, tax={us["totals"]["estimated_tax"]}, '
      f'net_after_tax={us["totals"]["net_after_tax"]}')
assert us["totals"]["gross_pnl"] == 1000.0, f'expected 1000.0 got {us["totals"]["gross_pnl"]}'
assert us["totals"]["estimated_tax"] == round(1000 * 0.37, 2), f'expected {1000*0.37} got {us["totals"]["estimated_tax"]}'
print(f'  [PASS] tax = 37% of gains = $370')

# ---------- TEST 2: All losses (negative PnL) ----------
print('\n[T2] All losses: -$50 × 10 trades')
j, holds = make_trades([-50] * 10)
us = j.tax_report('US')
print(f'  gross_pnl={us["totals"]["gross_pnl"]}, tax={us["totals"]["estimated_tax"]}, '
      f'net_after_tax={us["totals"]["net_after_tax"]}')
assert us["totals"]["gross_pnl"] == -500.0
assert us["totals"]["estimated_tax"] == 0.0, f'no tax on losses, got {us["totals"]["estimated_tax"]}'
print(f'  [PASS] tax = 0 on losses')

# ---------- TEST 3: Mixed PnL (5 wins, 5 losses) ----------
print('\n[T3] Mixed: +$100 × 5 wins, -$50 × 5 losses = $250 net')
j, holds = make_trades([100, 100, 100, 100, 100, -50, -50, -50, -50, -50])
us = j.tax_report('US')
print(f'  gross_pnl={us["totals"]["gross_pnl"]}, tax={us["totals"]["estimated_tax"]}')
assert us["totals"]["gross_pnl"] == 250.0
# Tax is on gross positive trades that fall in the short-term bucket
# All trades are short-term; short_pnl = sum of ALL pnl = 250 (losses included)
# So short_tax = 250 * 0.37 = $92.50
assert us["totals"]["estimated_tax"] == round(250 * 0.37, 2)
print(f'  [PASS] tax = 37% on net PnL $250 = $92.50')

# ---------- TEST 4: Long-term only (hold_bars >= 365 days equivalent) ----------
print('\n[T4] Long-term only: +$100 × 10 trades with 500-day holds')
j, _ = make_trades([100] * 10)
us = j.tax_report('US', trade_hold_days=[500] * 10)
print(f'  short_term_pnl={us["totals"]["short_term_pnl"]}, long_term_pnl={us["totals"]["long_term_pnl"]}, '
      f'tax={us["totals"]["estimated_tax"]}')
assert us["totals"]["short_term_pnl"] == 0
assert us["totals"]["long_term_pnl"] == 1000
# long_tax = 1000 * 0.20 = $200
assert us["totals"]["estimated_tax"] == 200.0
print(f'  [PASS] long-term tax = 20% = $200')

# ---------- TEST 5: Mixed short/long term ----------
print('\n[T5] Mixed: 5 short-term @ +$100, 5 long-term @ +$100')
j, _ = make_trades([100] * 10)
us = j.tax_report('US', trade_hold_days=[10, 10, 10, 10, 10, 500, 500, 500, 500, 500])
print(f'  short_pnl={us["totals"]["short_term_pnl"]}, long_pnl={us["totals"]["long_term_pnl"]}, '
      f'tax={us["totals"]["estimated_tax"]}')
# Short-term gains: 500, taxed at 37% = $185
# Long-term gains: 500, taxed at 20% = $100
# Total: $285
assert us["totals"]["short_term_pnl"] == 500
assert us["totals"]["long_term_pnl"] == 500
assert us["totals"]["estimated_tax"] == round(500 * 0.37 + 500 * 0.20, 2)
print(f'  [PASS] mixed-term tax = $285')

# ---------- TEST 6: EU flat rate ----------
print('\n[T6] EU: +$100 × 10')
j, _ = make_trades([100] * 10)
eu = j.tax_report('EU')
print(f'  gross_pnl={eu["totals"]["gross_pnl"]}, tax={eu["totals"]["estimated_tax"]}, rate={eu["totals"]["flat_rate"]}')
assert eu["totals"]["estimated_tax"] == round(1000 * 0.25, 2)
print(f'  [PASS] EU tax = 25% = $250')

# ---------- TEST 7: EU with losses ----------
print('\n[T7] EU: -$50 × 10 (all losses)')
j, _ = make_trades([-50] * 10)
eu = j.tax_report('EU')
print(f'  gross_pnl={eu["totals"]["gross_pnl"]}, tax={eu["totals"]["estimated_tax"]}')
assert eu["totals"]["estimated_tax"] == 0
print(f'  [PASS] EU tax = 0 on losses')

# ---------- TEST 8: UK ----------
print('\n[T8] UK: +$100 × 10')
j, _ = make_trades([100] * 10)
uk = j.tax_report('UK')
print(f'  tax={uk["totals"]["estimated_tax"]}, rate={uk["totals"]["flat_rate"]}')
assert uk["totals"]["estimated_tax"] == round(1000 * 0.20, 2)
print(f'  [PASS] UK tax = 20% = $200')

# ---------- TEST 9: AE (no tax) ----------
print('\n[T9] AE: +$100 × 10')
j, _ = make_trades([100] * 10)
ae = j.tax_report('AE')
print(f'  tax={ae["totals"]["estimated_tax"]}, rate={ae["totals"]["flat_rate"]}')
assert ae["totals"]["estimated_tax"] == 0
print(f'  [PASS] AE tax = 0')

# ---------- TEST 10: Empty journal ----------
print('\n[T10] Empty journal')
j = TradeJournal(run_id='empty')
us = j.tax_report('US')
assert us["totals"] == {}
print(f'  [PASS] empty returns empty totals')

# ---------- TEST 11: Unknown jurisdiction ----------
print('\n[T11] Unknown jurisdiction')
j, _ = make_trades([100])
result = j.tax_report('XX')
assert 'error' in result
print(f'  [PASS] returns error dict')

print('\n' + '=' * 75)
print('ALL TAX TESTS PASSED')
print('=' * 75)
