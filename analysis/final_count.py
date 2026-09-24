"""Final deployable count + cross-validation summary."""
import json

# ROBUST singles (already validated)
robust_singles = {
    'adx': ['NAS', 'EUR'],
    'linda_macd_lenient': ['EUR'],
    'sc_s5_stoch': ['NAS', 'EUR'],
    'sc_s8_bb': ['NAS'],
    'ms': ['EUR'],
}

# New ROBUST merges found in this session
merges = [
    {'name': 'sc_s8_bb + sc_s7_macd (AND-gate)', 'asset': 'EUR', 'tuning': 26, 'val': 56, 'trades': 6, 'robust': True, 'high_wr': False, 'verdict': 'ROBUST but marginal (6 trades total)'},
    {'name': 'OR adx + sc_s6 (vote-1-of-2)', 'asset': 'EUR', 'tuning': 68, 'val': 62, 'trades': 10, 'robust': True, 'high_wr': True, 'verdict': 'ROBUST + HIGH-WR (87.5%/100%) but only 10 trades'},
    {'name': 'sc_s6_macross primary + adx filter (cw=5)', 'asset': 'EUR', 'tuning': -57, 'val': 92, 'trades': 36, 'robust': False, 'high_wr': True, 'verdict': 'HIGH-WR (66.7%/73.3%) but BAD RR — LOSES on tuning'},
]

print('=' * 80)
print('FINAL DEPLOYABLE COUNT — unique strategies (no profile duplicates)')
print('=' * 80)
print()
print('ROBUST SINGLES (5 algorithms, 7 profiles):')
for i, (algo, profiles) in enumerate(robust_singles.items(), 1):
    print(f'  {i}. {algo:30s} profiles: {", ".join(profiles)}')
print(f'  → 5 unique algorithms')
print()
print('NEW POTENTIAL ROBUST MERGES (found this session):')
for i, m in enumerate(merges, 6):
    flag = '✅ ROBUST' if m['robust'] else 'X'
    wr_flag = '🔥 HI-WR' if m['high_wr'] else ''
    print(f'  {i}. {m["name"]:50s} {flag} {wr_flag}')
    print(f'     T ${m["tuning"]:+} / V ${m["val"]:+} / {m["trades"]} trades — {m["verdict"]}')
print()
print(f'TOTAL candidates: 5 singles + 3 merges = 8')
print()
print('VERDICT:')
print('  - 5 ROBUST singles: confirmed deployable')
print('  - 2 ROBUST merges: ROBUST but marginal (low trade count = statistical noise)')
print('  - 1 HIGH-WR but loses tuning (sc_s6 + adx filter): potential if RR fixed')
