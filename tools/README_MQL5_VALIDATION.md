# MQL5 ↔ Python Harness Equivalence Validation

This is the **only** way to know if our Python port matches the MQL5 logic.

## Workflow

### Step 1: Run MT5 Strategy Tester
1. Open MT5 → View → Strategy Tester (Ctrl+R)
2. Pick: `multi strat newmeta.mq5` (the EA we patched)
3. Set:
   - Symbol: `EURUSD`
   - Timeframe: `H1`
   - Date: `2022.01.01` → `2024.12.31`
   - Modeling: `Every tick based on real ticks` (most accurate)
   - Deposit: `10000`
4. **Note the params you used** (just one strategy at a time to start)
5. Click ▶ Start

### Step 2: Export the trade list
1. Open MetaEditor (F4)
2. Create new script: `export_tester_deals.mq5` (from `tools/`)
3. Compile
4. In MT5: File → Open Data Folder → `MQL5/Scripts/` → drop the .ex5
5. Switch back to Tester tab → select the test result → right-click → "Open Chart" OR
6. From a regular chart of the tested symbol, run the script: Navigator → Scripts → drag `export_tester_deals` onto chart
7. The script writes `tester_trades.csv` to `MQL5/Files/`

### Step 3: Run the comparison
```powershell
cd "D:\Trading\TRADING\youha created EA\Multi strat ea\backtest_harness"
$env:PYTHONPATH = (Get-Location).Path

python -m analysis.mql5_compare `
    --mt5-trades "C:\Users\youha\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-4000\MQL5\Files\tester_trades.csv" `
    --strategy fbb `
    --data EURUSD_H1
```

### Step 4: Interpret output
```
mt5_count:       600
py_count:        614
matched:         547
match_pct:       91.2  ← ≥80% is PASS
mt5_total_pnl:   $2500
py_total_pnl:    $2580
pnl_drift_pct:   3.2%  ← <5% is PASS
verdict:         PASS
```

| match_pct | pnl_drift | verdict | meaning |
|---|---|---|---|
| ≥80% | <5% | PASS | Port is faithful. Backtest results are trustworthy. |
| 50-80% | <15% | INVESTIGATE | Small drift. Check for missing indicators / off-by-one. |
| <50% | any | FAIL | Port is broken. Do not trust backtest. |

## Why this matters

The MQL5 EA has 6 strategies × 8-28 cases each = potentially hundreds of logic branches.
Even after our 1:1 port, subtle bugs can creep in. This tool catches them.

## What to do if FAIL

1. Compare first 10 trades side by side
2. Look for: wrong direction (sign flip), wrong entry bar (off-by-one), missing signal (case not implemented), wrong level (×40000 scaling bug like MFI case-3)
3. Fix in `strategies/*.py`
4. Re-run comparison

## Limitations

- **Grid mode mismatch**: The Python harness simulates grid/recovery; MT5 tester uses the EA's grid. If you want pure signal comparison, set grid_mode=GRID_NONE in the comparison (already done).
- **Swap differences**: Python computes swap from swap_pips/day; MT5 uses broker-specific swap rates. Run with swap_enabled=False for cleanest comparison.
- **Slippage**: MQL5 tester has 0 slippage by default; Python harness adds configurable slippage. Set both to 0 for comparison.
- **Magic number**: The EA uses 333777. The exporter filters by magic number to ignore manual trades.

## Per-strategy comparison matrix

Run comparison for each strategy separately (set others to disabled in EA inputs):

| Strategy | Indicators | Cases | EA open_orders_type | Python class |
|---|---|---|---|---|
| AC+AO | iAC, iAO | 1-8 | AC_OpenOrdersType | `ac_ao` |
| ADX | iADX | 1-4 | ADX_OpenOrdersType | `adx` |
| DeM | iDeMarker | 1-4 | DeM_OpenOrdersType | `dem` |
| FBB | iBands, iForce | 1+1-8 | FBB_OpenOrdersType_1+2 | `fbb` |
| MFI | iMFI | 1-4 | MFI_OpenOrdersType | `mfi` |
| MS | iMACD, iStochastic | 1-11+1-28 | MS_OpenOrdersType_1+2 | `ms` |