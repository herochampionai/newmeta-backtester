"""Debug grid recovery."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from backtester.grid_recovery import GridRecoveryManager, GRID_NONE

np.random.seed(42)
n = 2000
idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
price = 1.10 * np.exp(np.cumsum(np.random.normal(0, 0.0005, n)))
high = price * (1 + np.abs(np.random.normal(0, 0.0007, n)))
low = price * (1 - np.abs(np.random.normal(0, 0.0007, n)))
opn = np.roll(price, 1); opn[0] = price[0]
df = pd.DataFrame({"open": opn, "high": high, "low": low, "close": price, "volume": 1000}, index=idx)

entries = pd.Series(False, index=df.index); entries.iloc[::100] = True
direction = pd.Series(0, index=df.index); direction.iloc[::100] = 1

mgr = GridRecoveryManager(grid_mode=GRID_NONE, base_lot=0.1, grid_take_profit=50.0, grid_stop_loss=200.0)
print("Trace first 200 bars:")
st = mgr._state("fbb")
for i in range(200):
    bar = df.iloc[i]
    sig_dir = int(direction.iloc[i]) if entries.iloc[i] else 0
    closed = mgr.on_bar_close("fbb", sig_dir, float(bar["high"]), float(bar["low"]),
                              float(bar["close"]), i, 0.7, 0.3)
    if closed:
        print(f"  Bar {i}: CLOSED  reason={closed[0].reason}  pnl=${closed[0].pnl:.2f}  layers={closed[0].layers}")
    if st.open_layers:
        pnl = mgr._current_grid_pnl(st.open_layers, float(bar["close"]))
        if i < 30 or abs(pnl) > 40 or i % 50 == 0:
            print(f"  Bar {i}: open@{st.open_layers[0].entry_price:.5f} bar.close={bar['close']:.5f} pnl=${pnl:.2f}")
print(f"\nFinal: {len(st.closed_trades)} closed, {len(st.open_layers)} still open")