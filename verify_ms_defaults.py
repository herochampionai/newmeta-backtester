"""Verify MS default-param behavior after confluence/type_2 default fixes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.cache import load as load_cache
from strategies.ms import MS_Strategy
from strategies.fbb import FBB_Strategy
from strategies.adx import ADX_Strategy

df = load_cache("EURUSD", "H1")
print(f"Data: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

# MS with NO params (pure defaults) — should now use confluence=True + type_2=22
ms = MS_Strategy(params={})
sig = ms.generate(df)
n = (sig.entries).sum()
print(f"\nMS default params (confluence=True, type_2=22): {n} entries")

# MS with confluence explicitly OFF (old behavior) for comparison
ms_off = MS_Strategy(params={"use_confluence_filter": False, "open_orders_type_2": 0})
sig_off = ms_off.generate(df)
print(f"MS confluence=False, type_2=0 (old smoke config): {(sig_off.entries).sum()} entries")

# MS with confluence ON but type_2=0 (isolate confluence effect)
ms_c = MS_Strategy(params={"use_confluence_filter": True, "open_orders_type_2": 0})
sig_c = ms_c.generate(df)
print(f"MS confluence=True, type_2=0: {(sig_c.entries).sum()} entries")

# FBB default
fbb = FBB_Strategy(params={})
sig_f = fbb.generate(df)
print(f"\nFBB default params (ot1=1, ot2=8): {(sig_f.entries).sum()} entries")

# ADX default
adx = ADX_Strategy(params={})
sig_a = adx.generate(df)
print(f"ADX default params: {(sig_a.entries).sum()} entries")
