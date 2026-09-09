"""Final re-apply: target MFI case 3 specifically by looking for the MFI context."""
import re
from pathlib import Path

SRC = Path(r"D:\Trading\TRADING\youha created EA\Multi strat ea\multi strat newmeta.mq5")
raw = SRC.read_bytes()
text = raw.decode("utf-16", errors="replace")
original = text

# Step 1: re-apply patches NOT yet applied (header + 3 defaults)
PATCHES = [
    {
        "find": '#property copyright   "Copyright 2023, Nikolaos Pantzos (Merged by Gemini)"',
        "replace": '#property copyright   "Copyright 2025, NewMeta (Gemini 1.41 patch)"',
        "name": "header copyright",
    },
    {
        "find": '#property link        "https://www.mql5.com/en/users/pannik"',
        "replace": '#property link        "Charts.newmeta.ai"',
        "name": "header link",
    },
    {
        "find": '#property version     "1.40"',
        "replace": '#property version     "1.41"',
        "name": "header version",
    },
    {
        "find": "input int    AC_OpenOrdersType      = 5;//AC+AO Type Of Open Orders",
        "replace": "input int    AC_OpenOrdersType      = 1;//AC+AO Type Of Open Orders",
        "name": "AC_OpenOrdersType 5 -> 1",
    },
    {
        "find": "input int    AC_CloseOrdersType     = 1;//AC+AO Type Of Close Orders",
        "replace": "input int    AC_CloseOrdersType     = 4;//AC+AO Type Of Close Orders",
        "name": "AC_CloseOrdersType 1 -> 4",
    },
    {
        "find": "input int    ADX_OpenOrdersType     = 2;//ADX Type Of Open Orders",
        "replace": "input int    ADX_OpenOrdersType     = 1;//ADX Type Of Open Orders",
        "name": "ADX_OpenOrdersType 2 -> 1",
    },
]

for p in PATCHES:
    if p["find"] in text:
        text = text.replace(p["find"], p["replace"], 1)
        print(f"[ok]    {p['name']}")
    else:
        print(f"[SKIP]  {p['name']} (likely already applied)")

# Step 2: Footer (regex)
lines = text.splitlines(keepends=True)
for i, line in enumerate(lines):
    if ' - 2023 by Pannik' in line:
        lines[i] = re.sub(r'" - \d{4} by [A-Za-z]+\s*\S?', '" - 2025 by NewMeta r', line, count=1)
        print(f"[ok]    footer 2023->2025 (line {i})")
        break
    elif ' - 2025 by NewMeta' in line:
        print(f"[SKIP]  footer already updated")
        break
text = "".join(lines)

# Step 3: MFI case 3 — find via MFI_GetSignals context
lines = text.splitlines(keepends=True)
mfi_start = None
mfi_end = None
for i, line in enumerate(lines):
    if "void MFI_GetSignals" in line:
        mfi_start = i
    elif mfi_start is not None and line.strip().startswith("void ") and "MFI_GetSignals" not in line:
        mfi_end = i
        break

print(f"\nMFI_GetSignals: line {mfi_start+1} to {mfi_end+1 if mfi_end else 'end'}")

# Within MFI_GetSignals, find the OPEN section's case 3 (not close section).
# The "Signals open orders" comment marks the open block.
open_section_start = None
for i in range(mfi_start, mfi_end or len(lines)):
    if "Signals open orders" in lines[i]:
        open_section_start = i
        break

assert open_section_start is not None, "Could not find MFI open section"

case3_idx = None
case4_idx = None
# Match any whitespace + "case 3:" (UTF-16 decode may strip 1 space)
for i in range(open_section_start, mfi_end or len(lines)):
    line = lines[i]
    stripped = line.strip()
    if stripped == "case 3:":
        case3_idx = i
    elif stripped == "case 4:" and case3_idx is not None:
        case4_idx = i
        break

print(f"MFI OPEN case 3 at line {case3_idx+1}, case 4 at line {case4_idx+1}")
assert case3_idx is not None, "Could not find MFI open case 3"

buy_line = lines[case3_idx + 1]
sell_line = lines[case3_idx + 3]   # skip the body line `MFI_OpenBuy_1=true;`
print(f"  buy line:  {repr(buy_line[:80])}")
print(f"  sell line: {repr(sell_line[:80])}")

# Skip if already fixed (look for FIXED comment)
if "FIXED" in buy_line:
    print("[SKIP] MFI case 3 already fixed")
else:
    assert 'MFI_Value_0<LevelOpenDn' in buy_line, f"unexpected: {buy_line}"
    assert 'MFI_Value_0>LevelOpenUp' in sell_line, f"unexpected: {sell_line}"
    lines[case3_idx + 1] = '             if((MFI_Value_0 > LevelOpenDn) && (MFI_Value_1 < LevelOpenDn)) // <-- FIXED: Buys on UP-cross (exiting oversold)\n'
    lines[case3_idx + 3] = '             if((MFI_Value_0 < LevelOpenUp) && (MFI_Value_1 > LevelOpenUp)) // <-- FIXED: Sells on DOWN-cross (exiting overbought)\n'
    print(f"[ok]    MFI case 3 buy flipped (line {case3_idx + 2})")
    print(f"[ok]    MFI case 3 sell flipped (line {case3_idx + 4})")

text = "".join(lines)
new_raw = text.encode("utf-16")
SRC.write_bytes(new_raw)
print(f"\n[write] {SRC}  ({len(original)} -> {len(text)} chars)")