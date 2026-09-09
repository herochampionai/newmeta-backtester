"""Apply 5 patches from Multi Gemini.txt to multi strat newmeta.mq5.
Patch list (excluding MS_OpenOrdersType_2 per user):
  1. Header: copyright + link + version
  2. Line 48: AC_OpenOrdersType 5 -> 1
  3. Line 51: AC_CloseOrdersType 1 -> 4
  4. Line 84: ADX_OpenOrdersType 2 -> 1
  5. Line 4255: footer author string
  6+7. Lines 5331, 5333: MFI case-3 polarity flip (the real bug fix)
"""
import shutil
import sys
from pathlib import Path

SRC = Path(r"D:\Trading\TRADING\youha created EA\Multi strat ea\multi strat newmeta.mq5")
BAK = SRC.with_suffix(".mq5.bak.prepatch")

# Backup first
if not BAK.exists():
    shutil.copy2(SRC, BAK)
    print(f"[backup] {BAK}")
else:
    print(f"[backup] {BAK} (exists, skipping)")

text = SRC.read_text(encoding="utf-16")
original = text

PATCHES = [
    # 1. Header (3 lines)
    {
        "find": '#property copyright   "Copyright 2023, Nikolaos Pantzos (Merged by Gemini)"\n'
                '#property link        "https://www.mql5.com/en/users/pannik"\n'
                '#property version     "1.40"',
        "replace": '#property copyright   "Copyright 2025, NewMeta (Gemini 1.41 patch)"\n'
                   '#property link        "Charts.newmeta.ai"\n'
                   '#property version     "1.41"',
        "name": "header (copyright/link/version)",
    },
    # 2. Line 48: AC_OpenOrdersType
    {
        "find": "input int    AC_OpenOrdersType      = 5;//AC+AO Type Of Open Orders (0=Not Use)(from 1 to 8)",
        "replace": "input int    AC_OpenOrdersType      = 1;//AC+AO Type Of Open Orders (0=Not Use)(from 1 to 8)",
        "name": "AC_OpenOrdersType 5 -> 1",
    },
    # 3. Line 51: AC_CloseOrdersType
    {
        "find": "input int    AC_CloseOrdersType     = 1;//AC+AO Type Of Close Orders (0=Not Use)(from 1 to 8)",
        "replace": "input int    AC_CloseOrdersType     = 4;//AC+AO Type Of Close Orders (0=Not Use)(from 1 to 8)",
        "name": "AC_CloseOrdersType 1 -> 4",
    },
    # 4. Line 84: ADX_OpenOrdersType
    {
        "find": "input int    ADX_OpenOrdersType     = 2;//ADX Type Of Open Orders (0=Not Use)(from 1 to 4)",
        "replace": "input int    ADX_OpenOrdersType     = 1;//ADX Type Of Open Orders (0=Not Use)(from 1 to 4)",
        "name": "ADX_OpenOrdersType 2 -> 1",
    },
    # 5. Line 4255: footer author string in DisplayText Com1
    {
        "find": "Multipanel_EA - Ver 1.40  - 2023 by Pannik r",
        "replace": "Multipanel_EA - Ver 1.41  - 2025 by NewMeta r",
        "name": "footer author 2023 Pannik -> 2025 NewMeta",
    },
    # 6. Line 5331: MFI case-3 buy polarity flip
    {
        "find": "             if((MFI_Value_0<LevelOpenDn)&&(MFI_Value_1>LevelOpenDn))\n"
                "                MFI_OpenBuy_1=true;",
        "replace": "             if((MFI_Value_0 > LevelOpenDn) && (MFI_Value_1 < LevelOpenDn)) // <-- FIXED: Buys on UP-cross (exiting oversold)\n"
                   "                MFI_OpenBuy_1=true;",
        "name": "MFI case-3 buy polarity flip",
    },
    # 7. Line 5333: MFI case-3 sell polarity flip
    {
        "find": "             if((MFI_Value_0>LevelOpenUp)&&(MFI_Value_1<LevelOpenUp))\n"
                "                MFI_OpenSell_1=true;",
        "replace": "             if((MFI_Value_0 < LevelOpenUp) && (MFI_Value_1 > LevelOpenUp)) // <-- FIXED: Sells on DOWN-cross (exiting overbought)\n"
                   "                MFI_OpenSell_1=true;",
        "name": "MFI case-3 sell polarity flip",
    },
]

applied = []
for p in PATCHES:
    if p["find"] in text:
        text = text.replace(p["find"], p["replace"], 1)
        applied.append(p["name"])
        print(f"[ok]    {p['name']}")
    else:
        print(f"[MISS]  {p['name']}  (find string not found)")
        # Show a snippet for debugging
        # Try fuzzy match: first 60 chars
        snippet = p["find"][:60]
        if snippet in text:
            print(f"        note: only prefix matched — check whitespace")
        applied.append(None)

if text != original:
    SRC.write_text(text, encoding="utf-16")
    print(f"\n[write] {SRC}  ({len(original)} -> {len(text)} chars)")
    print(f"applied: {sum(1 for a in applied if a)}/{len(PATCHES)}")
else:
    print("\n[no-op] file unchanged")
    sys.exit(1)