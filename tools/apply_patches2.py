"""Apply remaining 3 patches using line-based replacement (handles CRLF, encoding).
Run after apply_patches.py first."""
import sys
from pathlib import Path

SRC = Path(r"D:\Trading\TRADING\youha created EA\Multi strat ea\multi strat newmeta.mq5")
# Read as UTF-16 LE (the file's encoding)
raw = SRC.read_bytes()
# Decode — first 2 bytes are BOM (FF FE) for UTF-16 LE
text = raw.decode("utf-16", errors="replace")
original = text

def replace_line_containing(text: str, needle: str, new_line: str) -> tuple[str, bool]:
    """Replace the line containing `needle` with `new_line`. Returns (text, success)."""
    lines = text.splitlines(keepends=True)
    found = False
    for i, line in enumerate(lines):
        if needle in line:
            lines[i] = new_line if new_line.endswith("\n") else new_line + "\n"
            found = True
            break
    return "".join(lines), found

PATCHES = [
    # 5. Footer author string (line 4255)
    {
        "find_in_line": ' - 2023 by Pannik r',
        "new_line": '      DisplayText("Com1",StringSubstr(ExpertName,0,ExpertNameLen)+" - 2025 by NewMeta r",11,"Arial Black",ColorExpertName,10,16);',
        "name": "footer 2023 Pannik -> 2025 NewMeta",
    },
    # 6. MFI case-3 buy polarity flip (line 5331)
    {
        "find_in_line": 'if((MFI_Value_0<LevelOpenDn)&&(MFI_Value_1>LevelOpenDn))',
        "new_line": '             if((MFI_Value_0 > LevelOpenDn) && (MFI_Value_1 < LevelOpenDn)) // <-- FIXED: Buys on UP-cross (exiting oversold)',
        "name": "MFI case-3 buy polarity flip",
    },
    # 7. MFI case-3 sell polarity flip (line 5333)
    {
        "find_in_line": 'if((MFI_Value_0>LevelOpenUp)&&(MFI_Value_1<LevelOpenUp))',
        "new_line": '             if((MFI_Value_0 < LevelOpenUp) && (MFI_Value_1 > LevelOpenUp)) // <-- FIXED: Sells on DOWN-cross (exiting overbought)',
        "name": "MFI case-3 sell polarity flip",
    },
]

applied = []
for p in PATCHES:
    text, ok = replace_line_containing(text, p["find_in_line"], p["new_line"])
    if ok:
        applied.append(p["name"])
        print(f"[ok]    {p['name']}")
    else:
        print(f"[MISS]  {p['name']}  (line not found)")
        applied.append(None)

if text != original:
    # Write back as UTF-16 LE (with BOM)
    new_raw = text.encode("utf-16")
    SRC.write_bytes(new_raw)
    print(f"\n[write] {SRC}  ({len(original)} -> {len(text)} chars)")
    print(f"applied: {sum(1 for a in applied if a)}/{len(PATCHES)}")
else:
    print("\n[no-op] file unchanged")
    sys.exit(1)