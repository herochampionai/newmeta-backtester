"""Final patch: footer text (line 4255). Match on prefix that decodes cleanly."""
from pathlib import Path

SRC = Path(r"D:\Trading\TRADING\youha created EA\Multi strat ea\multi strat newmeta.mq5")
raw = SRC.read_bytes()
text = raw.decode("utf-16", errors="replace")
original = text

# Find the line via clean prefix
PREFIX = 'DisplayText("Com1",StringSubstr(ExpertName,0,ExpertNameLen)+" - 2023 by Pannik'
NEW_LINE = '      DisplayText("Com1",StringSubstr(ExpertName,0,ExpertNameLen)+" - 2025 by NewMeta r",11,"Arial Black",ColorExpertName,10,16);'

lines = text.splitlines(keepends=True)
found = False
for i, line in enumerate(lines):
    if PREFIX in line:
        # Replace everything between the opening quote and `r",11,...`
        # We need to handle the corrupted character. Just substitute the year string.
        # The line structure is: DisplayText("Com1",StringSubstr(ExpertName,0,ExpertNameLen)+" - YYYY by NAME r",11,...)
        # Replace the ` - 2023 by Pannik <char>` substring with ` - 2025 by NewMeta r`
        old_text = line
        # Find the YYYY by NAME pattern
        import re
        new_line = re.sub(r'" - \d{4} by [A-Za-z]+\s*\S?', '" - 2025 by NewMeta r', old_text, count=1)
        if new_line != old_text:
            lines[i] = new_line
            found = True
            print(f"[ok] replaced line {i}")
            print(f"  old: {repr(old_text[:120])}")
            print(f"  new: {repr(new_line[:120])}")
        else:
            print(f"[MISS] regex didn't substitute")
        break

if found:
    text = "".join(lines)
    new_raw = text.encode("utf-16")
    SRC.write_bytes(new_raw)
    print(f"\n[write] {SRC}  ({len(original)} -> {len(text)} chars)")
else:
    print("[FAIL] could not find footer line")
    import sys; sys.exit(1)