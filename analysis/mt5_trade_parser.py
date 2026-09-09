"""Parse MT5 strategy tester trade export (CSV) and pair entry/exit deals into round-trip trades.
Output format matches what the Python harness produces."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np


def parse_mt5_deals_csv(csv_path: str | Path) -> pd.DataFrame:
    """Parse the deals CSV produced by `tools/export_tester_deals.mq5`.
    Returns a DataFrame of round-trip trades with:
      - entry_time, exit_time
      - direction (+1 long / -1 short)
      - entry_price, exit_price
      - lots
      - pnl (profit + swap + commission)
      - strategy (heuristic: based on entry_time + magic)
    """
    raw = pd.read_csv(csv_path)
    if raw.empty:
        return pd.DataFrame()
    # Convert timestamps
    raw["deal_time"] = pd.to_datetime(raw["deal_time"], errors="coerce")
    # Filter only entry+exit deals
    raw = raw[raw["deal_type"].isin([0, 1])].copy()
    # type 0 = BUY (entry for long, exit for short)
    # type 1 = SELL (exit for long, entry for short)
    # We use entry flag from exporter
    raw["is_entry"] = raw["entry"].astype(str) == "1"
    entries = raw[raw["is_entry"]].copy()
    exits = raw[~raw["is_entry"]].copy()
    trades = []
    # Group by position_id
    for pid, grp in raw.groupby("position_id"):
        grp = grp.sort_values("deal_time")
        entry_row = grp[grp["is_entry"]].iloc[0] if any(grp["is_entry"]) else None
        exit_row = grp[~grp["is_entry"]].iloc[-1] if any(~grp["is_entry"]) else None
        if entry_row is None or exit_row is None:
            continue
        # Direction: BUY entry = +1 long, SELL entry = -1 short
        direction = 1 if entry_row["deal_type"] == 0 else -1
        trades.append({
            "position_id": pid,
            "entry_time": entry_row["deal_time"],
            "exit_time": exit_row["deal_time"],
            "direction": direction,
            "entry_price": float(entry_row["price"]),
            "exit_price": float(exit_row["price"]),
            "lots": float(entry_row["volume"]),
            "profit": float(exit_row["profit"]),
            "swap": float(exit_row["swap"]),
            "commission": float(exit_row["commission"]),
            "pnl": float(exit_row["profit"]) + float(exit_row["swap"]) + float(exit_row["commission"]),
            "magic": int(entry_row["magic"]),
            "symbol": entry_row["symbol"],
        })
    return pd.DataFrame(trades)


def parse_mt5_html_report(html_path: str | Path) -> pd.DataFrame:
    """Parse MT5 strategy tester HTML report (Deals tab).
    Less reliable than CSV — CSV export is preferred."""
    from html.parser import HTMLParser
    text = Path(html_path).read_text(encoding="utf-8", errors="replace")
    # Find the Deals table rows: <tr><td>time</td><td>type</td>...
    # Use regex to grab each row
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', text, re.DOTALL)
    parsed = []
    headers = None
    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        if not cells:
            continue
        if headers is None and "Time" in cells[0] and "Type" in cells[1]:
            headers = cells
            continue
        if headers is None or len(cells) < len(headers):
            continue
        rec = dict(zip(headers, cells))
        try:
            parsed.append({
                "deal_time": pd.to_datetime(rec.get("Time", ""), errors="coerce"),
                "deal_type": 0 if "buy" in rec.get("Type", "").lower() else 1,
                "symbol": rec.get("Symbol", ""),
                "volume": float(rec.get("Volume", "0") or 0),
                "price": float(rec.get("Price", "0") or 0),
                "profit": float(rec.get("Profit", "0") or 0),
                "swap": float(rec.get("Swap", "0") or 0),
                "commission": float(rec.get("Commission", "0") or 0),
                "magic": int(rec.get("Magic", "0") or 0),
            })
        except (ValueError, KeyError):
            continue
    if not parsed:
        return pd.DataFrame()
    df = pd.DataFrame(parsed)
    # Heuristic: assume first half are entries, second half are exits (not robust)
    return df  # caller should use the CSV version when possible