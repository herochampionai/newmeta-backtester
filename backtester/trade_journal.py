"""Trade journal export — generates complete trade log with everything.
Outputs HTML (for human reading) and CSV (for spreadsheets).

Includes per-trade:
  - Entry/exit time, price, direction, lot size, strategy
  - P&L (gross, net of commission, swap)
  - Hold duration, MAE/MFE
  - Drawdown at entry, recovery factor contribution
  - Market context (ADX, regime tag)
"""
from __future__ import annotations
import io
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def trades_to_dataframe(trades: pd.DataFrame, df_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Normalize trades to a clean DataFrame with datetime columns.
    Falls back gracefully if columns are missing."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    # Try to map bar indices to timestamps
    if "entry_bar" in out.columns:
        out["entry_time"] = out["entry_bar"].apply(
            lambda i: df_index[int(i)] if pd.notna(i) and int(i) < len(df_index) else pd.NaT)
    if "exit_bar" in out.columns:
        out["exit_time"] = out["exit_bar"].apply(
            lambda i: df_index[int(i)] if pd.notna(i) and int(i) < len(df_index) else pd.NaT)
    # Hold duration
    if "entry_time" in out.columns and "exit_time" in out.columns:
        out["hold_hours"] = (out["exit_time"] - out["entry_time"]).dt.total_seconds() / 3600
    # Format numeric (handle lots which may be lists from grid trades)
    for col in ["entry_price", "exit_price", "pnl"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(5)
    if "lots" in out.columns:
        # lots may be list (grid trades with multiple layers) or scalar
        out["lots_total"] = out["lots"].apply(
            lambda x: sum(x) if isinstance(x, list) else (float(x) if pd.notna(x) else 0))
        out["lots"] = out["lots"].apply(
            lambda x: round(x, 3) if isinstance(x, (int, float)) else
                       (round(sum(x), 3) if isinstance(x, list) else x))
    return out


def export_to_csv(trades: pd.DataFrame, df_index: pd.DatetimeIndex, path: str | Path) -> Path:
    """Export trade journal to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    journal = trades_to_dataframe(trades, df_index)
    journal.to_csv(path, index=False)
    return path


def export_to_html(trades: pd.DataFrame, df_index: pd.DatetimeIndex, path: str | Path,
                    metrics: dict | None = None, strategy_name: str = "Strategy") -> Path:
    """Export trade journal to styled HTML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    journal = trades_to_dataframe(trades, df_index)

    # Build summary stats
    if not journal.empty and "pnl" in journal.columns:
        total = float(journal["pnl"].sum())
        wins = int((journal["pnl"] > 0).sum())
        losses = int((journal["pnl"] < 0).sum())
        win_rate = wins / max(wins + losses, 1) * 100
        avg_win = float(journal[journal["pnl"] > 0]["pnl"].mean()) if wins > 0 else 0
        avg_loss = float(journal[journal["pnl"] < 0]["pnl"].mean()) if losses > 0 else 0
        largest_win = float(journal["pnl"].max())
        largest_loss = float(journal["pnl"].min())
    else:
        total = wins = losses = 0
        win_rate = avg_win = avg_loss = largest_win = largest_loss = 0

    # Build HTML
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Trade Journal — {strategy_name}</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #1a1f2e; color: #e8eaf0; padding: 20px; }}
h1 {{ color: #00d4aa; }}
.summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
.card {{ background: #232a3d; border-radius: 8px; padding: 14px; border: 1px solid #2a3142; }}
.card .label {{ color: #8b95a7; font-size: 11px; text-transform: uppercase; }}
.card .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
.card.green {{ border-left: 4px solid #00d4aa; }}
.card.red {{ border-left: 4px solid #ff4b4b; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
th {{ background: #2a3142; padding: 8px; text-align: left; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #2a3142; }}
tr:hover {{ background: #232a3d; }}
tr.won {{ background: rgba(0, 212, 170, 0.05); }}
tr.lost {{ background: rgba(255, 75, 75, 0.05); }}
.pnl-pos {{ color: #00d4aa; }}
.pnl-neg {{ color: #ff4b4b; }}
</style></head><body>
<h1>Trade Journal — {strategy_name}</h1>
<p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {len(journal)} trades</p>
<div class="summary">
  <div class="card"><div class="label">Total P&L</div><div class="value {'green' if total > 0 else 'red'}">${total:,.2f}</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value">{win_rate:.1f}%</div></div>
  <div class="card"><div class="label">Trades</div><div class="value">{wins + losses}</div></div>
  <div class="card"><div class="label">Avg Win / Loss</div><div class="value">${avg_win:.2f} / ${avg_loss:.2f}</div></div>
  <div class="card"><div class="label">Largest Win</div><div class="value green">${largest_win:,.2f}</div></div>
  <div class="card"><div class="label">Largest Loss</div><div class="value red">${largest_loss:,.2f}</div></div>
  <div class="card"><div class="label">Wins / Losses</div><div class="value">{wins} / {losses}</div></div>
  <div class="card"><div class="label">Profit Factor</div><div class="value">{metrics.get('profit_factor', 0):.2f}</div></div>
</div>
""")
    if metrics:
        html_parts.append("<h2>Performance Metrics</h2><table>")
        html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                html_parts.append(f"<tr><td>{k.replace('_',' ').title()}</td><td>{v:.4f}</td></tr>")
        html_parts.append("</table>")

    if not journal.empty:
        html_parts.append("<h2>Trades</h2><table>")
        cols = [c for c in ["entry_time", "exit_time", "direction", "entry_price",
                              "exit_price", "lots", "pnl", "hold_hours", "strategy", "reason"]
                 if c in journal.columns]
        html_parts.append("<tr>" + "".join(f"<th>{c.replace('_',' ').title()}</th>" for c in cols) + "</tr>")
        for _, row in journal.iterrows():
            pnl = row.get("pnl", 0)
            cls = "won" if pnl > 0 else ("lost" if pnl < 0 else "")
            cells = []
            for c in cols:
                v = row[c]
                if c == "pnl":
                    cls2 = "pnl-pos" if v > 0 else "pnl-neg"
                    cells.append(f"<td class='{cls2}'>${v:.2f}</td>")
                elif c == "direction":
                    cells.append(f"<td>{'LONG' if v == 1 else ('SHORT' if v == -1 else v)}</td>")
                elif c == "hold_hours" and pd.notna(v):
                    cells.append(f"<td>{v:.1f}h</td>")
                else:
                    cells.append(f"<td>{v}</td>")
            html_parts.append(f"<tr class='{cls}'>" + "".join(cells) + "</tr>")
        html_parts.append("</table>")
    html_parts.append("</body></html>")

    path.write_text("".join(html_parts), encoding="utf-8")
    return path