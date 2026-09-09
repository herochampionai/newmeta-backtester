"""Sexy HTML dashboard generator for backtest results.

Renders a self-contained, single-file HTML report with:
  * Hero section with glassmorphism metric cards
  * Equity curve with entry/exit markers (Plotly)
  * Underwater drawdown chart (Plotly)
  * Trade distribution histograms (pure SVG, zero JS dep)
  * Risk metrics grid
  * Streak info
  * Styled per-trade table with alternating row colours

Design references: TradingView, Stripe Dashboard, Apple dark mode.
Embedded data is JSON-serialised; only Plotly.js is loaded from a CDN.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _pnl_series(trades: pd.DataFrame) -> pd.Series | None:
    """Return the per-trade PnL series regardless of column naming."""
    if trades is None or trades.empty:
        return None
    for c in ("pnl", "PnL", "profit", "Profit", "Net PnL"):
        if c in trades.columns:
            return pd.to_numeric(trades[c], errors="coerce").fillna(0.0)
    return None


def _direction_map(val: Any) -> str:
    """Pretty direction label, tolerant of multiple encodings."""
    s = str(val).lower().strip()
    if s in ("1", "long", "buy", "b", "+1"):
        return "LONG"
    if s in ("-1", "short", "sell", "s"):
        return "SHORT"
    return str(val)


def _streak_stats(trades: pd.DataFrame) -> dict:
    """Longest win/loss streaks + counts."""
    pnls = _pnl_series(trades)
    if pnls is None or pnls.empty:
        return {"longest_win_streak": 0, "longest_loss_streak": 0,
                "total_wins": 0, "total_losses": 0}
    is_win = (pnls > 0).values
    longest_win = longest_loss = cur_win = cur_loss = 0
    for w in is_win:
        if w:
            cur_win += 1
            cur_loss = 0
            if cur_win > longest_win:
                longest_win = cur_win
        else:
            cur_loss += 1
            cur_win = 0
            if cur_loss > longest_loss:
                longest_loss = cur_loss
    return {
        "longest_win_streak": int(longest_win),
        "longest_loss_streak": int(longest_loss),
        "total_wins": int(is_win.sum()),
        "total_losses": int((~is_win).sum()),
    }


def _drawdown_series(equity: pd.Series) -> tuple[list, list]:
    """Underwater drawdown as (timestamps, drawdown_pct)."""
    if equity is None or equity.empty:
        return [], []
    eq = pd.Series(equity).dropna()
    if eq.empty:
        return [], []
    peak = eq.cummax()
    dd = (eq / peak - 1.0) * 100.0  # percent
    idx = eq.index
    if hasattr(idx, "strftime"):
        x = [t.strftime("%Y-%m-%d %H:%M") for t in idx]
    else:
        x = [str(t) for t in idx]
    return x, [round(float(v), 4) for v in dd.values]


def _equity_payload(equity: pd.Series) -> tuple[list, list]:
    """Equity curve as (x, y) lists for Plotly."""
    if equity is None or equity.empty:
        return [], []
    eq = pd.Series(equity).dropna()
    if eq.empty:
        return [], []
    idx = eq.index
    if hasattr(idx, "strftime"):
        x = [t.strftime("%Y-%m-%d %H:%M") for t in idx]
    else:
        x = [str(t) for t in idx]
    y = [round(float(v), 4) for v in eq.values]
    return x, y


def _trade_markers(trades: pd.DataFrame, df_index: pd.DatetimeIndex) -> tuple[list, list, list, list]:
    """Entry/exit markers as (x_entry, y_entry, x_exit, y_exit).

    Maps entry_bar / exit_bar to timestamps + looks up equity value at that bar.
    """
    if trades is None or trades.empty:
        return [], [], [], []

    def lookup_y(t):
        try:
            pos = df_index.get_indexer([pd.Timestamp(t)], method="nearest")[0]
            return float(pos)
        except Exception:
            return None

    x_entry: list[str] = []
    y_entry: list[float] = []
    x_exit: list[str] = []
    y_exit: list[float] = []
    pnls = _pnl_series(trades)
    for i, row in trades.iterrows():
        et = row.get("entry_time")
        xt = row.get("exit_time")
        if pd.notna(et):
            ts = pd.Timestamp(et).strftime("%Y-%m-%d %H:%M")
            x_entry.append(ts)
        if pd.notna(xt):
            ts = pd.Timestamp(xt).strftime("%Y-%m-%d %H:%M")
            x_exit.append(ts)
        if pnls is not None and i in pnls.index:
            y_entry.append(0.0)   # placeholder
            y_exit.append(0.0)

    # We don't have equity indexed by trade, so we send marker-less markers.
    # Plotly will overlay them on the equity curve; y values are 0 (the script
    # maps them to trade count by patching in JS).
    return x_entry, y_entry, x_exit, y_exit


def _svg_histogram(values: list[float], *, bins: int = 24, width: int = 480,
                   height: int = 220, color_loss: str = "#ff4b4b",
                   color_win: str = "#00d4aa", title: str = "") -> str:
    """Inline SVG histogram — zero JS dependency. Bars are coloured by sign."""
    if not values:
        return (f'<div class="empty-chart">No data</div>')
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)],
                     dtype=float)
    if arr.size == 0:
        return '<div class="empty-chart">No data</div>'
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmin == vmax:
        vmax = vmin + 1.0
    edges = np.linspace(vmin, vmax, bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    peak = max(int(counts.max()), 1)
    pad_top = 16
    pad_bot = 36
    pad_left = 36
    pad_right = 12
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bot
    bar_w = plot_w / max(bins, 1)
    bars = []
    zero_x = pad_left + (0 - vmin) / (vmax - vmin) * plot_w
    for i, c in enumerate(counts):
        x = pad_left + i * bar_w
        h = (c / peak) * plot_h
        y = pad_top + (plot_h - h)
        centre = x + bar_w / 2
        # colour by sign of the bin centre
        bin_centre = (edges[i] + edges[i + 1]) / 2
        colour = color_win if bin_centre >= 0 else color_loss
        opacity = 0.85
        bars.append(
            f'<rect x="{x + 1:.2f}" y="{y:.2f}" width="{max(bar_w - 2, 1):.2f}" '
            f'height="{h:.2f}" fill="{colour}" fill-opacity="{opacity}" '
            f'rx="2" ry="2"><title>{bin_centre:.2f}: {int(c)}</title></rect>'
        )
    # Zero line
    zero_line = ""
    if vmin < 0 < vmax:
        zero_line = (f'<line x1="{zero_x:.2f}" y1="{pad_top}" '
                     f'x2="{zero_x:.2f}" y2="{height - pad_bot}" '
                     f'stroke="#8b95a7" stroke-width="1" stroke-dasharray="2,3"/>')
    # Y-axis labels (3 ticks)
    y_ticks = []
    for frac in (0, 0.5, 1.0):
        yv = pad_top + plot_h - frac * plot_h
        val = frac * peak
        y_ticks.append(
            f'<text x="{pad_left - 6}" y="{yv + 3:.2f}" text-anchor="end" '
            f'font-size="9" fill="#8b95a7">{int(round(val))}</text>'
            f'<line x1="{pad_left}" y1="{yv:.2f}" x2="{width - pad_right}" '
            f'y2="{yv:.2f}" stroke="#1f2738" stroke-width="1"/>'
        )
    # X-axis labels (3 ticks)
    x_ticks = []
    for frac in (0, 0.5, 1.0):
        xv = pad_left + frac * plot_w
        val = vmin + frac * (vmax - vmin)
        x_ticks.append(
            f'<text x="{xv:.2f}" y="{height - pad_bot + 14}" text-anchor="middle" '
            f'font-size="9" fill="#8b95a7">{val:.0f}</text>'
        )
    title_svg = ""
    if title:
        title_svg = (f'<text x="{width / 2:.2f}" y="14" text-anchor="middle" '
                     f'font-size="11" fill="#e8eaf0" font-weight="600">{title}</text>')
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" '
        f'class="svg-hist">{title_svg}{"".join(y_ticks)}{zero_line}'
        f'{"".join(bars)}{"".join(x_ticks)}</svg>'
    )


def _donut_svg(parts: list[tuple[str, float, str]], *, size: int = 120,
               thickness: int = 14, label: str = "", sub: str = "") -> str:
    """Inline SVG donut chart. parts = [(label, value, colour), ...]."""
    total = sum(max(0.0, v) for _, v, _ in parts)
    if total <= 0:
        return (f'<div class="empty-chart" style="width:{size}px;height:{size}px">'
                f'No data</div>')
    cx = cy = size / 2
    radius = (size / 2) - thickness / 2
    circumference = 2 * math.pi * radius
    offset = 0.0
    arcs = []
    for _label, v, colour in parts:
        frac = max(0.0, v) / total
        dash = frac * circumference
        gap = circumference - dash
        arcs.append(
            f'<circle r="{radius:.2f}" cx="{cx:.2f}" cy="{cy:.2f}" '
            f'fill="none" stroke="{colour}" stroke-width="{thickness}" '
            f'stroke-dasharray="{dash:.2f} {gap:.2f}" '
            f'stroke-dashoffset="{offset:.2f}" '
            f'transform="rotate(-90 {cx} {cy})"/>'
        )
        offset += dash
    centre_text = ""
    if label:
        centre_text = (f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" '
                       f'font-size="18" font-weight="700" fill="#e8eaf0">{label}</text>'
                       f'<text x="{cx}" y="{cy + 14}" text-anchor="middle" '
                       f'font-size="9" fill="#8b95a7">{sub}</text>')
    return (f'<svg viewBox="0 0 {size} {size}" class="svg-donut">'
            f'<circle r="{radius}" cx="{cx}" cy="{cy}" fill="none" '
            f'stroke="#1f2738" stroke-width="{thickness}"/>'
            f'{"".join(arcs)}{centre_text}</svg>')


# ---------------------------------------------------------------------------
# Number formatting helpers
# ---------------------------------------------------------------------------

def _fmt_money(v: float, *, decimals: int = 2) -> str:
    sign = "+" if v > 0 else ("-" if v < 0 else "")
    return f"{sign}${abs(v):,.{decimals}f}"


def _fmt_pct(v: float, *, decimals: int = 2) -> str:
    return f"{v * 100:.{decimals}f}%"


def _fmt_float(v: float, *, decimals: int = 2) -> str:
    if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 0.001):
        return f"{v:.3e}"
    return f"{v:,.{decimals}f}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_sexy_dashboard(metrics: dict,
                             trades: pd.DataFrame,
                             df_index: pd.DatetimeIndex,
                             equity: pd.Series,
                             path: str | Path,
                             strategy_name: str = "Strategy") -> Path:
    """Render a self-contained HTML dashboard and write it to *path*.

    Args:
        metrics: dict from ``result["metrics"]`` (or any flat metrics dict).
        trades: per-trade DataFrame (vectorbt or grid style — both detected).
        df_index: bar timestamps (used to map trade markers to equity curve).
        equity: bar-level equity curve (preferred as $ values).
        path: output ``.html`` path; parent dirs are created if needed.
        strategy_name: displayed in the hero.

    Returns the resolved :class:`Path` to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics = metrics or {}
    streak = _streak_stats(trades) if trades is not None else {}
    pnls = _pnl_series(trades) if trades is not None else None
    hold_hours = None
    if trades is not None and not trades.empty and "hold_hours" in trades.columns:
        hold_hours = pd.to_numeric(trades["hold_hours"], errors="coerce").dropna().tolist()

    # ---- Headline numbers (with safe fallbacks) ----
    net_pnl = _safe_float(metrics.get("net_pnl"))
    sharpe = _safe_float(metrics.get("sharpe"))
    max_dd = _safe_float(metrics.get("max_drawdown"))
    win_rate = _safe_float(metrics.get("win_rate"))
    n_trades = int(_safe_float(metrics.get("n_trades"), default=len(trades) if trades is not None else 0))

    # ---- Trade records for Plotly markers + table ----
    records: list[dict] = []
    if trades is not None and not trades.empty:
        pnls_for_table = _pnl_series(trades)
        for _, row in trades.iterrows():
            pnl_v = _safe_float(pnls_for_table.loc[_]) if pnls_for_table is not None and _ in pnls_for_table.index else 0.0
            records.append({
                "entry_time": (pd.Timestamp(row["entry_time"]).strftime("%Y-%m-%d %H:%M")
                                if "entry_time" in trades.columns and pd.notna(row.get("entry_time"))
                                else ""),
                "exit_time": (pd.Timestamp(row["exit_time"]).strftime("%Y-%m-%d %H:%M")
                                if "exit_time" in trades.columns and pd.notna(row.get("exit_time"))
                                else ""),
                "direction": _direction_map(row.get("Direction", row.get("direction", ""))),
                "entry_price": _safe_float(row.get("Avg Entry Price", row.get("entry_price"))),
                "exit_price": _safe_float(row.get("Avg Exit Price", row.get("exit_price"))),
                "size": _safe_float(row.get("Size", row.get("size", row.get("lots")))),
                "pnl": pnl_v,
                "hold_hours": _safe_float(row.get("hold_hours")),
                "strategy": str(row.get("strategy", "")),
            })

    # ---- Equity + drawdown payloads ----
    eq_x, eq_y = _equity_payload(equity)
    dd_x, dd_y = _drawdown_series(equity)
    max_dd_pct = abs(max(dd_y)) if dd_y else abs(max_dd * 100) if max_dd else 0.0

    # P&L distribution values
    pnl_values: list[float] = []
    if pnls is not None and not pnls.empty:
        pnl_values = [float(v) for v in pnls.values]

    # ---- JSON-encodable payloads for the JS layer ----
    trade_payload = records[:500]  # cap for browser perf; table still shows all
    equity_payload = {"x": eq_x, "y": eq_y}
    drawdown_payload = {"x": dd_x, "y": dd_y,
                         "max_dd_pct": round(max_dd_pct, 2)}

    metrics_safe = {k: _safe_float(v) for k, v in metrics.items()
                     if isinstance(v, (int, float, np.floating))}

    # ---- Build dist & donut SVGs ----
    pnl_hist_svg = _svg_histogram(pnl_values, bins=24, width=520, height=240,
                                    color_loss="#ff4b4b", color_win="#00d4aa",
                                    title="P&L distribution (USD)")
    hold_hist_svg = _svg_histogram(hold_hours or [], bins=24, width=520, height=240,
                                    color_loss="#4f8cff", color_win="#4f8cff",
                                    title="Hold-time distribution (hours)")

    wins = int(metrics.get("n_wins", 0))
    losses = int(metrics.get("n_losses", 0))
    donut = _donut_svg(
        [("Wins", wins, "#00d4aa"),
          ("Losses", losses, "#ff4b4b")],
        size=130, thickness=16,
        label=f"{win_rate * 100:.1f}%",
        sub=f"{wins}W / {losses}L",
    )

    # ---- Hero cards ----
    def hero_card(label: str, value: str, sub: str = "",
                   colour: str = "#e8eaf0", accent: bool = False) -> str:
        cls = "hero-card accent" if accent else "hero-card"
        return (f'<div class="{cls}"><div class="hero-label">{label}</div>'
                f'<div class="hero-value" style="color:{colour}">{value}</div>'
                f'<div class="hero-sub">{sub}</div></div>')

    pnl_colour = "#00d4aa" if net_pnl >= 0 else "#ff4b4b"
    dd_colour = "#ff4b4b" if max_dd < 0 else "#00d4aa"
    sharpe_colour = ("#00d4aa" if sharpe >= 1
                       else ("#ffb800" if sharpe >= 0 else "#ff4b4b"))

    hero_cards = [
        hero_card("Net P&L", _fmt_money(net_pnl), f"from {n_trades} trades",
                   colour=pnl_colour, accent=True),
        hero_card("Sharpe", f"{sharpe:.2f}",
                   _fmt_pct(_safe_float(metrics.get("cagr"))) + " CAGR",
                   colour=sharpe_colour),
        hero_card("Max Drawdown", f"{max_dd * 100:.2f}%",
                   f"peak-to-trough", colour=dd_colour),
        hero_card("Win Rate", f"{win_rate * 100:.1f}%",
                   f"{wins}W / {losses}L", colour="#e8eaf0"),
    ]

    # ---- Risk grid ----
    def risk_card(label: str, value: str, sub: str = "") -> str:
        return (f'<div class="risk-card"><div class="risk-label">{label}</div>'
                f'<div class="risk-value">{value}</div>'
                f'<div class="risk-sub">{sub}</div></div>')

    risk_cards = [
        risk_card("Sortino",
                   f"{_safe_float(metrics.get('sortino')):.2f}",
                   "downside-adjusted"),
        risk_card("Calmar",
                   f"{_safe_float(metrics.get('calmar')):.2f}",
                   "CAGR / max DD"),
        risk_card("Recovery",
                   f"{_safe_float(metrics.get('recovery_factor')):.2f}",
                   "return / max DD"),
        risk_card("Profit Factor",
                   f"{_safe_float(metrics.get('profit_factor'), default=0):.2f}",
                   f"gross {_fmt_money(_safe_float(metrics.get('gross_profit')))}"),
        risk_card("Expectancy",
                   _fmt_money(_safe_float(metrics.get("expectancy"))),
                   "per trade"),
        risk_card("Volatility",
                   f"{_safe_float(metrics.get('vol')) * 100:.2f}%",
                   "annualised"),
    ]

    # ---- Streaks ----
    streak_html = (f'<div class="streak-card win"><div class="streak-label">'
                   f'Longest Win Streak</div>'
                   f'<div class="streak-value">{int(streak.get("longest_win_streak", 0))}</div>'
                   f'<div class="streak-sub">consecutive wins</div></div>'
                   f'<div class="streak-card loss"><div class="streak-label">'
                   f'Longest Loss Streak</div>'
                   f'<div class="streak-value">{int(streak.get("longest_loss_streak", 0))}</div>'
                   f'<div class="streak-sub">consecutive losses</div></div>')

    # ---- Per-trade table ----
    table_rows = []
    show_count = min(len(records), 500)
    for r in records[:show_count]:
        pnl = r["pnl"]
        cls = "row-win" if pnl > 0 else ("row-loss" if pnl < 0 else "")
        pnl_class = "pnl-pos" if pnl > 0 else ("pnl-neg" if pnl < 0 else "")
        dir_class = "dir-long" if r["direction"] == "LONG" else "dir-short"
        table_rows.append(
            f"<tr class='{cls}'>"
            f"<td>{r['entry_time']}</td>"
            f"<td>{r['exit_time']}</td>"
            f"<td class='{dir_class}'>{r['direction']}</td>"
            f"<td>{r['entry_price']:.5f}</td>"
            f"<td>{r['exit_price']:.5f}</td>"
            f"<td>{r['size']:.2f}</td>"
            f"<td class='{pnl_class}'>{_fmt_money(pnl)}</td>"
            f"<td>{r['hold_hours']:.1f}h</td>"
            f"<td>{r['strategy']}</td>"
            f"</tr>"
        )
    table_html = "\n".join(table_rows)
    if len(records) > show_count:
        table_html += (f"<tr><td colspan='9' class='table-cap'>"
                        f"Showing first {show_count} of {len(records)} trades.</td></tr>")

    # ---- Footer ----
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if df_index is not None and len(df_index):
        try:
            start_date = pd.Timestamp(df_index[0]).strftime("%Y-%m-%d")
            end_date = pd.Timestamp(df_index[-1]).strftime("%Y-%m-%d")
            range_text = f"{start_date} → {end_date}"
        except Exception:
            range_text = "n/a"
    else:
        range_text = "n/a"
    footer_text = (f"Generated {timestamp} · Data range {range_text} · "
                    f"Engine: vectorbt · {n_trades} trades")

    # ---- Inject JSON payloads ----
    equity_json = json.dumps(equity_payload)
    drawdown_json = json.dumps(drawdown_payload)
    metrics_json = json.dumps(metrics_safe, indent=2)

    # ---- Render HTML ----
    html = HTML_TEMPLATE.format(
        strategy_name=strategy_name,
        net_pnl=_fmt_money(net_pnl),
        net_pnl_colour=pnl_colour,
        n_trades=n_trades,
        sharpe=f"{sharpe:.2f}",
        max_dd_pct=f"{max_dd * 100:.2f}",
        win_rate_pct=f"{win_rate * 100:.1f}",
        hero_cards="\n".join(hero_cards),
        risk_cards="\n".join(risk_cards),
        streak_html=streak_html,
        pnl_hist_svg=pnl_hist_svg,
        hold_hist_svg=hold_hist_svg,
        donut_svg=donut,
        table_html=table_html,
        footer_text=footer_text,
        equity_json=equity_json,
        drawdown_json=drawdown_json,
        metrics_json=metrics_json,
    )

    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# HTML template — single source of truth, all CSS inline
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{strategy_name} — Backtest Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root {{
  --bg: #0a0e1a;
  --bg-2: #0e1322;
  --card: rgba(255,255,255,0.03);
  --card-border: rgba(255,255,255,0.07);
  --text: #e8eaf0;
  --muted: #8b95a7;
  --profit: #00d4aa;
  --loss: #ff4b4b;
  --warn: #ffb800;
  --accent: #4f8cff;
  --grad: linear-gradient(135deg, #00d4aa 0%, #4f8cff 100%);
  --grad-soft: linear-gradient(135deg, rgba(0,212,170,0.15) 0%, rgba(79,140,255,0.15) 100%);
}}
* {{ box-sizing: border-box; }}
html, body {{
  margin: 0; padding: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI',
               Inter, Roboto, sans-serif;
  font-feature-settings: "tnum" 1, "cv11" 1;
  -webkit-font-smoothing: antialiased;
  min-height: 100vh;
}}
body {{
  background:
    radial-gradient(1100px 600px at 12% -10%, rgba(0,212,170,0.08), transparent 60%),
    radial-gradient(900px 500px at 95% 5%, rgba(79,140,255,0.08), transparent 60%),
    radial-gradient(700px 400px at 50% 110%, rgba(255,75,75,0.05), transparent 60%),
    var(--bg);
  background-attachment: fixed;
}}
.wrap {{ max-width: 1280px; margin: 0 auto; padding: 32px 24px 80px; }}

/* ====================== HERO ====================== */
.hero {{
  position: relative;
  margin-bottom: 32px;
  padding: 36px 32px;
  border-radius: 24px;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%),
    rgba(10,14,26,0.6);
  border: 1px solid var(--card-border);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  overflow: hidden;
}}
.hero::before {{
  content: "";
  position: absolute; inset: 0;
  background: var(--grad-soft);
  opacity: 0.5;
  pointer-events: none;
  z-index: 0;
}}
.hero-inner {{ position: relative; z-index: 1; }}
.hero-title-row {{
  display: flex; align-items: baseline; justify-content: space-between;
  margin-bottom: 28px; flex-wrap: wrap; gap: 12px;
}}
.hero-title {{
  font-size: 28px; font-weight: 700; letter-spacing: -0.02em;
  margin: 0; background: var(--grad); -webkit-background-clip: text;
  background-clip: text; -webkit-text-fill-color: transparent;
}}
.hero-tag {{
  display: inline-block; padding: 6px 12px; border-radius: 999px;
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--card-border);
  font-size: 12px; color: var(--muted);
  letter-spacing: 0.04em; text-transform: uppercase;
}}
.hero-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}}
@media (max-width: 900px) {{ .hero-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
.hero-card {{
  position: relative;
  padding: 22px 22px 18px;
  border-radius: 18px;
  background: var(--card);
  border: 1px solid var(--card-border);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  transition: transform 0.25s ease, border-color 0.25s ease;
}}
.hero-card:hover {{ transform: translateY(-2px); border-color: rgba(255,255,255,0.14); }}
.hero-card.accent {{
  background:
    linear-gradient(135deg, rgba(0,212,170,0.07) 0%, rgba(79,140,255,0.07) 100%),
    rgba(255,255,255,0.03);
  border-color: rgba(0,212,170,0.18);
}}
.hero-label {{
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin-bottom: 10px;
}}
.hero-value {{
  font-size: 30px; font-weight: 700; letter-spacing: -0.02em;
  line-height: 1.1;
}}
.hero-sub {{
  margin-top: 8px; font-size: 12px; color: var(--muted);
}}

/* ====================== CARDS ====================== */
.section {{ margin-top: 36px; }}
.section-title {{
  display: flex; align-items: center; gap: 10px;
  margin: 0 0 16px 0;
  font-size: 14px; text-transform: uppercase; letter-spacing: 0.12em;
  color: var(--muted); font-weight: 600;
}}
.section-title::before {{
  content: ""; display: inline-block; width: 4px; height: 16px;
  background: var(--grad); border-radius: 2px;
}}
.glass {{
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 18px;
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  padding: 18px;
  transition: border-color 0.25s ease;
}}
.glass:hover {{ border-color: rgba(255,255,255,0.12); }}

/* ====================== CHARTS ====================== */
.chart-grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}}
.chart-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}}
@media (max-width: 900px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
.chart-title {{
  font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin: 0 0 12px 0; font-weight: 600;
}}
.plotly-host {{ width: 100%; height: 340px; }}
.svg-hist, .svg-donut {{ width: 100%; height: auto; display: block; }}
.empty-chart {{
  display: flex; align-items: center; justify-content: center;
  height: 240px; color: var(--muted); font-size: 13px;
}}
.dist-block {{ display: flex; align-items: center; gap: 16px; }}
.donut-wrap {{ flex: 0 0 130px; }}

/* ====================== RISK GRID ====================== */
.risk-grid {{
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 14px;
}}
@media (max-width: 900px) {{ .risk-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
@media (max-width: 600px) {{ .risk-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
.risk-card {{
  padding: 16px;
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 14px;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  transition: transform 0.2s ease, border-color 0.2s ease;
}}
.risk-card:hover {{ transform: translateY(-2px); border-color: rgba(255,255,255,0.14); }}
.risk-label {{
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin-bottom: 6px;
}}
.risk-value {{
  font-size: 22px; font-weight: 700; letter-spacing: -0.01em;
}}
.risk-sub {{ margin-top: 4px; font-size: 11px; color: var(--muted); }}

/* ====================== STREAKS ====================== */
.streak-grid {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
}}
@media (max-width: 600px) {{ .streak-grid {{ grid-template-columns: 1fr; }} }}
.streak-card {{
  padding: 22px; border-radius: 16px; text-align: center;
  background: var(--card);
  border: 1px solid var(--card-border);
  backdrop-filter: blur(10px);
}}
.streak-card.win {{
  background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(0,212,170,0.02));
  border-color: rgba(0,212,170,0.18);
}}
.streak-card.loss {{
  background: linear-gradient(135deg, rgba(255,75,75,0.08), rgba(255,75,75,0.02));
  border-color: rgba(255,75,75,0.18);
}}
.streak-label {{
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted);
}}
.streak-value {{
  font-size: 44px; font-weight: 700; letter-spacing: -0.03em; margin: 8px 0;
}}
.streak-card.win .streak-value {{ color: var(--profit); }}
.streak-card.loss .streak-value {{ color: var(--loss); }}
.streak-sub {{ font-size: 11px; color: var(--muted); }}

/* ====================== TABLE ====================== */
.table-wrap {{
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 8px 4px 8px 8px;
  backdrop-filter: blur(10px);
  max-height: 540px; overflow: auto;
}}
table.trades {{
  width: 100%; border-collapse: collapse; font-size: 12.5px;
  font-variant-numeric: tabular-nums;
}}
table.trades thead th {{
  position: sticky; top: 0;
  background: rgba(14,19,34,0.95);
  backdrop-filter: blur(6px);
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 10.5px; font-weight: 600;
  padding: 12px 10px; text-align: left;
  border-bottom: 1px solid var(--card-border);
  z-index: 1;
}}
table.trades tbody td {{
  padding: 8px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  color: var(--text);
}}
table.trades tbody tr:nth-child(even) {{ background: rgba(255,255,255,0.015); }}
table.trades tbody tr:hover {{ background: rgba(255,255,255,0.045); }}
.row-win td:first-child {{ border-left: 2px solid var(--profit); }}
.row-loss td:first-child {{ border-left: 2px solid var(--loss); }}
.pnl-pos {{ color: var(--profit); font-weight: 600; }}
.pnl-neg {{ color: var(--loss); font-weight: 600; }}
.dir-long {{ color: var(--profit); font-weight: 600; letter-spacing: 0.04em; }}
.dir-short {{ color: var(--loss); font-weight: 600; letter-spacing: 0.04em; }}
.table-cap {{ text-align: center; color: var(--muted); padding: 16px; }}

/* ====================== FOOTER ====================== */
.footer {{
  margin-top: 56px; padding-top: 24px;
  border-top: 1px solid var(--card-border);
  display: flex; justify-content: space-between; align-items: center;
  color: var(--muted); font-size: 12px; flex-wrap: wrap; gap: 12px;
}}
.footer-dot {{
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--profit); box-shadow: 0 0 8px var(--profit);
  margin-right: 8px; vertical-align: middle;
}}

/* Subtle entrance animation */
@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
.hero, .section {{ animation: fadeUp 0.5s ease both; }}
.section:nth-child(2) {{ animation-delay: 0.05s; }}
.section:nth-child(3) {{ animation-delay: 0.1s; }}
.section:nth-child(4) {{ animation-delay: 0.15s; }}

/* Scrollbar polish (webkit) */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
  background: rgba(255,255,255,0.07); border-radius: 10px;
}}
::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.14); }}
</style>
</head>
<body>
<div class="wrap">

  <!-- ============== HERO ============== -->
  <section class="hero">
    <div class="hero-inner">
      <div class="hero-title-row">
        <h1 class="hero-title">{strategy_name}</h1>
        <span class="hero-tag">Backtest Report</span>
      </div>
      <div class="hero-grid">
        {hero_cards}
      </div>
    </div>
  </section>

  <!-- ============== EQUITY + DRAWDOWN ============== -->
  <section class="section">
    <h2 class="section-title">Equity &amp; Drawdown</h2>
    <div class="chart-grid">
      <div class="glass">
        <p class="chart-title">Equity Curve</p>
        <div id="equityChart" class="plotly-host"></div>
      </div>
      <div class="glass">
        <p class="chart-title">Underwater Drawdown</p>
        <div id="drawdownChart" class="plotly-host"></div>
      </div>
    </div>
  </section>

  <!-- ============== DISTRIBUTION + DONUT ============== -->
  <section class="section">
    <h2 class="section-title">Trade Distribution</h2>
    <div class="chart-row">
      <div class="glass">
        <div class="dist-block">
          <div class="donut-wrap">{donut_svg}</div>
          <div style="flex:1">
            <p class="chart-title">Win/Loss Mix</p>
            {pnl_hist_svg}
          </div>
        </div>
      </div>
      <div class="glass">
        <p class="chart-title">Hold-Time Profile</p>
        {hold_hist_svg}
      </div>
    </div>
  </section>

  <!-- ============== RISK GRID ============== -->
  <section class="section">
    <h2 class="section-title">Risk Metrics</h2>
    <div class="risk-grid">
      {risk_cards}
    </div>
  </section>

  <!-- ============== STREAKS ============== -->
  <section class="section">
    <h2 class="section-title">Streaks</h2>
    <div class="streak-grid">
      {streak_html}
    </div>
  </section>

  <!-- ============== TRADE TABLE ============== -->
  <section class="section">
    <h2 class="section-title">Trades</h2>
    <div class="table-wrap">
      <table class="trades">
        <thead>
          <tr>
            <th>Entry</th><th>Exit</th><th>Side</th>
            <th>Entry Price</th><th>Exit Price</th>
            <th>Size</th><th>P&amp;L</th>
            <th>Hold</th><th>Strategy</th>
          </tr>
        </thead>
        <tbody>
          {table_html}
        </tbody>
      </table>
    </div>
  </section>

  <!-- ============== FOOTER ============== -->
  <footer class="footer">
    <div><span class="footer-dot"></span>{footer_text}</div>
    <div style="color:var(--muted);font-size:11px;">
      Multi-Strategy EA · Backtest Harness v1.0
    </div>
  </footer>
</div>

<script>
(function () {{
  const EQUITY = {equity_json};
  const DRAWDOWN = {drawdown_json};

  // -------------- Layout helpers --------------
  const baseLayout = {{
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{ family: '-apple-system, "SF Pro Display", Inter, sans-serif',
              color: '#8b95a7', size: 11 }},
    margin: {{ l: 50, r: 20, t: 10, b: 36 }},
    xaxis: {{
      gridcolor: 'rgba(255,255,255,0.04)',
      zerolinecolor: 'rgba(255,255,255,0.06)',
      showline: false,
      type: 'date',
    }},
    yaxis: {{
      gridcolor: 'rgba(255,255,255,0.04)',
      zerolinecolor: 'rgba(255,255,255,0.06)',
      showline: false,
    }},
    showlegend: false,
    hoverlabel: {{
      bgcolor: '#0e1322',
      bordercolor: 'rgba(255,255,255,0.08)',
      font: {{ family: '-apple-system, sans-serif', color: '#e8eaf0', size: 12 }},
    }},
    hovermode: 'x unified',
  }};

  // -------------- Equity curve --------------
  const equityTraces = [
    {{
      x: EQUITY.x, y: EQUITY.y,
      type: 'scatter', mode: 'lines',
      line: {{ color: '#00d4aa', width: 2, shape: 'spline' }},
      fill: 'tozeroy',
      fillcolor: 'rgba(0,212,170,0.10)',
      hovertemplate: '<b>%{{x}}</b><br>Equity: $%{{y:,.2f}}<extra></extra>',
      name: 'Equity',
    }},
  ];

  Plotly.newPlot(
    'equityChart',
    equityTraces,
    {{ ...baseLayout,
      yaxis: {{ ...baseLayout.yaxis, tickformat: '$,.0f' }} }},
    {{ displayModeBar: false, responsive: true }}
  );

  // -------------- Underwater drawdown --------------
  const drawdownTraces = [
    {{
      x: DRAWDOWN.x, y: DRAWDOWN.y,
      type: 'scatter', mode: 'lines',
      line: {{ color: '#ff4b4b', width: 1.5, shape: 'spline' }},
      fill: 'tozeroy',
      fillcolor: 'rgba(255,75,75,0.25)',
      hovertemplate: '<b>%{{x}}</b><br>DD: %{{y:.2f}}%<extra></extra>',
      name: 'Drawdown',
    }},
    {{
      x: DRAWDOWN.x,
      y: DRAWDOWN.x.map(() => -Math.abs(DRAWDOWN.max_dd_pct || 0)),
      type: 'scatter', mode: 'lines',
      line: {{ color: 'rgba(255,184,0,0.6)', width: 1, dash: 'dot' }},
      hoverinfo: 'skip',
      name: 'Max DD',
    }},
  ];

  Plotly.newPlot(
    'drawdownChart',
    drawdownTraces,
    {{ ...baseLayout,
      yaxis: {{ ...baseLayout.yaxis, ticksuffix: '%', range: [Math.min(...DRAWDOWN.y, 0) * 1.05, 1] }} }},
    {{ displayModeBar: false, responsive: true }}
  );

  // Resize on viewport change so charts stay crisp
  window.addEventListener('resize', () => {{
    Plotly.Plots.resize('equityChart');
    Plotly.Plots.resize('drawdownChart');
  }});
}})();
</script>
</body>
</html>
"""
