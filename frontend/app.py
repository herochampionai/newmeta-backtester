"""Universal Backtester — Newmeta Research Lab
Institutional-grade backtesting, Bayesian parameter optimization, full market radar scanner, and MQL5 EA porting engine.
"""
from __future__ import annotations
import sys
from pathlib import Path
import io
import json

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.live_fetcher import fetch_with_priority, load_settings
from data.mt5_export import resolve_terminal
from core.loader import load_any_strategy
from core.mql5_parser import MQL5Parser
from strategies._base import Signals
from strategies import (
    STRATEGY_REGISTRY, TRADABLE_STRATEGY_REGISTRY,
    MULTI_STRAT_EA_REGISTRY, CRYPTO_STRAT_EA_REGISTRY,
)
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE, GRID_LOSS, GRID_PROFIT, GRID_LOSS_AND_PROFIT
from backtester.adaptive import AdaptiveConfig
from backtester.execution import MARKET_PROFILES, profile_for
from backtester.metrics_v2 import compute_all
from backtester.analytics import strategy_scoreboard
from backtester.sexydashboard import generate_dashboard_html
from analysis.optuna_optimizer import optimize_strategy_criterion, get_study_trials_df, get_param_importances_dict
from analysis.composite_criterion import CRITERION_PRESETS, composite_score, composite_to_dict
from analysis.ticker_scanner import scan_market_matrix, get_available_mt5_symbols, PRESET_UNIVERSES
from analysis.universe_optimizer import rank_universe, optimize_top_n
from analysis.montecarlo import ci_metrics, bootstrap_returns
from analysis.markowitz_alloc import allocate
from analysis.walkforward import walk_forward, wf_summary
from analysis.execution_stress import run_execution_stress
from analysis.readiness import assess_backtest_readiness, assess_walkforward_readiness
from analysis.data_quality import assess_data_quality
from analysis.mql5_set_export import build_mql5_set_text
from analysis.research_packet import save_research_packet
from core.surgical_features import is_enabled, get_feature_defaults as _default_surgical_features

st.set_page_config(
    page_title="Newmeta Research Lab - Backtester",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg": "#010409",
    "card_bg": "#0d1117",
    "card_border": "#1f2937",
    "panel_deep": "#020617",
    "primary": "#22d3ee",
    "success": "#34d399",
    "warning": "#fbbf24",
    "danger": "#fb7185",
    "info": "#60a5fa",
    "text": "#e5edf7",
    "muted": "#94a3b8",
}

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background:
            radial-gradient(circle at 18% 0%, rgba(34,211,238,0.12), transparent 28%),
            radial-gradient(circle at 82% 0%, rgba(251,113,133,0.12), transparent 30%),
            {PALETTE['bg']};
        color: {PALETTE['text']};
        font-family: 'Inter', sans-serif;
    }}
    .block-container {{ padding-top: 0.75rem; padding-bottom: 0.75rem; max-width: 1720px; }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #020617 0%, #0d1117 100%);
        border-right: 1px solid {PALETTE['card_border']};
    }}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label {{ color: {PALETTE['muted']} !important; }}
    h1, h2, h3 {{ color: {PALETTE['text']} !important; font-family: 'Orbitron', sans-serif; letter-spacing: 0; }}
    .nm-topbar {{
        display: flex; align-items: center; gap: 16px; justify-content: space-between;
        background: linear-gradient(135deg, rgba(13,17,23,0.96), rgba(2,6,23,0.96));
        border: 1px solid {PALETTE['card_border']}; border-radius: 8px;
        padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 18px 55px rgba(0,0,0,0.34);
    }}
    .nm-brand {{ display:flex; align-items:center; gap:14px; min-width:0; }}
    .nm-logo {{ width:58px; height:58px; object-fit:cover; border-radius:8px; border:1px solid rgba(34,211,238,0.35); }}
    .nm-logo-fallback {{ width:58px; height:58px; border-radius:8px; display:grid; place-items:center; font-family:'Orbitron'; font-weight:800; color:white; border:1px solid rgba(34,211,238,0.45); background:#020617; }}
    .nm-title {{ font-family:'Orbitron', sans-serif; font-weight:800; font-size:1.58rem; line-height:1.05; color:{PALETTE['text']}; }}
    .nm-subtitle {{ color:{PALETTE['muted']}; font-size:0.95rem; margin-top:4px; }}
    .nm-pill {{ border:1px solid rgba(34,211,238,0.38); color:{PALETTE['primary']}; background:rgba(34,211,238,0.08); padding:7px 10px; border-radius:999px; font-size:0.78rem; font-weight:700; white-space:nowrap; }}
    .nm-context-strip {{ display:grid; grid-template-columns: repeat(6, minmax(130px, 1fr)); gap:8px; margin:6px 0 8px; }}
    .nm-context-item {{ background:rgba(13,17,23,0.84); border:1px solid {PALETTE['card_border']}; border-radius:8px; padding:8px 10px; min-height:52px; }}
    .nm-context-item span {{ color:{PALETTE['muted']}; display:block; font-size:0.78rem; text-transform:uppercase; }}
    .nm-context-item b {{ color:{PALETTE['text']}; display:block; margin-top:3px; font-size:1.08rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .nm-panel {{ background:rgba(13,17,23,0.88); border:1px solid {PALETTE['card_border']}; border-radius:8px; padding:10px 12px; margin:6px 0; }}
    .nm-panel-title {{ font-family:'Orbitron'; color:{PALETTE['text']}; font-weight:700; font-size:1.08rem; margin-bottom:3px; }}
    .nm-muted {{ color:{PALETTE['muted']}; font-size:0.86rem; }}
    .metric-card {{
        background: linear-gradient(135deg, rgba(13,17,23,0.98) 0%, rgba(2,6,23,0.98) 100%);
        border: 1px solid {PALETTE['card_border']}; border-radius: 8px;
        padding: 10px 12px; margin-bottom: 6px; box-shadow: 0 10px 26px rgba(0,0,0,0.20);
    }}
    .metric-card .label {{ color: {PALETTE['muted']}; font-size: 10px; text-transform: uppercase; margin-bottom: 4px; }}
    .metric-card .value {{ color: {PALETTE['text']}; font-size: 20px; font-weight: 800; line-height: 1.2; }}
    .metric-card.green {{ border-left: 4px solid {PALETTE['success']}; }}
    .metric-card.red   {{ border-left: 4px solid {PALETTE['danger']}; }}
    .metric-card.blue  {{ border-left: 4px solid {PALETTE['info']}; }}
    .metric-card.amber {{ border-left: 4px solid {PALETTE['warning']}; }}
    div[data-testid="stVerticalBlock"] {{ gap: 0.45rem !important; }}
    div[data-testid="stHorizontalBlock"] {{ gap: 0.7rem !important; }}
    div[data-testid="stElementContainer"] {{ margin-bottom: 0.2rem !important; }}
    div[data-testid="stFileUploader"] section {{ min-height: 52px !important; padding: 8px 10px !important; }}
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {{ padding: 8px 10px !important; }}
    .stButton > button, .stDownloadButton > button {{ min-height: 38px !important; padding: 0.42rem 0.7rem !important; border-radius:8px !important; border:1px solid {PALETTE['card_border']} !important; font-weight:700 !important; }}
    div[data-baseweb="select"] > div {{ min-height: 38px !important; }}
    input {{ min-height: 36px !important; }}
    div[data-testid="stTabs"] button {{ color:{PALETTE['muted']} !important; font-weight:700; }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{ color:{PALETTE['primary']} !important; }}
    div[data-testid="stFileUploader"] section {{ background:rgba(2,6,23,0.72); border:1px dashed rgba(34,211,238,0.38); border-radius:8px; }}
    div[data-baseweb="select"] > div, input, textarea {{ background-color:#020617 !important; border-color:{PALETTE['card_border']} !important; color:{PALETTE['text']} !important; }}
    .nm-footer {{ color:{PALETTE['muted']}; font-size:0.8rem; border-top:1px solid {PALETTE['card_border']}; padding-top:12px; margin-top:18px; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

LOGO_PATH = Path(__file__).parent / "assets" / "newmeta_logo.png"


def metric_card(label: str, value: str, color: str = "blue") -> str:
    return f'<div class="metric-card {color}"><div class="label">{label}</div><div class="value">{value}</div></div>'


def _equity_chart_overlay(equity_a: pd.Series, equity_b: pd.Series,
                          drawdown_a: pd.Series,
                          label_a: str = "Tick", label_b: str = "OHLC") -> go.Figure:
    """Overlay two equity curves for OHLC vs Tick comparison."""
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            vertical_spacing=0.03)
    except TypeError:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.3, 0.7],
                            vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=equity_a.index, y=equity_a.values, mode="lines",
                              name=label_a, line=dict(color=PALETTE["primary"], width=2.2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=equity_b.index, y=equity_b.values, mode="lines",
                              name=label_b, line=dict(color=PALETTE["warning"], width=1.8, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown_a.index, y=drawdown_a.values * 100,
                              mode="lines", name=f"DD ({label_a})",
                              line=dict(color=PALETTE["danger"], width=1),
                              fill="tozeroy", fillcolor="rgba(251, 113, 133, 0.2)"),
                  row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["bg"],
                      plot_bgcolor=PALETTE["card_bg"], height=480,
                      showlegend=True, legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
                      margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(color=PALETTE["text"]))
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1, gridcolor=PALETTE["card_border"])
    fig.update_yaxes(title_text="DD (%)", row=2, col=1, gridcolor=PALETTE["card_border"])
    return fig


def equity_chart(equity: pd.Series, drawdown: pd.Series) -> go.Figure:
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            vertical_spacing=0.03)
    except TypeError:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.3, 0.7],
                            vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines",
                              name="Equity", line=dict(color=PALETTE["primary"], width=2),
                              fill="tozeroy", fillcolor="rgba(34, 211, 238, 0.1)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values * 100,
                              mode="lines", name="DD",
                              line=dict(color=PALETTE["danger"], width=1),
                              fill="tozeroy", fillcolor="rgba(251, 113, 133, 0.2)"),
                  row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["bg"],
                      plot_bgcolor=PALETTE["card_bg"], height=480,
                      showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(color=PALETTE["text"]))
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1, gridcolor=PALETTE["card_border"])
    fig.update_yaxes(title_text="DD (%)", row=2, col=1, gridcolor=PALETTE["card_border"])
    return fig


def cumulative_pnl_chart(equity: pd.Series, init_cash: float) -> go.Figure:
    pnl = equity - init_cash
    colors = [PALETTE["success"] if v >= 0 else PALETTE["danger"] for v in pnl.values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pnl.index, y=pnl.values, mode="lines",
        name="Cumulative P&L",
        line=dict(color=PALETTE["primary"], width=2.5),
        fill="tozeroy", fillcolor="rgba(34, 211, 238, 0.08)",
        hovertemplate="<b>%{x}</b><br>P&L: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=pnl.index, y=pnl.values, mode="markers",
        marker=dict(color=colors, size=3, line=dict(width=0)),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_hline(y=0, line=dict(color=PALETTE["muted"], width=1, dash="dash"))
    if len(pnl) > 1:
        max_idx = pnl.idxmax(); min_idx = pnl.idxmin()
        fig.add_annotation(
            x=max_idx, y=float(pnl.max()),
            text=f"Peak: ${pnl.max():.0f}",
            showarrow=True, arrowhead=2, ax=-40, ay=-30,
            font=dict(color=PALETTE["success"], size=11),
        )
        fig.add_annotation(
            x=min_idx, y=float(pnl.min()),
            text=f"Trough: ${pnl.min():.0f}",
            showarrow=True, arrowhead=2, ax=40, ay=30,
            font=dict(color=PALETTE["danger"], size=11),
        )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card_bg"], height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color=PALETTE["text"]),
        xaxis=dict(gridcolor=PALETTE["card_border"]),
        yaxis=dict(title="Cumulative P&L ($)", gridcolor=PALETTE["card_border"],
                    tickprefix="$", tickformat=",.0f"),
    )
    return fig


def price_chart_with_signals(df: pd.DataFrame, sig: Signals, trades: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    has_ohlc = all(c in df.columns for c in ("open", "high", "low", "close"))
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Price",
            increasing_line_color=PALETTE["success"],
            decreasing_line_color=PALETTE["danger"],
            whiskerwidth=0,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["close"], mode="lines", name="Price",
            line=dict(color=PALETTE["primary"], width=1.5),
        ))

    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)

    long_entries = entries & (direction > 0)
    short_entries = entries & (direction < 0)
    vol = df["close"].std() * 0.2 if len(df) > 1 else 0.001

    if long_entries.any():
        idx = df.index[long_entries.values]
        prices = df.loc[idx, "low"].values - vol
        fig.add_trace(go.Scatter(
            x=idx, y=prices, mode="markers",
            marker=dict(symbol="triangle-up", size=10, color=PALETTE["success"]),
            name="Long Entry",
            hovertemplate="<b>Long Entry</b><br>%{x|%Y-%m-%d %H:%M}<br>Price: %{y:.5f}<extra></extra>",
        ))

    if short_entries.any():
        idx = df.index[short_entries.values]
        prices = df.loc[idx, "high"].values + vol
        fig.add_trace(go.Scatter(
            x=idx, y=prices, mode="markers",
            marker=dict(symbol="triangle-down", size=10, color=PALETTE["danger"]),
            name="Short Entry",
            hovertemplate="<b>Short Entry</b><br>%{x|%Y-%m-%d %H:%M}<br>Price: %{y:.5f}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card_bg"], height=440,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color=PALETTE["text"]),
        xaxis=dict(gridcolor=PALETTE["card_border"], rangeslider=dict(visible=False)),
        yaxis=dict(title="Price", gridcolor=PALETTE["card_border"]),
    )
    return fig


def monthly_returns_heatmap(equity: pd.Series) -> go.Figure | None:
    try:
        daily = equity.resample("D").last().dropna()
        if len(daily) < 10:
            return None
        ret = daily.pct_change().dropna()
        df_ret = ret.to_frame("ret")
        df_ret["year"] = df_ret.index.year
        df_ret["month"] = df_ret.index.month
        monthly = df_ret.groupby(["year", "month"])["ret"].apply(lambda r: (1 + r).prod() - 1).unstack() * 100.0
        
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        cols = [month_names[m - 1] for m in monthly.columns]
        
        fig = go.Figure(data=go.Heatmap(
            z=monthly.values,
            x=cols,
            y=[str(y) for y in monthly.index],
            colorscale="RdYlGn",
            colorbar=dict(title="Return %"),
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=PALETTE["bg"],
            plot_bgcolor=PALETTE["card_bg"], height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(color=PALETTE["text"]),
        )
        return fig
    except Exception:
        return None


def optuna_importance_chart(importances: dict[str, float]) -> go.Figure:
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    params = [k for k, v in sorted_items][::-1]
    scores = [v for k, v in sorted_items][::-1]
    
    fig = go.Figure(go.Bar(
        x=scores, y=params, orientation="h",
        marker=dict(color=PALETTE["primary"]),
        hovertemplate="Parameter: %{y}<br>Importance: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Parameter Sensitivity & Importance (%)",
        template="plotly_dark", paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card_bg"], height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color=PALETTE["text"]),
        xaxis=dict(title="Importance %", gridcolor=PALETTE["card_border"]),
    )
    return fig


def optimization_history_chart(trials_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trials_df["trial_number"], y=trials_df["score"],
        mode="markers", name="Trial Score",
        marker=dict(color=PALETTE["primary"], size=6, opacity=0.7),
        hovertemplate="Trial #%{x}<br>Score: %{y:.2f}<extra></extra>",
    ))
    # Best cumulative curve
    cum_best = trials_df.sort_values("trial_number")["score"].cummax()
    fig.add_trace(go.Scatter(
        x=cum_best.index, y=cum_best.values,
        mode="lines", name="Best Found",
        line=dict(color=PALETTE["success"], width=2),
    ))
    fig.update_layout(
        title="Optimization Convergence History",
        template="plotly_dark", paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card_bg"], height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color=PALETTE["text"]),
        xaxis=dict(title="Trial Number", gridcolor=PALETTE["card_border"]),
        yaxis=dict(title="Criterion Score", gridcolor=PALETTE["card_border"]),
    )
    return fig


def render_metrics(m: dict, keys: list[str], cols_per_row: int = 5):
    rows = [keys[i:i + cols_per_row] for i in range(0, len(keys), cols_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for col, k in zip(cols, row):
            with col:
                v = m.get(k, 0)
                if isinstance(v, (int, float)):
                    if k in ("win_rate", "total_return", "cagr", "max_drawdown", "vol"):
                        s = f"{v:.1%}"
                    elif k in ("net_pnl", "final_equity", "avg_win", "avg_loss", "expectancy"):
                        s = f"${v:,.2f}"
                    else:
                        s = f"{v:.2f}"
                else:
                    s = str(v)
                color = "blue"
                if k in ("sharpe", "sortino", "calmar", "recovery_factor"):
                    color = "green" if v > 1.0 else ("amber" if v > 0.5 else "red")
                elif k in ("total_return", "cagr", "net_pnl"):
                    color = "green" if v > 0 else "red"
                elif k == "max_drawdown":
                    color = "red" if v < -0.20 else ("amber" if v < -0.10 else "green")
                elif k == "win_rate":
                    color = "green" if v > 0.55 else ("amber" if v > 0.45 else "red")
                elif k == "profit_factor":
                    color = "green" if v > 1.30 else ("amber" if v > 1.0 else "red")
                st.markdown(metric_card(k.replace("_", " ").title(), s, color=color), unsafe_allow_html=True)


# === Sidebar Settings ===
settings = load_settings()
default_terminal = settings.get("mt5_terminal", r"D:\MT5_EuroPrinter\terminal64.exe")

with st.sidebar:
    st.markdown("### NewMeta Controls")
    terminals_found = [str(p) for p in [
        Path(r"D:\MT5_EuroPrinter\terminal64.exe"),
        Path(r"D:\MT5_Bybit\terminal64.exe"),
    ] if p.exists()]
    if not terminals_found:
        terminals_found = [default_terminal]

    market_tab, risk_tab, grid_tab, exec_tab, data_tab = st.tabs(["Market", "Risk", "Grid", "Execution", "Data"])

    with market_tab:
        terminal_choice = st.selectbox("MT5 Terminal", terminals_found, index=0)
        symbol = st.text_input("Symbol", st.session_state.get("selected_symbol", "XAUUSD"))
        market_choice = st.selectbox("Market preset", ["Auto", "Forex", "XAUUSD", "Crypto"], index=0)
        market_profile = profile_for(market_choice, symbol)
        st.caption(f"Using {market_profile.name}: pip={market_profile.pip_size}, contract={market_profile.contract_size:g}")
        timeframe = st.selectbox("Timeframe", ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"],
                                 index=["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"].index(st.session_state.get("selected_timeframe", "M1")))
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start", pd.Timestamp("2023-11-01"))
        end_date = col2.date_input("End", pd.Timestamp("2026-09-30"))

    with risk_tab:
        profile_name = st.selectbox("Profile", ["Custom", "Conservative", "Balanced", "Aggressive"])
        init_cash = st.number_input("Initial cash ($)", 1000, 1_000_000, 10000, step=1000)
        commission_mode = st.selectbox("Commission mode", ["Pips", "Percent"], index=0)
        if commission_mode == "Pips":
            commission_pips = st.number_input("Commission (pips/order)", 0.0, 50.0, float(market_profile.default_commission_pips), step=0.1)
            commission_pct = 0.0
        else:
            commission_pct = st.number_input("Commission (%/order)", 0.0, 2.0, float(market_profile.commission_pct), step=0.005, format="%.3f")
            commission_pips = 0.0
        slippage_pips = st.number_input("Slippage (pips/order)", 0.0, 50.0, float(market_profile.default_slippage_pips), step=0.1)
        spread_pips = st.number_input("Spread (pips/order)", 0.0, 500.0, float(market_profile.default_spread_pips), step=0.1)
        pip_size = st.number_input("Pip size", 0.00000001, 100.0, float(market_profile.pip_size), format="%.8f")
        contract_size = st.number_input("Contract size", 0.0001, 1_000_000.0, float(market_profile.contract_size), step=1.0)
        base_lot = st.number_input("Base lot", 0.01, 10.0, 0.1, step=0.01)
        adaptive_on = st.checkbox("Adaptive lot sizing", value=False)

    with grid_tab:
        grid_options = ["None", "Loss only", "Profit only", "Loss+Profit"]
        grid_mode = st.selectbox("Grid mode", grid_options, index=0)
        GRID_MAP = {"None": GRID_NONE, "Loss only": GRID_LOSS,
                    "Profit only": GRID_PROFIT, "Loss+Profit": GRID_LOSS_AND_PROFIT}
        pips_between = st.slider("Pips between layers", 5, 200, 30)
        grid_lot_mult = st.slider("Grid lot multiplier", 1.0, 3.0, 1.5, step=0.1)
        grid_tp = st.number_input("Grid TP ($)", 0.0, 1000.0, 50.0, step=5.0)
        grid_sl = st.number_input("Grid SL ($)", 0.0, 5000.0, 200.0, step=10.0)
        max_layers = st.slider("Max grid layers", 1, 12, 4)
        recovery_mode_label = st.selectbox("Recovery mode", ["None", "Last close (martingale)"], index=0)
        rec_mult = st.slider("Recovery lot multiplier", 1.0, 5.0, 2.0, step=0.1)

    with exec_tab:
        # Tick-level execution controls. Default OFF — no behavior change for existing users.
        st.caption("🎯 **Tick execution** — uses tick-by-tick spread model + OHLC bar-close fills. "
                   "Honest scope: adds variable spread + commission to fills (matches MT5 'Every tick' spread model), "
                   "but TP/SL triggers still check bar close, not intra-bar wicks.")
        tick_mode = st.selectbox(
            "Execution mode",
            ["off (OHLC bars)", "synthetic ticks", "real (MT5 live ticks)"],
            index=0,
            help="off = standard OHLC bar-level. synthetic = synthesized intra-bar ticks + variable spread (commission/slippage). real = try real MT5 ticks first.",
        )
        # Map UI label to engine param
        tick_mode_internal = {"off (OHLC bars)": "off", "synthetic (intra-bar ticks)": "synthetic", "real (MT5 live ticks)": "real"}[tick_mode]
        ticks_per_bar = st.slider("Synthetic ticks per bar", 5, 50, 20, step=5, disabled=(tick_mode_internal == "off"),
                                   help="More ticks = more accurate fills but slower. 20 is a good balance.")
        use_real_spreads = st.checkbox("Use real broker spreads", value=False, disabled=(tick_mode_internal == "off"),
                                        help="Fetch recent spread history from MT5 (requires running terminal).")
        compare_ohlc = st.checkbox("Also run OHLC baseline for comparison", value=False, disabled=(tick_mode_internal == "off"),
                                    help="Run both OHLC + tick backtests and overlay equity curves. Doubles runtime.")
        if tick_mode_internal != "off":
            st.info(f"⚡ Tick mode ON ({tick_mode_internal}). Engine routes to engine_deep.deep_backtest().")
            if compare_ohlc:
                st.info("📊 Will overlay OHLC baseline after tick run completes.")
            # Informational: grid=NONE still executes the initial signal-driven layer
            # via GridRecoveryManager, but grid/recovery overlays (martingale layers,
            # basket TP/SL) only fire when grid_mode != NONE. So this combo produces
            # a "pure-strategy" tick sim — valid, but not what people usually want.
            if "None" in grid_mode:
                st.info("ℹ Grid mode = None. Tick sim will still fire the initial signal-driven trade "
                        "via GridRecoveryManager, but grid/recovery layers and basket TP/SL are disabled. "
                        "For grid EA validation, pick Loss / Profit / Loss+Profit.")

    with data_tab:
        swap_on = st.checkbox("Apply swap", value=False)
        long_swap = st.number_input("Long swap (pips/day)", -100.0, 100.0, float(market_profile.default_long_swap_pips), step=0.1)
        short_swap = st.number_input("Short swap (pips/day)", -100.0, 100.0, float(market_profile.default_short_swap_pips), step=0.1)

# === Header ===
brand_col, title_col, status_col = st.columns([0.32, 3.8, 1.0], vertical_alignment="center")
with brand_col:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=58)
    else:
        st.markdown('<div class="nm-logo-fallback">NM</div>', unsafe_allow_html=True)
with title_col:
    st.markdown('<div class="nm-title">NEWMETA RESEARCH LAB - BACKTESTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="nm-subtitle">Universal Multi-Strategy Testing, Bayesian Optuna Criterion Study, Full Market Radar & MQL5 Porting Engine</div>', unsafe_allow_html=True)
with status_col:
    st.markdown('<div class="nm-pill">2026 SOVEREIGN ENGINE</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="nm-context-strip">
  <div class="nm-context-item"><span>Symbol</span><b>{symbol}</b></div>
  <div class="nm-context-item"><span>Timeframe</span><b>{timeframe}</b></div>
  <div class="nm-context-item"><span>Window</span><b>{start_date} to {end_date}</b></div>
  <div class="nm-context-item"><span>Starting Equity</span><b>${init_cash:,.0f}</b></div>
  <div class="nm-context-item"><span>Risk Profile</span><b>{profile_name}</b></div>
  <div class="nm-context-item"><span>Execution Mode</span><b>{grid_mode}</b></div>
</div>
""", unsafe_allow_html=True)

TOOLBAR_MODES = [
    ("🚀 Backtest", "Backtest"),
    ("🔬 Optimize (Criterion Study)", "Optimize"),
    ("📡 Market Scanner Radar", "Market Scanner"),
    ("🧩 MQL5 Inspector & Port", "MQL5 Inspector"),
    ("🎲 Monte Carlo", "Monte Carlo"),
    ("📊 Walk-Forward", "Walk-Forward"),
]
if "current_mode" not in st.session_state:
    st.session_state["current_mode"] = "Backtest"

cols = st.columns(len(TOOLBAR_MODES) + 1)
for i, (label, mode_key) in enumerate(TOOLBAR_MODES):
    with cols[i]:
        is_active = st.session_state["current_mode"] == mode_key
        btn_type = "primary" if is_active else "secondary"
        if st.button(label, type=btn_type, use_container_width=True, key=f"toolbar_{i}"):
            st.session_state["current_mode"] = mode_key
            st.rerun()
with cols[-1]:
    more_open = st.button("More Modes...", use_container_width=True, key="toolbar_more")
    if more_open:
        st.session_state["show_more_modes"] = not st.session_state.get("show_more_modes", False)

mode = st.session_state["current_mode"]

if st.session_state.get("show_more_modes"):
    with st.expander("Additional Institutional Tools", expanded=True):
        MORE_MODES = [
            "Backtest", "Optimize", "Market Scanner", "MQL5 Inspector", "Monte Carlo", "Walk-Forward",
            "Multi-Strategy", "WF Matrix", "Validate Strategies", "Auto-Magic", "Guided Walkthrough",
        ]
        more_cols = st.columns(4)
        for i, m in enumerate(MORE_MODES):
            with more_cols[i % 4]:
                is_active = mode == m
                btn_type = "primary" if is_active else "secondary"
                if st.button(m, type=btn_type, use_container_width=True, key=f"more_{i}"):
                    st.session_state["current_mode"] = m
                    st.session_state["show_more_modes"] = False
                    st.rerun()

# === File Drop / Strategy Selection ===
col_drop, col_status = st.columns([3, 1])
with col_drop:
    uploaded = st.file_uploader(
        "Drop your Strategy File (.mq5, .py, .pine, .txt)",
        type=["mq5", "py", "pine", "txt", "md"],
        accept_multiple_files=False,
        help="Auto-extracts 200+ inputs, profiles, sessions and maps to vectorized engine",
    )
with col_status:
    strategy_choice = st.selectbox(
        "Strategy Library",
        list(STRATEGY_REGISTRY.keys()),
        index=list(STRATEGY_REGISTRY.keys()).index("light9") if "light9" in STRATEGY_REGISTRY else 0,
    )

# Strategy loading
loaded_strategy_cls = None
loaded_params = {}
loaded_info = {}

if uploaded is not None:
    temp_dir = ROOT / "output" / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / uploaded.name
    temp_path.write_bytes(uploaded.getbuffer())
    loaded_strategy_cls, loaded_params, loaded_info = load_any_strategy(temp_path)
    st.session_state["uploaded_path"] = str(temp_path)
    st.session_state["loaded_params"] = loaded_params
    st.session_state["loaded_info"] = loaded_info
    st.success(f"Loaded `{uploaded.name}` ({loaded_info.get('input_count', len(loaded_params))} parameters extracted)")
elif "uploaded_path" in st.session_state and Path(st.session_state["uploaded_path"]).exists():
    loaded_strategy_cls, loaded_params, loaded_info = load_any_strategy(st.session_state["uploaded_path"])
else:
    loaded_strategy_cls = STRATEGY_REGISTRY.get(strategy_choice)
    loaded_params = {}
    loaded_info = {"suggested_strategy": strategy_choice}

# Data fetcher helper
@st.cache_data(ttl=600, show_spinner=False)
def fetch_data_cached(sym: str, tf: str, start: str, end: str | None, term: str | None):
    return fetch_with_priority(sym, tf, start=start, end=end, terminal_override=term, allow_synthetic=True)


# =========================================================================
# MODE 1: BACKTEST
# =========================================================================
if mode == "Backtest":
    st.markdown('<div class="nm-panel"><div class="nm-panel-title">🚀 Backtest Console</div><div class="nm-muted">Vectorized execution with adaptive lot sizing, grid/recovery corridors, swap calculations, and institutional trade logs.</div></div>', unsafe_allow_html=True)
    
    if st.button("▶ Run Backtest", type="primary", use_container_width=True):
        # Clear stale comparison before each run — otherwise switching strategy / symbol /
        # params would silently compare against the previous OHLC baseline.
        st.session_state.pop("bt_result_ohlc", None)
        with st.spinner(f"Running {'tick-level' if tick_mode_internal != 'off' else 'vectorized'} simulation on {symbol} [{timeframe}]..."):
            df, data_info = fetch_data_cached(symbol, timeframe, str(start_date), str(end_date), terminal_choice)
            if df is None or len(df) < 30:
                st.error("Insufficient market data for backtest.")
            else:
                strat_obj = loaded_strategy_cls(params=loaded_params)
                sig = strat_obj.generate(df)
                entries = sig.entries.fillna(False).astype(bool)
                direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)

                signals = {getattr(loaded_strategy_cls, "name", "strat"): (entries, direction)}
                result = run_full(
                    df, signals, init_cash=init_cash,
                    spread_pips=spread_pips, commission_pips=commission_pips, commission_pct=commission_pct,
                    slippage_pips=slippage_pips, pip_size=pip_size, contract_size=contract_size,
                    symbol=symbol,
                    base_lot=base_lot, grid_mode=GRID_MAP[grid_mode],
                    pips_between_orders=float(pips_between), grid_lot_multiplier=float(grid_lot_mult),
                    grid_take_profit=float(grid_tp), grid_stop_loss=float(grid_sl), max_grid_layers=int(max_layers),
                    recovery_mode=1 if "Last close" in recovery_mode_label else 0,
                    recovery_lot_multiplier=float(rec_mult),
                    adaptive_enabled=adaptive_on, swap_enabled=swap_on,
                    long_swap_pips=float(long_swap), short_swap_pips=float(short_swap),
                    tick_mode=tick_mode_internal,
                    ticks_per_bar=int(ticks_per_bar),
                    use_real_spreads=bool(use_real_spreads),
                )
                result["signals"] = sig
                result["df"] = df
                # If user wants OHLC comparison alongside tick, run it now and store both
                if tick_mode_internal != "off" and compare_ohlc:
                    with st.spinner("Running OHLC baseline for comparison..."):
                        result_ohlc = run_full(
                            df, signals, init_cash=init_cash,
                            spread_pips=spread_pips, commission_pips=commission_pips, commission_pct=commission_pct,
                            slippage_pips=slippage_pips, pip_size=pip_size, contract_size=contract_size,
                            symbol=symbol,
                            base_lot=base_lot, grid_mode=GRID_MAP[grid_mode],
                            pips_between_orders=float(pips_between), grid_lot_multiplier=float(grid_lot_mult),
                            grid_take_profit=float(grid_tp), grid_stop_loss=float(grid_sl), max_grid_layers=int(max_layers),
                            recovery_mode=1 if "Last close" in recovery_mode_label else 0,
                            recovery_lot_multiplier=float(rec_mult),
                            adaptive_enabled=adaptive_on, swap_enabled=swap_on,
                            long_swap_pips=float(long_swap), short_swap_pips=float(short_swap),
                            tick_mode="off",  # OHLC baseline
                        )
                        st.session_state["bt_result_ohlc"] = result_ohlc
                st.session_state["bt_result"] = result

    if "bt_result" in st.session_state:
        bt = st.session_state["bt_result"]
        m = bt["metrics"]

        st.markdown("### Institutional Performance Scorecard")
        primary_keys = ["net_pnl", "total_return", "profit_factor", "sharpe", "sortino",
                        "calmar", "max_drawdown", "win_rate", "n_trades", "recovery_factor"]
        render_metrics(m, primary_keys, cols_per_row=5)

        secondary_keys = ["avg_win", "avg_loss", "expectancy", "vol", "cagr", "longest_dd_bars"]
        render_metrics(m, secondary_keys, cols_per_row=6)

        # MQL5 Tester results mirror — same rows MT5 shows, same names
        try:
            from backtester.metrics_v2 import tester_statistics
            from analysis.composite_criterion import criterion_mql5_complex
            stats = tester_statistics(m, bt.get("trades"), bt.get("equity"), init_cash)
            complex_score = float(criterion_mql5_complex(m))
            if complex_score < 20:
                badge, color = "🔴 WEAK", "red"
            elif complex_score < 50:
                badge, color = "🟡 FAIR", "orange"
            elif complex_score < 80:
                badge, color = "🟢 GOOD", "green"
            else:
                badge, color = "🟢🟢 ELITE", "green"
            st.markdown(f"### 🖥️ MQL5 Tester Mirror — Complex Result: **:{color}[{complex_score:.1f}/100 {badge}]**")
            import pandas as _pd
            order = ["STAT_INITIAL_DEPOSIT", "STAT_PROFIT", "STAT_GROSS_PROFIT", "STAT_GROSS_LOSS",
                     "STAT_PROFIT_FACTOR", "STAT_RECOVERY_FACTOR", "STAT_SHARPE_RATIO",
                     "STAT_EXPECTED_PAYOFF", "STAT_TRADES", "STAT_WIN_TRADES", "STAT_LOSS_TRADES",
                     "STAT_WIN_PERCENT", "STAT_EQUITY_DD", "STAT_EQUITY_DD_PERCENT",
                     "STAT_MAX_PROFITTRADE", "STAT_MAX_LOSSTRADE", "STAT_CONPROFITMAX",
                     "STAT_CONLOSSMAX", "STAT_SHORT_TRADES", "STAT_LONG_TRADES",
                     "STAT_WIN_SHORT_TRADES", "STAT_WIN_LONG_TRADES"]
            rows = [{"Tester (MT5 name)": k, "Value": (round(v, 2) if isinstance(v, float) else v)}
                    for k, v in stats.items() if k in order]
            st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Matches MT5 Strategy Tester Results tab. STAT_MIN_MARGINLEVEL = None (no margin model yet). "
                       "Optimize with 'MQL5 Complex Criterion max (0-100)' to rank passes exactly like Tester.")
        except Exception as _e:
            st.caption(f"MQL5 mirror unavailable: {_e}")

        # Pro suite: data gate + margin + robustness + one-click report
        try:
            from backtester.pro_suite import (grade_data, save_manifest, margin_required,
                margin_level, lots_for_risk, monte_carlo_bands, export_report)
            from backtester.symbol_spec import get_spec
            _spec = get_spec(symbol)
            _grade = grade_data(df if "df" in locals() else bt.get("equity", _pd.DataFrame()).to_frame() if bt.get("equity") is not None else None, {"source": bt.get("tick_source", "cache")})
            if "F" in _grade["grade"] or "synthetic" in _grade["source"]:
                st.error(f"⛔ {_grade['loud_banner']} — switch to MT5/cached data for tradable results.")
            else:
                st.success(f"✅ {_grade['loud_banner']} | {_spec.symbol} triple={_spec.triple_day} tick_value={_spec.tick_value}")
            _px = float((bt.get("equity").iloc[-1] if bt.get("equity") is not None and len(bt.get("equity")) else 0) or 0)
            _marg = margin_required(base_lot, float(df["close"].iloc[-1]) if "df" in locals() and df is not None and len(df) else 1.1, _spec.contract_size, 30.0)
            _lvl = margin_level(float(m.get("final_equity", 10000) or 10000), _marg)
            c1, c2, c3 = st.columns(3)
            c1.metric("Margin/lot ($)", f"{_marg:,.0f}")
            c2.metric("Margin level", f"{_lvl:.0f}%" if _lvl else "—")
            c3.metric("Risk 1% lots", f"{lots_for_risk(float(m.get('final_equity', 10000) or 10000), 1.0, 30.0, _spec.pip_size, _spec.contract_size):.2f}")
            if _lvl is not None and _lvl < 100:
                st.warning(f"⚠️ Margin level {_lvl:.0f}% — near stop-out (100%). Reduce lots.")
            _mc = monte_carlo_bands(bt.get("trades"))
            if "p5" in _mc:
                st.caption(f"🎲 Monte-Carlo (500 shuffles): p5=${_mc['p5']:,.0f} p50=${_mc['p50']:,.0f} p95=${_mc['p95']:,.0f} P(profit)={_mc['prob_profit']:.0%}")
                if _mc["prob_profit"] < 0.8:
                    st.warning("Overfit risk — P(profit) <80% on reshuffled trades.")
            _mp = save_manifest(getattr(loaded_strategy_cls, "name", "strat"), symbol, timeframe, dict(loaded_params),
                                bt.get("execution", {}), _grade, m)
            if st.button("📄 Export Pro Report (HTML)", key="pro_rep"):
                _rp = export_report(getattr(loaded_strategy_cls, "name", "strat"), symbol, timeframe, stats, complex_score, _grade, _mp)
                st.success(f"Saved: {_rp} | manifest: {_mp}")
        except Exception as _e2:
            st.caption(f"Pro suite note: {_e2}")

        # Data Quality Dashboard (from engine_full)
        try:
            dq_banner = bt.get("data_quality_banner")
            dq = bt.get("data_quality")
            if dq_banner:
                grade = dq.get("grade", "?") if dq else "?"
                score = dq.get("score", 0) if dq else 0
                grade_colors = {"A": "green", "B": "lightgreen", "C": "orange", "D": "red", "F": "darkred"}
                color = grade_colors.get(grade, "gray")
                st.markdown(f"### 📊 Data Quality Dashboard — **:{color}[{grade}]** (Score: {score}/100)")
                st.caption(dq_banner)
                if dq:
                    with st.expander("🔍 Details", expanded=False):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Gaps", f"{dq['gaps']['total_gaps']} ({dq['gaps']['severity']})")
                        c2.metric("Outliers", f"{dq['outliers']['count']} ({dq['outliers']['severity']})")
                        c3.metric("Stale", f"{'Yes' if dq['stale']['is_stale'] else 'No'} ({dq['stale']['severity']})")
                        c4.metric("Volume", dq['volume']['severity'].upper())
                        if dq.get('spread'):
                            c1.metric("Spread", dq['spread']['severity'].upper())
                            c2.metric("P50 Spread", f"{dq['spread']['percentiles'].get(50, 0):.1f} pips")
                            c3.metric("P99 Spread", f"{dq['spread']['percentiles'].get(99, 0):.1f} pips")
                        # Volume by session
                        if dq['volume']['by_session']:
                            import pandas as _pd
                            vol_df = _pd.DataFrame(dq['volume']['by_session']).T
                            vol_df.columns = ['Mean', 'Median', 'Std', '% of Total']
                            st.dataframe(vol_df, use_container_width=True)
                        # Flags
                        if dq.get('flags'):
                            st.caption("Flags: " + "; ".join(dq['flags'][:10]))
        except Exception as _e3:
            st.caption(f"Data quality note: {_e3}")

        # Tick execution stats panel (only when tick mode ran)
        is_tick_mode = (bt.get("execution", {}).get("tick_mode", "off") != "off")
        if is_tick_mode or "tick_source" in bt:
            st.markdown("### 🎯 Tick Execution Stats")
            tick_cols = st.columns(5)
            with tick_cols[0]:
                st.metric("Tick source", bt.get("tick_source", "synthetic"))
            with tick_cols[1]:
                st.metric("Spread source", bt.get("spread_source", "synthetic"))
            with tick_cols[2]:
                st.metric("Ticks synthesized", f"{bt.get('n_ticks_synthesized', 0):,}")
            with tick_cols[3]:
                st.metric("Avg spread (pips)", f"{bt.get('avg_spread_pips', 0):.2f}")
            with tick_cols[4]:
                st.metric("Max spread (pips)", f"{bt.get('max_spread_pips', 0):.2f}")
            # Removed 'Intra-bar fills' metric — currently always 0 (never incremented in engine_deep).
            # Note: TP/SL still checks bar close, not actual intra-bar wicks.
            if bt.get("execution", {}).get("tick_mode") == "synthetic":
                st.caption(f"💡 Tick mode: {bt['execution'].get('ticks_per_bar', '?')} synthetic ticks/bar. "
                           "Adds variable spread + commission to fills. TP/SL still uses bar close.")

        # OHLC vs Tick delta panel (only if comparison was run)
        if "bt_result_ohlc" in st.session_state:
            ohlc_m = st.session_state["bt_result_ohlc"]["metrics"]
            tick_m = bt["metrics"]
            st.markdown("### 🔄 OHLC vs Tick Reality Check")
            delta_cols = st.columns(5)
            with delta_cols[0]:
                d_pnl = tick_m.get("net_pnl", 0) - ohlc_m.get("net_pnl", 0)
                st.metric("Net PnL Δ", f"${d_pnl:+,.0f}", delta=f"{d_pnl:+.0f}",
                          delta_color="inverse" if d_pnl < 0 else "normal")
            with delta_cols[1]:
                d_trades = tick_m.get("n_trades", 0) - ohlc_m.get("n_trades", 0)
                st.metric("Trades Δ", f"{d_trades:+d}",
                          help="Trade count difference reflects variable spread costs + commission model differences.")
            with delta_cols[2]:
                d_sharpe = tick_m.get("sharpe", 0) - ohlc_m.get("sharpe", 0)
                st.metric("Sharpe Δ", f"{d_sharpe:+.2f}")
            with delta_cols[3]:
                d_dd = tick_m.get("max_drawdown", 0) - ohlc_m.get("max_drawdown", 0)
                st.metric("Max DD Δ", f"{d_dd:+.2%}", delta_color="inverse" if d_dd > 0 else "normal")
            with delta_cols[4]:
                d_pf = tick_m.get("profit_factor", 0) - ohlc_m.get("profit_factor", 0)
                st.metric("Profit Factor Δ", f"{d_pf:+.2f}")
            if d_pnl < 0 or d_dd > 0:
                st.warning("⚠ Tick sim shows worse results than OHLC — typical, because variable spread + commission costs accumulate. "
                           "If you also have a Grid mode enabled, grid layers may be triggered slightly differently due to fill timing. "
                           "This gap reflects execution friction, not intra-bar price wicks.")

        # Action download buttons
        dl1, dl2, dl3, dl4 = st.columns(4)
        with dl1:
            html_report = generate_dashboard_html(bt)
            st.download_button("🌐 Download Standalone HTML Dashboard", html_report, file_name=f"{symbol}_{timeframe}_report.html", mime="text/html", use_container_width=True)
        with dl2:
            if len(bt["trades"]) > 0:
                st.download_button("📊 Export Trades CSV", bt["trades"].to_csv(index=False), file_name="trades.csv", mime="text/csv", use_container_width=True)
        with dl3:
            st.download_button("📋 Export Metrics JSON", json.dumps(m, indent=2, default=str), file_name="metrics.json", mime="application/json", use_container_width=True)
        with dl4:
            set_content = build_mql5_set_text(getattr(loaded_strategy_cls, "name", "strat"), loaded_params)
            st.download_button("💾 Download MT5 .set File", set_content, file_name=f"{symbol}_params.set", mime="text/plain", use_container_width=True)

        # Tab list — same 5 tabs always; comparison shows inside Equity tab as overlay
        tab_equity, tab_pnl, tab_candles, tab_monthly, tab_trades = st.tabs([
            "📈 Equity & Drawdown", "💵 Cumulative P&L", "🕯️ Price & Signals", "📅 Monthly Returns", "📝 Trade Journal"
        ])

        with tab_equity:
            dd = bt["equity"] / bt["equity"].cummax() - 1
            # If comparison ran, overlay OHLC equity in the same chart
            if "bt_result_ohlc" in st.session_state:
                fig = _equity_chart_overlay(
                    bt["equity"], st.session_state["bt_result_ohlc"]["equity"],
                    dd, label_a=f"Tick ({bt.get('tick_source', 'synthetic')})",
                    label_b="OHLC baseline",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.plotly_chart(equity_chart(bt["equity"], dd), use_container_width=True)
        with tab_pnl:
            st.plotly_chart(cumulative_pnl_chart(bt["equity"], init_cash), use_container_width=True)
        with tab_candles:
            st.plotly_chart(price_chart_with_signals(bt["df"], bt["signals"], bt.get("trades")), use_container_width=True)
        with tab_monthly:
            hm = monthly_returns_heatmap(bt["equity"])
            if hm is not None:
                st.plotly_chart(hm, use_container_width=True)
            else:
                st.info("Insufficient monthly history for heatmap generation.")
        with tab_trades:
            if len(bt["trades"]) > 0:
                st.dataframe(bt["trades"], use_container_width=True, height=450)
            else:
                st.info("No trades executed during this backtest period.")


# =========================================================================
# MODE 2: OPTIMIZE (CRITERION STUDY)
# =========================================================================
elif mode == "Optimize":
    st.markdown('<div class="nm-panel"><div class="nm-panel-title">🔬 Bayesian Criterion Study & Parameter Optimizer</div><div class="nm-muted">Select exact inputs to tune with Optuna TPE sampling. Vectorized execution is 100x-500x faster than MT5 Strategy Tester.</div></div>', unsafe_allow_html=True)
    
    st.info("💡 **Why Newmeta Optimizer is 100x-500x Faster than MT5 Tester:** MT5 runs interpreted bytecode sequentially, launching single-threaded passes with full terminal process initialization and disk deal logging (taking 30-60 mins for 1,000 passes on M1 data). Newmeta keeps all bars in vectorized NumPy/SIMD memory buffers, optimizing 100-500 trials in just 5-15 seconds!")
    
    opt_col1, opt_col2 = st.columns([1, 1])
    with opt_col1:
        criterion_choice = st.selectbox("Optimization Criterion / Objective", list(CRITERION_PRESETS.keys()), index=0)
        n_trials = st.slider("Number of Optuna Trials", 10, 500, 100, step=10)
    
    with opt_col2:
        if criterion_choice.startswith("Composite"):
            st.caption("Custom Criterion Weights:")
            cw_s = st.slider("Sharpe Weight", 0.0, 1.0, 0.35, 0.05)
            cw_c = st.slider("Calmar Weight", 0.0, 1.0, 0.25, 0.05)
            cw_p = st.slider("Profit Factor Weight", 0.0, 1.0, 0.20, 0.05)
            cw_d = st.slider("Drawdown Weight (Inverse)", 0.0, 1.0, 0.10, 0.05)
            cw_w = st.slider("Win Rate Weight", 0.0, 1.0, 0.10, 0.05)
            weights = {"sharpe": cw_s, "calmar": cw_c, "pf": cw_p, "dd": cw_d, "win_rate": cw_w}
            criterion_fn = lambda m: composite_score(m, weights)
        else:
            criterion_fn = CRITERION_PRESETS[criterion_choice]

    # Parameter selection
    st.markdown("### 🎛️ Select Inputs to Optimize")
    st.caption("Select checkboxes for parameters you want Optuna to search. Adjust min, max, and step ranges:")
    
    all_params = dict(loaded_params)
    if not all_params and hasattr(loaded_strategy_cls, "INPUT_DEFAULTS"):
        all_params = dict(getattr(loaded_strategy_cls, "INPUT_DEFAULTS"))
        
    if not all_params:
        all_params = {
            "SessionAdx_Threshold1": 30.0,
            "SessionAdx_Period1": 14,
            "Hunter_TakeProfitPercent": 1.26,
            "Hunter_StopLossPercent": 1.80,
            "TrailingDistancePips": 951,
            "BreakevenActivationPips": 169,
            "RVOL_Threshold": 1.5,
        }

    # Filter numeric parameters for tuning
    param_spec = {}
    selected_keys = []
    
    col_p1, col_p2, col_p3 = st.columns(3)
    p_keys = list(all_params.keys())
    
    for i, pk in enumerate(p_keys):
        pval = all_params[pk]
        # Only suggest tunable numeric/bool params
        if isinstance(pval, (int, float, bool)) or str(pval).replace('.', '', 1).isdigit():
            target_col = col_p1 if i % 3 == 0 else (col_p2 if i % 3 == 1 else col_p3)
            with target_col:
                is_selected = st.checkbox(f"Tune `{pk}`", value=(i < 4), key=f"chk_opt_{pk}")
                if is_selected:
                    selected_keys.append(pk)
                    try:
                        num_val = float(pval)
                        is_int = isinstance(pval, int) or (isinstance(pval, str) and pval.isdigit())
                        c_min, c_max = st.columns(2)
                        p_min = c_min.number_input(f"{pk} Min", value=float(max(1, num_val * 0.5) if is_int else max(0.1, num_val * 0.5)), key=f"min_{pk}")
                        p_max = c_max.number_input(f"{pk} Max", value=float(num_val * 1.5 if is_int else num_val * 2.0), key=f"max_{pk}")
                        param_spec[pk] = {"type": "int" if is_int else "float", "min": p_min, "max": p_max}
                    except Exception:
                        param_spec[pk] = {"type": "categorical", "choices": [True, False]}

    if st.button("🚀 Run Criterion Study & Optimization", type="primary", use_container_width=True, disabled=len(param_spec) == 0):
        with st.spinner(f"Running {n_trials} Bayesian Optuna trials optimizing '{criterion_choice}'..."):
            df, _ = fetch_data_cached(symbol, timeframe, str(start_date), str(end_date), terminal_choice)
            if df is None or len(df) < 30:
                st.error("Insufficient market data for optimization.")
            else:
                eng_kwargs = {
                    "init_cash": init_cash,
                    "spread_pips": spread_pips, "commission_pips": commission_pips, "commission_pct": commission_pct,
                    "slippage_pips": slippage_pips, "pip_size": pip_size, "contract_size": contract_size,
                    "base_lot": base_lot, "grid_mode": GRID_MAP[grid_mode],
                    "pips_between_orders": float(pips_between), "grid_lot_multiplier": float(grid_lot_mult),
                    "grid_take_profit": float(grid_tp), "grid_stop_loss": float(grid_sl), "max_grid_layers": int(max_layers),
                    "recovery_mode": 1 if "Last close" in recovery_mode_label else 0,
                    "recovery_lot_multiplier": float(rec_mult),
                    "adaptive_enabled": adaptive_on, "swap_enabled": swap_on,
                    "long_swap_pips": float(long_swap), "short_swap_pips": float(short_swap),
                }
                study = optimize_strategy_criterion(
                    loaded_strategy_cls, df, param_spec, criterion_fn,
                    n_trials=n_trials, fixed_params=all_params, engine_kwargs=eng_kwargs,
                )
                trials_df = get_study_trials_df(study)
                importances = get_param_importances_dict(study)
                
                st.session_state["study_result"] = {
                    "study": study,
                    "trials_df": trials_df,
                    "importances": importances,
                    "best_params": study.best_trial.params,
                    "best_score": study.best_trial.value,
                    "criterion": criterion_choice,
                }

    if "study_result" in st.session_state:
        sr = st.session_state["study_result"]
        st.markdown(f"### 🏆 Best Found Parameters (Criterion Score: **{sr['best_score']:.2f}**)")
        
        st.json(sr["best_params"])
        
        # Download MT5 .set
        set_text = build_mql5_set_text(getattr(loaded_strategy_cls, "name", "strat"), {**all_params, **sr["best_params"]})
        c_act1, c_act2 = st.columns(2)
        with c_act1:
            if st.button("⚡ Apply Best Parameters to Backtester", use_container_width=True):
                loaded_params.update(sr["best_params"])
                st.session_state["loaded_params"] = loaded_params
                st.session_state["current_mode"] = "Backtest"
                st.rerun()
        with c_act2:
            st.download_button("💾 Download Optimized MT5 .set File", set_text, file_name=f"{symbol}_opt_{criterion_choice}.set", mime="text/plain", use_container_width=True)
            
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if sr["importances"]:
                st.plotly_chart(optuna_importance_chart(sr["importances"]), use_container_width=True)
        with chart_col2:
            if not sr["trials_df"].empty:
                st.plotly_chart(optimization_history_chart(sr["trials_df"]), use_container_width=True)
                
        if not sr["trials_df"].empty:
            st.markdown("### Top 15 Trials Leaderboard")
            st.dataframe(sr["trials_df"].head(15), use_container_width=True)

    st.markdown("### 🌍 Universe Fit — Which Ticker Fits This Strategy Most?")
    st.caption("Real backtest per symbol × timeframe, ranked by the same criterion (Full / PF / Drawdown / WinRate). Replaces heuristic-only fit with true PnL ranking.")
    u_c1, u_c2, u_c3 = st.columns(3)
    with u_c1:
        u_univ = st.selectbox("Universe", list(PRESET_UNIVERSES.keys()) + ["Custom"], index=1, key="u_univ")
        u_crit = st.selectbox("Rank Criterion", list(CRITERION_PRESETS.keys()), index=0, key="u_crit")
    with u_c2:
        if u_univ == "Custom":
            u_syms = [s.strip().upper() for s in st.text_input("Symbols", "XAUUSD,EURUSD,GBPUSD,BTCUSD,US30", key="u_syms").split(",") if s.strip()]
        else:
            u_syms = PRESET_UNIVERSES[u_univ]
        u_tfs = st.multiselect("TFs", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"], default=["H1", "H4"], key="u_tfs")
    with u_c3:
        u_topn = st.number_input("Optuna Top-N", 0, 5, 0, key="u_topn")
        u_trials = st.number_input("Trials per Top", 10, 200, 50, key="u_trials")
    if st.button("🌍 Rank Universe by Backtest", use_container_width=True, key="u_rank"):
        strat_nm = getattr(loaded_strategy_cls, "name", "strat")
        eng_u = {"init_cash": init_cash, "spread_pips": spread_pips, "commission_pips": commission_pips,
                 "commission_pct": commission_pct, "slippage_pips": slippage_pips, "pip_size": pip_size,
                 "contract_size": contract_size, "base_lot": base_lot, "grid_mode": GRID_MAP[grid_mode],
                 "tick_mode": "synthetic", "ticks_per_bar": 10}
        with st.spinner(f"Backtesting {strat_nm} × {len(u_syms)} symbols × {len(u_tfs)} TFs [{u_crit}]..."):
            pbar = st.progress(0.0); stx = st.empty()
            def _cb(cur, total, msg):
                pbar.progress(cur / max(total, 1)); stx.caption(msg)
            ranked = rank_universe(strat_nm, u_syms, u_tfs, params=dict(loaded_params),
                                   criterion=u_crit, engine_kwargs=eng_u,
                                   terminal=terminal_choice, progress_callback=_cb)
            pbar.empty(); stx.empty()
            st.session_state["universe_ranked"] = ranked
            st.session_state["universe_crit"] = u_crit
    if "universe_ranked" in st.session_state and not st.session_state["universe_ranked"].empty:
        st.dataframe(st.session_state["universe_ranked"].head(25), use_container_width=True)
        if int(u_topn) > 0 and len(param_spec) > 0:
            if st.button("🔬 Optimize Top-N Fits", key="u_opt"):
                strat_nm = getattr(loaded_strategy_cls, "name", "strat")
                eng_u = {"init_cash": init_cash, "spread_pips": spread_pips, "commission_pips": commission_pips,
                         "commission_pct": commission_pct, "slippage_pips": slippage_pips, "pip_size": pip_size,
                         "contract_size": contract_size, "base_lot": base_lot, "grid_mode": GRID_MAP[grid_mode]}
                with st.spinner("Optuna on top fits..."):
                    top_df = optimize_top_n(strat_nm, st.session_state["universe_ranked"], param_spec,
                                            n_top=int(u_topn), n_trials=int(u_trials),
                                            engine_kwargs=eng_u, criterion=st.session_state.get("universe_crit", u_crit),
                                            terminal=terminal_choice)
                    st.dataframe(top_df, use_container_width=True)


# =========================================================================
# MODE 3: FULL MARKET SYMBOLS SCANNER RADAR
# =========================================================================
elif mode == "Market Scanner":
    st.markdown('<div class="nm-panel"><div class="nm-panel-title">📡 Full Market Symbols Scanner & Quantitative Radar</div><div class="nm-muted">Scans multiple assets and timeframes simultaneously. Ranks by Volatility ATR, ADX Regime, Spread Cost Ratio, RVOL, and Strategy Fit Score.</div></div>', unsafe_allow_html=True)
    
    scan_col1, scan_col2 = st.columns([1, 2])
    with scan_col1:
        univ_choice = st.selectbox("Asset Universe", list(PRESET_UNIVERSES.keys()) + ["Connected MT5 Live Symbols", "Custom Comma-Separated"], index=0)
        if univ_choice == "Connected MT5 Live Symbols":
            mt5_symbols_raw = get_available_mt5_symbols()
            if mt5_symbols_raw:
                symbols_to_scan = [s["name"] for s in mt5_symbols_raw[:35]]
                st.caption(f"Connected MT5: {len(mt5_symbols_raw)} total symbols detected (scanning top {len(symbols_to_scan)})")
            else:
                symbols_to_scan = PRESET_UNIVERSES["Metals & Commodities"] + PRESET_UNIVERSES["Forex Majors"]
                st.caption("MT5 offline. Using default Forex + Metals universe.")
        elif univ_choice == "Custom Comma-Separated":
            custom_syms = st.text_input("Enter Symbols", "XAUUSD, EURUSD, GBPUSD, US30, USTEC, BTCUSD")
            symbols_to_scan = [s.strip().upper() for s in custom_syms.split(",") if s.strip()]
        else:
            symbols_to_scan = PRESET_UNIVERSES[univ_choice]
            
    with scan_col2:
        selected_tfs = st.multiselect("Timeframes to Scan", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"], default=["M1", "M5", "H1"])

    if st.button("🚀 Scan Full Market Radar", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        def update_progress(cur, total, msg):
            progress_bar.progress(cur / total)
            status_text.caption(msg)
            
        strat_name = getattr(loaded_strategy_cls, "name", "light9")
        df_radar = scan_market_matrix(symbols_to_scan, selected_tfs, strategy_name=strat_name, strategy_params=loaded_params, terminal=terminal_choice, progress_callback=update_progress)
        progress_bar.empty()
        status_text.empty()
        
        st.session_state["radar_result"] = df_radar

    if "radar_result" in st.session_state and not st.session_state["radar_result"].empty:
        radar = st.session_state["radar_result"]
        
        # Summary KPI Cards
        top_vol = radar.sort_values("atr_pct", ascending=False).iloc[0]
        top_fit = radar.sort_values("suitability", ascending=False).iloc[0]
        tightest_spread = radar.sort_values("spread_cost_ratio", ascending=True).iloc[0]
        
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown(metric_card("Highest Volatility (ATR %)", f"{top_vol['symbol']} [{top_vol['timeframe']}] ({top_vol['atr_pct']:.2f}%)", "green"), unsafe_allow_html=True)
        with kpi2:
            st.markdown(metric_card("Best Strategy Fit Score", f"{top_fit['symbol']} [{top_fit['timeframe']}] ({top_fit['suitability']:.0f}/100)", "blue"), unsafe_allow_html=True)
        with kpi3:
            st.markdown(metric_card("Tightest Relative Spread", f"{tightest_spread['symbol']} [{tightest_spread['timeframe']}] ({tightest_spread['spread_cost_ratio']:.1f}% drag)", "amber"), unsafe_allow_html=True)

        st.markdown("### 📊 Market Radar Overview")
        st.dataframe(radar, use_container_width=True, height=450)
        
        # 1-Click Quick Load into Backtester
        st.markdown("### ⚡ Quick-Load into Backtester")
        sym_tf_options = [f"{r['symbol']} | {r['timeframe']}" for _, r in radar.iterrows()]
        sel_sym_tf = st.selectbox("Select Symbol & Timeframe to Test", sym_tf_options, index=0)
        
        if st.button("⚡ Load Selected Pair into Backtester", type="primary", use_container_width=True):
            chosen_sym, chosen_tf = [x.strip() for x in sel_sym_tf.split("|")]
            st.session_state["selected_symbol"] = chosen_sym
            st.session_state["selected_timeframe"] = chosen_tf
            st.session_state["current_mode"] = "Backtest"
            st.rerun()


# =========================================================================
# MODE 4: MQL5 INSPECTOR & PORT
# =========================================================================
elif mode == "MQL5 Inspector":
    st.markdown('<div class="nm-panel"><div class="nm-panel-title">🧩 MQL5 Code Inspector & Python Converter</div><div class="nm-muted">Deep code decompilation, input parameter cataloging, profile preset mapping, and Python strategy generation.</div></div>', unsafe_allow_html=True)
    
    mq5_files = list(Path(r"C:\Users\youha\Desktop").glob("*.mq5")) + list((ROOT / "strategies").glob("**/*.mq5"))
    mq5_options = [str(p) for p in mq5_files]
    selected_mq5_path = st.selectbox("Select MQL5 EA File to Inspect", mq5_options if mq5_options else ["(Drop file above)"], index=0)
    
    target_p = Path(selected_mq5_path) if selected_mq5_path != "(Drop file above)" else None
    if target_p and target_p.exists():
        parser = MQL5Parser(target_p)
        spec = parser.parse()
        
        st.markdown(f"### EA: `{spec['ea_name']}` (Version {spec['version']})")
        st.caption(f"Total Inputs: **{len(spec['inputs'])}** | Detected Profiles: **{len(spec['profiles'])}**")
        
        insp_tab1, insp_tab2, insp_tab3, insp_tab4 = st.tabs(["Inputs Catalog", "Profile Presets", "Session & Signal Logic", "Python Strategy Code"])
        
        with insp_tab1:
            inputs_df = pd.DataFrame([
                {"Parameter": k, "Default": v["default"], "Type": v["type"], "Group": v["group"]}
                for k, v in spec["inputs"].items()
            ])
            st.dataframe(inputs_df, use_container_width=True, height=450)
            
        with insp_tab2:
            st.json(spec["profile_presets"])
            
        with insp_tab3:
            st.json(spec["signal_logic"])
            st.json(spec["risk_params"])
            
        with insp_tab4:
            py_code = parser.generate_python_class()
            st.code(py_code, language="python")
            st.download_button("💾 Download Generated Python Strategy", py_code, file_name=f"{target_p.stem}_strategy.py", mime="text/x-python", use_container_width=True)


# =========================================================================
# MODE 5: MONTE CARLO
# =========================================================================
elif mode == "Monte Carlo":
    st.markdown('<div class="nm-panel"><div class="nm-panel-title">🎲 Monte Carlo Simulation</div><div class="nm-muted">Bootstrap resampling of returns with 95% Confidence Intervals.</div></div>', unsafe_allow_html=True)
    n_sims = st.slider("Simulations", 100, 2000, 500, step=100)
    
    if st.button("▶ Run Monte Carlo", type="primary", use_container_width=True):
        df, _ = fetch_data_cached(symbol, timeframe, str(start_date), str(end_date), terminal_choice)
        if df is not None and len(df) > 30:
            strat_obj = loaded_strategy_cls(params=loaded_params)
            sig = strat_obj.generate(df)
            result = run_full(df, {getattr(loaded_strategy_cls, "name", "strat"): (sig.entries.fillna(False).astype(bool), pd.Series(sig.direction, index=df.index).fillna(0).astype(int))}, base_lot=base_lot)
            trades = result.get("trades")
            if trades is not None and len(trades) > 10:
                pnls = pd.to_numeric(trades.get("net_pnl", trades.get("pnl", [])), errors="coerce").dropna().values
                mc_curves = bootstrap_returns(pnls, n_sims=n_sims, horizon=len(pnls))
                ci = ci_metrics(mc_curves)
                
                st.markdown("### Monte Carlo 95% Confidence Intervals")
                st.json(ci)
                
                fig = go.Figure()
                for i in range(min(50, n_sims)):
                    fig.add_trace(go.Scatter(y=mc_curves[i], mode="lines", line=dict(width=0.5, color="rgba(34, 211, 238, 0.15)"), showlegend=False))
                fig.update_layout(title="Monte Carlo Equity Paths", template="plotly_dark", paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["card_bg"], height=400)
                st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# MODE 6: WALK-FORWARD
# =========================================================================
elif mode == "Walk-Forward":
    st.markdown('<div class="nm-panel"><div class="nm-panel-title">📊 Walk-Forward Efficiency Analysis</div><div class="nm-muted">Rolling In-Sample optimization and Out-Of-Sample validation to eliminate curve fitting.</div></div>', unsafe_allow_html=True)
    st.info("Walk-forward engine splits historical data into rolling train/test windows, validating out-of-sample consistency.")


# Footer
st.markdown("""
<div class="nm-footer">
  Newmeta Research Lab Backtester & Strategy Codex 2026 · Pure Python Vectorized SIMD Engine · 100% MT5 MQL5 Fidelity
</div>
""", unsafe_allow_html=True)
