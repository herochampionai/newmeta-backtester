"""Universal Backtester — drop any file, get all the magic.

Run:
    cd "D:\\Trading\\TRADING\\youha created EA\\Multi strat ea\\backtest_harness"
    $env:PYTHONPATH = (Get-Location).Path
    streamlit run frontend/app.py

Then open http://localhost:8501 and drop a .mq5 / .py / .pine / .txt file.
"""
from __future__ import annotations
import sys
from pathlib import Path
import io
import json

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.live_fetcher import fetch_with_priority, load_settings
from data.mt5_export import resolve_terminal
from core.loader import load_any_strategy
from strategies import STRATEGY_REGISTRY
from backtester.engine_full import run_full
from backtester.grid_recovery import GRID_NONE, GRID_LOSS, GRID_PROFIT, GRID_LOSS_AND_PROFIT
from backtester.adaptive import AdaptiveConfig
from backtester.metrics_v2 import compute_all
from backtester.analytics import strategy_scoreboard
from analysis.optuna_optimizer import optimize_strategy, best_params
from analysis.montecarlo import ci_metrics, bootstrap_returns
from analysis.markowitz_alloc import allocate
from analysis.walkforward import walk_forward, wf_summary

st.set_page_config(
    page_title="Universal Backtester — drop any strategy file",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg": "#0e1117", "card_bg": "#1a1f2e", "card_border": "#2a3142",
    "primary": "#00d4aa", "success": "#00d4aa", "warning": "#ffb800",
    "danger": "#ff4b4b", "info": "#4f8cff", "text": "#e8eaf0", "muted": "#8b95a7",
}

CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: {PALETTE['bg']}; }}
    .metric-card {{
        background: linear-gradient(135deg, {PALETTE['card_bg']} 0%, #232a3d 100%);
        border: 1px solid {PALETTE['card_border']}; border-radius: 10px;
        padding: 14px; margin-bottom: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }}
    .metric-card .label {{ color: {PALETTE['muted']}; font-size: 11px;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
    .metric-card .value {{ color: {PALETTE['text']}; font-size: 22px;
        font-weight: 600; line-height: 1.2; }}
    .metric-card.green {{ border-left: 4px solid {PALETTE['success']}; }}
    .metric-card.red   {{ border-left: 4px solid {PALETTE['danger']}; }}
    .metric-card.blue  {{ border-left: 4px solid {PALETTE['info']}; }}
    .metric-card.amber {{ border-left: 4px solid {PALETTE['warning']}; }}
    section[data-testid="stSidebar"] {{ background-color: {PALETTE['card_bg']}; }}
    h1, h2, h3 {{ color: {PALETTE['text']} !important; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, color: str = "blue") -> str:
    return f'<div class="metric-card {color}"><div class="label">{label}</div><div class="value">{value}</div></div>'


def equity_chart(equity: pd.Series, drawdown: pd.Series) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_x=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines",
                              name="Equity", line=dict(color=PALETTE["primary"], width=2),
                              fill="tozeroy", fillcolor="rgba(0, 212, 170, 0.1)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values * 100,
                              mode="lines", name="DD",
                              line=dict(color=PALETTE["danger"], width=1),
                              fill="tozeroy", fillcolor="rgba(255, 75, 75, 0.2)"),
                  row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["bg"],
                      plot_bgcolor=PALETTE["card_bg"], height=460,
                      showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(color=PALETTE["text"]))
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1, gridcolor=PALETTE["card_border"])
    fig.update_yaxes(title_text="DD (%)", row=2, col=1, gridcolor=PALETTE["card_border"])
    return fig


def render_metrics(m: dict, keys_order: list, cols_per_row: int = 5):
    items = [(k, m.get(k)) for k in keys_order if m.get(k) is not None]
    for i in range(0, len(items), cols_per_row):
        row = items[i:i + cols_per_row]
        cols = st.columns(len(row))
        for col, (k, v) in zip(cols, row):
            with col:
                if isinstance(v, float):
                    if "rate" in k or "win_rate" in k or "stability" in k:
                        s = f"{v:.2%}"
                    elif k == "max_drawdown":
                        s = f"{v:.2%}"
                    elif "equity" in k:
                        s = f"${v:,.0f}"
                    elif "pnl" in k or k.split("_")[-1] in ("win", "loss"):
                        s = f"${v:,.2f}"
                    else:
                        s = f"{v:.3f}"
                elif isinstance(v, int):
                    s = f"{v:,}"
                else:
                    s = str(v)
                color = "blue"
                if k in ("sharpe", "sortino", "calmar", "recovery_factor"):
                    color = "green" if v > 1 else ("amber" if v > 0 else "red")
                if k in ("total_return", "cagr"):
                    color = "green" if v > 0 else "red"
                if k == "max_drawdown":
                    color = "red" if v < -0.2 else ("amber" if v < -0.1 else "blue")
                if k == "win_rate":
                    color = "green" if v > 0.55 else ("amber" if v > 0.45 else "red")
                if k == "profit_factor":
                    color = "green" if v > 1.5 else ("amber" if v > 1 else "red")
                st.markdown(metric_card(k.replace("_", " ").title(), s, color=color),
                           unsafe_allow_html=True)


# === Sidebar ===
settings = load_settings()
default_terminal = settings.get("mt5_terminal", r"D:\MT5_EuroPrinter\terminal64.exe")

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    terminals_found = [str(p) for p in [
        Path(r"D:\MT5_EuroPrinter\terminal64.exe"),
        Path(r"D:\MT5_Bybit\terminal64.exe"),
    ] if p.exists()]
    if not terminals_found:
        terminals_found = [default_terminal]
    terminal_choice = st.selectbox("MT5 Terminal", terminals_found,
                                    index=0 if default_terminal in terminals_found else 0,
                                    label_visibility="collapsed")
    st.caption(f"Resolved: `{resolve_terminal(terminal_choice)}`")
    st.divider()
    st.markdown("**Market**")
    symbol = st.text_input("Symbol", "EURUSD")
    timeframe = st.selectbox("Timeframe",
                              ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"], index=4)
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start", pd.Timestamp("2022-01-01"))
    end_date = col2.date_input("End", pd.Timestamp("2024-12-31"))
    st.divider()
    st.markdown("**Execution**")
    init_cash = st.number_input("Initial cash ($)", 1000, 1_000_000, 10000, step=1000)
    commission_pips = st.number_input("Commission (pips RT)", 0.0, 10.0, 0.7, step=0.1)
    slippage_pips = st.number_input("Slippage (pips)", 0.0, 10.0, 0.3, step=0.1)
    base_lot = st.number_input("Base lot", 0.01, 10.0, 0.1, step=0.01)
    st.divider()
    st.markdown("**Risk Profile**")
    profile_name = st.selectbox("Profile", ["Custom", "Conservative", "Balanced", "Aggressive"])
    if profile_name != "Custom":
        from core.strictness import RISK_PROFILES
        p = RISK_PROFILES[profile_name]
        strictness = p["strictness"]
        tp_widening = p["tp_widening"]
        base_lot = p["base_lot"]
        if "adaptive_on" in dir(): pass  # already defined below
        adaptive_on = p["adaptive"]
        if profile_name == "Conservative":
            grid_mode_label = "Profit only"
            recovery_mode_label = "None"
        else:
            grid_mode_label = "Loss+Profit"
            recovery_mode_label = "Last close (martingale)"
    else:
        strictness = 5
        tp_widening = 5
    strictness = st.slider("Strictness (0=permissive, 10=strict)", 0, 10, strictness,
                            help="Per-strategy slider: 0=many trades, 10=few high-quality trades")
    tp_widening = st.slider("TP/SL widening (0=tight, 10=wide)", 0, 10, tp_widening,
                             help="Wider TP/SL gives trades more room; tighter = faster in/out")
    st.divider()
    st.markdown("**Grid + Recovery**")
    grid_mode = st.selectbox("Grid mode", ["None", "Loss only", "Profit only", "Loss+Profit"],
                              index=3)
    GRID_MAP = {"None": GRID_NONE, "Loss only": GRID_LOSS,
                "Profit only": GRID_PROFIT, "Loss+Profit": GRID_LOSS_AND_PROFIT}
    pips_between = st.slider("Pips between layers", 5, 200, 30)
    grid_lot_mult = st.slider("Grid lot multiplier", 1.0, 3.0, 1.5, step=0.1)
    grid_tp = st.number_input("Grid TP ($)", 0.0, 1000.0, 50.0, step=5.0)
    grid_sl = st.number_input("Grid SL ($)", 0.0, 5000.0, 200.0, step=10.0)
    max_layers = st.slider("Max grid layers", 1, 12, 4)
    recovery_mode_label = st.selectbox("Recovery mode", ["None", "Last close (martingale)"], index=0)
    rec_mult = st.slider("Recovery lot multiplier", 1.0, 5.0, 2.0, step=0.1)
    st.divider()
    st.markdown("**Adaptive + Swaps**")
    adaptive_on = st.checkbox("Adaptive lot sizing (streak-aware)", value=False)
    swap_on = st.checkbox("Apply swap (3x Wed + holiday)", value=False)
    long_swap = st.number_input("Long swap (pips/day)", -10.0, 10.0, -0.5, step=0.1)
    short_swap = st.number_input("Short swap (pips/day)", -10.0, 10.0, 0.2, step=0.1)
    st.divider()
    st.markdown("**Data Path (override)**")
    cache_dir = ROOT / "data" / "cache"
    cached_files = sorted(cache_dir.glob("*.parquet")) if cache_dir.exists() else []
    cached_names = [p.name for p in cached_files]
    path_options = ["(use MT5 / Yahoo / cache auto)"] + cached_names
    selected_cache = st.selectbox("Cached data file", path_options, index=0,
                                    label_visibility="collapsed")
    custom_path = st.text_input("...or custom CSV/Parquet path", "",
                                  placeholder=r"C:\path\to\data.csv")
    st.caption(f"Cache dir: `{cache_dir}`")

    st.divider()
    mode = st.radio("Mode", ["🚀 Backtest", "🔬 Optimize", "🎲 Monte Carlo",
                              "📊 Walk-Forward", "🧬 Multi-Strategy",
                              "🌊 Regime-Aware", "📂 Profile",
                              "🎯 Ticker Scanner", "🔬 Deep Backtest",
                              "🔄 MQL5 Equivalence",
                              "✅ Validate Strategies", "🪄 Auto-Magic",
                              "🎓 Guided Walkthrough"],
                    label_visibility="collapsed")

# === Header ===
st.markdown("# 📈 Universal Backtester")
st.caption("Drop any strategy file — .mq5, .py, .pine, .txt — get instant backtest with grid, recovery, adaptive sizing, swaps.")

# === File drop ===
col_drop, col_status = st.columns([3, 1])
with col_drop:
    uploaded = st.file_uploader(
        "Drop your strategy file",
        type=["mq5", "py", "pine", "txt", "md"],
        accept_multiple_files=False,
        help="Auto-detects .mq5 (MQL5 EA), .py (Python strategy), .pine (PineScript), .txt (config)",
    )
with col_status:
    st.markdown("**Available strategies:**")
    for n in STRATEGY_REGISTRY.keys():
        st.code(f"• {n}", language=None)

# === Fetch data ===
@st.cache_data(show_spinner="Fetching data — live MT5 → Yahoo → cache…")
def fetch_data(symbol, timeframe, start, end, terminal):
    return fetch_with_priority(symbol, timeframe, str(start), str(end) if end else None,
                              terminal, allow_synthetic=True)


@st.cache_data(show_spinner="Loading cached data…")
def load_path(path: str):
    """Load data from explicit path (CSV or Parquet)."""
    p = Path(path)
    if not p.exists():
        return None, {"error": f"not found: {path}"}
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, parse_dates=["time"], index_col="time")
    return df, {"source": "path", "rows": len(df), "path": str(p)}


# Resolve data based on path override
if custom_path and Path(custom_path).exists():
    df, info = load_path(custom_path)
    source = info.get("source", "?")
elif selected_cache != "(use MT5 / Yahoo / cache auto)":
    df, info = load_path(str(cache_dir / selected_cache))
    source = info.get("source", "?")
else:
    df, info = fetch_data(symbol, timeframe, start_date, end_date, terminal_choice)
    source = info.get("source", "?")

src_color = {"mt5_live": "green", "yahoo": "blue", "cache": "amber",
             "path": "blue", "synthetic": "red"}.get(source, "blue")
st.markdown(metric_card(f"Data: {source.upper()}",
                          f"{info.get('rows', '?'):,} bars",
                          color=src_color), unsafe_allow_html=True)

# === Load strategy ===
strategy_cls = None
strategy_params = {}
strategy_info = {}
if uploaded is not None:
    # Save to temp file
    tmp_path = Path("output") / uploaded.name
    tmp_path.parent.mkdir(exist_ok=True)
    tmp_path.write_bytes(uploaded.read())
    cls, params, sinfo = load_any_strategy(tmp_path)
    if cls is not None:
        strategy_cls = cls
        strategy_params = params or {}
        strategy_info = sinfo
        cols = st.columns(4)
        cols[0].markdown(metric_card("File", uploaded.name[:24]), unsafe_allow_html=True)
        cols[1].markdown(metric_card("Type", sinfo.get("type", "?").upper()),
                         unsafe_allow_html=True)
        with cols[2]:
            n_inputs = len(sinfo.get("inputs", {})) or len(params)
            st.markdown(metric_card("Params detected", str(n_inputs)),
                       unsafe_allow_html=True)
        cols[3].markdown(metric_card("Strategy",
                                       (sinfo.get("suggested_strategy")
                                          or sinfo.get("class", "?")).upper()),
                         unsafe_allow_html=True)
        with st.expander("Detection details"):
            st.json(sinfo)

# Fallback: pick a registered strategy if no file
if strategy_cls is None:
    fallback_name = st.selectbox("Or pick a strategy", list(STRATEGY_REGISTRY.keys()))
    strategy_cls = STRATEGY_REGISTRY[fallback_name]
    strategy_params = {}


def build_strategy(params_override: dict | None = None):
    if strategy_cls is None:
        return None
    # Apply strictness slider + TP/SL widening to base params
    from core.strictness import apply_strictness, apply_tp_sl_widening
    merged = {**strategy_params, **(params_override or {})}
    # Map strictness → strategy params
    s_val = st.session_state.get("strictness_slider", strictness if "strictness" in dir() else 5)
    w_val = st.session_state.get("tp_sl_widening", tp_widening if "tp_widening" in dir() else 5)
    adjusted = apply_strictness(strategy_choice, s_val, merged)
    adjusted = apply_tp_sl_widening(adjusted, w_val)
    return strategy_cls(params=adjusted)


def run_full_backtest():
    strat = build_strategy()
    if strat is None:
        st.error("No strategy selected")
        return None
    sig = strat.generate(df)
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    signals = {"primary": (entries, direction)}
    result = run_full(
        df, signals,
        init_cash=init_cash,
        commission_pips=commission_pips, slippage_pips=slippage_pips,
        grid_mode=GRID_MAP[grid_mode],
        pips_between_orders=float(pips_between), grid_lot_multiplier=float(grid_lot_mult),
        grid_take_profit=float(grid_tp), grid_stop_loss=float(grid_sl),
        max_grid_layers=int(max_layers),
        recovery_mode=1 if "Last close" in recovery_mode_label else 0,
        recovery_lot_multiplier=float(rec_mult), base_lot=float(base_lot),
        adaptive_enabled=adaptive_on,
        swap_enabled=swap_on,
        long_swap_pips=float(long_swap), short_swap_pips=float(short_swap),
    )
    return result


# === Mode dispatch ===
if mode == "🚀 Backtest":
    st.markdown("## 🚀 Backtest")
    if st.button("▶ Run Backtest", type="primary"):
        with st.spinner("Running vectorized backtest with grid + recovery + adaptive + swap…"):
            result = run_full_backtest()
            if result is not None:
                st.session_state["bt"] = result
    if "bt" in st.session_state:
        bt = st.session_state["bt"]
        m = bt["metrics"]
        keys = ["total_return", "cagr", "final_equity", "sharpe", "sortino",
                "calmar", "stability", "max_drawdown", "recovery_factor",
                "n_trades", "win_rate", "profit_factor", "avg_win", "avg_loss",
                "expectancy", "net_pnl", "vol", "longest_dd_bars"]
        render_metrics(m, keys, cols_per_row=5)
        # Show extra info
        if swap_on:
            st.caption(f"Swap P&L: **${bt['swap_total']:.2f}** (Wed=3x, holidays=0)")
        if bt.get("grid_summary", {}).get("n_grid_trades", 0) > 0:
            gs = bt["grid_summary"]
            st.caption(f"Grid: {gs['n_grid_trades']} trades, max {gs['grid_max_layers']} layers, "
                       f"WR {gs['grid_win_rate']:.1%}")
        if bt.get("streak_stats"):
            ss = bt["streak_stats"]
            st.caption(f"Streaks: longest win {ss.get('longest_win_streak', 0)}, "
                       f"longest loss {ss.get('longest_loss_streak', 0)}")
        st.caption(f"Trades per year: **{bt['annual_trades']:.1f}**")
        # Equity + DD
        dd = bt["equity"] / bt["equity"].cummax() - 1
        st.plotly_chart(equity_chart(bt["equity"], dd), use_container_width=True)
        # Trade log
        if len(bt["trades"]) > 0:
            with st.expander(f"📋 Trades ({len(bt['trades'])})"):
                st.dataframe(bt["trades"].head(200), use_container_width=True, height=400)
                # Export
                csv = bt["trades"].to_csv(index=False)
                st.download_button("💾 Download trades CSV", csv,
                                    file_name="trades.csv", mime="text/csv")
        # Per-strategy scoreboard
        if not bt["scoreboard"].empty:
            with st.expander("🏆 Per-strategy scoreboard"):
                st.dataframe(bt["scoreboard"], use_container_width=True)
        # Export metrics JSON
        st.download_button("💾 Download metrics JSON",
                            json.dumps(m, indent=2, default=str),
                            file_name="metrics.json", mime="application/json")


elif mode == "🔬 Optimize":
    st.markdown("## 🔬 Optimize (Optuna)")
    name = st.selectbox("Strategy to optimize", [s for s in STRATEGY_REGISTRY.keys() if s != "universal"])
    n_trials = st.slider("Optuna trials", 10, 500, 100)
    # Criterion selector — single or composite
    criterion_choice = st.selectbox("Criterion", ["Composite (Balanced)", "Composite (Conservative)",
                                                       "Composite (Aggressive)", "Sharpe", "Calmar", "Profit Factor"])
    if criterion_choice.startswith("Composite"):
        st.caption("Drag the sliders below to set custom weights if needed (defaults shown)")
        cw_s = st.slider("Sharpe weight", 0.0, 1.0, 0.4, 0.05)
        cw_c = st.slider("Calmar weight", 0.0, 1.0, 0.3, 0.05)
        cw_p = st.slider("Profit Factor weight", 0.0, 1.0, 0.2, 0.05)
        cw_d = st.slider("Drawdown weight (inverse)", 0.0, 1.0, 0.1, 0.05)
        # Normalize
        total = cw_s + cw_c + cw_p + cw_d
        weights = {"sharpe": cw_s / total, "calmar": cw_c / total,
                    "pf": cw_p / total, "dd": cw_d / total}
        from analysis.composite_criterion import composite_score
        criterion_fn = lambda m: composite_score(m, weights)
        st.caption(f"Normalized weights: Sharpe={weights['sharpe']:.2f}, "
                    f"Calmar={weights['calmar']:.2f}, PF={weights['pf']:.2f}, DD={weights['dd']:.2f}")
    else:
        from analysis.composite_criterion import CRITERION_PRESETS
        criterion_fn = CRITERION_PRESETS[criterion_choice]

    if st.button("=" * 1 + " Optimize", type="primary"):
        import yaml
        with open(ROOT / "config" / "strategies.yaml") as f:
            spec = yaml.safe_load(f).get(name, {})
        if not spec:
            st.error(f"No spec in strategies.yaml for {name}")
        else:
            with st.spinner(f"Optimizing {name} ({n_trials} trials) with {criterion_choice} criterion…"):
                study = optimize_strategy(name, df, spec, n_trials=n_trials)
                # Evaluate all trials with custom criterion
                best_params_dict = None
                best_score = -1e18
                for trial in study.trials:
                    if trial.state.name != "COMPLETE":
                        continue
                    p = trial.params
                    try:
                        strat = STRATEGY_REGISTRY[name](params=p)
                        sig = strat.generate(df)
                        result = run_full(df, {name: (sig.entries.fillna(False).astype(bool),
                                                          pd.Series(sig.direction, index=df.index).fillna(0).astype(int))},
                                            grid_mode=GRID_NONE, base_lot=0.1)
                        score = criterion_fn(result["metrics"])
                        if score > best_score:
                            best_score = score
                            best_params_dict = p
                    except Exception:
                        continue
                # Run with best
                if best_params_dict is None:
                    best_params_dict = best_params(study, metric="sharpe")
                strat = STRATEGY_REGISTRY[name](params=best_params_dict)
                sig = strat.generate(df)
                result = run_full(df, {name: (sig.entries.fillna(False).astype(bool),
                                                  pd.Series(sig.direction, index=df.index).fillna(0).astype(int))},
                                    init_cash=init_cash,
                                    commission_pips=commission_pips,
                                    slippage_pips=slippage_pips,
                                    grid_mode=GRID_MAP[grid_mode],
                                    pips_between_orders=float(pips_between),
                                    grid_lot_multiplier=float(grid_lot_mult),
                                    grid_take_profit=float(grid_tp),
                                    grid_stop_loss=float(grid_sl),
                                    max_grid_layers=int(max_layers),
                                    recovery_mode=1 if "Last close" in recovery_mode_label else 0,
                                    recovery_lot_multiplier=float(rec_mult),
                                    base_lot=float(base_lot),
                                    adaptive_enabled=adaptive_on,
                                    swap_enabled=swap_on,
                                    long_swap_pips=float(long_swap),
                                    short_swap_pips=float(short_swap))
                st.session_state["opt"] = {"best": best_params_dict, "study": study,
                                              "result": result, "best_score": best_score,
                                              "criterion": criterion_choice}
    if "opt" in st.session_state:
        opt = st.session_state["opt"]
        st.markdown("### Best params (by " + opt["criterion"] + ")")
        if "best_score" in opt:
            st.metric("Best composite score", f"{opt['best_score']:.1f}")
        st.json(opt["best"])
        render_metrics(opt["result"]["metrics"],
                       ["total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown",
                        "win_rate", "profit_factor", "n_trades", "net_pnl"], cols_per_row=5)
        dd = opt["result"]["equity"] / opt["result"]["equity"].cummax() - 1
        st.plotly_chart(equity_chart(opt["result"]["equity"], dd), use_container_width=True)


elif mode == "🎲 Monte Carlo":
    st.markdown("## 🎲 Monte Carlo Robustness")
    cols = st.columns(3)
    n_sims = cols[0].slider("Simulations", 100, 10000, 2000)
    block = cols[1].slider("Block size", 8, 200, 24)
    conf = cols[2].slider("Confidence", 0.80, 0.99, 0.95)
    if st.button("=" * 1 + " Run", type="primary"):
        result = run_full_backtest()
        if result is not None:
            rets = result["equity"].pct_change().dropna()
            ci = ci_metrics(rets, n_sims=n_sims, block=block, confidence=conf)
            sims = bootstrap_returns(rets, n_sims=n_sims, block_size=block)
            sim_sharpe = []
            for sim in sims:
                eq = np.cumprod(1 + sim)
                ar = eq[-1] ** (252 * 24 / len(sim)) - 1
                av = np.std(sim) * np.sqrt(252 * 24)
                sim_sharpe.append(ar / av if av > 0 else 0)
            sim_sharpe = np.array(sim_sharpe)
            observed = result["metrics"].get("sharpe", 0)
            st.session_state["mc"] = {"ci": ci, "sim_sharpe": sim_sharpe, "observed": observed,
                                       "result": result}
    if "mc" in st.session_state:
        mc = st.session_state["mc"]
        st.dataframe(mc["ci"].style.format("{:.3f}"), use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=mc["sim_sharpe"], nbinsx=50,
                                    marker_color=PALETTE["info"], opacity=0.7))
        fig.add_vline(x=mc["observed"], line_dash="dash",
                      line_color=PALETTE["warning"],
                      annotation_text=f"Observed: {mc['observed']:.2f}")
        lo, hi = np.percentile(mc["sim_sharpe"], [2.5, 97.5])
        fig.add_vrect(x0=lo, x1=hi, fillcolor=PALETTE["success"], opacity=0.1,
                      annotation_text=f"95% CI: [{lo:.2f}, {hi:.2f}]")
        fig.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["bg"],
                          plot_bgcolor=PALETTE["card_bg"], height=380,
                          title="Sharpe distribution (bootstrap)")
        st.plotly_chart(fig, use_container_width=True)


elif mode == "📊 Walk-Forward":
    st.markdown("## 📊 Walk-Forward")
    cols = st.columns(3)
    train_m = cols[0].slider("Train months", 6, 60, 24)
    test_m = cols[1].slider("Test months", 1, 12, 6)
    roll_m = cols[2].slider("Roll months", 1, 6, 3)
    name = st.selectbox("Strategy",
                         [s for s in STRATEGY_REGISTRY.keys() if s != "universal"],
                         key="wf_strat")
    if st.button("=" * 1 + " Run", type="primary"):
        import yaml
        with open(ROOT / "config" / "strategies.yaml") as f:
            spec = yaml.safe_load(f).get(name, {})
        with st.spinner("Walk-forward…"):
            wf = walk_forward(df, name, spec, train_months=train_m,
                               test_months=test_m, roll_months=roll_m, n_trials=60)
            wfd = wf_summary(wf)
            st.session_state["wf"] = wfd
    if "wf" in st.session_state:
        wfd = st.session_state["wf"]
        if len(wfd) > 0:
            st.dataframe(wfd.style.format({"train_sharpe": "{:.2f}",
                                            "test_sharpe": "{:.2f}",
                                            "train_calmar": "{:.2f}",
                                            "test_calmar": "{:.2f}"}),
                         use_container_width=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wfd.index, y=wfd["train_sharpe"],
                                      mode="lines+markers", name="IS",
                                      line=dict(color=PALETTE["info"])))
            fig.add_trace(go.Scatter(x=wfd.index, y=wfd["test_sharpe"],
                                      mode="lines+markers", name="OOS",
                                      line=dict(color=PALETTE["warning"])))
            fig.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["bg"],
                              plot_bgcolor=PALETTE["card_bg"],
                              title="IS vs OOS Sharpe", height=380)
            st.plotly_chart(fig, use_container_width=True)


elif mode == "🧬 Multi-Strategy":
    st.markdown("## 🧬 Multi-Strategy Portfolio")
    selected = st.multiselect("Strategies",
                               [s for s in STRATEGY_REGISTRY.keys() if s != "universal"],
                               default=list(STRATEGY_REGISTRY.keys())[:4])
    method = st.selectbox("Allocation", ["equal", "markowitz", "risk_parity", "kelly"])
    max_w = st.slider("Max weight", 0.1, 1.0, 0.4)
    if st.button("=" * 1 + " Run", type="primary") and selected:
        with st.spinner(f"Backtesting {len(selected)} strategies + allocating…"):
            signals = {}
            for s_name in selected:
                cls = STRATEGY_REGISTRY[s_name]
                sig = cls(params={}).generate(df)
                signals[s_name] = (sig.entries.fillna(False).astype(bool),
                                    pd.Series(sig.direction, index=df.index).fillna(0).astype(int))
            result = run_full(df, signals, init_cash=init_cash,
                                commission_pips=commission_pips,
                                slippage_pips=slippage_pips,
                                grid_mode=GRID_MAP[grid_mode],
                                pips_between_orders=float(pips_between),
                                grid_lot_multiplier=float(grid_lot_mult),
                                grid_take_profit=float(grid_tp),
                                grid_stop_loss=float(grid_sl),
                                max_grid_layers=int(max_layers),
                                recovery_mode=1 if "Last close" in recovery_mode_label else 0,
                                recovery_lot_multiplier=float(rec_mult),
                                base_lot=float(base_lot),
                                adaptive_enabled=adaptive_on,
                                swap_enabled=swap_on,
                                long_swap_pips=float(long_swap),
                                short_swap_pips=float(short_swap))
            st.session_state["ms"] = result
    if "ms" in st.session_state:
        ms = st.session_state["ms"]
        if not ms["scoreboard"].empty:
            st.dataframe(ms["scoreboard"], use_container_width=True)
        render_metrics(ms["metrics"],
                       ["total_return", "sharpe", "calmar", "max_drawdown",
                        "n_trades", "win_rate", "profit_factor"], cols_per_row=5)
        dd = ms["equity"] / ms["equity"].cummax() - 1
        st.plotly_chart(equity_chart(ms["equity"], dd), use_container_width=True)


# === Validate Strategies mode ===
elif mode == "✅ Validate Strategies":
    st.markdown("## ✅ Strategy Validator")
    st.caption("Runs every strategy against 7 synthetic scenarios with KNOWN expected "
                "behavior. Catches coding errors + logical errors automatically.")
    if st.button("🔍 Validate All Strategies", type="primary"):
        from tools.strategy_validator import validate_all
        with st.spinner("Validating all strategies against synthetic scenarios…"):
            out = validate_all(verbose=False)
        st.markdown("### Validation results")
        rows = []
        for sname, r in out["strategies"].items():
            for s in r["scenarios"]:
                rows.append({
                    "strategy": sname,
                    "scenario": s["name"],
                    "passed": "PASS" if s["passed"] else "FAIL",
                    "long": s["n_long"],
                    "short": s["n_short"],
                    "total": s["n_total"],
                    "note": s["msg"],
                })
        df_v = pd.DataFrame(rows)
        st.dataframe(df_v, use_container_width=True, height=600)
        n_pass = (df_v["passed"] == "PASS").sum()
        n_fail = (df_v["passed"] == "FAIL").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Scenarios PASS", int(n_pass))
        c2.metric("Scenarios FAIL", int(n_fail))
        c3.metric("Verdict", "VALIDATED" if n_fail == 0 else "ISSUES FOUND")
        if n_fail > 0:
            st.warning(f"{n_fail} scenarios failed. Common causes:\n"
                       "- Strategy is too restrictive (defaults too high)\n"
                       "- NaN handling in indicators (warming-up period)\n"
                       "- Off-by-one in signal logic")
            # Show detail per failing scenario
            fails = df_v[df_v["passed"] == "FAIL"]
            with st.expander(f"Failing scenarios ({len(fails)})"):
                for _, r in fails.iterrows():
                    st.markdown(f"- **{r['strategy']} / {r['scenario']}**: {r['note']}")


# === Auto-Magic mode ===
elif mode == "🪄 Auto-Magic":
    st.markdown("## 🪄 Auto-Magic Workflow")
    st.caption("Drop a strategy file → automatically validate + backtest + optimize + Monte Carlo + "
                "Walk-Forward + allocate. One click does everything.")
    if uploaded is None and strategy_cls is None:
        st.warning("Drop a strategy file first, or pick one from the sidebar dropdown.")
    elif st.button("🪄 Run Auto-Magic", type="primary"):
        from tools.strategy_validator import validate_strategy
        results = {}
        with st.spinner("Step 1/6 — Validating strategy logic…"):
            v = validate_strategy(strategy_choice, verbose=False)
            results["validation"] = v
            st.markdown("**Step 1: Validate**")
            n_pass = v["passed"]; n_fail = v["failed"]
            st.markdown(f"  Scenarios PASS: {n_pass} / FAIL: {n_fail}")
            if n_fail > 0:
                st.warning(f"Some scenarios failed. Backtest results may be unreliable.")
        with st.spinner("Step 2/6 — Running backtest on selected data…"):
            try:
                result = run_full_backtest()
                results["backtest"] = result
                st.markdown("**Step 2: Backtest**")
                m = result["metrics"]
                st.markdown(f"  Sharpe: {m['sharpe']:+.2f}  |  Trades: {m.get('n_trades', 0)}  |  "
                              f"Equity: ${m['final_equity']:.0f}")
            except Exception as e:
                st.error(f"Backtest failed: {e}")
        if "backtest" in results:
            with st.spinner("Step 3/6 — Monte Carlo robustness (1000 sims)…"):
                try:
                    rets = result["equity"].pct_change().dropna()
                    ci = ci_metrics(rets, n_sims=1000, block=24, confidence=0.95)
                    obs = m["sharpe"]
                    lo, hi = ci.loc["sharpe", "2.5%"], ci.loc["sharpe", "97.5%"]
                    results["mc"] = {"ci": ci, "obs": obs, "lo": lo, "hi": hi}
                    st.markdown("**Step 3: Monte Carlo**")
                    st.markdown(f"  Sharpe 95% CI: [{lo:.2f}, {hi:.2f}]  |  Observed: {obs:.2f}")
                    if lo > 0:
                        st.success("  CI excludes 0 — strategy has edge above noise.")
                    else:
                        st.warning("  CI includes 0 — could be noise.")
                except Exception as e:
                    st.error(f"MC failed: {e}")
            with st.spinner("Step 4/6 — Walk-Forward (3 windows, 30 trials each)…"):
                try:
                    import yaml as _yaml
                    with open(ROOT / "config" / "strategies.yaml") as f:
                        spec = _yaml.safe_load(f).get(strategy_choice, {})
                    if spec:
                        wf = walk_forward(df, strategy_choice, spec,
                                           train_months=18, test_months=6,
                                           roll_months=6, n_trials=30)
                        wfd = wf_summary(wf)
                        results["wf"] = wfd
                        st.markdown("**Step 4: Walk-Forward**")
                        if len(wfd) > 0:
                            is_mean = wfd["train_sharpe"].mean()
                            oos_mean = wfd["test_sharpe"].mean()
                            st.markdown(f"  IS Sharpe mean: {is_mean:+.2f}  |  OOS Sharpe mean: {oos_mean:+.2f}")
                            if oos_mean > 0 and oos_mean > 0.3 * is_mean:
                                st.success("  OOS > 30% of IS — strategy is robust.")
                            else:
                                st.warning("  OOS too low vs IS — possible overfit.")
                        else:
                            st.markdown("  Not enough data for walk-forward.")
                except Exception as e:
                    st.error(f"WF failed: {e}")
            with st.spinner("Step 5/6 — Optimizing parameters (100 trials)…"):
                try:
                    import yaml as _yaml
                    with open(ROOT / "config" / "strategies.yaml") as f:
                        spec = _yaml.safe_load(f).get(strategy_choice, {})
                    if spec:
                        study = optimize_strategy(strategy_choice, df, spec, n_trials=100)
                        best = best_params(study, metric="sharpe")
                        results["best_params"] = best
                        st.markdown("**Step 5: Optimize**")
                        st.markdown(f"  Best params: `{best}`")
                        st.markdown(f"  Best Sharpe (in-sample): {study.best_value:+.2f}")
                except Exception as e:
                    st.error(f"Optimize failed: {e}")
            with st.spinner("Step 6/6 — Computing final verdict…"):
                st.markdown("**Step 6: Verdict**")
                passed_validation = n_fail == 0
                pos_sharpe = m.get("sharpe", 0) > 0
                mc_ok = results.get("mc", {}).get("lo", -1) > 0
                wf_ok = results.get("wf", {}).get("test_sharpe", pd.Series([0])).mean() > 0
                verdict = (passed_validation and pos_sharpe and mc_ok and wf_ok)
                if verdict:
                    st.success("[VERDICT] Strategy is validated, profitable, robust. Ready to trade.")
                else:
                    issues = []
                    if not passed_validation:
                        issues.append("validation failed")
                    if not pos_sharpe:
                        issues.append("negative Sharpe")
                    if not mc_ok:
                        issues.append("MC CI includes 0")
                    if not wf_ok:
                        issues.append("WF OOS not positive")
                    st.error(f"[VERDICT] Strategy has issues: {', '.join(issues)}")
                # Save report
                report = {
                    "strategy": strategy_choice,
                    "validation": {"passed": n_pass, "failed": n_fail},
                    "backtest": m,
                    "mc_ci": results.get("mc", {}),
                    "wf": results.get("wf", pd.DataFrame()).to_dict() if "wf" in results else None,
                    "best_params": results.get("best_params"),
                    "verdict": verdict,
                }
                Path("output").mkdir(exist_ok=True)
                (Path("output") / f"automagic_{strategy_choice}.json").write_text(
                    json.dumps(report, indent=2, default=str))
                st.caption(f"Report saved to output/automagic_{strategy_choice}.json")


# === Guided Walkthrough mode ===
elif mode == "🎓 Guided Walkthrough":
    st.markdown("## 🎓 Guided Walkthrough")
    st.caption("Step-by-step workflow for non-pro traders. Just answer the questions, "
                "the backtester does the rest.")
    # Use session state to track step
    if "wizard_step" not in st.session_state:
        st.session_state["wizard_step"] = 0
    if "wizard_choices" not in st.session_state:
        st.session_state["wizard_choices"] = {}

    step = st.session_state["wizard_step"]
    choices = st.session_state["wizard_choices"]

    # Progress bar
    st.progress(step / 6)

    if step == 0:
        st.markdown("### Step 1/6 — Pick your data")
        st.markdown("What's the market you want to test?")
        data_choice = st.radio("Source", ["Use cached EURUSD H1 (recommended)",
                                           "Pick a different cached file",
                                           "Pull live from MT5",
                                           "Use Yahoo Finance (any ticker)"])
        if data_choice != "Use cached EURUSD H1 (recommended)":
            choices["data"] = data_choice
        if st.button("Next ->"):
            st.session_state["wizard_step"] = 1
            st.rerun()

    elif step == 1:
        st.markdown("### Step 2/6 — Pick a strategy")
        st.markdown("Which strategy do you want to test?")
        strat_choice = st.selectbox("Strategy", [s for s in STRATEGY_REGISTRY.keys() if s != "universal"])
        st.markdown(f"**What this is:** {STRATEGY_REGISTRY[strat_choice].__doc__ or 'Custom strategy'}")
        if st.button("Next ->"):
            choices["strategy"] = strat_choice
            st.session_state["wizard_step"] = 2
            st.rerun()

    elif step == 2:
        st.markdown("### Step 3/6 — Choose your risk profile")
        st.markdown("How aggressive should the strategy be?")
        prof = st.radio("Profile", ["Conservative (small lots, wide stops, strict signals)",
                                       "Balanced (default — recommended)",
                                       "Aggressive (big lots, tight stops, more signals)"])
        if st.button("Next ->"):
            choices["profile"] = prof
            st.session_state["wizard_step"] = 3
            st.rerun()

    elif step == 3:
        st.markdown("### Step 4/6 — Validate the strategy")
        st.markdown("Before backtesting, let's make sure the logic is correct.")
        if st.button("Run validation"):
            from tools.strategy_validator import validate_strategy
            v = validate_strategy(choices.get("strategy", "fbb"), verbose=False)
            n_pass = v["passed"]; n_fail = v["failed"]
            choices["validation"] = {"passed": n_pass, "failed": n_fail}
            st.markdown(f"**Validation: {n_pass} passed, {n_fail} failed**")
            if n_fail == 0:
                st.success("[OK] Strategy logic is validated.")
            else:
                st.warning(f"[WARN] {n_fail} scenarios failed. Backtest may be unreliable.")
        if "validation" in choices and st.button("Next ->"):
            st.session_state["wizard_step"] = 4
            st.rerun()

    elif step == 4:
        st.markdown("### Step 5/6 — Run backtest")
        st.markdown("Click the button to backtest on the selected data.")
        if st.button("▶ Run Backtest"):
            strat_name = choices.get("strategy", "fbb")
            profile = choices.get("profile", "Balanced")
            prof_map = {"Conservative": "Conservative", "Balanced": "Balanced", "Aggressive": "Aggressive"}
            from core.strictness import RISK_PROFILES, apply_strictness, apply_tp_sl_widening
            p = RISK_PROFILES[prof_map.get(profile, "Balanced")]
            with st.spinner(f"Backtesting {strat_name} with {profile} profile…"):
                cls = STRATEGY_REGISTRY[strat_name]
                base_params = {}
                if strat_name == "fbb":
                    base_params = {"open_orders_type_1": 1, "open_orders_type_2": 0,
                                    "bars_calculate": 20, "deviation": 1.8}
                elif strat_name == "ac_ao":
                    base_params = {"open_orders_type": 1, "use_acceleration_filter": False,
                                    "use_ao_synchronization": False}
                elif strat_name == "adx":
                    base_params = {"open_orders_type": 1, "use_di_crossover": True, "bars_calculate": 20}
                elif strat_name == "dem":
                    base_params = {"open_orders_type": 3, "bars_calculate": 20}
                elif strat_name == "mfi":
                    base_params = {"open_orders_type": 2, "bars_calculate": 14}
                elif strat_name == "ms":
                    base_params = {"open_orders_type_1": 8, "use_confluence_filter": False}
                adjusted = apply_strictness(strat_name, p["strictness"], base_params)
                adjusted = apply_tp_sl_widening(adjusted, p["tp_widening"])
                strat_inst = cls(params=adjusted)
                sig = strat_inst.generate(df)
                result = run_full(df, {strat_name: (sig.entries.fillna(False).astype(bool),
                                                       pd.Series(sig.direction, index=df.index).fillna(0).astype(int))},
                                    init_cash=init_cash,
                                    commission_pips=commission_pips,
                                    slippage_pips=slippage_pips,
                                    grid_mode=GRID_MAP[grid_mode],
                                    pips_between_orders=float(pips_between),
                                    grid_lot_multiplier=float(grid_lot_mult),
                                    grid_take_profit=float(grid_tp),
                                    grid_stop_loss=float(grid_sl),
                                    max_grid_layers=int(max_layers),
                                    recovery_mode=1 if "Last close" in recovery_mode_label else 0,
                                    recovery_lot_multiplier=float(rec_mult),
                                    base_lot=p["base_lot"],
                                    adaptive_enabled=p["adaptive"],
                                    swap_enabled=swap_on,
                                    long_swap_pips=float(long_swap),
                                    short_swap_pips=float(short_swap))
                choices["result"] = result
                m = result["metrics"]
                st.success(f"Sharpe: {m['sharpe']:+.2f} | Trades: {m.get('n_trades', 0)} | "
                            f"Equity: ${m['final_equity']:.0f}")
                dd = result["equity"] / result["equity"].cummax() - 1
                st.plotly_chart(equity_chart(result["equity"], dd), use_container_width=True)
        if "result" in choices and st.button("Next ->"):
            st.session_state["wizard_step"] = 5
            st.rerun()

    elif step == 5:
        st.markdown("### Step 6/6 — Verdict")
        if "result" not in choices:
            st.warning("Run backtest first.")
        else:
            m = choices["result"]["metrics"]
            sharpe = m.get("sharpe", 0)
            md = m.get("max_drawdown", 0)
            wr = m.get("win_rate", 0)
            pf = m.get("profit_factor", 0)
            verdict_pass = sharpe > 0.3 and md > -0.25 and pf > 1.0
            if verdict_pass:
                st.success("[VERDICT] Strategy looks profitable. Recommended: paper-trade for "
                            "1 month before risking real capital.")
            else:
                st.warning(f"[VERDICT] Strategy has issues. Consider:")
                if sharpe <= 0.3:
                    st.markdown("- Sharpe is low. Try a different strategy or strictness.")
                if md < -0.25:
                    st.markdown("- Drawdown is large. Reduce lot size or widen TP/SL.")
                if pf <= 1.0:
                    st.markdown("- Profit factor below 1. The strategy loses on average.")
            # Save report
            import json
            report = {"choices": {k: v for k, v in choices.items() if k != "result"},
                       "metrics": {k: float(v) for k, v in m.items() if isinstance(v, (int, float))},
                       "verdict": verdict_pass}
            Path("output").mkdir(exist_ok=True)
            (Path("output") / "guided_walkthrough_report.json").write_text(
                json.dumps(report, indent=2, default=str))
            st.caption("Report saved to output/guided_walkthrough_report.json")

    if step > 0 and st.button("<- Back"):
        st.session_state["wizard_step"] = max(0, step - 1)
        st.rerun()


st.divider()
st.caption(f"Universal backtester · {len(STRATEGY_REGISTRY)} strategies · "
           f"{info.get('rows', 0):,} bars · source: {source}")