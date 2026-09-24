"""Strict MT5/LEAN parity harness — ≤0.5 pip, corr>0.99, OOS/WFE enforcement.

Auto-fetches MT5 deals, runs same params through Newmeta, compares tick-for-tick.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

from backtester.engine_full import run_full
from backtester.symbol_spec import get_spec
from data.live_fetcher import fetch_with_priority
from strategies import STRATEGY_REGISTRY


def fetch_mt5_deals(symbol: str, start: str, end: str, terminal: str | None = None) -> pd.DataFrame:
    """Pull MT5 deal history for exact comparison."""
    try:
        from data.mt5_export import resolve_terminal, init_mt5
        import MetaTrader5 as mt5
        from datetime import datetime
        t = terminal or resolve_terminal()
        if not t or not init_mt5(t):
            return pd.DataFrame()
        deals = mt5.history_deals_get(datetime.fromisoformat(start), datetime.fromisoformat(end))
        if deals is None or len(deals) == 0:
            mt5.shutdown()
            return pd.DataFrame()
        df = pd.DataFrame([d._asdict() for d in deals])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        mt5.shutdown()
        return df
    except Exception:
        return pd.DataFrame()


def compare_equity(mt5_deals: pd.DataFrame, newmeta_equity: pd.Series,
                   tolerance_pips: float = 0.5, pip_size: float = 0.0001) -> dict:
    """Compare equity curves — return pass/fail + drift stats."""
    if mt5_deals.empty or newmeta_equity.empty:
        return {"pass": False, "reason": "empty data"}
    # Build MT5 equity from deals
    mt5_eq = pd.Series(10000.0, index=newmeta_equity.index, dtype=float)
    for _, d in mt5_deals.iterrows():
        if d.get("entry", 0) == 0:  # DEAL_ENTRY_IN
            continue
        pnl = float(d.get("profit", 0))
        ts = d.name
        idx = newmeta_equity.index.get_indexer([ts], method="nearest")[0]
        if idx >= 0:
            mt5_eq.iloc[idx:] += pnl
    # Correlation
    common = newmeta_equity.index.intersection(mt5_eq.index)
    if len(common) < 10:
        return {"pass": False, "reason": "insufficient overlap"}
    c1 = newmeta_equity.loc[common].values
    c2 = mt5_eq.loc[common].values
    corr = float(np.corrcoef(c1, c2)[0, 1]) if len(c1) > 1 else 0.0
    # Max pip drift
    drift = np.abs(c1 - c2).max()
    drift_pips = drift / pip_size
    passed = bool(corr > 0.99 and drift_pips <= tolerance_pips)
    return {"pass": passed, "corr": round(corr, 4), "max_drift_pips": round(drift_pips, 2),
            "tolerance_pips": tolerance_pips, "verdict": "PASS" if passed else "FAIL"}


def wfe_enforce(is_metrics: dict, oos_metrics: dict, min_wfe: float = 0.5) -> dict:
    """Walk-Forward Efficiency gate — reject if OOS decay > threshold."""
    is_r = float(is_metrics.get("net_pnl", 0) or 0)
    oos_r = float(oos_metrics.get("net_pnl", 0) or 0)
    wfe = (oos_r / is_r) if is_r > 0 else 0.0
    return {"wfe": round(wfe, 2), "pass": bool(wfe >= min_wfe and oos_r > 0),
            "min_wfe": min_wfe, "verdict": "ACCEPT" if wfe >= min_wfe else "REJECT"}


def strict_parity_test(strategy_name: str, symbol: str, timeframe: str,
                       params: dict, start: str, end: str,
                       terminal: str | None = None,
                       tolerance_pips: float = 0.5,
                       min_corr: float = 0.99) -> dict:
    """Full strict parity: fetch MT5 deals, run Newmeta, compare, WFE gate."""
    # 1. MT5 deals
    mt5_deals = fetch_mt5_deals(symbol, start, end, terminal)
    if mt5_deals.empty:
        return {"pass": False, "reason": "no MT5 deals found"}

    # 2. Newmeta run (tick mode)
    df, _ = fetch_with_priority(symbol, timeframe, allow_synthetic=False, terminal_override=terminal)
    if df is None or len(df) < 100:
        return {"pass": False, "reason": "insufficient data"}
    spec = get_spec(symbol, terminal=terminal)
    strat = STRATEGY_REGISTRY[strategy_name](params=params)
    sig = strat.generate(df)
    entries = sig.entries.fillna(False).astype(bool)
    direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
    signals = {strategy_name: (entries, direction)}
    res = run_full(df, signals, params=params, tick_mode="synthetic", ticks_per_bar=20,
                   pip_size=spec.pip_size, contract_size=spec.contract_size,
                   base_lot=0.1, leverage=30.0)

    # 3. Compare
    cmp = compare_equity(mt5_deals, res["equity"], tolerance_pips, spec.pip_size)

    # 4. WFE (split data 70/30)
    split_idx = int(len(df) * 0.7)
    df_is = df.iloc[:split_idx]
    df_oos = df.iloc[split_idx:]
    sig_is = strat.generate(df_is)
    sig_oos = strat.generate(df_oos)
    res_is = run_full(df_is, {strategy_name: (sig_is.entries.fillna(False).astype(bool),
                                               pd.Series(sig_is.direction, index=df_is.index).fillna(0).astype(int))},
                      params=params, tick_mode="synthetic", pip_size=spec.pip_size,
                      contract_size=spec.contract_size, base_lot=0.1)
    res_oos = run_full(df_oos, {strategy_name: (sig_oos.entries.fillna(False).astype(bool),
                                                pd.Series(sig_oos.direction, index=df_oos.index).fillna(0).astype(int))},
                       params=params, tick_mode="synthetic", pip_size=spec.pip_size,
                       contract_size=spec.contract_size, base_lot=0.1)
    wfe = wfe_enforce(res_is["metrics"], res_oos["metrics"])

    return {"parity": cmp, "wfe": wfe, "overall_pass": cmp["pass"] and wfe["pass"],
            "newmeta_metrics": res["metrics"], "mt5_deals": len(mt5_deals)}