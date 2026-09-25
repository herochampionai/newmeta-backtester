"""Full Market Symbols Scanner & Quantitative Radar.
Scans symbols across multiple timeframes (M1, M5, M15, M30, H1, H4, D1).
Calculates Spread, Volatility ATR, ADX Regime, RSI, RVOL, Strategy Signals, and Fit Score.
Works seamlessly with live MT5 or fallback data.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.live_fetcher import fetch_with_priority
from strategies import STRATEGY_REGISTRY
from strategies.indicators import adx, rsi

PRESET_UNIVERSES = {
    "Metals & Commodities": ["XAUUSD", "XAUAUD", "XAGUSD", "USOIL", "UKOIL"],
    "Forex Majors": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"],
    "Forex Crosses": ["EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "CADJPY", "NZDJPY", "EURAUD", "GBPCHF"],
    "Global Indices": ["US30", "USTEC", "NAS100", "SPX500", "GER40", "UK100", "JP225"],
    "Crypto": ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "DOGEUSD", "LTCUSD"],
}


def get_available_mt5_symbols(category: str | None = None) -> list[dict]:
    """Query live MT5 for symbol list and metadata."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return []

        symbols = mt5.symbols_get()
        if not symbols:
            mt5.shutdown()
            return []

        result = []
        for s in symbols:
            cat = s.path.split("\\")[0] if "\\" in s.path else "Other"
            if category and category.lower() != "all" and category.lower() not in cat.lower() and category.lower() not in s.name.lower():
                continue

            result.append({
                "name": s.name,
                "category": cat,
                "path": s.path,
                "spread": s.spread,
                "digits": s.digits,
                "point": s.point,
                "description": s.description or s.name,
                "visible": s.visible,
            })

        mt5.shutdown()
        return result
    except Exception:
        return []


def calculate_symbol_metrics(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    spread_points: float | None = None,
    point_size: float = 0.01,
    digits: int = 2,
    strategy_obj: Any = None,
) -> dict:
    """Calculate pro quantitative radar metrics on a symbol's OHLCV dataframe."""
    if df is None or len(df) < 30:
        return {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1000, index=df.index)

    cur_price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    chg_pct = ((cur_price - prev_close) / prev_close) * 100.0

    # 24-period change
    lookback_24 = min(len(close) - 1, 24)
    price_24_ago = float(close.iloc[-lookback_24])
    chg_24h_pct = ((cur_price - price_24_ago) / price_24_ago) * 100.0

    # ATR (14)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
    atr_pct = (atr_14 / cur_price) * 100.0

    pip_scale = 0.0001 if digits == 5 or digits == 4 else (0.01 if digits == 3 or digits == 2 else 1.0)
    atr_pips = atr_14 / pip_scale

    # ADX (14)
    adx_val, pdi, ndi = adx(high, low, close, length=14)
    adx_cur = float(adx_val.iloc[-1]) if len(adx_val) > 0 and pd.notna(adx_val.iloc[-1]) else 20.0
    pdi_cur = float(pdi.iloc[-1]) if len(pdi) > 0 and pd.notna(pdi.iloc[-1]) else 20.0
    ndi_cur = float(ndi.iloc[-1]) if len(ndi) > 0 and pd.notna(ndi.iloc[-1]) else 20.0

    if adx_cur >= 25:
        regime = "Bullish Trend" if pdi_cur > ndi_cur else "Bearish Trend"
        regime_badge = "🟢 Strong Bull" if pdi_cur > ndi_cur else "🔴 Strong Bear"
    elif adx_cur >= 18:
        regime = "Mild Bull" if pdi_cur > ndi_cur else "Mild Bear"
        regime_badge = "↗️ Mild Bull" if pdi_cur > ndi_cur else "↘️ Mild Bear"
    else:
        regime = "Ranging / Chop"
        regime_badge = "⚪ Consolidation"

    # RSI (14)
    rsi_s = rsi(close, length=14)
    rsi_cur = float(rsi_s.iloc[-1]) if len(rsi_s) > 0 and pd.notna(rsi_s.iloc[-1]) else 50.0

    # RVOL (Relative Volume)
    vol_mean = vol.rolling(20, min_periods=1).mean()
    rvol = float(vol.iloc[-1] / max(vol_mean.iloc[-1], 1.0))

    # Spread in pips
    spread_pips = (spread_points * point_size / pip_scale) if spread_points is not None else (atr_pips * 0.05)
    spread_cost_ratio = (spread_pips / max(atr_pips, 0.001)) * 100.0  # Spread as % of bar range

    # Strategy Signal Check (if strategy supplied)
    signal_label = "NEUTRAL"
    if strategy_obj is not None:
        try:
            sig = strategy_obj.generate(df)
            entries = sig.entries.fillna(False).values
            direction = sig.direction if hasattr(sig, "direction") else np.zeros(len(df))
            if entries[-1]:
                signal_label = "BUY 🟢" if direction[-1] > 0 else "SELL 🔴"
            elif len(entries) > 2 and entries[-2]:
                signal_label = "RECENT BUY" if direction[-2] > 0 else "RECENT SELL"
        except Exception:
            pass

    # Breakout / Strategy Suitability Score (0-100)
    # Rewards: high volatility ATR, clear ADX trend, low spread cost ratio, RVOL surge
    vol_score = min(atr_pct / 0.5, 1.0) * 35.0
    trend_score = min(adx_cur / 40.0, 1.0) * 35.0
    spread_penalty = min(spread_cost_ratio / 15.0, 1.0) * 20.0
    rvol_bonus = min(rvol / 2.0, 1.0) * 10.0

    suitability = max(0.0, min(100.0, vol_score + trend_score + rvol_bonus - spread_penalty))

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "price": cur_price,
        "change_pct": round(chg_pct, 2),
        "change_24h_pct": round(chg_24h_pct, 2),
        "spread_pips": round(spread_pips, 1),
        "spread_cost_ratio": round(spread_cost_ratio, 1),
        "atr_pips": round(atr_pips, 1),
        "atr_pct": round(atr_pct, 2),
        "adx": round(adx_cur, 1),
        "regime": regime_badge,
        "rsi": round(rsi_cur, 1),
        "rvol": round(rvol, 2),
        "suitability": round(suitability, 1),
        "signal": signal_label,
        "bars": len(df),
    }


def scan_market_matrix(
    symbols: list[str],
    timeframes: list[str] = ["M1", "M5", "M15", "H1", "H4", "D1"],
    strategy_name: str | None = None,
    strategy_params: dict | None = None,
    terminal: str | None = None,
    progress_callback: Any = None,
) -> pd.DataFrame:
    """Scan a universe of symbols across multiple timeframes."""
    rows = []
    total = len(symbols) * len(timeframes)
    count = 0

    strat_obj = None
    if strategy_name and strategy_name in STRATEGY_REGISTRY:
        strat_obj = STRATEGY_REGISTRY[strategy_name](params=strategy_params or {})

    for sym in symbols:
        sym_clean = sym.strip().upper()
        for tf in timeframes:
            count += 1
            if progress_callback:
                progress_callback(count, total, f"Scanning {sym_clean} [{tf}]...")

            try:
                df, info = fetch_with_priority(sym_clean, tf, allow_synthetic=True, terminal_override=terminal)
                if df is not None and len(df) >= 30:
                    metrics = calculate_symbol_metrics(df, sym_clean, tf, strategy_obj=strat_obj)
                    if metrics:
                        rows.append(metrics)
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    df_result = pd.DataFrame(rows).sort_values(["suitability", "atr_pct"], ascending=[False, False])
    return df_result
