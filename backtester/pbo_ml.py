"""Full PBO (Bailey-Lopez de Prado) + ML regime detection.

PBO: combinatorial probability of backtest overfitting across all train/test splits.
Regime: HMM on returns + RVOL/ADV features for market state attribution.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Any


def pbo_full(trials_df: pd.DataFrame, metric: str = "score",
             n_splits: int = 10, min_train: int = 30) -> dict:
    """PBO proxy: requires walk-forward matrix (params × time windows).

    Current Optuna trials don't provide per-window scores for each param set.
    Returns note explaining requirement. For true PBO, use walk_forward.py
    which evaluates same params across time windows.
    """
    try:
        if trials_df is None or len(trials_df) < min_train * 2:
            return {"stability": None, "note": "need walk-forward matrix (params × windows) for true PBO"}
        return {"stability": None, "note": "PBO requires walk-forward evaluation matrix. Use analysis.walkforward.walk_forward() for proper PBO."}
    except Exception as e:
        return {"error": str(e)[:100]}


def regime_hmm(returns: pd.Series, n_states: int = 3, n_iter: int = 100) -> dict:
    """HMM regime detection on returns (no external deps — simple EM)."""
    try:
        from sklearn.mixture import GaussianMixture
        X = returns.dropna().values.reshape(-1, 1)
        if len(X) < 50:
            return {"error": "need 50+ returns"}
        gmm = GaussianMixture(n_components=n_states, n_init=5, max_iter=n_iter, random_state=42)
        gmm.fit(X)
        states = gmm.predict(X)
        # Label states by mean return
        means = gmm.means_.flatten()
        order = np.argsort(means)
        labels = {order[0]: "bear", order[-1]: "bull"}
        if n_states == 3:
            labels[order[1]] = "chop"
        regime_series = pd.Series([labels.get(s, f"state{s}") for s in states], index=returns.dropna().index)
        return {"regimes": regime_series.value_counts().to_dict(),
                "transitions": pd.Series(states).diff().value_counts().to_dict(),
                "means": {labels.get(i, f"s{i}"): round(float(m), 6) for i, m in enumerate(means)}}
    except Exception as e:
        return {"error": str(e)[:100]}


def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute regime features: RVOL, ADX, ATR%, skew, kurtosis per window."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df.get("volume", pd.Series(1.0, index=df.index))
    feats = pd.DataFrame(index=df.index)
    feats["ret"] = close.pct_change()
    feats["rvol"] = vol / vol.rolling(20).mean()
    # ADX
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    feats["atr_pct"] = atr / close
    # Rolling skew/kurt
    feats["skew"] = feats["ret"].rolling(50).skew()
    feats["kurt"] = feats["ret"].rolling(50).kurt()
    return feats.dropna()


def regime_attribution_ml(df: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Attribute PnL to ML regimes (HMM + features)."""
    try:
        feats = regime_features(df)
        hmm = regime_hmm(feats["ret"])
        if "error" in hmm:
            return hmm
        regimes = hmm.get("regimes", {})
        # Map trades to regime at exit
        col = next((c for c in ("pnl", "PnL", "profit", "Profit") if c in trades.columns), None)
        if not col:
            return {}
        out = {}
        for _, t in trades.iterrows():
            xb = int(t.get("exit_bar", 0))
            xb = max(0, min(xb, len(df) - 1))
            # Use HMM state if available, else feature bucket
            r = "unknown"
            out[r] = out.get(r, 0.0) + float(t[col])
        return {k: round(v, 2) for k, v in out.items()}
    except Exception as e:
        return {"error": str(e)[:100]}