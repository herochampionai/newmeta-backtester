"""Regime detector — classifies market state to choose which strategies should fire.

Regimes (based on ADX + Bollinger BandWidth + volatility):
  - TRENDING_UP:    ADX > 25 + price above EMA50 → use trend-following (ADX, MS, AC+AO)
  - TRENDING_DOWN:  ADX > 25 + price below EMA50 → same
  - RANGING:        ADX < 20 + low BBWidth → use mean-reversion (FBB, DeM)
  - OVEREXTENDED:   BBWidth > 1.5× median → use reversal (MFI, DeM contrarian)
  - VOLATILE:       recent vol > 1.5× historical → reduce size / use tighter stops
  - CHOPPY:         ADX 20-25 + medium BBWidth → use only confirmation strategies

The RegimeAwareStrategy class wraps the normal signal pipeline and SILENCES
strategies that aren't appropriate for the current regime.

Use:
    from core.regime import RegimeAwareStrategy
    strat = RegimeAwareStrategy(
        df,
        strategy_map={
            "trending_up": ["ms", "adx"],
            "trending_down": ["ms", "adx"],
            "ranging": ["fbb", "dem"],
            "overextended": ["mfi", "dem"],
            "volatile": [],
            "choppy": ["fbb"],
        },
    )
    sig = strat.generate()
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable
# Lazy imports: avoid circular dependency with strategies/__init__.py
# (which imports RegimeAwareStrategy from this module)
def _get_registry():
    from strategies import STRATEGY_REGISTRY
    return STRATEGY_REGISTRY
def _get_signals_class():
    from strategies._base import Signals
    return Signals


def detect_regimes(df: pd.DataFrame,
                    adx_period: int = 14,
                    ema_period: int = 50,
                    bb_period: int = 20,
                    bb_dev: float = 2.0,
                    lookback: int = 100) -> pd.Series:
    """Classify each bar into a market regime.

    Returns pd.Series of regime strings aligned to df.index.
    Possible values: 'trending_up', 'trending_down', 'ranging', 'overextended',
                    'volatile', 'choppy', 'unknown'
    """
    if len(df) < max(adx_period, ema_period, bb_period, lookback) + 10:
        return pd.Series("unknown", index=df.index)

    # ADX
    from strategies.indicators import adx as adx_fn, bollinger as bb_fn
    adx_v, pdi, mdi = adx_fn(df["high"], df["low"], df["close"], adx_period)

    # EMA for trend direction
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()

    # Bollinger BandWidth
    mid, up, lo = bb_fn(df["close"], bb_period, bb_dev)
    bb_width = (up - lo) / mid
    bb_width_median = bb_width.rolling(lookback).median()
    bb_ratio = bb_width / bb_width_median  # 1.0 = normal, 1.5+ = overextended

    # Realized volatility
    ret = df["close"].pct_change()
    vol = ret.rolling(20).std()
    vol_median = vol.rolling(lookback).median()
    vol_ratio = vol / vol_median

    regimes = pd.Series("unknown", index=df.index, dtype=object)
    for i in range(len(df)):
        a = adx_v.iloc[i]
        ema_diff = (df["close"].iloc[i] - ema.iloc[i]) / ema.iloc[i]
        bb_r = bb_ratio.iloc[i] if pd.notna(bb_ratio.iloc[i]) else 1.0
        v_r = vol_ratio.iloc[i] if pd.notna(vol_ratio.iloc[i]) else 1.0
        if pd.isna(a):
            regimes.iloc[i] = "unknown"
            continue
        # Priority: volatile > overextended > trending > ranging
        if v_r > 1.8:
            regimes.iloc[i] = "volatile"
        elif bb_r > 1.5:
            regimes.iloc[i] = "overextended"
        elif a > 25 and ema_diff > 0.001:
            regimes.iloc[i] = "trending_up"
        elif a > 25 and ema_diff < -0.001:
            regimes.iloc[i] = "trending_down"
        elif a < 20 and bb_r < 0.8:
            regimes.iloc[i] = "ranging"
        elif 20 <= a <= 25:
            regimes.iloc[i] = "choppy"
        else:
            # Default to ranging if not strongly trending
            regimes.iloc[i] = "ranging"
    return regimes


class RegimeAwareStrategy:
    """Wraps multiple strategies and silences them based on detected regime.

    Example strategy_map:
        {
            "trending_up":   ["ms", "adx"],      # MS + ADX fire only in uptrends
            "trending_down": ["ms", "adx"],
            "ranging":       ["fbb", "dem"],     # FBB + DeM fire only in ranges
            "overextended":  ["mfi", "dem"],     # MFI + DeM contrarian in extremes
            "volatile":      [],                  # no strategies in chaos
            "choppy":        ["fbb"],             # only FBB in transitional zones
        }
    """
    name = "regime_aware"

    def __init__(self,
                  df: pd.DataFrame,
                  strategy_map: dict[str, list[str]] | None = None,
                  params_map: dict[str, dict] | None = None,
                  regimes: pd.Series | None = None):
        STRATEGY_REGISTRY = _get_registry()
        self.df = df
        if regimes is None:
            self.regimes = detect_regimes(df)
        else:
            self.regimes = regimes
        self.strategy_map = strategy_map or self._default_strategy_map()
        self.params_map = params_map or {}
        # Pre-build strategy instances
        self.strategies = {}
        for strat_name in set().union(*self.strategy_map.values()):
            if strat_name in STRATEGY_REGISTRY:
                params = self.params_map.get(strat_name, {})
                self.strategies[strat_name] = STRATEGY_REGISTRY[strat_name](params=params)

    def _default_strategy_map(self) -> dict:
        """Sensible default mapping — each strategy shines in its regime."""
        return {
            "trending_up":   ["ms", "adx", "ac_ao"],
            "trending_down": ["ms", "adx", "ac_ao"],
            "ranging":       ["fbb", "dem"],
            "overextended":  ["mfi", "dem"],
            "volatile":      ["fbb"],  # only mean-reversion in volatility
            "choppy":        ["fbb", "dem"],
        }

    def generate(self) -> Signals:
        """Produce merged signals based on regime at each bar."""
        Signals = _get_signals_class()
        sig = Signals(
            entries=pd.Series(False, index=self.df.index),
            exits=pd.Series(False, index=self.df.index),
            direction=pd.Series(0, index=self.df.index, dtype=int),
        )
        for strat_name, strat in self.strategies.items():
            try:
                inner_sig = strat.generate(self.df)
            except Exception:
                continue
            # Determine which bars allow this strategy
            allowed = pd.Series(False, index=self.df.index)
            for regime, allowed_strats in self.strategy_map.items():
                if strat_name in allowed_strats:
                    regime_mask = (self.regimes == regime)
                    allowed = allowed | regime_mask
            # Apply allowed mask
            sig.entries = sig.entries | (inner_sig.entries & allowed)
            # Direction: only where allowed (handle ndarray or Series)
            inner_dir = inner_sig.direction
            if not isinstance(inner_dir, pd.Series):
                inner_dir = pd.Series(inner_dir, index=self.df.index)
            new_dir = inner_dir.where(allowed, 0).fillna(0).astype(int).values
            # Combine: first non-zero wins
            cur_dir = sig.direction if isinstance(sig.direction, np.ndarray) else sig.direction.values
            sig.direction = np.where(cur_dir != 0, cur_dir, new_dir)
        return sig


def regime_performance_summary(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Compute per-regime performance from a trade log.

    Uses entry time to determine regime for each trade.
    Returns DataFrame with columns: regime, n_trades, win_rate, net_pnl, avg_pnl.
    """
    if trades.empty or "entry_time" not in trades.columns:
        return pd.DataFrame()
    # Detect regimes
    regimes = detect_regimes(df)
    trades = trades.copy()
    # Map each trade's entry time to nearest bar
    nearest_idx = df.index.get_indexer(trades["entry_time"], method="nearest")
    trades["regime"] = [regimes.iloc[i] if i < len(regimes) else "unknown"
                          for i in nearest_idx]
    # Aggregate
    summary = trades.groupby("regime").agg(
        n_trades=("pnl", "size"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
        net_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    ).reset_index()
    return summary