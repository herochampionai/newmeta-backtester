"""Smart merge strategies — OR-gate, vote, confirmation window, primary+filter.

BUGFIX 2026-09-18: BaseStrategy is @dataclass, so passing params as positional
arg overwrites `name`. Use super().__init__(name=...) and store raw params separately.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind
from ._base import BaseStrategy, Signals


def _empty_signals(idx):
    z = pd.Series(False, index=idx)
    return Signals(entries=z.copy(), exits=z.copy(), direction=pd.Series(0, index=idx, dtype=int))


class VoteStrategy(BaseStrategy):
    """N strategies vote BUY/SELL/HOLD. Trade fires when |net votes| >= threshold.
    vote_threshold=1 means "any strategy can fire it" (OR-gate).
    vote_threshold=N means all N must agree (AND-gate).
    """
    name = "vote"

    def __init__(self, params=None):
        super().__init__(name="vote")
        self._raw_params = params or {}
        self._strategies = []
        self._params_list = []
        if self._raw_params:
            n = int(self._raw_params.get('n_strategies', 2))
            for i in range(n):
                cls = self._raw_params.get(f'_strategy{i+1}')
                p = self._raw_params.get(f'_strategy{i+1}_params', {})
                if cls is not None:
                    self._strategies.append(cls)
                    self._params_list.append(p)

    def generate(self, df):
        if not self._strategies:
            return _empty_signals(df.index)
        threshold = int(self._raw_params.get('vote_threshold', len(self._strategies) // 2 + 1))
        cooldown = int(self._raw_params.get('cooldown', 0))

        votes = np.zeros(len(df), dtype=int)
        for cls, p in zip(self._strategies, self._params_list):
            sub = cls(p)
            sig = sub.generate(df)
            d = sig.direction
            if isinstance(d, np.ndarray):
                d = pd.Series(d, index=df.index)
            votes += d.fillna(0).astype(int).values

        # Fire when |votes| >= threshold
        direction = np.where(votes >= threshold, 1, np.where(votes <= -threshold, -1, 0))
        direction = pd.Series(direction, index=df.index, dtype=int)

        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


class PrimaryWithFilterStrategy(BaseStrategy):
    """PRIMARY fires entry. FILTER must agree within N bars."""
    name = "primary_with_filter"

    def __init__(self, params=None):
        super().__init__(name="primary_with_filter")
        self._raw_params = params or {}
        self._primary = None
        self._filter = None
        if self._raw_params:
            p_cls = self._raw_params.get('_primary')
            p_p = self._raw_params.get('_primary_params', {})
            f_cls = self._raw_params.get('_filter')
            f_p = self._raw_params.get('_filter_params', {})
            if p_cls is not None:
                self._primary = p_cls(p_p)
            if f_cls is not None:
                self._filter = f_cls(f_p)

    def generate(self, df):
        if self._primary is None or self._filter is None:
            return _empty_signals(df.index)
        confirm_window = int(self._raw_params.get('confirm_window', 5))
        cooldown = int(self._raw_params.get('cooldown', 0))

        sig_p = self._primary.generate(df)
        sig_f = self._filter.generate(df)
        d_p = sig_p.direction.values if hasattr(sig_p.direction, 'values') else sig_p.direction
        d_f = sig_f.direction.values if hasattr(sig_f.direction, 'values') else sig_f.direction
        if isinstance(d_p, np.ndarray) is False:
            d_p = np.array(d_p)
        if isinstance(d_f, np.ndarray) is False:
            d_f = np.array(d_f)

        direction = np.zeros(len(df), dtype=int)
        for i in range(len(df)):
            if d_p[i] == 0:
                continue
            end = min(i + confirm_window + 1, len(df))
            for j in range(i + 1, end):
                if d_f[j] == d_p[i]:
                    direction[i] = d_p[i]
                    break

        direction = pd.Series(direction, index=df.index, dtype=int)

        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig


class WeightedVotingStrategy(BaseStrategy):
    """Each strategy has a weight. Trade fires when weighted sum > threshold."""
    name = "weighted_voting"

    def __init__(self, params=None):
        super().__init__(name="weighted_voting")
        self._raw_params = params or {}
        self._strategies = []
        self._weights = []
        if self._raw_params:
            n = int(self._raw_params.get('n_strategies', 2))
            for i in range(n):
                cls = self._raw_params.get(f'_strategy{i+1}')
                p = self._raw_params.get(f'_strategy{i+1}_params', {})
                w = float(self._raw_params.get(f'_weight{i+1}', 1.0))
                if cls is not None:
                    self._strategies.append(cls)
                    self._weights.append(w)

    def generate(self, df):
        if not self._strategies:
            return _empty_signals(df.index)
        threshold = float(self._raw_params.get('threshold', 0.0))
        cooldown = int(self._raw_params.get('cooldown', 0))

        score = np.zeros(len(df), dtype=float)
        for cls, w in zip(self._strategies, self._weights):
            sub = cls({})
            sig = sub.generate(df)
            d = sig.direction
            if isinstance(d, np.ndarray):
                d = pd.Series(d, index=df.index)
            score += w * d.fillna(0).astype(float).values

        direction = np.where(score > threshold, 1, np.where(score < -threshold, -1, 0))
        direction = pd.Series(direction, index=df.index, dtype=int)

        if cooldown > 0:
            new_dir = direction.values.copy()
            last_idx = -999
            for i in range(len(direction)):
                if new_dir[i] != 0:
                    if i - last_idx < cooldown:
                        new_dir[i] = 0
                    else:
                        last_idx = i
            direction = pd.Series(new_dir, index=df.index, dtype=int)

        sig = _empty_signals(df.index)
        sig.entries = direction != 0
        sig.direction = direction
        return sig
