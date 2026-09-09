"""Adaptive position sizing — streak-aware, drawdown-aware.

Two regimes:
  - normal: full base lot
  - cautious: reduced lot (during losing streak or DD)
  - hot:      modestly increased lot (during winning streak)
  - pause:   no trades (during severe DD)

Detection:
  - recent N trades: if all losses > cautious threshold
  - rolling drawdown: max DD over last K trades > pause threshold
  - win rate window: if win rate < 40% over last 20 trades -> cautious

Inputs (typical):
  base_lot: 0.1
  cautious_threshold: 3 consecutive losses
  pause_threshold:    -10% rolling DD
  hot_threshold:      5 consecutive wins
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass


@dataclass
class AdaptiveConfig:
    base_lot: float = 0.1
    cautious_after: int = 3          # consecutive losses
    hot_after: int = 5               # consecutive wins
    cautious_multiplier: float = 0.5  # size in cautious regime
    hot_multiplier: float = 1.2       # size in hot regime
    pause_dd_pct: float = 0.10        # rolling DD to pause
    pause_window: int = 30            # lookback window for rolling DD
    max_lot: float = 5.0


class AdaptiveSizer:
    """Stream lot sizes based on streak + rolling drawdown."""
    def __init__(self, cfg: AdaptiveConfig | None = None):
        self.cfg = cfg or AdaptiveConfig()
        self.recent_results: deque[bool] = deque(maxlen=20)  # True = win
        self.recent_pnls: deque[float] = deque(maxlen=self.cfg.pause_window)
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.paused = False
        self.pause_until_equity = None

    def on_trade_close(self, pnl: float) -> None:
        """Called after each trade closes."""
        self.recent_results.append(pnl > 0)
        self.recent_pnls.append(pnl)
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        # Check pause condition
        if len(self.recent_pnls) >= self.cfg.pause_window:
            equity = 10000 + sum(self.recent_pnls)  # simple sum (good enough for rolling check)
            peak = max(10000, max(10000 + sum(list(self.recent_pnls)[:i+1])
                                     for i in range(len(self.recent_pnls))))
            dd = (equity - peak) / peak
            if dd <= -self.cfg.pause_dd_pct:
                self.paused = True

    def get_lot(self, base_lot: float | None = None) -> float:
        """Return current lot size. 0 if paused."""
        if self.paused:
            return 0.0
        base = base_lot or self.cfg.base_lot
        if self.consecutive_losses >= self.cfg.cautious_after:
            return base * self.cfg.cautious_multiplier
        if self.consecutive_wins >= self.cfg.hot_after:
            return min(base * self.cfg.hot_multiplier, self.cfg.max_lot)
        return base

    def get_state(self) -> dict:
        return {
            "paused": self.paused,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "recent_win_rate": (sum(self.recent_results) / len(self.recent_results)
                                  if self.recent_results else 0.0),
            "current_lot": self.get_lot(),
        }

    def reset(self):
        self.recent_results.clear()
        self.recent_pnls.clear()
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.paused = False