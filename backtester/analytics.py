"""Trade-frequency analytics — answers the user's questions:
  - How many trades per year per strategy
  - Lowest-performing strategies
  - Streak distribution (longest win/loss streak)
  - Sharpe, profit factor per strategy
  - Adaptive sizer state over time
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from collections import Counter


def trades_per_year(trades: pd.DataFrame, year_col: str = "exit_bar",
                    bar_index_to_date: pd.Series | None = None) -> dict:
    """Returns dict: {year: n_trades, total: avg_per_year}."""
    if trades.empty:
        return {}
    if bar_index_to_date is not None and "exit_time" in trades.columns:
        trades = trades.copy()
        trades["year"] = pd.to_datetime(trades["exit_time"]).dt.year
    elif "year" not in trades.columns:
        return {"total": len(trades)}
    counts = trades["year"].value_counts().sort_index()
    return dict(counts)


def streak_stats(trades: pd.DataFrame, pnl_col: str = "pnl") -> dict:
    """Compute longest win/loss streaks from trade list.
    Auto-detects PnL column name (pnl, PnL, profit, Profit)."""
    if trades.empty:
        return {}
    for c in (pnl_col, "pnl", "PnL", "profit", "Profit"):
        if c in trades.columns:
            pnls = trades[c].values
            break
    else:
        return {}
    is_win = pnls > 0
    longest_win = 0
    longest_loss = 0
    cur_win = 0
    cur_loss = 0
    for w in is_win:
        if w:
            cur_win += 1
            cur_loss = 0
            longest_win = max(longest_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            longest_loss = max(longest_loss, cur_loss)
    return {
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
        "total_wins": int(is_win.sum()),
        "total_losses": int((~is_win).sum()),
    }


def strategy_scoreboard(per_strategy_metrics: dict[str, dict],
                          rank_by: str = "sharpe") -> pd.DataFrame:
    """Build a ranked scoreboard across strategies."""
    rows = []
    for strat, m in per_strategy_metrics.items():
        rows.append({
            "strategy": strat,
            "sharpe": m.get("sharpe", 0),
            "calmar": m.get("calmar", 0),
            "win_rate": m.get("win_rate", 0),
            "profit_factor": m.get("profit_factor", 0),
            "net_pnl": m.get("net_pnl", 0),
            "n_trades": m.get("n_trades", 0),
            "max_dd": m.get("max_drawdown", 0),
            "expectancy": m.get("expectancy", 0),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    return df


def annual_trade_count(n_trades: int, n_bars: int, periods_per_year: int) -> float:
    """Trades per year = n_trades * (periods_per_year / n_bars)."""
    if n_bars == 0:
        return 0
    return n_trades * (periods_per_year / n_bars)


def lowest_performers(scoreboard: pd.DataFrame, n: int = 3,
                      by: str = "sharpe") -> pd.DataFrame:
    """Return bottom-N strategies by given metric."""
    if scoreboard.empty:
        return scoreboard
    return scoreboard.nsmallest(n, by).reset_index(drop=True)