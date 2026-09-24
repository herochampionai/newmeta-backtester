"""Comprehensive metrics — every metric a quant cares about.
Returns dict, easy to render as cards in Streamlit."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _safe_scalar(x, default: float = 0.0) -> float:
    """Convert any near-scalar (Series, ndarray, list, None, NaN/inf) to float.
    Returns default if conversion fails or value is non-finite.
    """
    try:
        if x is None:
            return default
        if isinstance(x, (pd.Series, pd.DataFrame)):
            if len(x) == 0:
                return default
            x = x.iloc[0] if hasattr(x, "iloc") else x
        elif isinstance(x, (list, tuple, np.ndarray)):
            x = x[0] if len(x) > 0 else default
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except (TypeError, ValueError, IndexError):
        return default


def compute_all(returns: pd.Series, trades: pd.DataFrame | None = None,
                equity: pd.Series | None = None, periods_per_year: int = 252 * 24,
                risk_free: float = 0.0, init_cash: float = 10000.0) -> dict:
    """returns: bar-level returns (e.g. H1). trades: optional per-trade log.
    equity: optional equity curve; computed from returns if not provided."""
    r = returns.dropna()
    if r.empty:
        return {"error": "empty returns"}
    if equity is None:
        equity = (1 + r).cumprod()
    n = len(r)
    years = n / periods_per_year

    # Basic returns (equity is in dollars, normalize) — defensive coercion
    final_dollars = _safe_scalar(equity.iloc[-1] if len(equity) > 0 else None, init_cash)
    initial_dollars = _safe_scalar(equity.iloc[0] if len(equity) > 0 else None, init_cash)
    if initial_dollars <= 0 or not np.isfinite(initial_dollars):
        initial_dollars = init_cash
    if final_dollars <= 0 or not np.isfinite(final_dollars):
        final_dollars = init_cash
    total_ret = (final_dollars / initial_dollars) - 1
    cagr = (final_dollars / initial_dollars) ** (1 / max(years, 1e-9)) - 1
    cagr = _safe_scalar(cagr)
    total_ret = _safe_scalar(total_ret)
    vol = _safe_scalar(r.std() * np.sqrt(periods_per_year))
    if vol > 0 and np.isfinite(vol):
        sharpe = float((cagr - risk_free) / vol)
    else:
        sharpe = 0.0
    downside = r[r < 0]
    dd_std = _safe_scalar(downside.std() * np.sqrt(periods_per_year)) if len(downside) else 0.0
    if dd_std > 0 and np.isfinite(dd_std):
        sortino = float((cagr - risk_free) / dd_std)
    else:
        sortino = 0.0

    # Drawdown
    peak = equity.cummax()
    dd = (equity / peak - 1)
    max_dd = _safe_scalar(dd.min())
    calmar = float(cagr / abs(max_dd)) if (max_dd < 0 and max_dd != 0) else 0.0
    # Drawdown duration
    is_dd = dd < 0
    dd_groups = (is_dd != is_dd.shift()).cumsum()
    dd_durations = dd.groupby(dd_groups).size()
    longest_dd_bars = int(dd_durations.max()) if len(dd_durations) else 0

    # Recovery factor = total return / max DD
    recovery_factor = float(total_ret / abs(max_dd)) if (max_dd < 0 and max_dd != 0) else 0.0

    # Stability = R² of log equity vs time (higher = smoother equity curve)
    if n > 2:
        try:
            log_eq = np.log(np.asarray(equity.values, dtype=float))
            # Guard against -inf / nan in log
            if not np.all(np.isfinite(log_eq)):
                stability = 0.0
            else:
                t = np.arange(n)
                slope, intercept = np.polyfit(t, log_eq, 1)
                ss_res = np.sum((log_eq - (slope * t + intercept)) ** 2)
                ss_tot = np.sum((log_eq - log_eq.mean()) ** 2)
                stability = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        except Exception:
            stability = 0.0
    else:
        stability = 0.0

    out = {
        # Headline
        "total_return": total_ret,
        "cagr": cagr,
        "final_equity": float(equity.iloc[-1]),
        # Risk-adjusted
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "stability": stability,
        "vol": vol,
        # Risk
        "max_drawdown": max_dd,
        "longest_dd_bars": longest_dd_bars,
        "recovery_factor": recovery_factor,
        # Trade stats
        "n_bars": n,
        "n_positive_bars": int((r > 0).sum()),
        "win_rate_bars": float((r > 0).mean()),
    }

    # Per-trade metrics (if log provided)
    if trades is not None and len(trades) > 0:
        # Normalize PnL column (vectorbt uses 'PnL', grid uses 'pnl')
        pnl_col = None
        for c in ("pnl", "PnL", "profit", "Profit"):
            if c in trades.columns:
                pnl_col = c
                break
        if pnl_col is None:
            # Try to compute from column
            return out
        pnls = trades[pnl_col]
        if len(pnls) > 0:
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            n_trades = len(pnls)
            n_wins = len(wins)
            n_losses = len(losses)
            win_rate = n_wins / n_trades if n_trades else 0.0
            avg_win = float(wins.mean()) if len(wins) else 0.0
            avg_loss = float(losses.mean()) if len(losses) else 0.0
            # Profit factor
            gross_profit = float(wins.sum()) if len(wins) else 0.0
            gross_loss = float(-losses.sum()) if len(losses) else 0.0
            # PF is undefined when gross_loss == 0. Use None (caller decides display)
            # and a numeric proxy "n_losses" so the absence of losses is unambiguous.
            if gross_loss > 0:
                pf = gross_profit / gross_loss
            elif gross_profit > 0:
                pf = float("inf")  # all winners, no losers — mathematically undefined
            else:
                pf = 0.0  # no winners, no losers (no trades at all)
            # Expectancy = (win_rate * avg_win) - ((1-win_rate) * abs(avg_loss))
            expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
            # Largest win/loss
            largest_win = float(pnls.max())
            largest_loss = float(pnls.min())
            # Avg bars held
            if "entry_time" in trades.columns and "exit_time" in trades.columns:
                held = (pd.to_datetime(trades["exit_time"]) -
                        pd.to_datetime(trades["entry_time"])).dt.total_seconds() / 3600
                avg_held = float(held.mean()) if len(held) else 0.0
                out["avg_held_hours"] = avg_held
            out.update({
                "n_trades": n_trades,
                "n_wins": n_wins,
                "n_losses": n_losses,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "largest_win": largest_win,
                "largest_loss": largest_loss,
                # When PF is undefined (no losses), use a clearly-finite numeric cap of 99
                # so it sorts/ranks correctly. UI should render this as "—" or "∞".
                "profit_factor": (99.0 if pf == float("inf") else float(pf)),
                "profit_factor_undefined": pf == float("inf"),  # flag for UI
                "expectancy": expectancy,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "net_pnl": float(pnls.sum()),
            })
    return out


def to_card_metrics(m: dict) -> dict:
    """Format metric values for display (round, add units)."""
    out = {}
    for k, v in m.items():
        if isinstance(v, float):
            if abs(v) < 10 and abs(v) > -10 and k not in ("final_equity", "n_bars", "n_trades", "n_wins", "n_losses"):
                out[k] = round(v, 4)
            else:
                out[k] = round(v, 2)
        else:
            out[k] = v
    return out


def tester_statistics(metrics: dict, trades: pd.DataFrame | None = None,
                      equity: pd.Series | None = None,
                      init_cash: float = 10000.0) -> dict:
    """MQL5 TesterStatistics() mirror — same names MT5 shows in Results tab.

    Keys follow STAT_* naming so OnTester-style custom criteria port 1:1:
    STAT_PROFIT, STAT_TRADES, STAT_PROFIT_FACTOR, STAT_RECOVERY_FACTOR,
    STAT_SHARPE_RATIO, STAT_EXPECTED_PAYOFF, STAT_EQUITY_DD / _PERCENT /
    _RELATIVE, STAT_BALANCE_DD*, STAT_WIN_*, STAT_CONPROFITMAX/MIN etc.
    Missing broker-only fields (margin level) return None explicitly.
    """
    m = dict(metrics or {})
    net = float(m.get("net_pnl", 0) or 0)
    gross_p = float(m.get("gross_profit", 0) or 0)
    gross_l = float(m.get("gross_loss", 0) or 0)
    pf = float(m.get("profit_factor", 0) or 0)
    rec = float(m.get("recovery_factor", 0) or 0)
    sharpe = float(m.get("sharpe", 0) or 0)
    n = int(m.get("n_trades", 0) or 0)
    n_wins = int(m.get("n_wins", 0) or 0)
    exp_payoff = float(m.get("expectancy", (net / n) if n else 0) or 0)
    max_dd = float(m.get("max_drawdown", 0) or 0)  # negative fraction

    # Drawdowns in money + percent from equity curve (balance == equity here:
    # single-asset, no open-position carry — matches MT5 when no overnight holds)
    eq_dd_money = 0.0
    eq_dd_pct = 0.0
    if equity is not None and len(equity) > 1:
        try:
            peak = equity.cummax()
            dd_money = (equity - peak).min()
            eq_dd_money = abs(float(dd_money))
            peak_at_dd = float(peak[(equity - peak).idxmin()])
            eq_dd_pct = (eq_dd_money / peak_at_dd * 100.0) if peak_at_dd else 0.0
        except Exception:
            pass
    if not eq_dd_money and max_dd < 0:
        eq_dd_money = abs(max_dd) * float(init_cash)
        eq_dd_pct = abs(max_dd) * 100.0

    # Consecutive + direction splits from trade log
    con_profit_max = con_loss_max = 0
    max_win_trade = float(m.get("largest_win", 0) or 0)
    max_loss_trade = float(m.get("largest_loss", 0) or 0)
    short_trades = long_trades = win_short = win_long = 0
    try:
        if trades is not None and len(trades) > 0:
            pnl_col = next((c for c in ("pnl", "PnL", "profit", "Profit") if c in trades.columns), None)
            dir_col = next((c for c in ("direction", "Direction", "type") if c in trades.columns), None)
            if pnl_col:
                signs = (trades[pnl_col] > 0).astype(int).tolist()
                best = cur = 0
                for s in signs:
                    cur = cur + 1 if s == 1 else 0
                    best = max(best, cur)
                con_profit_max = int(best)
                best = cur = 0
                for s in signs:
                    cur = cur + 1 if s == 0 else 0
                    best = max(best, cur)
                con_loss_max = int(best)
            if dir_col is not None and pnl_col:
                longs = trades[trades[dir_col] > 0]
                shorts = trades[trades[dir_col] < 0]
                long_trades, short_trades = len(longs), len(shorts)
                win_long = int((longs[pnl_col] > 0).sum()) if len(longs) else 0
                win_short = int((shorts[pnl_col] > 0).sum()) if len(shorts) else 0
    except Exception:
        pass

    return {
        "STAT_INITIAL_DEPOSIT": float(init_cash),
        "STAT_PROFIT": net,
        "STAT_GROSS_PROFIT": gross_p,
        "STAT_GROSS_LOSS": gross_l,
        "STAT_PROFIT_FACTOR": pf,
        "STAT_RECOVERY_FACTOR": rec,
        "STAT_SHARPE_RATIO": sharpe,
        "STAT_EXPECTED_PAYOFF": exp_payoff,
        "STAT_TRADES": n,
        "STAT_WIN_TRADES": n_wins,
        "STAT_LOSS_TRADES": max(n - n_wins, 0),
        "STAT_WIN_PERCENT": (n_wins / n * 100.0) if n else 0.0,
        "STAT_EQUITY_DD": eq_dd_money,
        "STAT_EQUITY_DD_PERCENT": eq_dd_pct,
        "STAT_EQUITY_DD_RELATIVE": eq_dd_pct,
        "STAT_BALANCE_DD": eq_dd_money,
        "STAT_BALANCE_DD_PERCENT": eq_dd_pct,
        "STAT_MAX_PROFITTRADE": max_win_trade,
        "STAT_MAX_LOSSTRADE": max_loss_trade,
        "STAT_CONPROFITMAX": con_profit_max,
        "STAT_CONLOSSMAX": con_loss_max,
        "STAT_SHORT_TRADES": short_trades,
        "STAT_LONG_TRADES": long_trades,
        "STAT_WIN_SHORT_TRADES": win_short,
        "STAT_WIN_LONG_TRADES": win_long,
        "STAT_MIN_MARGINLEVEL": None,  # needs margin model — not simulated
    }