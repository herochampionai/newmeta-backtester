"""Universe screener — which ticker fits which strategy.

Runs a strategy x symbol matrix over cached data and ranks every cell by a
criterion (default: composite score). Answers the two questions that matter
before any fine-tuning:

    best symbol for strategy X?   -> best_per_strategy
    best strategy for symbol Y?    -> best_per_symbol

Faster than MQL5's one-symbol-at-a-time optimization: cells run in parallel
workers (ProcessPoolExecutor) with automatic serial fallback, shared cache
reads, and a single JSON report.

Usage:
    from analysis.universe_screener import screen, discover_symbols
    rep = screen([{"name": "adx", "params": {...}}],
                 symbols=["EURUSD", "GBPUSD"], timeframe="H1")
    print(rep.best_per_strategy)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time


def discover_symbols(timeframe: str = "H1") -> list[str]:
    """Find symbols with cached data for this timeframe.

    Handles EURUSD_H1_<hash>.parquet, EURUSD_H1_duka_<hash>.parquet,
    XAUUSD_M1.parquet naming. Prefers duka files when both exist.
    """
    cache = Path("data/cache")
    if not cache.exists():
        return []
    tf = timeframe.upper()
    seen: dict[str, Path] = {}
    for p in sorted(cache.glob("*.parquet"), key=lambda x: x.stat().st_mtime,
                    reverse=True):
        parts = p.stem.split("_")
        if len(parts) < 2:
            continue
        sym, file_tf = parts[0].upper(), parts[1].upper()
        if file_tf != tf or not sym.isalpha() or len(sym) < 6:
            continue
        # Prefer duka (real ticks) over Yahoo when both cached.
        if sym not in seen or "_duka_" in p.name:
            seen[sym] = p
    return sorted(seen)


def _load_cached(symbol: str, timeframe: str, min_bars: int = 200):
    """Cache-only load (never downloads — screener must stay fast).

    Tries candidates newest-first (duka preferred) and takes the first with
    enough bars — a thin partial cache must not shadow a full one.
    """
    import pandas as pd
    cands = sorted(Path("data/cache").glob(f"{symbol.upper()}_{timeframe.upper()}*.parquet"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    # Prefer duka files.
    cands.sort(key=lambda p: ("_duka_" not in p.name, -p.stat().st_mtime))
    for c in cands:
        try:
            df = pd.read_parquet(c)
        except Exception:
            continue
        if df is not None and len(df) >= min_bars:
            return df
    return None


def _screen_cell(job: tuple) -> dict:
    """One strategy x symbol cell. Module-level so workers can pickle it."""
    strategy_name, params, symbol, timeframe, capital, exec_cfg, min_bars = job
    import pandas as pd
    from backtester.engine_full import run_full
    import importlib
    from strategies._base import BaseStrategy

    t0 = time.time()
    try:
        df = _load_cached(symbol, timeframe, min_bars=min_bars)
        if df is None:
            return {"strategy": strategy_name, "symbol": symbol, "ok": False,
                    "error": f"no cached data with >={min_bars} bars",
                    "elapsed_sec": round(time.time() - t0, 1)}
        # Honest costs by default (same rule as run_pipeline): explicit spread
        # wins, else bar mean spread_pips, else 0.5p. Zero-spread screening
        # ranks fantasy fills first.
        exec_cfg = dict(exec_cfg or {})
        exec_cfg.setdefault("slippage_pips", 0.3)
        if "spread_pips" not in exec_cfg:
            exec_cfg["spread_pips"] = (float(df["spread_pips"].mean())
                                       if "spread_pips" in df.columns else 0.5)
        if strategy_name == "adx":
            from strategies.adx import ADX_Strategy
            strat = ADX_Strategy(name="adx", params=params)
        else:
            mod = importlib.import_module(f"strategies.{strategy_name}")
            cls = next(v for v in vars(mod).values()
                       if isinstance(v, type) and issubclass(v, BaseStrategy)
                       and v is not BaseStrategy)
            strat = cls(name=strategy_name, params=params)
        sig = strat.generate(df)
        r = run_full(df, {strategy_name: (sig.entries.values.astype(int),
                                          sig.exits.values.astype(int))},
                     init_cash=capital, strict_data=False, **exec_cfg)
        m = r.get("metrics", {})
        trades = r.get("trades", pd.DataFrame())
        return {"strategy": strategy_name, "symbol": symbol, "ok": True,
                "net_pnl": round(float(m.get("net_pnl", 0) or 0), 2),
                "sharpe": round(float(m.get("sharpe", 0) or 0), 3),
                "max_drawdown": round(float(m.get("max_drawdown", 0) or 0), 4),
                "trades": len(trades),
                "win_rate": round(float(m.get("win_rate", 0) or 0), 4),
                "bars": len(df),
                "elapsed_sec": round(time.time() - t0, 1)}
    except Exception as e:
        return {"strategy": strategy_name, "symbol": symbol, "ok": False,
                "error": f"{type(e).__name__}: {str(e)[:120]}",
                "elapsed_sec": round(time.time() - t0, 1)}


@dataclass
class ScreenReport:
    """Ranked strategy x symbol matrix."""
    timeframe: str
    rows: list = field(default_factory=list)
    best_per_strategy: dict = field(default_factory=dict)
    best_per_symbol: dict = field(default_factory=dict)
    rejected_symbols: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, out_path: str | Path) -> str:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return str(out_path)


def _score_row(row: dict, criterion: str) -> float:
    if criterion == "sharpe":
        return row.get("sharpe", 0)
    if criterion == "net_pnl":
        return row.get("net_pnl", 0)
    if criterion == "profit_factor":
        return row.get("net_pnl", 0) / max(1.0, abs(row.get("max_drawdown", 0)) * 10000)
    # default: composite — Sharpe with drawdown + trade-count guards
    dd = abs(row.get("max_drawdown", 0))
    trades = row.get("trades", 0)
    if trades < 10 or row.get("net_pnl", 0) <= 0:
        return float("-inf")
    return row.get("sharpe", 0) - dd * 2.0 + min(trades, 200) / 200.0 * 0.1


def screen(strategies: list[dict], symbols: list[str] | None = None,
           timeframe: str = "H1", capital: float = 10000.0,
           criterion: str = "composite", exec_cfg: dict | None = None,
           jobs: int = 1, min_bars: int = 200, verbose: bool = True,
           min_avg_volume: float = 0.0, max_avg_spread_pips: float | None = None,
           pip_size: float = 0.0001) -> ScreenReport:
    """Run the matrix. Parallel workers with serial fallback.

    strategies: [{"name": "adx", "params": {...}}, ...]
    symbols: None = auto-discover cached symbols for the timeframe.
    exec_cfg: forwarded to run_full (spread_pips/slippage_pips/tick_mode/...).
    Tradability universe filters (LEAN-style selection discipline):
      min_avg_volume: skip symbols averaging less bar volume (0 = off).
      max_avg_spread_pips: skip symbols wider than this (needs spread_pips
        column; None = off). Rejected symbols are reported, not hidden.
    """
    import concurrent.futures
    t0 = time.time()
    if symbols is None:
        symbols = discover_symbols(timeframe)
    exec_cfg = exec_cfg or {}
    # Tradability pre-filter: measure each symbol once, reject thin/wide
    # markets up front (reported in rejected_symbols, never silent).
    rejected_symbols: dict[str, str] = {}
    if min_avg_volume > 0 or max_avg_spread_pips is not None:
        kept = []
        for sym in symbols:
            df0 = _load_cached(sym, timeframe, min_bars=min_bars)
            if df0 is None:
                rejected_symbols[sym] = "no cached data"
                continue
            avg_vol = float(df0["volume"].mean()) if "volume" in df0.columns else 0.0
            if min_avg_volume > 0 and avg_vol < min_avg_volume:
                rejected_symbols[sym] = f"avg_volume {avg_vol:.1f} < {min_avg_volume}"
                continue
            if max_avg_spread_pips is not None and "spread_pips" in df0.columns:
                avg_sp = float(df0["spread_pips"].mean())
                if avg_sp > max_avg_spread_pips:
                    rejected_symbols[sym] = (f"avg_spread {avg_sp:.2f}p "
                                             f"> {max_avg_spread_pips}p")
                    continue
            kept.append(sym)
        symbols = kept
        if verbose:
            for sym, why in rejected_symbols.items():
                print(f"  [universe] reject {sym}: {why}")
    jobspec = [(s["name"], s.get("params", {}), sym, timeframe, capital, exec_cfg, min_bars)
               for s in strategies for sym in symbols]
    rows: list[dict] = []
    if jobs > 1 and len(jobspec) > 1:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
                for row in ex.map(_screen_cell, jobspec):
                    rows.append(row)
                    if verbose:
                        _print_row(row, criterion)
        except Exception as e:
            print(f"  [screen] parallel failed ({type(e).__name__}), serial fallback")
            rows = []
    if not rows:
        for job in jobspec:
            row = _screen_cell(job)
            rows.append(row)
            if verbose:
                _print_row(row, criterion)
    ok_rows = [r for r in rows if r.get("ok")]
    for r in ok_rows:
        r["_score"] = _score_row(r, criterion)
    best_per_strategy: dict = {}
    best_per_symbol: dict = {}
    for s in strategies:
        cands = [r for r in ok_rows if r["strategy"] == s["name"]]
        passing = [r for r in cands if r["_score"] != float("-inf")]
        pool = passing or cands
        if pool:
            b = max(pool, key=lambda r: r["_score"])
            best_per_strategy[s["name"]] = {
                "symbol": b["symbol"],
                "score": (round(b["_score"], 3) if b["_score"] != float("-inf") else None),
                "passing": bool(passing),
                "net_pnl": b["net_pnl"], "sharpe": b["sharpe"],
                "trades": b["trades"]}
    for sym in symbols:
        cands = [r for r in ok_rows if r["symbol"] == sym]
        passing = [r for r in cands if r["_score"] != float("-inf")]
        pool = passing or cands
        if pool:
            b = max(pool, key=lambda r: r["_score"])
            best_per_symbol[sym] = {
                "strategy": b["strategy"],
                "score": (round(b["_score"], 3) if b["_score"] != float("-inf") else None),
                "passing": bool(passing),
                "net_pnl": b["net_pnl"], "sharpe": b["sharpe"],
                "trades": b["trades"]}
    for r in rows:
        r.pop("_score", None)
    return ScreenReport(timeframe=timeframe, rows=rows,
                        best_per_strategy=best_per_strategy,
                        best_per_symbol=best_per_symbol,
                        rejected_symbols=rejected_symbols,
                        elapsed_sec=round(time.time() - t0, 1))


def _print_row(row: dict, criterion: str):
    if not row.get("ok"):
        print(f"  {row['strategy']} x {row['symbol']}: FAIL {row.get('error', '')}")
    else:
        print(f"  {row['strategy']} x {row['symbol']}: "
              f"PnL=${row['net_pnl']:.2f} Sharpe={row['sharpe']:.3f} "
              f"n={row['trades']} ({row['elapsed_sec']:.1f}s)")


# ---------- Self-test ----------
if __name__ == "__main__":
    syms = discover_symbols("H1")
    print("discovered:", syms)
    assert "EURUSD" in syms
    rep = screen([{"name": "adx", "params": {"bars_calculate": 10}}],
                 symbols=[s for s in syms if s in ("EURUSD", "BTCUSDT", "XAUUSD")][:2],
                 timeframe="H1", jobs=1, verbose=True)
    print("best_per_strategy:", rep.best_per_strategy)
    assert rep.rows and any(r.get("ok") for r in rep.rows)
    p = rep.save("output/reports/_selftest_screen.json")
    print("saved:", p)
    print("SELF-TEST PASS")
