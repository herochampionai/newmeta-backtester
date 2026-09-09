"""Orchestrator. Loads config + cached data, runs backtest for each strategy,
allocates via Markowitz, runs walk-forward + Monte Carlo, writes outputs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

from data.cache import load as load_cache
from data.mt5_export import fetch_bars, find_terminal, init_mt5
from strategies import STRATEGY_REGISTRY
from backtester.engine import run_direction, run_portfolio
from analysis.optuna_optimizer import optimize_strategy, best_params, pareto_df
from analysis.markowitz_alloc import allocate
from analysis.walkforward import walk_forward, wf_summary
from analysis.montecarlo import ci_metrics

ROOT = Path(__file__).parent
OUT = ROOT / "output"


def load_config() -> dict:
    with open(ROOT / "config" / "strategies.yaml") as f:
        return yaml.safe_load(f)


def fetch_or_load(cfg: dict) -> pd.DataFrame:
    sym = cfg["common"]["symbol"]
    tf = cfg["common"]["timeframe"]
    try:
        df, meta = load_cache(sym, tf)
        print(f"[data] cache hit {sym} {tf} rows={len(df)} sha={meta['sha']}")
        return df
    except FileNotFoundError:
        pass
    print(f"[data] no cache; fetching {sym} {tf} from MT5...")
    init_mt5(find_terminal())
    df = fetch_bars(sym, tf, cfg["common"]["start"], cfg["common"].get("end"))
    mt5.shutdown()
    return df


def run_single(strategy_name: str, df: pd.DataFrame, params: dict, cfg: dict):
    cls = STRATEGY_REGISTRY[strategy_name]
    strat = cls(params=params)
    sig = strat.generate(df)
    pf, summary = run_direction(df, sig.entries, sig.direction,
                                init_cash=cfg["common"]["initial_cash"],
                                commission=cfg["common"]["commission"],
                                slippage=cfg["common"]["slippage"])
    return pf, summary, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="all",
                    help="strategy name or 'all'")
    ap.add_argument("--optimize", action="store_true", help="run Optuna before backtest")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--montecarlo", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    df = fetch_or_load(cfg)

    OUT.mkdir(exist_ok=True)
    if args.strategy == "all":
        names = [n for n, sp in cfg.items()
                 if n not in ("common", "allocation", "optimization")
                 and isinstance(sp, dict) and sp.get("enabled", True)]
    else:
        names = [args.strategy]

    summaries = {}
    signals_dict = {}
    for name in names:
        spec = cfg[name]
        params = {k: v[0] if isinstance(v, list) and len(v) == 2
                  else v for k, v in spec.items() if k != "enabled"}
        if args.optimize:
            print(f"[opt] optimizing {name} ({args.n_trials} trials)...")
            study = optimize_strategy(name, df, spec, n_trials=args.n_trials)
            params = best_params(study, metric="sharpe")
            pd.DataFrame(study.best_trials).to_csv(OUT / f"{name}_optuna.csv", index=False)
            print(f"[opt] {name} best sharpe params: {params}")

        pf, summary, sig = run_single(name, df, params, cfg)
        summaries[name] = summary
        signals_dict[name] = sig
        print(f"[{name}] {summary}")
        pd.Series(summary).to_json(OUT / f"{name}_summary.json")

    # Portfolio allocation (if multiple strategies)
    if len(summaries) > 1 and "allocation" in cfg:
        method = cfg["allocation"].get("method", "markowitz")
        max_w = cfg["allocation"].get("max_weight", 0.4)
        max_dd = cfg["allocation"].get("max_drawdown_constraint", 0.15)
        # Per-strategy bar returns
        rets = pd.DataFrame({n: pf_returns(df, signals_dict[n]) for n in names})
        rets = rets.fillna(0)
        # Daily aggregation
        daily = (1 + rets).groupby(rets.index.date).prod() - 1
        daily.index = pd.to_datetime(daily.index)
        w = allocate(daily, method=method, max_w=max_w, max_dd=max_dd)
        port_summary = run_portfolio(rets, w, init_cash=cfg["common"]["initial_cash"])
        print(f"[portfolio] weights ({method}): {dict(zip(names, w.round(3)))}")
        print(f"[portfolio] summary: {port_summary}")
        pd.Series(port_summary).to_json(OUT / "portfolio_summary.json")

    # Walk-forward
    if args.walkforward and len(names) == 1:
        name = names[0]
        spec = cfg[name]
        wf = walk_forward(df, name, spec, n_trials=max(50, args.n_trials // 4))
        wf_df = wf_summary(wf)
        wf_df.to_csv(OUT / f"{name}_walkforward.csv", index=False)
        print(f"[wf] {name} OOS sharpe mean={wf_df['test_sharpe'].mean():.2f}, "
              f"calmar mean={wf_df['test_calmar'].mean():.2f}")

    # Monte Carlo
    if args.montecarlo and len(names) == 1:
        name = names[0]
        rets = pf_returns(df, signals_dict[name]).dropna()
        ci = ci_metrics(rets, n_sims=2000, block=24)
        ci.to_csv(OUT / f"{name}_montecarlo.csv")
        print(f"[mc] {name} Sharpe 95% CI:\n{ci.loc['sharpe']}")


def pf_returns(df: pd.DataFrame, sig) -> pd.Series:
    """Compute bar-level strategy returns from signals."""
    pf, _ = run_direction(df, sig.entries, sig.direction)
    return pf.returns()


if __name__ == "__main__":
    main()