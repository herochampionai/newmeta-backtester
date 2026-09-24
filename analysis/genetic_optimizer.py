"""NSGA-II genetic optimizer for parameter discovery.

Replaces Optuna TPE with NSGA-II (Non-dominated Sorting Genetic Algorithm II) —
the same genetic-algorithm approach MT5 Strategy Tester uses for multi-objective
optimization.  Three objectives: maximize Sharpe, maximize Calmar, minimize
max drawdown.

Usage:
    PYTHONPATH=. python -m analysis.genetic_optimizer --strategy fbb --symbol EURUSD --timeframe H1 --n-gen 50 --pop 40

Output:
    output/genetic_<strategy>_<symbol>_pareto.csv  — Pareto front
    output/genetic_<strategy>_<symbol>_best.json   — best params per objective
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.optuna_optimizer import _sample  # noqa: E402
from backtester.engine_full import GRID_NONE, run_full  # noqa: E402
from core.surgical_features import get_feature_defaults  # noqa: E402
from data.cache import load as load_cache  # noqa: E402
from strategies import STRATEGY_REGISTRY  # noqa: E402


def load_strategy_spec(strategy_name: str) -> dict:
    """Load parameter ranges from config/strategies.yaml."""
    import yaml
    spec_path = ROOT / "config" / "strategies.yaml"
    with open(spec_path) as f:
        all_specs = yaml.safe_load(f)
    spec = all_specs.get(strategy_name, {})
    # Remove non-parameter keys
    spec.pop("enabled", None)
    return spec


def objective_factory(df: pd.DataFrame, strategy_name: str, spec: dict):
    """Create an Optuna objective that runs run_full with the sampled params."""
    cls = STRATEGY_REGISTRY[strategy_name]
    base_defaults = get_feature_defaults()

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        params = _sample(trial, spec)
        # Merge with surgical feature defaults (all OFF)
        params = {**base_defaults, **params}

        # Generate signals on the full df (training window)
        inst = cls(params=params)
        sig = inst.generate(df)
        signals = {strategy_name: (sig.entries, sig.direction)}

        # Run grid-less backtest (pure signal validation first)
        try:
            result = run_full(df, signals, grid_mode=GRID_NONE, params=params)
            m = result["metrics"]
            sharpe = float(m.get("sharpe", 0.0))
            calmar = float(m.get("calmar", 0.0))
            max_dd = float(m.get("max_drawdown", 0.0))
            if sharpe != sharpe:  # NaN check
                sharpe = -10.0
            if calmar != calmar:
                calmar = -10.0
            if max_dd != max_dd:
                max_dd = 0.0
            return sharpe, calmar, abs(max_dd)
        except Exception:
            return -10.0, -10.0, 999.0

    return objective


def run_genetic(df: pd.DataFrame, strategy_name: str, spec: dict,
                pop_size: int = 40, n_gen: int = 50, seed: int = 42) -> optuna.Study:
    """Run NSGA-II multi-objective optimization."""
    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(
            population_size=pop_size, seed=seed,
            crossover_prob=0.9, mutation_prob=0.1 / max(len(spec), 1),
        ),
        study_name=f"genetic_{strategy_name}",
    )
    total_trials = pop_size * n_gen
    study.optimize(objective_factory(df, strategy_name, spec),
                   n_trials=total_trials, show_progress_bar=False)
    return study


def main():
    ap = argparse.ArgumentParser(description="NSGA-II genetic optimizer")
    ap.add_argument("--strategy", default="fbb")
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--pop", type=int, default=40, help="Population size")
    ap.add_argument("--n-gen", type=int, default=50, help="Number of generations")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-fraction", type=float, default=0.7,
                    help="Fraction of data for training (rest = validation)")
    args = ap.parse_args()

    df, meta = load_cache(args.symbol, args.timeframe)
    print(f"[load] {meta.get('symbol', args.symbol)} {args.timeframe}: {len(df)} bars")

    spec = load_strategy_spec(args.strategy)
    if not spec:
        print(f"[ERROR] no spec found for strategy '{args.strategy}' in config/strategies.yaml")
        return 1
    print(f"[spec] {len(spec)} parameters for {args.strategy}")

    # Split into train / validation
    split = int(len(df) * args.test_fraction)
    df_train = df.iloc[:split]
    df_val = df.iloc[split:]
    print(f"[split] train: {len(df_train)} bars | val: {len(df_val)} bars")

    # Optimize on training window
    print(f"[optim] NSGA-II: pop={args.pop} gen={args.n_gen} total_trials={args.pop * args.n_gen}")
    study = run_genetic(df_train, args.strategy, spec,
                        pop_size=args.pop, n_gen=args.n_gen, seed=args.seed)

    pareto = study.trials_dataframe(attrs=("values", "params"))
    pareto.columns = ["_".join(c) for c in pareto.columns]

    out = ROOT / "output"
    out.mkdir(exist_ok=True)

    # Save Pareto front
    pareto_path = out / f"genetic_{args.strategy}_{args.symbol}_pareto.csv"
    pareto.to_csv(pareto_path, index=False)
    print(f"\n[saved] Pareto front → {pareto_path}  ({len(pareto)} trials)")

    # Extract best params per objective
    best = {}
    obj_names = ["sharpe", "calmar", "neg_max_dd"]
    for i, (direction, name) in enumerate(zip(study.directions, obj_names)):
        best_trial = None
        for t in study.best_trials:
            if t.values[i] is not None:
                if best_trial is None or \
                   (direction == optuna.study.StudyDirection.MAXIMIZE and
                    t.values[i] > best_trial.values[i]) or \
                   (direction == optuna.study.StudyDirection.MINIMIZE and
                    t.values[i] < best_trial.values[i]):
                    best_trial = t
        if best_trial:
            best[name] = {
                "params": best_trial.params,
                "value": float(best_trial.values[i]),
            }

    best_path = out / f"genetic_{args.strategy}_{args.symbol}_best.json"
    best_path.write_text(json.dumps(best, indent=2))
    print(f"[saved] Best params → {best_path}")

    # Validation on out-of-sample
    if best:
        print("\n[validation] OOS on remaining %d bars..." % len(df_val))
        for obj_name, info in best.items():
            val_params = {**get_feature_defaults(), **info["params"]}
            inst = STRATEGY_REGISTRY[args.strategy](params=val_params)
            sig = inst.generate(df_val)
            signals = {args.strategy: (sig.entries, sig.direction)}
            try:
                res = run_full(df_val, signals, grid_mode=GRID_NONE, params=val_params)
                m = res["metrics"]
                print(f"  {obj_name:10s} train={info['value']:.3f}  "
                      f"val Sharpe={m.get('sharpe', 0):+.3f}  "
                      f"Calmar={m.get('calmar', 0):+.3f}  "
                      f"maxDD={m.get('max_drawdown', 0):.3f}")
            except Exception as e:
                print(f"  {obj_name:10s} val: [ERROR] {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
