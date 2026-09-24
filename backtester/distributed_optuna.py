"""Distributed Optuna — RDBStorage for multi-worker optimization.

Run: `optuna-dashboard sqlite:///optuna.db` to monitor.
Workers: `python -m backtester.distributed_optuna worker --study study_name`
"""
from __future__ import annotations
import argparse
import optuna
from typing import Any, Callable
from pathlib import Path


DEFAULT_STORAGE = "sqlite:///optuna.db"


def create_study(study_name: str, direction: str = "maximize",
                 storage: str = DEFAULT_STORAGE) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )


def run_worker(study_name: str, objective: Callable[[optuna.Trial], float],
               n_trials: int = 100, storage: str = DEFAULT_STORAGE) -> None:
    """Worker process — attaches to existing study and runs trials."""
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)


def get_best_params(study_name: str, storage: str = DEFAULT_STORAGE) -> dict:
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study.best_trial.params if study.best_trial else {}


def get_param_importances(study_name: str, storage: str = DEFAULT_STORAGE) -> dict[str, float]:
    study = optuna.load_study(study_name=study_name, storage=storage)
    try:
        imp = optuna.importance.get_param_importances(study)
        return {k: round(float(v) * 100, 2) for k, v in imp.items()}
    except Exception:
        return {}


def pareto_front(study_name: str, metrics: list[str], storage: str = DEFAULT_STORAGE) -> list[dict]:
    """Multi-objective pareto front from study user_attrs."""
    study = optuna.load_study(study_name=study_name, storage=storage)
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, **t.params}
        for m in metrics:
            row[m] = t.user_attrs.get("metrics", {}).get(m, 0)
        rows.append(row)
    return rows


# CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["create", "worker", "best", "importance", "pareto"])
    parser.add_argument("--study", required=True)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--direction", default="maximize")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    if args.cmd == "create":
        create_study(args.study, args.direction, args.storage)
        print(f"Created study '{args.study}' at {args.storage}")
    elif args.cmd == "worker":
        # Import objective from analysis.optuna_optimizer
        from analysis.optuna_optimizer import optimize_strategy_criterion
        # This is a placeholder — worker needs objective fn passed in real use
        print("Worker mode: import and pass objective function in your script")
    elif args.cmd == "best":
        print(get_best_params(args.study, args.storage))
    elif args.cmd == "importance":
        print(get_param_importances(args.study, args.storage))
    elif args.cmd == "pareto":
        print(pareto_front(args.study, ["sharpe", "profit_factor", "max_drawdown"], args.storage))