"""Optuna multi-objective optimizer. Searches parameter space per strategy.
Outputs Pareto front: (Sharpe, Calmar) pairs + best params per objective."""
from __future__ import annotations
from pathlib import Path
import json
import optuna
import numpy as np
import pandas as pd
from dataclasses import asdict
from strategies import STRATEGY_REGISTRY
from backtester.engine import run_direction


def _sample(trial: optuna.Trial, spec: dict, prefix: str = "") -> dict:
    """Recursively sample from a spec dict of ranges."""
    out = {}
    for k, v in spec.items():
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, bool) for x in v):
            out[k] = bool(trial.suggest_categorical(f"{prefix}{k}", v))
        elif isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            lo, hi = v
            key = f"{prefix}{k}"
            if isinstance(lo, int) and isinstance(hi, int):
                out[k] = trial.suggest_int(key, lo, hi)
            else:
                out[k] = trial.suggest_float(key, float(lo), float(hi))
        else:
            out[k] = v
    return out


def optimize_strategy(strategy_name: str, df: pd.DataFrame, spec: dict,
                      n_trials: int = 200, init_cash: float = 10_000.0,
                      commission: float = 7e-5, slippage: float = 3e-5,
                      study_name: str | None = None) -> optuna.Study:
    """Multi-objective: maximize Sharpe, maximize Calmar."""
    cls = STRATEGY_REGISTRY[strategy_name]
    study_name = study_name or f"{strategy_name}_v1"

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = _sample(trial, spec)
        try:
            strat = cls(params=params)
            sig = strat.generate(df)
            _, summary = run_direction(df, sig.entries, sig.direction,
                                       init_cash=init_cash, commission=commission,
                                       slippage=slippage)
            sharpe = summary.get("sharpe", 0.0)
            calmar = summary.get("calmar", 0.0)
        except Exception:
            return -10.0, -10.0
        if not np.isfinite(sharpe):
            sharpe = -10.0
        if not np.isfinite(calmar):
            calmar = -10.0
        return float(sharpe), float(calmar)

    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def pareto_df(study: optuna.Study) -> pd.DataFrame:
    rows = []
    for t in study.best_trials:
        vals = t.values or (-np.inf, -np.inf)
        rows.append({"sharpe": vals[0], "calmar": vals[1], **t.params})
    return pd.DataFrame(rows)


def best_params(study: optuna.Study, metric: str = "sharpe") -> dict:
    """Return params of best trial by metric. metric ∈ {sharpe, calmar}."""
    idx = 0 if metric == "sharpe" else 1
    best = max(study.best_trials, key=lambda t: (t.values or [-1e9]*2)[idx])
    return best.params