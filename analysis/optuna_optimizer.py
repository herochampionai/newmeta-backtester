"""Optuna intelligent optimizer with custom criterion study.
Searches parameter space per strategy with Bayesian TPE sampling.
100-500x faster than MT5 Strategy Tester via vectorized in-memory execution.
"""
from __future__ import annotations
from pathlib import Path
import json
import optuna
import numpy as np
import pandas as pd
from typing import Callable, Any

from strategies import STRATEGY_REGISTRY
from backtester.engine_full import run_full, GRID_NONE
from backtester.metrics_v2 import compute_all


def _sample_param(trial: optuna.Trial, key: str, cfg: Any) -> Any:
    """Sample a parameter based on its specification dictionary or list."""
    if isinstance(cfg, dict):
        ptype = cfg.get("type", "float")
        if ptype == "int":
            low = int(cfg.get("min", 1))
            high = int(cfg.get("max", 100))
            step = int(cfg.get("step", 1))
            return trial.suggest_int(key, low, high, step=step)
        elif ptype == "float":
            low = float(cfg.get("min", 0.0))
            high = float(cfg.get("max", 100.0))
            step = cfg.get("step")
            if step is not None:
                return trial.suggest_float(key, low, high, step=float(step))
            return trial.suggest_float(key, low, high)
        elif ptype in ("categorical", "choice"):
            choices = cfg.get("choices", [True, False])
            return trial.suggest_categorical(key, choices)
        elif ptype == "bool":
            return bool(trial.suggest_categorical(key, [True, False]))
        else:
            return cfg.get("default", 0)
    elif isinstance(cfg, list):
        if len(cfg) == 2 and all(isinstance(x, bool) for x in cfg):
            return bool(trial.suggest_categorical(key, cfg))
        elif len(cfg) == 2 and all(isinstance(x, int) for x in cfg):
            return trial.suggest_int(key, cfg[0], cfg[1])
        elif len(cfg) == 2 and all(isinstance(x, (int, float)) for x in cfg):
            return trial.suggest_float(key, float(cfg[0]), float(cfg[1]))
        elif len(cfg) > 2:
            return trial.suggest_categorical(key, cfg)
    return cfg


def optimize_strategy_criterion(
    strategy_obj_or_cls: Any,
    df: pd.DataFrame,
    param_spec: dict,
    criterion_fn: Callable[[dict], float],
    n_trials: int = 100,
    direction: str = "maximize",
    fixed_params: dict | None = None,
    engine_kwargs: dict | None = None,
    study_name: str | None = None,
    seed: int = 42,
) -> optuna.Study:
    """Run Optuna TPE optimization on a strategy using an arbitrary criterion scoring function.
    
    Args:
        strategy_obj_or_cls: Strategy class or instance
        df: Historical OHLCV DataFrame
        param_spec: Dict of parameter ranges to optimize
        criterion_fn: Callable taking metrics dict and returning float score
        n_trials: Number of Bayesian trials
        direction: 'maximize' or 'minimize'
        fixed_params: Base parameters not being optimized
        engine_kwargs: Execution settings (spread, commission, base_lot, etc.)
    """
    study_name = study_name or f"opt_{getattr(strategy_obj_or_cls, 'name', 'strat')}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    fixed = dict(fixed_params or {})
    eng_kwargs = dict(engine_kwargs or {})
    
    if isinstance(strategy_obj_or_cls, str):
        cls = STRATEGY_REGISTRY[strategy_obj_or_cls]
        strat_name = strategy_obj_or_cls
    elif hasattr(strategy_obj_or_cls, "generate") and not isinstance(strategy_obj_or_cls, type):
        cls = type(strategy_obj_or_cls)
        strat_name = getattr(strategy_obj_or_cls, "name", "custom_strat")
    else:
        cls = strategy_obj_or_cls
        strat_name = getattr(cls, "name", "strat")

    def objective(trial: optuna.Trial) -> float:
        sampled = {}
        for k, v in param_spec.items():
            sampled[k] = _sample_param(trial, k, v)
        
        merged_params = {**fixed, **sampled}
        
        try:
            strat = cls(params=merged_params)
            sig = strat.generate(df)
            entries = sig.entries.fillna(False).astype(bool)
            direction_series = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
            
            signals = {strat_name: (entries, direction_series)}
            result = run_full(df, signals, **eng_kwargs)
            metrics = result.get("metrics", {})
            
            score = criterion_fn(metrics)
            if not np.isfinite(score):
                return -1e9 if direction == "maximize" else 1e9
            
            trial.set_user_attr("metrics", {
                "sharpe": metrics.get("sharpe", 0.0),
                "calmar": metrics.get("calmar", 0.0),
                "sortino": metrics.get("sortino", 0.0),
                "profit_factor": metrics.get("profit_factor", 0.0),
                "net_pnl": metrics.get("net_pnl", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "win_rate": metrics.get("win_rate", 0.0),
                "n_trades": metrics.get("n_trades", 0),
            })
            return float(score)
        except Exception:
            return -1e9 if direction == "maximize" else 1e9

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def get_study_trials_df(study: optuna.Study) -> pd.DataFrame:
    """Convert all complete study trials into a rich DataFrame with params and metrics."""
    rows = []
    for t in study.trials:
        if t.state.name != "COMPLETE":
            continue
        row = {
            "trial_number": t.number,
            "score": round(float(t.value), 4) if t.value is not None else -1e9,
        }
        for k, v in t.user_attrs.get("metrics", {}).items():
            row[k] = v
        for pk, pv in t.params.items():
            row[pk] = pv
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    df_trials = pd.DataFrame(rows).sort_values("score", ascending=False)
    return df_trials


def get_param_importances_dict(study: optuna.Study) -> dict[str, float]:
    """Calculate parameter importance percentages using fANOVA / Tree-importance."""
    try:
        if len(study.trials) < 6:
            return {}
        importances = optuna.importance.get_param_importances(study)
        return {k: round(float(v) * 100, 2) for k, v in importances.items()}
    except Exception:
        return {}


def _sample(trial: optuna.Trial, spec: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in spec.items():
        out[k] = _sample_param(trial, f"{prefix}{k}", v)
    return out


def optimize_strategy(strategy_name: str, df: pd.DataFrame, spec: dict,
                      n_trials: int = 200, init_cash: float = 10_000.0,
                      commission: float = 7e-5, slippage: float = 3e-5,
                      study_name: str | None = None) -> optuna.Study:
    """Legacy multi-objective helper."""
    cls = STRATEGY_REGISTRY[strategy_name]
    study_name = study_name or f"{strategy_name}_v1"

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = _sample(trial, spec)
        try:
            strat = cls(params=params)
            sig = strat.generate(df)
            from backtester.engine import run_direction
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
    """Return params of best trial by metric."""
    if hasattr(study, "best_trial") and study.best_trial is not None and study.best_trial.value is not None:
        return study.best_trial.params
    if hasattr(study, "best_trials") and len(study.best_trials) > 0:
        idx = 0 if metric == "sharpe" else 1
        best = max(study.best_trials, key=lambda t: (t.values or [-1e9]*2)[min(idx, len(t.values or []) - 1)])
        return best.params
    return {}
