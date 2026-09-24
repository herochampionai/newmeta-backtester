"""Fine-tuner — Optuna TPE parameter search, faster than MQL5's grid.

MQL5's genetic optimizer evaluates full grids; TPE (Tree-structured Parzen
Estimator) models the objective and samples where improvement is likely,
reaching good configs in a fraction of the trials. The objective is
OOS-honest by construction:

    train split  -> maximize Sharpe with guardrails
                    (min trades, drawdown cap; violations score -inf)
    holdout split -> the winner is RE-EVALUATED here and both numbers are
                    reported. train >> test gap = overfit, printed plainly.

Faster than MQL5 two ways: fewer trials for equal quality (TPE vs grid)
plus parallel trials (n_jobs) with deterministic seeding.

Usage:
    from analysis.fine_tuner import fine_tune
    rep = fine_tune(df, run_bt,
                    space={"bars_calculate": (5, 30, "int"),
                           "level_open_orders_1": (15, 45, "int")},
                    n_trials=40, seed=7)
    print(rep.best_params, rep.train_sharpe, rep.test_sharpe)
    # run_bt(params, df_slice) -> metrics dict
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time


@dataclass
class FineTuneReport:
    """Tuning outcome with the honest train/test pair."""
    best_params: dict = field(default_factory=dict)
    best_value: float = 0.0
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_trades: int = 0
    test_trades: int = 0
    n_trials: int = 0
    n_completed: int = 0
    elapsed_sec: float = 0.0
    history: list = field(default_factory=list)

    def overfit_gap(self) -> float:
        return self.train_sharpe - self.test_sharpe

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overfit_gap"] = round(self.overfit_gap(), 3)
        return d

    def save(self, out_path: str | Path) -> str:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return str(out_path)


def _suggest(trial, space: dict) -> dict:
    """Map {name: (low, high, type)} to Optuna suggestions."""
    import optuna
    params = {}
    for name, spec in space.items():
        low, high, typ = spec
        if typ == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        elif typ == "float":
            params[name] = trial.suggest_float(name, float(low), float(high))
        elif typ == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec[0])
                                                     if isinstance(spec[0], (list, tuple))
                                                     else [low, high])
        else:
            raise ValueError(f"unknown space type {typ!r} for {name}")
    return params


def fine_tune(df, run_bt, space: dict, n_trials: int = 40, timeout: float | None = None,
              train_frac: float = 0.7, min_trades: int = 10, max_dd: float = 0.30,
              n_jobs: int = 1, seed: int = 7, verbose: bool = True) -> FineTuneReport:
    """TPE search on train split, honest re-evaluation on holdout.

    Args:
        df: full price data (time-ordered; split preserves order).
        run_bt: callable(params, df_slice) -> metrics dict with sharpe,
            max_drawdown, and a trade count under 'trades' or 'n_trades'.
        space: {param: (low, high, 'int'|'float'|'categorical')}.
        n_trials/timeout: budget (whichever hits first).
        train_frac: IS fraction.
        min_trades/max_dd: guardrails — violations score -inf (pruned).
        n_jobs: parallel trials (1 = deterministic serial).
        seed: sampler seed for reproducibility.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    cut = int(len(df) * train_frac)
    df_train, df_test = df.iloc[:cut], df.iloc[cut:]
    history: list = []

    def _trades(m: dict) -> int:
        return int(m.get("trades", m.get("n_trades", 0)) or 0)

    def objective(trial) -> float:
        params = _suggest(trial, space)
        try:
            m = run_bt(params, df_train) or {}
        except Exception:
            return float("-inf")
        sharpe = float(m.get("sharpe", 0) or 0)
        if _trades(m) < min_trades:
            return float("-inf")
        if abs(float(m.get("max_drawdown", 0) or 0)) > max_dd:
            return float("-inf")
        history.append({"params": params, "value": round(sharpe, 4)})
        return sharpe

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs,
                   show_progress_bar=False)
    completed = [t for t in study.trials
                 if t.state.name == "COMPLETE" and t.value != float("-inf")]

    rep = FineTuneReport(n_trials=n_trials, n_completed=len(completed),
                         elapsed_sec=round(time.time() - t0, 1),
                         history=history[-20:])
    if not completed:
        if verbose:
            print("  tuner: no completing trial passed guardrails")
        return rep
    best = max(completed, key=lambda t: t.value)
    rep.best_params = dict(best.params)
    rep.best_value = round(float(best.value), 4)

    # Honest holdout re-evaluation of the single winner.
    try:
        mtr = run_bt(rep.best_params, df_train) or {}
        mte = run_bt(rep.best_params, df_test) or {}
    except Exception as e:
        if verbose:
            print(f"  tuner: holdout re-eval failed ({type(e).__name__})")
        return rep
    rep.train_sharpe = round(float(mtr.get("sharpe", 0) or 0), 3)
    rep.test_sharpe = round(float(mte.get("sharpe", 0) or 0), 3)
    rep.train_trades = _trades(mtr)
    rep.test_trades = _trades(mte)
    if verbose:
        print(f"  tuner: best={rep.best_params} train={rep.train_sharpe:.3f} "
              f"test={rep.test_sharpe:.3f} gap={rep.overfit_gap():.3f} "
              f"({rep.n_completed}/{n_trials} completed, {rep.elapsed_sec:.1f}s)")
    return rep


# ---------- Self-test ----------
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(3)

    # Peak at period=14, noisy evaluations, train/test same shape.
    def fake_bt(p, _df):
        q = -abs(p["period"] - 14) / 10.0
        return {"sharpe": q + rng.normal(0, 0.05), "max_drawdown": -0.05,
                "trades": 40}

    df = pd.DataFrame({"close": [1.0] * 300})
    rep = fine_tune(df, fake_bt, {"period": (5, 25, "int")},
                    n_trials=15, seed=7, verbose=True)
    print("best:", rep.best_params, "train:", rep.train_sharpe, "test:", rep.test_sharpe)
    assert rep.n_completed > 0
    assert abs(rep.best_params["period"] - 14) <= 3, rep.best_params
    p = rep.save("output/reports/_selftest_tune.json")
    print("saved:", p)

    # Guardrail path: nothing passes min_trades.
    def thin_bt(p, _df):
        return {"sharpe": 5.0, "max_drawdown": -0.01, "trades": 2}
    rep2 = fine_tune(df, thin_bt, {"period": (5, 25, "int")},
                     n_trials=5, min_trades=10, verbose=False)
    assert rep2.n_completed == 0 and not rep2.best_params
    print("guardrail path OK")
    print("SELF-TEST PASS")
