"""Walk-forward analysis. Splits data into rolling train/test windows,
optimizes on train, validates on test, collects out-of-sample metrics."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from backtester.metrics import metrics_from_returns


@dataclass
class WFWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: dict
    test_metrics: dict
    params: dict


def walk_forward(df: pd.DataFrame, strategy_name: str, param_spec: dict,
                 train_months: int = 36, test_months: int = 6,
                 roll_months: int = 3, n_trials: int = 80,
                 periods_per_year: int = 252 * 24,
                 ) -> list[WFWindow]:
    """Returns list of WFWindow. Caller is responsible for running Optuna per window."""
    from strategies import STRATEGY_REGISTRY
    from backtester.engine import run_direction
    from analysis.optuna_optimizer import _sample
    import optuna

    cls = STRATEGY_REGISTRY[strategy_name]
    start = df.index[0]
    end = df.index[-1]
    out: list[WFWindow] = []

    cursor = start
    while True:
        tr_end = cursor + pd.DateOffset(months=train_months)
        te_end = tr_end + pd.DateOffset(months=test_months)
        if te_end > end:
            break
        train = df[(df.index >= cursor) & (df.index < tr_end)]
        test = df[(df.index >= tr_end) & (df.index < te_end)]
        if len(train) < 200 or len(test) < 50:
            cursor += pd.DateOffset(months=roll_months)
            continue

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))

        def obj(trial):
            params = _sample(trial, param_spec)
            try:
                sig = cls(params=params).generate(train)
                _, m = run_direction(train, sig.entries, sig.direction)
                return m.get("sharpe", -10)
            except Exception:
                return -10

        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        sig_tr = cls(params=best).generate(train)
        _, m_tr = run_direction(train, sig_tr.entries, sig_tr.direction)
        sig_te = cls(params=best).generate(test)
        _, m_te = run_direction(test, sig_te.entries, sig_te.direction)
        out.append(WFWindow(cursor, tr_end, tr_end, te_end, m_tr, m_te, best))
        cursor += pd.DateOffset(months=roll_months)

    return out


def wf_summary(windows: list) -> pd.DataFrame:
    rows = []
    for w in windows:
        rows.append({
            "train_start": w.train_start,
            "train_end": w.train_end,
            "test_start": w.test_start,
            "test_end": w.test_end,
            "train_sharpe": w.train_metrics.get("sharpe"),
            "test_sharpe": w.test_metrics.get("sharpe"),
            "train_calmar": w.train_metrics.get("calmar"),
            "test_calmar": w.test_metrics.get("calmar"),
            "test_max_dd": w.test_metrics.get("max_drawdown"),
        })
    return pd.DataFrame(rows)