"""Criterion study — which selection criterion actually picks OOS winners?

Having 22 criteria is only useful if you know which ones work. This module
backtests N param configs on an in-sample half, scores every config with
every criterion, then checks the picks against the out-of-sample half:

    per criterion: Spearman(IS rank, OOS rank), top-1 regret
                   (best OOS Sharpe minus OOS Sharpe of the pick),
                   top-3 hit rate.

A criterion with high rank correlation and low regret earns its place in
the optimizer. One with ~0 correlation is decoration — stop using it.

Usage:
    from analysis.criterion_study import study_criteria
    rep = study_criteria(df, run_bt, param_grid,
                         criteria=["Sharpe Ratio", "Deflated Sharpe Ratio (DSR)"])
    # run_bt(params, df_slice) -> metrics dict with sharpe/net_pnl
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time


@dataclass
class CriterionScore:
    """How one criterion performed as a selector."""
    criterion: str
    spearman: float
    top1_regret: float
    top3_hit: bool
    picked_config: int
    picked_oos_sharpe: float
    best_oos_sharpe: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spearman"] = round(self.spearman, 3)
        d["top1_regret"] = round(self.top1_regret, 3)
        d["picked_oos_sharpe"] = round(self.picked_oos_sharpe, 3)
        d["best_oos_sharpe"] = round(self.best_oos_sharpe, 3)
        return d


@dataclass
class CriterionStudyReport:
    """Ranked criteria + audit trail."""
    n_configs: int
    train_bars: int
    test_bars: int
    scores: list = field(default_factory=list)
    elapsed_sec: float = 0.0

    def ranking(self) -> list:
        # Primary: low regret. Tiebreak: high rank correlation.
        return sorted(self.scores,
                      key=lambda s: (s.top1_regret, -s.spearman))

    def to_dict(self) -> dict:
        return {"n_configs": self.n_configs, "train_bars": self.train_bars,
                "test_bars": self.test_bars,
                "elapsed_sec": round(self.elapsed_sec, 1),
                "ranking": [s.to_dict() for s in self.ranking()]}

    def save(self, out_path: str | Path) -> str:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return str(out_path)


def _spearman(a: list[float], b: list[float]) -> float:
    from scipy.stats import spearmanr
    import numpy as np
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return 0.0
    r = spearmanr(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return float(r.statistic) if r.statistic == r.statistic else 0.0  # NaN guard


def study_criteria(df, run_bt, param_grid: list[dict],
                   criteria: list[str] | None = None,
                   train_frac: float = 0.5, verbose: bool = True) -> CriterionStudyReport:
    """Run the study.

    Args:
        df: full price data (time-split into IS/OOS halves).
        run_bt: callable(params, df_slice) -> metrics dict with at least
            'sharpe' and 'net_pnl' keys.
        param_grid: list of param dicts (the candidate configs).
        criteria: subset of CRITERION_PRESETS names (default: all 22).
        train_frac: IS fraction (time order preserved).
    """
    from analysis.composite_criterion import CRITERION_PRESETS
    t0 = time.time()
    names = criteria or list(CRITERION_PRESETS.keys())
    unknown = [c for c in names if c not in CRITERION_PRESETS]
    if unknown:
        raise ValueError(f"unknown criteria: {unknown}")

    cut = int(len(df) * train_frac)
    df_is, df_oos = df.iloc[:cut], df.iloc[cut:]
    is_metrics, oos_sharpes = [], []
    for i, p in enumerate(param_grid):
        try:
            mi = run_bt(p, df_is) or {}
            mo = run_bt(p, df_oos) or {}
        except Exception as e:
            if verbose:
                print(f"  config {i}: runner failed ({type(e).__name__}), skipped")
            continue
        is_metrics.append((i, p, mi))
        oos_sharpes.append(float(mo.get("sharpe", 0) or 0))
    if not is_metrics:
        raise RuntimeError("no configs produced metrics")

    oos_best = max(oos_sharpes)
    oos_order = sorted(range(len(oos_sharpes)), key=lambda i: oos_sharpes[i],
                       reverse=True)
    oos_rank_of = {idx: r for r, idx in enumerate(oos_order)}

    scores = []
    for name in names:
        fn = CRITERION_PRESETS[name]
        scored = []
        for i, p, mi in is_metrics:
            try:
                scored.append((i, float(fn(mi) or 0)))
            except Exception:
                scored.append((i, float("-inf")))
        scored.sort(key=lambda t: t[1], reverse=True)
        is_rank_of = {idx: r for r, (idx, _) in enumerate(scored)}
        idxs = [i for i, _, _ in is_metrics]
        rho = _spearman([is_rank_of[i] for i in idxs],
                        [oos_rank_of[j] for j in range(len(idxs))])
        pick = scored[0][0]
        pick_oos = oos_sharpes[pick]
        scores.append(CriterionScore(
            criterion=name, spearman=rho,
            top1_regret=oos_best - pick_oos,
            top3_hit=oos_rank_of[pick] < 3,
            picked_config=pick, picked_oos_sharpe=pick_oos,
            best_oos_sharpe=oos_best))
        if verbose:
            print(f"  {name[:38]:38s} rho={rho:+.3f} regret={oos_best - pick_oos:.3f} "
                  f"top3={'Y' if oos_rank_of[pick] < 3 else '.'}")

    return CriterionStudyReport(n_configs=len(is_metrics),
                                train_bars=len(df_is), test_bars=len(df_oos),
                                scores=scores,
                                elapsed_sec=round(time.time() - t0, 1))


# ---------- Self-test ----------
if __name__ == "__main__":
    # Synthetic: config quality = -abs(period - 14); OOS adds noise.
    import numpy as np
    rng = np.random.default_rng(11)

    def fake_bt(p, _df):
        q = -abs(p["period"] - 14)
        noise = rng.normal(0, 0.5)
        return {"sharpe": q + noise, "net_pnl": (q + noise) * 100,
                "max_drawdown": -0.05, "win_rate": 0.55, "trades": 40}

    import pandas as pd
    df = pd.DataFrame({"close": [1.0] * 200})
    grid = [{"period": v} for v in (5, 10, 14, 20, 25)]
    rep = study_criteria(df, fake_bt, grid,
                         criteria=["Sharpe Ratio", "Net P&L ($)", "Win Rate (%) (Guarded)"],
                         verbose=True)
    print("ranking:", [(s.criterion[:20], round(s.spearman, 2)) for s in rep.ranking()])
    assert rep.n_configs == 5
    p = rep.save("output/reports/_selftest_criterion.json")
    print("saved:", p)

    try:
        study_criteria(df, fake_bt, grid, criteria=["Nope"])
        raise SystemExit("should have raised")
    except ValueError as e:
        print("unknown criterion rejected:", str(e)[:60])
    print("SELF-TEST PASS")
