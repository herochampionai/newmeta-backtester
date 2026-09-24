"""R024: Auto-iterate loop — WF verdict drives param refinement.

When walk-forward says OVERFIT (or too few windows pass), blindly accepting
the backtest config is how strategies die live. This loop does the honest
thing automatically:

    round 0: WF over user params -> verdict + best OOS params
    round N: shrink bounds around best OOS params, WF again
    stop: verdict ACCEPT, or budget (max_rounds) exhausted

Every round is recorded (params tried, verdict, passed/failed, best OOS
Sharpe) so the audit trail shows exactly what was attempted — a documented
negative result beats a silent overfit.

Usage:
    from analysis.auto_iterate import auto_iterate, bounds_from_params
    report = auto_iterate(
        df=df, strategy_name="adx",
        base_params={"bars_calculate": 10, "level_open_orders_1": 25},
        run_wf_fn=my_wf_runner,   # (param_spec) -> dict with wf report fields
        max_rounds=3,
    )
    print(report.accepted, report.best_params)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time


@dataclass
class IterateRound:
    """One WF round inside the loop."""
    round_idx: int
    param_spec: dict
    verdict: str
    passed: int
    failed: int
    n_windows: int
    best_params: dict = field(default_factory=dict)
    best_oos_sharpe: float = 0.0
    elapsed_sec: float = 0.0


@dataclass
class AutoIterateReport:
    """Full audit trail of the loop."""
    strategy_name: str
    base_params: dict
    rounds: list = field(default_factory=list)
    accepted: bool = False
    best_params: dict = field(default_factory=dict)
    best_oos_sharpe: float = 0.0
    stop_reason: str = ""
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "base_params": self.base_params,
            "accepted": self.accepted,
            "best_params": self.best_params,
            "best_oos_sharpe": round(self.best_oos_sharpe, 4),
            "stop_reason": self.stop_reason,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "rounds": [asdict(r) for r in self.rounds],
        }

    def save(self, out_path: str | Path) -> str:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return str(out_path)


def bounds_from_params(params: dict, radius: float = 0.5) -> dict:
    """Build a 5-point grid per numeric param centered on the given values.

    Int params stay ints (midpoints rounded) so strategies calling int()
    get the intended values instead of silently truncated floats.
    """
    spec = {}
    for k, v in params.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v == 0:
            continue
        if isinstance(v, int):
            low = max(1, int(v * (1 - radius)))
            high = int(v * (1 + radius)) + 1
            spec[k] = [low, round((low + v) / 2), v, round((v + high) / 2), high]
        else:
            low = v * (1 - radius)
            high = v * (1 + radius)
            spec[k] = [low, (low + v) / 2, v, (v + high) / 2, high]
    return spec


def shrink_bounds(best: dict, radius: float = 0.25) -> dict:
    """Tighten the grid around the best OOS params for the next round."""
    return bounds_from_params(best, radius=radius)


def auto_iterate(df, strategy_name: str, base_params: dict, run_wf_fn,
                 max_rounds: int = 3, target_verdict: str = "ACCEPT",
                 shrink_radius: float = 0.25, verbose: bool = True) -> AutoIterateReport:
    """Run the WF -> shrink -> WF loop.

    Args:
        df: price data.
        strategy_name: for the audit trail.
        base_params: starting config (round 0 centers here).
        run_wf_fn: callable(param_spec) -> dict with keys:
            verdict, passed, failed, n_windows, best_params (dict),
            best_oos_sharpe (float). Raises on genuinely broken configs.
        max_rounds: budget. Round 0 always runs; refinements follow.
        target_verdict: stop when reached.
        shrink_radius: bound radius for rounds 1+ (round 0 uses 0.5).
        verbose: print per-round summary.

    Returns:
        AutoIterateReport with accepted flag + full round history.
    """
    t0 = time.time()
    report = AutoIterateReport(strategy_name=strategy_name, base_params=dict(base_params))

    current_center = dict(base_params)
    radius = 0.5
    stalls = 0
    prev_best: dict = {}
    for rnd in range(max_rounds + 1):
        spec = bounds_from_params(current_center, radius=radius)
        if not spec:
            report.stop_reason = "no numeric params to vary"
            break
        rt0 = time.time()
        try:
            out = run_wf_fn(spec)
        except Exception as e:
            report.rounds.append(IterateRound(
                round_idx=rnd, param_spec=spec, verdict="ERROR",
                passed=0, failed=0, n_windows=0,
                elapsed_sec=round(time.time() - rt0, 1)))
            report.stop_reason = f"wf runner raised: {type(e).__name__}: {str(e)[:120]}"
            break

        verdict = str(out.get("verdict", "UNKNOWN"))
        rnd_rec = IterateRound(
            round_idx=rnd, param_spec=spec, verdict=verdict,
            passed=int(out.get("passed", 0)), failed=int(out.get("failed", 0)),
            n_windows=int(out.get("n_windows", 0)),
            best_params=dict(out.get("best_params", {}) or {}),
            best_oos_sharpe=float(out.get("best_oos_sharpe", 0) or 0),
            elapsed_sec=round(time.time() - rt0, 1))
        report.rounds.append(rnd_rec)

        # Track global best across rounds by OOS Sharpe.
        if rnd_rec.best_oos_sharpe > report.best_oos_sharpe or rnd == 0:
            report.best_oos_sharpe = rnd_rec.best_oos_sharpe
            report.best_params = dict(rnd_rec.best_params) or dict(current_center)

        if verbose:
            print(f"  [iterate r{rnd}] verdict={verdict} "
                  f"passed={rnd_rec.passed}/{rnd_rec.n_windows} "
                  f"best_oos_sharpe={rnd_rec.best_oos_sharpe:.3f} "
                  f"({rnd_rec.elapsed_sec:.1f}s)")

        if verdict == target_verdict:
            report.accepted = True
            report.stop_reason = f"{target_verdict} at round {rnd}"
            break

        # Stall detection: same best params with no Sharpe gain means the
        # discrete grid is exhausted here — widen and re-explore once, then
        # stop instead of burning budget on identical rounds.
        if rnd_rec.best_params == prev_best:
            stalls += 1
            if stalls >= 2:
                report.stop_reason = (f"converged (no improvement for "
                                      f"{stalls} rounds, best OOS Sharpe "
                                      f"{report.best_oos_sharpe:.3f})")
                break
            radius = 0.5  # re-expand to escape the local grid
        else:
            stalls = 0
            radius = shrink_radius
        prev_best = dict(rnd_rec.best_params)

        # Refine around this round's best OOS params and continue.
        if rnd_rec.best_params:
            current_center = dict(rnd_rec.best_params)
    else:
        report.stop_reason = f"budget exhausted ({max_rounds + 1} rounds)"

    if not report.best_params:
        report.best_params = dict(base_params)
    report.elapsed_sec = round(time.time() - t0, 1)
    return report


# ---------- Self-test ----------
if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({"close": [1.0] * 100})

    # Grid math first (no WF needed).
    g = bounds_from_params({"bars_calculate": 10, "threshold": 25.0})
    assert g["bars_calculate"] == [5, 8, 10, 13, 16], g
    assert all(isinstance(x, int) for x in g["bars_calculate"])
    assert g["threshold"][2] == 25.0
    s = shrink_bounds({"bars_calculate": 13}, radius=0.25)
    assert s["bars_calculate"][2] == 13, s  # shrinks around best
    print("grid math OK")

    # Accept path: fake ACCEPTs when the grid's best pick is 13
    # (reachable in round 0 from base 10).
    def fake_accept(spec):
        grid = spec.get("bars_calculate", [10])
        best = min(grid, key=lambda x: abs(x - 13))
        ok = best == 13
        return {
            "verdict": "ACCEPT" if ok else "OVERFIT",
            "passed": 6 if ok else 1, "failed": 2 if ok else 7,
            "n_windows": 8, "best_params": {"bars_calculate": best},
            "best_oos_sharpe": 0.9 if ok else 0.2,
        }

    rep = auto_iterate(df, "adx", {"bars_calculate": 10}, fake_accept,
                       max_rounds=3, verbose=True)
    print("accepted:", rep.accepted)
    print("best:", rep.best_params, "stop:", rep.stop_reason)
    assert rep.accepted and rep.best_params["bars_calculate"] == 13
    p = rep.save("output/reports/_selftest_iterate.json")
    print("saved:", p)

    # Converge path: same best every round -> stops early, not budget.
    rep2 = auto_iterate(df, "adx", {"bars_calculate": 10},
                        lambda spec: {"verdict": "OVERFIT", "passed": 0, "failed": 8,
                                      "n_windows": 8, "best_params": {"bars_calculate": 10},
                                      "best_oos_sharpe": 0.1},
                        max_rounds=5, verbose=False)
    assert not rep2.accepted and "converged" in rep2.stop_reason, rep2.stop_reason
    assert len(rep2.rounds) < 6  # stopped before budget
    print("converge path OK:", rep2.stop_reason, f"({len(rep2.rounds)} rounds)")

    # Error path: runner raises -> recorded, loop stops cleanly.
    def boom(spec):
        raise RuntimeError("wf exploded")
    rep3 = auto_iterate(df, "adx", {"bars_calculate": 10}, boom,
                        max_rounds=3, verbose=False)
    assert not rep3.accepted and "raised" in rep3.stop_reason
    print("error path OK:", rep3.stop_reason)
    print("SELF-TEST PASS")
