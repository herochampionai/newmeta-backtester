"""R025: Review gate — independent PASS/FAIL before paper.

The pipeline produces numbers; this module turns them into a decision with
named reasons. Every threshold is explicit and overridable — no silent
judgment. A strategy reaches paper only when gate passes; otherwise the
report says exactly which checks failed.

Checks (defaults):
  profitable      net_pnl > 0
  enough_trades   trades >= min_trades (30)
  sharpe_floor    sharpe >= min_sharpe (0.3)
  walkforward     verdict == ACCEPT, or pass_ratio >= min_wf_pass_ratio (0.5)
  significance    PSR verdict STRONG or DSR verdict STRONG or bootstrap
                    verdict STRONG (*)
  stress          verdict != FRAGILE
  drawdown_cap    |max_drawdown| <= max_dd (0.25)

(*) PSR/DSR STRONG at 40 trials is a high bar; the default accepts either
one STRONG. Loosen via require_both_strong=False (default) semantics.

Usage:
    from analysis.review_gate import review
    gate = review(backtest_metrics={...}, wf_verdict="OVERFIT", ...,
                 Trades=78, psr_verdict="STRONG", dsr_verdict="STRONG",
                  stress_verdict="ROBUST")
    print(gate["pass"], gate["reasons"])
"""
from __future__ import annotations


DEFAULTS = {
    "min_trades": 30,
    "min_sharpe": 0.3,
    "min_wf_pass_ratio": 0.5,
    "max_dd": 0.25,
    "accept_wf_verdicts": ("ACCEPT", "ROBUST"),
    "accept_psr_verdicts": ("STRONG",),
    "accept_dsr_verdicts": ("STRONG",),
    "accept_perm_verdicts": ("STRONG",),
    "forbid_stress_verdicts": ("FRAGILE",),
}


def review(backtest_metrics: dict | None = None, wf_verdict: str = "UNKNOWN",
           wf_passed: int = 0, wf_windows: int = 0,
           psr_verdict: str = "UNKNOWN", dsr_verdict: str = "UNKNOWN",
           perm_verdict: str = "UNKNOWN",
           stress_verdict: str = "UNKNOWN", trades: int = 0,
           **overrides) -> dict:
    """Evaluate all checks. Returns {pass, reasons, checks}."""
    cfg = {**DEFAULTS, **overrides}
    m = backtest_metrics or {}
    checks: dict[str, dict] = {}

    def _add(name: str, ok: bool, detail: str):
        checks[name] = {"pass": bool(ok), "detail": detail}

    pnl = float(m.get("net_pnl", 0) or 0)
    _add("profitable", pnl > 0, f"net_pnl=${pnl:.2f}")

    _add("enough_trades", trades >= cfg["min_trades"],
         f"{trades} trades (min {cfg['min_trades']})")

    sharpe = float(m.get("sharpe", 0) or 0)
    _add("sharpe_floor", sharpe >= cfg["min_sharpe"],
         f"sharpe={sharpe:.3f} (min {cfg['min_sharpe']})")

    ratio = (wf_passed / wf_windows) if wf_windows else 0.0
    wf_ok = (wf_verdict in cfg["accept_wf_verdicts"]
             or ratio >= cfg["min_wf_pass_ratio"])
    _add("walkforward", wf_ok,
         f"verdict={wf_verdict} passed={wf_passed}/{wf_windows}")

    sig_ok = (psr_verdict in cfg["accept_psr_verdicts"]
              or dsr_verdict in cfg["accept_dsr_verdicts"]
              or perm_verdict in cfg["accept_perm_verdicts"])
    _add("significance", sig_ok,
         f"PSR={psr_verdict} DSR={dsr_verdict} BOOT={perm_verdict}")

    _add("stress", stress_verdict not in cfg["forbid_stress_verdicts"],
         f"stress={stress_verdict}")

    dd = abs(float(m.get("max_drawdown", 0) or 0))
    _add("drawdown_cap", dd <= cfg["max_dd"],
         f"max_dd={dd * 100:.2f}% (cap {cfg['max_dd'] * 100:.0f}%)")

    failed = [k for k, v in checks.items() if not v["pass"]]
    reasons = [f"FAIL {k}: {checks[k]['detail']}" for k in failed]
    passed = [k for k in checks if k not in failed]
    return {
        "pass": not failed,
        "reasons": reasons,
        "checks": checks,
        "summary": f"{len(passed)}/{len(checks)} checks passed",
    }


# ---------- Self-test ----------
if __name__ == "__main__":
    good = review(
        backtest_metrics={"net_pnl": 297.08, "sharpe": 0.594, "max_drawdown": -0.0418},
        wf_verdict="ACCEPT", wf_passed=6, wf_windows=8,
        psr_verdict="STRONG", dsr_verdict="STRONG",
        stress_verdict="ROBUST", trades=78)
    print("good:", good["pass"], good["summary"])
    assert good["pass"] and not good["reasons"]

    bad = review(
        backtest_metrics={"net_pnl": -266.94, "sharpe": -0.557, "max_drawdown": -0.09},
        wf_verdict="OVERFIT", wf_passed=1, wf_windows=8,
        psr_verdict="POOR", dsr_verdict="OVERFIT",
        stress_verdict="ROBUST", trades=82)
    print("bad:", bad["pass"])
    for r in bad["reasons"]:
        print("  ", r)
    assert not bad["pass"] and len(bad["reasons"]) >= 3

    thin = review(backtest_metrics={"net_pnl": 50.0, "sharpe": 1.2, "max_drawdown": -0.02},
                  wf_verdict="ACCEPT", wf_passed=4, wf_windows=4,
                  psr_verdict="STRONG", dsr_verdict="STRONG",
                  stress_verdict="ROBUST", trades=5)
    assert not thin["pass"] and any("enough_trades" in r for r in thin["reasons"])
    print("thin-data correctly rejected")

    # Bootstrap-only evidence carries the significance leg.
    boot_only = review(
        backtest_metrics={"net_pnl": 300.0, "sharpe": 0.6, "max_drawdown": -0.04},
        wf_verdict="ACCEPT", wf_passed=6, wf_windows=8,
        psr_verdict="POOR", dsr_verdict="WEAK", perm_verdict="STRONG",
        stress_verdict="ROBUST", trades=80)
    assert boot_only["checks"]["significance"]["pass"], boot_only["checks"]
    assert boot_only["pass"], boot_only["reasons"]
    print("bootstrap-only significance accepted")
    print("SELF-TEST PASS")
