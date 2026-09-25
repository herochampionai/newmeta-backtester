"""Parameter sensitivity analysis — Sobol indices, tornado charts, interaction detection.

R010: Quantifies how much each parameter contributes to the variance of the backtest
objective. Done BEFORE optimization (to inform the search space) and AFTER (to
verify the optimal isn't a sharp spike).

Features:
- Sobol first-order indices (Saltelli sampling, Jansen estimator)
- Sobol total-order indices (detects interactions)
- Tornado chart data (one-at-a-time sensitivity)
- Interaction detection (param pairs with high total - first)
- Tornado-style ranking by importance
- CLI for quick analysis
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def sobol_sampling(n: int, param_specs: list[dict]) -> np.ndarray:
    """Saltelli sampling scheme: generates N*(2D+2) samples.

    Each param_spec is dict with 'name', 'low', 'high', 'type' ('float' or 'int').
    Returns array of shape (N*(2D+2), D).
    """
    D = len(param_specs)
    # Saltelli: A, B, AB_i matrices for i in 1..D
    # A and B: N x D random in [0,1]
    # AB_i: A with column i replaced by B[:, i]
    n_base = n
    n_total = n_base * (2 * D + 2)

    # Generate base samples
    A = np.random.uniform(0, 1, size=(n_base, D))
    B = np.random.uniform(0, 1, size=(n_base, D))

    samples = [A, B]  # First two matrices
    for i in range(D):
        AB = A.copy()
        AB[:, i] = B[:, i]
        samples.append(AB)

    X = np.vstack(samples)  # (N*(2D+2), D)

    # Scale to actual ranges
    scaled = np.zeros_like(X)
    for j, spec in enumerate(param_specs):
        low = spec['low']
        high = spec['high']
        scaled[:, j] = low + X[:, j] * (high - low)
        if spec.get('type') == 'int':
            scaled[:, j] = np.round(scaled[:, j]).astype(int)
    return scaled


def _evaluate_params(samples: np.ndarray, param_specs: list[dict],
                     evaluator: Callable) -> np.ndarray:
    """Evaluate objective for each sample row."""
    n = len(samples)
    Y = np.zeros(n)
    for i in range(n):
        params = {spec['name']: samples[i, j] for j, spec in enumerate(param_specs)}
        try:
            Y[i] = float(evaluator(params))
        except Exception:
            Y[i] = np.nan
    return Y


def sobol_indices(samples: np.ndarray, Y: np.ndarray, D: int) -> dict:
    """Compute Sobol first-order (S_i) and total-order (ST_i) indices.

    Jansen estimator for S_i: V_i / V
    Jansen estimator for ST_i: 1 - V_{-i} / V
    """
    n_base = len(Y) // (2 * D + 2)

    # Reshape into A, B, AB_1, ..., AB_D
    A = Y[:n_base]
    B = Y[n_base:2 * n_base]
    AB_list = []
    for i in range(D):
        AB_list.append(Y[(2 + i) * n_base:(3 + i) * n_base])

    # Total variance
    Y_clean = Y[~np.isnan(Y)]
    if len(Y_clean) == 0:
        return {"S": [], "ST": [], "V": 0.0}
    V = float(np.var(Y_clean))

    if V == 0:
        return {"S": [0.0] * D, "ST": [0.0] * D, "V": 0.0}

    S = []
    ST = []
    for i in range(D):
        # First-order (Jansen): V_i = mean(B * (AB_i - A))
        Vi = float(np.mean(B * (AB_list[i] - A)))
        Si = Vi / V

        # Total-order (Jansen): V_{-i} = mean(A * (AB_i - B)) → ST_i = 1 - V_{-i}/V
        V_minus_i = float(np.mean(A * (AB_list[i] - B)))
        STi = 1.0 - V_minus_i / V

        # Clamp to [0, 1] to handle Jansen estimator noise
        Si_clamped = max(0.0, min(1.0, Si))
        STi_clamped = max(0.0, min(1.0, STi))

        S.append(round(Si_clamped, 4))
        ST.append(round(STi_clamped, 4))

    return {"S": S, "ST": ST, "V": round(V, 4)}


def tornado_chart_data(samples: np.ndarray, Y: np.ndarray,
                       param_specs: list[dict]) -> pd.DataFrame:
    """Compute one-at-a-time sensitivity (correlation-based proxy).

    For each param, split samples into low/high (median split) and measure
    mean objective in each bucket. Tornado = range of means.
    """
    rows = []
    valid = ~np.isnan(Y)
    if valid.sum() == 0:
        return pd.DataFrame()

    for j, spec in enumerate(param_specs):
        col = samples[valid, j]
        y = Y[valid]
        median = np.median(col)
        low_mask = col <= median
        high_mask = col > median

        mean_low = float(np.mean(y[low_mask])) if low_mask.sum() > 0 else 0.0
        mean_high = float(np.mean(y[high_mask])) if high_mask.sum() > 0 else 0.0
        delta = mean_high - mean_low

        # Also compute Spearman correlation
        from scipy import stats
        try:
            corr, _ = stats.spearmanr(col, y)
            corr = float(corr) if np.isfinite(corr) else 0.0
        except Exception:
            corr = 0.0

        rows.append({
            "param": spec["name"],
            "low_mean": round(mean_low, 4),
            "high_mean": round(mean_high, 4),
            "delta": round(delta, 4),
            "abs_delta": round(abs(delta), 4),
            "spearman_corr": round(corr, 3),
            "low": spec["low"],
            "high": spec["high"],
        })

    return pd.DataFrame(rows).sort_values("abs_delta", ascending=False).reset_index(drop=True)


def detect_interactions(sobol_result: dict, param_names: list[str]) -> pd.DataFrame:
    """Detect parameter interactions from Sobol indices.

    Interaction indicator: ST_i - S_i > 0.2 means the param has significant
    interactions with others.
    """
    S = sobol_result.get("S", [])
    ST = sobol_result.get("ST", [])

    if not S or not ST or len(S) != len(param_names):
        return pd.DataFrame()

    rows = []
    for i, name in enumerate(param_names):
        si = S[i]
        sti = ST[i]
        interaction = max(0.0, sti - si)
        rows.append({
            "param": name,
            "first_order": si,
            "total_order": sti,
            "interaction": round(interaction, 4),
            "verdict": (
                "STRONG_INTERACTION" if interaction > 0.4 else
                "MEDIUM_INTERACTION" if interaction > 0.2 else
                "LOW_INTERACTION"
            )
        })

    return pd.DataFrame(rows).sort_values("interaction", ascending=False).reset_index(drop=True)


def analyze_parameter_sensitivity(
    param_specs: list[dict],
    evaluator: Callable,
    n_samples: int = 256,
    seed: int = 42
) -> dict:
    """Full sensitivity analysis: Sobol + tornado + interactions.

    Args:
        param_specs: list of dicts with 'name', 'low', 'high', optional 'type'
        evaluator: function(params_dict) -> scalar objective (higher = better)
        n_samples: base sample count (total = N*(2D+2))
        seed: random seed

    Returns:
        dict with:
        - sobol: first/total indices per param
        - tornado: tornado chart data
        - interactions: interaction detection
        - importance_ranking: params sorted by total importance
        - samples: the actual sampled params (for debugging)
        - Y: the objective values
    """
    np.random.seed(seed)
    D = len(param_specs)
    if D == 0:
        return {"sobol": {}, "tornado": pd.DataFrame(), "interactions": pd.DataFrame(),
                "importance_ranking": [], "samples": np.array([]), "Y": np.array([])}

    # Generate samples
    samples = sobol_sampling(n_samples, param_specs)

    # Evaluate
    Y = _evaluate_params(samples, param_specs, evaluator)

    # Sobol
    sobol_result = sobol_indices(samples, Y, D)

    # Tornado
    tornado = tornado_chart_data(samples, Y, param_specs)

    # Interactions
    param_names = [s["name"] for s in param_specs]
    interactions = detect_interactions(sobol_result, param_names)

    # Importance ranking
    importance = []
    if len(sobol_result.get("ST", [])) == len(param_names):
        for name, sti in zip(param_names, sobol_result["ST"]):
            importance.append({"param": name, "total_importance": sti})
        importance.sort(key=lambda x: x["total_importance"], reverse=True)

    return {
        "sobol": sobol_result,
        "tornado": tornado,
        "interactions": interactions,
        "importance_ranking": importance,
        "samples": samples,
        "Y": Y,
    }


def export_sensitivity_report(result: dict, out_path: str | Path) -> str:
    """Export sensitivity report as HTML with tornado chart."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sobol = result.get("sobol", {})
    tornado = result.get("tornado", pd.DataFrame())
    interactions = result.get("interactions", pd.DataFrame())
    ranking = result.get("importance_ranking", [])

    # Build tornado rows
    tornado_rows = ""
    if not tornado.empty:
        for _, r in tornado.iterrows():
            tornado_rows += f"<tr><td>{r['param']}</td><td>{r['low_mean']:.3f}</td><td>{r['high_mean']:.3f}</td><td>{r['delta']:+.3f}</td><td>{r['spearman_corr']:+.3f}</td></tr>"

    # Sobol rows
    sobol_rows = ""
    if ranking:
        for r in ranking:
            idx = next((i for i, p in enumerate(result.get("interactions", pd.DataFrame()).to_dict("records")) if p["param"] == r["param"]), -1)
            si = sobol.get("S", [])[idx] if idx >= 0 and idx < len(sobol.get("S", [])) else 0
            sti = r["total_importance"]
            sobol_rows += f"<tr><td>{r['param']}</td><td>{si:.3f}</td><td>{sti:.3f}</td><td>{max(0, sti-si):.3f}</td></tr>"

    # Interactions rows
    interaction_rows = ""
    if not interactions.empty:
        for _, r in interactions.iterrows():
            cls = "high" if r["interaction"] > 0.4 else ("medium" if r["interaction"] > 0.2 else "low")
            interaction_rows += f"<tr class='{cls}'><td>{r['param']}</td><td>{r['first_order']:.3f}</td><td>{r['total_order']:.3f}</td><td>{r['interaction']:.3f}</td><td>{r['verdict']}</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Parameter Sensitivity Report</title>
<style>
body{{font-family:'Segoe UI',Arial;background:#0d1117;color:#e6edf3;margin:0;padding:24px}}
h1{{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:8px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:16px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{border:1px solid #30363d;padding:8px;text-align:left}}
th{{background:#21262d;color:#8b949e}}
tr.high{{background:#3a1a1a}}
tr.medium{{background:#3a2a1a}}
tr.low{{background:#1a2a1a}}
.high{{color:#ff6b6b;font-weight:600}}
.medium{{color:#ffb800;font-weight:600}}
.low{{color:#7bd88f;font-weight:600}}
</style></head><body>
<h1>📊 Parameter Sensitivity Report</h1>
<div class='card'><h3>🌪️ Tornado Chart (one-at-a-time)</h3>
<table>
<tr><th>Param</th><th>Low Mean</th><th>High Mean</th><th>Delta</th><th>Spearman ρ</th></tr>
{tornado_rows or "<tr><td colspan=5>No data</td></tr>"}
</table></div>

<div class='card'><h3>📈 Sobol Indices</h3>
<table>
<tr><th>Param</th><th>First-order (S)</th><th>Total-order (ST)</th><th>Interaction (ST-S)</th></tr>
{sobol_rows or "<tr><td colspan=4>No data</td></tr>"}
</table>
<p><small>S = direct effect, ST = total effect including interactions. ST-S = interaction strength.</small></p>
</div>

<div class='card'><h3>🔗 Interactions</h3>
<table>
<tr><th>Param</th><th>First</th><th>Total</th><th>Interaction</th><th>Verdict</th></tr>
{interaction_rows or "<tr><td colspan=5>No data</td></tr>"}
</table></div>
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


# ---------- Self-test ----------
if __name__ == "__main__":
    # Test with a known function: f(x, y) = x^2 + 0.5*y
    # x has higher sensitivity (Sobol S_x ~ 0.8)
    # y has lower sensitivity (Sobol S_y ~ 0.2)

    np.random.seed(42)
    param_specs = [
        {"name": "x", "low": -1.0, "high": 1.0, "type": "float"},
        {"name": "y", "low": -1.0, "high": 1.0, "type": "float"},
    ]

    def test_eval(params):
        x = params["x"]
        y = params["y"]
        return x + 0.3 * y + 0.2 * x * y  # x is dominant, y is secondary, interaction

    result = analyze_parameter_sensitivity(param_specs, test_eval, n_samples=512)
    print("Sobol S:", result["sobol"]["S"])
    print("Sobol ST:", result["sobol"]["ST"])
    print()
    print("Tornado:")
    print(result["tornado"])
    print()
    print("Interactions:")
    print(result["interactions"])
    print()
    print("Ranking:")
    for r in result["importance_ranking"]:
        print(f"  {r['param']}: {r['total_importance']:.3f}")

    export_sensitivity_report(result, "output/sensitivity_test.html")
    print()
    print("Report: output/sensitivity_test.html")
