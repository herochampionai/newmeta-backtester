"""R007: Factor attribution — PCA factors, HMM regimes, Brinson attribution.

For PORTFOLIO-level analysis only (requires ≥3 assets for meaningful results).

Features:
- PCA on portfolio returns: extract latent factors (market, carry, momentum, vol proxies)
- HMM regime detection: bull/bear/crisis/transition states
- Regime-conditioned metrics: Sharpe/Sortino per regime
- Brinson attribution: allocation + selection + interaction effects vs benchmark
- Factor exposure report: how much of returns come from each factor

Usage:
    attribution = FactorAttribution(portfolio_returns, benchmark_returns)
    attribution.compute_all()
    report = attribution.report()
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class FactorResult:
    """PCA factor analysis result."""
    n_components: int = 0
    explained_variance: list[float] = field(default_factory=list)
    cumulative_variance: list[float] = field(default_factory=list)
    loadings: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_returns: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class RegimeResult:
    """HMM regime detection result."""
    n_regimes: int = 0
    regime_labels: list[str] = field(default_factory=list)
    regime_assignments: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    current_regime: str = "unknown"


@dataclass
class BrinsonResult:
    """Brinson attribution vs benchmark."""
    total_allocation_effect: float = 0.0
    total_selection_effect: float = 0.0
    total_interaction_effect: float = 0.0
    per_asset: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class AttributionReport:
    factors: Optional[FactorResult] = None
    regimes: Optional[RegimeResult] = None
    brinson: Optional[BrinsonResult] = None
    summary: dict = field(default_factory=dict)


class FactorAttribution:
    """Portfolio-level factor, regime, and Brinson attribution."""

    def __init__(self, returns: pd.DataFrame, benchmark_returns: pd.Series | None = None,
                 weights: np.ndarray | None = None, n_regimes: int = 3,
                 min_assets_for_pca: int = 3):
        """Args:
            returns: DataFrame of asset returns (columns = assets)
            benchmark_returns: Series of benchmark returns (optional, for Brinson)
            weights: portfolio weights (default: equal-weighted)
            n_regimes: number of HMM regimes
            min_assets_for_pca: minimum assets needed for PCA
        """
        self.returns = returns
        self.benchmark = benchmark_returns
        self.n_regimes = n_regimes
        self.min_assets = min_assets_for_pca
        n = returns.shape[1]
        self.weights = weights if weights is not None else np.ones(n) / n

    def compute_all(self) -> AttributionReport:
        report = AttributionReport()
        report.factors = self.compute_pca()
        report.regimes = self.compute_regimes()
        if self.benchmark is not None:
            report.brinson = self.compute_brinson()
        report.summary = self._summary(report)
        return report

    def compute_pca(self) -> FactorResult | None:
        """PCA on asset returns."""
        n_assets = self.returns.shape[1]
        if n_assets < self.min_assets:
            return None

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return None

        # Standardize returns
        clean = self.returns.dropna()
        if len(clean) < 30:
            return None

        scaler = StandardScaler()
        X = scaler.fit_transform(clean)

        # PCA
        n_components = min(n_assets, 3)
        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(X)

        explained = list(pca.explained_variance_ratio_)
        cum = np.cumsum(explained).tolist()

        loadings = pd.DataFrame(
            pca.components_,
            columns=self.returns.columns,
            index=[f"Factor{i+1}" for i in range(n_components)]
        )

        factor_returns = pd.DataFrame(
            factors,
            index=clean.index,
            columns=[f"Factor{i+1}" for i in range(n_components)]
        )

        return FactorResult(
            n_components=n_components,
            explained_variance=[round(float(x), 4) for x in explained],
            cumulative_variance=[round(float(x), 4) for x in cum],
            loadings=loadings,
            factor_returns=factor_returns,
        )

    def compute_regimes(self) -> RegimeResult | None:
        """HMM regime detection on portfolio returns."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            return self._simple_regime_detection()

        # Compute portfolio returns
        portfolio_ret = (self.returns * self.weights).sum(axis=1).dropna()
        if len(portfolio_ret) < 50:
            return None

        # Prepare features: returns + rolling volatility
        X = np.column_stack([
            portfolio_ret.values,
            portfolio_ret.rolling(20).std().fillna(0).values,
        ])

        try:
            model = GaussianHMM(n_components=self.n_regimes, covariance_type="full",
                                n_iter=100, random_state=42)
            model.fit(X)
            assignments = model.predict(X)

            # Label regimes by mean return (bull=highest, bear=lowest)
            means = [portfolio_ret.values[assignments == k].mean() for k in range(self.n_regimes)]
            labels_map = sorted(range(self.n_regimes), key=lambda i: means[i], reverse=True)
            label_names = ["BULL", "NEUTRAL", "BEAR", "CRISIS"][:self.n_regimes]
            new_labels = [None] * self.n_regimes
            for new_idx, old_idx in enumerate(labels_map):
                new_labels[old_idx] = label_names[new_idx] if new_idx < len(label_names) else f"R{new_idx}"

            regime_labels = [new_labels[a] for a in assignments]
            current_regime = regime_labels[-1]

            # Stats per regime
            stats_rows = []
            for k in range(self.n_regimes):
                mask = assignments == k
                if mask.sum() > 0:
                    stats_rows.append({
                        "regime": new_labels[k],
                        "n_periods": int(mask.sum()),
                        "pct_of_time": round(float(mask.sum() / len(assignments) * 100), 1),
                        "mean_return": round(float(portfolio_ret.values[mask].mean()), 5),
                        "volatility": round(float(portfolio_ret.values[mask].std()), 5),
                        "annualized_return": round(float(portfolio_ret.values[mask].mean() * 252), 3),
                        "annualized_vol": round(float(portfolio_ret.values[mask].std() * np.sqrt(252)), 3),
                    })

            return RegimeResult(
                n_regimes=self.n_regimes,
                regime_labels=regime_labels,
                regime_assignments=assignments,
                regime_stats=pd.DataFrame(stats_rows),
                transition_matrix=model.transmat_,
                current_regime=current_regime,
            )
        except Exception:
            return self._simple_regime_detection()

    def _simple_regime_detection(self) -> RegimeResult | None:
        """Fallback: vol-based regime detection (no hmmlearn required)."""
        portfolio_ret = (self.returns * self.weights).sum(axis=1).dropna()
        if len(portfolio_ret) < 30:
            return None

        rolling_vol = portfolio_ret.rolling(20).std().fillna(0)
        vol_p33 = rolling_vol.quantile(0.33)
        vol_p67 = rolling_vol.quantile(0.67)

        regimes = np.where(
            rolling_vol < vol_p33, 0,
            np.where(rolling_vol > vol_p67, 2, 1)
        )
        label_names = ["BULL", "NEUTRAL", "BEAR"][:self.n_regimes]
        regime_labels = [label_names[r] for r in regimes]

        stats_rows = []
        for k in range(self.n_regimes):
            mask = regimes == k
            if mask.sum() > 0:
                stats_rows.append({
                    "regime": label_names[k],
                    "n_periods": int(mask.sum()),
                    "pct_of_time": round(float(mask.sum() / len(regimes) * 100), 1),
                    "mean_return": round(float(portfolio_ret.values[mask].mean()), 5),
                    "volatility": round(float(portfolio_ret.values[mask].std()), 5),
                })

        return RegimeResult(
            n_regimes=self.n_regimes,
            regime_labels=regime_labels,
            regime_assignments=regimes,
            regime_stats=pd.DataFrame(stats_rows),
            transition_matrix=np.eye(self.n_regimes),
            current_regime=regime_labels[-1],
        )

    def compute_brinson(self) -> BrinsonResult | None:
        """Brinson-Fachler attribution: allocation, selection, interaction vs benchmark."""
        if self.benchmark is None:
            return None

        # Align dates
        common_dates = self.returns.index.intersection(self.benchmark.index)
        if len(common_dates) < 30:
            return None

        rets = self.returns.loc[common_dates]
        bench = self.benchmark.loc[common_dates]

        # Period returns (cumulative over period)
        period_rets = (1 + rets).prod() - 1
        period_bench = (1 + bench).prod() - 1

        # Portfolio return (weighted)
        port_ret = float((period_rets * self.weights).sum())

        # Brinson effects per asset:
        # Allocation = (w_p - w_b) * (R_b)
        # Selection = w_b * (R_p - R_b)
        # Interaction = (w_p - w_b) * (R_p - R_b)
        # Note: for benchmark, w_b = 1/N (cap-weighted proxy)
        n_assets = len(period_rets)
        w_b = np.ones(n_assets) / n_assets

        rows = []
        for i, asset in enumerate(period_rets.index):
            R_p = float(period_rets.iloc[i])
            R_b_period = float(period_bench)
            alloc = (self.weights[i] - w_b[i]) * R_b_period
            sel = w_b[i] * (R_p - R_b_period)
            inter = (self.weights[i] - w_b[i]) * (R_p - R_b_period)
            rows.append({
                "asset": asset,
                "weight_port": round(float(self.weights[i]), 4),
                "weight_bench": round(float(w_b[i]), 4),
                "return": round(R_p, 4),
                "allocation_effect": round(alloc, 4),
                "selection_effect": round(sel, 4),
                "interaction_effect": round(inter, 4),
            })

        per_asset = pd.DataFrame(rows)
        return BrinsonResult(
            total_allocation_effect=round(float(per_asset["allocation_effect"].sum()), 4),
            total_selection_effect=round(float(per_asset["selection_effect"].sum()), 4),
            total_interaction_effect=round(float(per_asset["interaction_effect"].sum()), 4),
            per_asset=per_asset,
        )

    def _summary(self, report: AttributionReport) -> dict:
        s = {}
        if report.factors:
            s["factors"] = {
                "n_components": report.factors.n_components,
                "first_factor_var": report.factors.explained_variance[0] if report.factors.explained_variance else 0,
                "first_3_factors_total_var": sum(report.factors.explained_variance[:3]),
            }
        if report.regimes:
            s["regimes"] = {
                "n_regimes": report.regimes.n_regimes,
                "current_regime": report.regimes.current_regime,
                "regime_distribution": (
                    report.regimes.regime_stats.set_index("regime")["pct_of_time"].to_dict()
                    if not report.regimes.regime_stats.empty else {}
                ),
            }
        if report.brinson:
            s["brinson"] = {
                "allocation": report.brinson.total_allocation_effect,
                "selection": report.brinson.total_selection_effect,
                "interaction": report.brinson.total_interaction_effect,
            }
        return s


def regime_conditioned_metrics(returns: pd.DataFrame, weights: np.ndarray | None = None,
                                n_regimes: int = 3) -> pd.DataFrame | None:
    """Compute Sharpe/Sortino/Calmar per regime."""
    try:
        fa = FactorAttribution(returns, weights=weights, n_regimes=n_regimes)
        regime_result = fa.compute_regimes()
        if regime_result is None or regime_result.regime_stats.empty:
            return None

        portfolio_ret = (returns * (weights or np.ones(returns.shape[1]) / returns.shape[1])).sum(axis=1)

        # Compute per-regime metrics
        rows = []
        for _, row in regime_result.regime_stats.iterrows():
            regime_name = row["regime"]
            regime_label = regime_name
            for k, label in enumerate(set(regime_result.regime_labels)):
                if label == regime_name:
                    mask = regime_result.regime_assignments == k
                    break
            else:
                continue

            ret = portfolio_ret.iloc[mask.values if hasattr(mask, 'values') else mask]
            if len(ret) > 1:
                sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
                # Sortino
                downside = ret[ret < 0].std()
                sortino = float(ret.mean() / downside * np.sqrt(252)) if downside > 0 else 0
                # Max DD
                cum = (1 + ret).cumprod()
                max_dd = float((cum / cum.cummax() - 1).min()) if len(cum) > 0 else 0
                # Calmar
                cagr = float((cum.iloc[-1]) ** (252 / len(ret)) - 1) if cum.iloc[-1] > 0 else 0
                calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0
                rows.append({
                    "regime": regime_name,
                    "n_periods": int(mask.sum()),
                    "sharpe": round(sharpe, 2),
                    "sortino": round(sortino, 2),
                    "max_dd": round(max_dd, 4),
                    "calmar": round(calmar, 2),
                    "ann_return": round(float(ret.mean() * 252), 4),
                    "ann_vol": round(float(ret.std() * np.sqrt(252)), 4),
                })
        return pd.DataFrame(rows)
    except Exception as e:
        return None


def export_attribution_report(report: AttributionReport, out_path: str | Path) -> str:
    """Export as HTML."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    s = report.summary

    factors_html = ""
    if report.factors:
        ev = ", ".join(f"{x:.1%}" for x in report.factors.explained_variance)
        factors_html = f"<div class='card'><h3>📊 PCA Factors ({report.factors.n_components})</h3><p>Explained variance: {ev}</p></div>"

    regimes_html = ""
    if report.regimes:
        rows = ""
        if not report.regimes.regime_stats.empty:
            for _, r in report.regimes.regime_stats.iterrows():
                rows += f"<tr><td>{r['regime']}</td><td>{r['n_periods']}</td><td>{r['pct_of_time']}%</td>"
                if "annualized_return" in r:
                    rows += f"<td>{r['annualized_return']:.3f}</td><td>{r['annualized_vol']:.3f}</td>"
                rows += "</tr>"
        regimes_html = f"<div class='card'><h3>🎯 Regime Detection ({report.regimes.n_regimes} states)</h3>"
        regimes_html += f"<p>Current: <b>{report.regimes.current_regime}</b></p>"
        regimes_html += f"<table><tr><th>Regime</th><th>N</th><th>% Time</th><th>Ann.Ret</th><th>Ann.Vol</th></tr>{rows}</table></div>"

    brinson_html = ""
    if report.brinson:
        b = report.brinson
        brinson_html = f"<div class='card'><h3>💼 Brinson Attribution</h3>"
        brinson_html += f"<p>Allocation: <b>{b.total_allocation_effect:+.4f}</b></p>"
        brinson_html += f"<p>Selection: <b>{b.total_selection_effect:+.4f}</b></p>"
        brinson_html += f"<p>Interaction: <b>{b.total_interaction_effect:+.4f}</b></p></div>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Factor Attribution Report</title>
<style>
body{{font-family:'Segoe UI',Arial;background:#0d1117;color:#e6edf3;margin:0;padding:24px}}
h1{{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:8px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:16px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{border:1px solid #30363d;padding:8px}}
th{{background:#21262d;color:#8b949e}}
</style></head><body>
<h1>📊 Factor Attribution Report</h1>
{factors_html}
{regimes_html}
{brinson_html}
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


# ---------- Self-test ----------
if __name__ == "__main__":
    np.random.seed(42)
    # Simulate portfolio with 4 assets and 500 days
    n_days = 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D", tz="UTC")

    # Common factor (market)
    market = np.cumsum(np.random.randn(n_days) * 0.01)

    # Asset returns = beta * market + idiosyncratic
    assets = {}
    for i, name in enumerate(["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]):
        beta = 0.5 + i * 0.2
        idio = np.random.randn(n_days) * 0.005
        assets[name] = market * beta + idio
    returns = pd.DataFrame(assets, index=dates)
    benchmark = pd.Series(market, index=dates)

    fa = FactorAttribution(returns, benchmark, weights=np.array([0.4, 0.3, 0.2, 0.1]))
    report = fa.compute_all()

    print("=== Summary ===")
    print(json.dumps(report.summary, indent=2, default=str))

    print()
    print("=== Regime Stats ===")
    if report.regimes:
        print(report.regimes.regime_stats.to_string())

    print()
    print("=== Brinson Attribution ===")
    if report.brinson:
        print(f"Allocation: {report.brinson.total_allocation_effect}")
        print(f"Selection: {report.brinson.total_selection_effect}")
        print(f"Interaction: {report.brinson.total_interaction_effect}")

    export_attribution_report(report, "output/attribution_test.html")
    print()
    print("Report: output/attribution_test.html")