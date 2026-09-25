"""Competition study — Newmeta vs LEAN (QuantConnect) vs MT5 Strategy Tester.

A scored, evidence-backed matrix, not marketing. Scores are 0-5 per
dimension with a dated evidence note so claims can be re-checked as the
landscape moves (it moves fast: MT5 shipped an MCP assistant in Sep 2026).

Two outputs:
  1. where we lead (protect these),
  2. GAPS: what the competition has better, each tagged by HOW it closes:
     code (we can build it) | money (licenses/cloud to rent) |
     time (track record, compounding trust) | structural (their servers,
     cannot be coded around — mitigate instead).

Usage:
    from analysis.competition_study import matrix, gaps, report
    print(report())
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

EVIDENCE_DATE = "2026-09-25"

# dimension -> {vendor: (score, evidence)}
MATRIX: dict[str, dict[str, tuple[int, str]]] = {
    "market_data": {
        "newmeta": (3, "Dukascopy FX ticks real + cached; Yahoo biased (survivorship, no delistings)"),
        "lean": (5, "400TB+, survivorship-free, point-in-time, 40+ alt vendors, multi-asset"),
        "mt5": (4, "Broker real ticks, M1-verified, years deep; per-broker feed, no alt data"),
    },
    "execution_realism": {
        "newmeta": (3, "Real spreads (duka spread_pips), bid/ask deep path, slippage/commission; single feed, no broker-specific behavior"),
        "lean": (4, "Event-driven fills, 20 brokerage models, co-located live fills"),
        "mt5": (5, "Every-tick-real-ticks from YOUR broker incl. their spreads; SL/TP ambiguity resolved like live"),
    },
    "statistical_rigor": {
        "newmeta": (5, "PSR/DSR/FDR, PBO, Sobol, t/Mann-Whitney+bootstrap, review gate — inside the loop"),
        "lean": (2, "TradeStatistics: win/loss, expectancy, durations. No PSR/DSR/PBO/Sobol natively"),
        "mt5": (1, "No significance testing; Complex Criterion is uncorrected"),
    },
    "optimization_search": {
        "newmeta": (3, "Optuna TPE (15 trials/17s), seeded/pruned/parallel; single machine"),
        "lean": (4, "Cloud optimization at scale"),
        "mt5": (5, "Genetic + MQL5 Cloud Network (~10k agents)"),
    },
    "validation_walkforward": {
        "newmeta": (5, "Anchored WF + verdicts + auto-iterate + stall detection + PBO"),
        "lean": (3, "Out-of-sample + robustness tooling, no verdict machine"),
        "mt5": (2, "Forward-test fraction only, no overfit statistics"),
    },
    "live_trading": {
        "newmeta": (1, "Paper bridge + kill switch; zero real adapters, zero live fills"),
        "lean": (5, "375k strategies deployed, $100B/mo notional, 20 integrations"),
        "mt5": (4, "One-click same-terminal deploy; broker-dependent quality"),
    },
    "scale_cloud": {
        "newmeta": (1, "Local ProcessPoolExecutor; sandbox breaks it"),
        "lean": (5, "Cloud cores, co-located live nodes"),
        "mt5": (5, "MQL5 Cloud Network: ~10k volunteer agents for genetic optimization"),
    },
    "agent_native": {
        "newmeta": (4, "MCP server (10 tools), NL builder, auto-iterate, review gate — full loop drivable"),
        "lean": (3, "AI assistance + research agents; execution still human-gated"),
        "mt5": (3, "Sep-2026 MCP assistant (reports/optimize/logs); live judgment explicitly blocked"),
    },
    "cost_access": {
        "newmeta": (5, "Free, local, no account needed"),
        "lean": (3, "Free tier; seats/data/cloud scale with bill"),
        "mt5": (4, "Free terminal; needs broker account"),
    },
    "ecosystem_trust": {
        "newmeta": (1, "Single-author research code, months old, no live record"),
        "lean": (5, "542k quants, 13yr record, 180+ engine contributors"),
        "mt5": (5, "Every retail broker ships it; market/signals/VPS economy"),
    },
    "paper_reconciliation": {
        "newmeta": (4, "Bridge log + reconcile + drift metrics + journal"),
        "lean": (4, "Paper and live share the engine"),
        "mt5": (3, "Visual tester + forward test, manual reconciliation"),
    },
    "tax_reporting": {
        "newmeta": (4, "Auto journal + US/EU/UK/AE reports"),
        "lean": (1, "No tax reporting; broker statements only"),
        "mt5": (1, "Deals history export only, no jurisdiction logic"),
    },
}


@dataclass
class Gap:
    """One thing the competition has better."""
    area: str
    leader: str  # lean | mt5 | both
    why_they_win: str
    closability: str  # code | money | time | structural
    our_status: str
    next_action: str
    priority: int = 3  # 1 = do now, 5 = accept


GAPS: list[Gap] = [
    Gap("survivorship-bias-free multi-asset data", "lean",
        "Delisted companies vanish from Yahoo; every equity backtest flatters itself",
        "money", "flagged (no_survivorship_control); FX scope unaffected",
        "License institutional data when funded; until then stay FX-first", 1),
    Gap("live brokerage execution + track record", "lean",
        "$100B/mo across 20 integrations vs our zero live fills",
        "time", "paper bridge + kill switch, no adapter",
        "Pick ONE broker (MT5 Manager API or Binance), then 12mo audited paper", 1),
    Gap("broker-native real ticks", "mt5",
        "Only MetaQuotes sits inside every broker's server room",
        "structural", "Dukascopy real ticks (different feed than any one broker)",
        "R026 divergence harness vs MT5 reports when terminal available", 2),
    Gap("tester-to-live code identity", "mt5",
        "Same MQL5, same terminal, one click; ours needs Python->API translation",
        "structural", "paper bridge only",
        "Keep strategy logic engine-agnostic so translation surface stays small", 3),
    Gap("optimization at cloud scale", "both",
        "~10k MT5 agents / QC cloud vs our one machine",
        "money", "TPE (fewer trials needed) + local parallel; no cloud runner",
        "Rent, don't build (QC cloud or MT5 Cloud for final sweeps)", 3),
    Gap("streaming no-lookahead architecture", "lean",
        "Event-driven core cannot see the future; our pandas paths rely on discipline",
        "code", "no systematic guard",
        "Purged/embargoed splits + lookahead audit of generate()", 2),
    Gap("universe selection models", "lean",
        "Index-style tradable universes reduce selection bias",
        "code", "screener ranks cached symbols only",
        "Liquidity/market-cap universe filters over multi-asset cache", 3),
    Gap("distribution + trust compounding", "both",
        "542k quants / every-broker distribution vs single author",
        "time", "GitHub repo, no public record",
        "Publish monthly audited paper reports; trust accrues or it doesn't", 4),
]


def matrix() -> dict:
    """Full scored matrix with evidence."""
    return {dim: {v: {"score": s, "evidence": e} for v, (s, e) in vendors.items()}
            for dim, vendors in MATRIX.items()}


def our_leads() -> list[str]:
    """Dimensions where newmeta strictly tops both rivals."""
    return [dim for dim, vendors in MATRIX.items()
            if vendors["newmeta"][0] > vendors["lean"][0]
            and vendors["newmeta"][0] > vendors["mt5"][0]]


def gaps(by_closability: str | None = None, priority_max: int = 5) -> list[dict]:
    """Gaps, optionally filtered, sorted by priority."""
    out = [g for g in GAPS if g.priority <= priority_max
           and (by_closability is None or g.closability == by_closability)]
    return [asdict(g) for g in sorted(out, key=lambda g: g.priority)]


def code_closable() -> list[dict]:
    """The gaps sessions like this one can actually kill."""
    return gaps(by_closability="code")


def report() -> str:
    """Markdown positioning memo."""
    lines = [f"# Competition study ({EVIDENCE_DATE})",
             "\n## Full matrix (0-5)",
             "",
             "| dimension | newmeta | lean | mt5 |",
             "|---|---|---|---|"]
    for dim, vendors in MATRIX.items():
        lines.append(f"| {dim} | {vendors['newmeta'][0]} | {vendors['lean'][0]} "
                     f"| {vendors['mt5'][0]} |")
    lines.append("\n## Where we lead (protect these)")
    for dim in our_leads():
        lines.append(f"- **{dim}**: us {MATRIX[dim]['newmeta'][0]} "
                     f"vs LEAN {MATRIX[dim]['lean'][0]} / MT5 {MATRIX[dim]['mt5'][0]}")
    lines.append("\n## Gaps: what they have better")
    for g in gaps():
        lines.append(f"\n### [{g['priority']}] {g['area']} — leader: {g['leader']} "
                     f"({g['closability']})")
        lines.append(f"- Why they win: {g['why_they_win']}")
        lines.append(f"- Our status: {g['our_status']}")
        lines.append(f"- Next: {g['next_action']}")
    lines.append("\n## Strategy in one line")
    lines.append("Stop chasing their moats (data licenses, broker servers, cloud "
                 "fleets, 13-year trust). Deepen ours: FX rigor + agentic velocity "
                 "+ published audited record — the position neither occupies.")
    return "\n".join(lines)


def save(out_dir: str | Path = "output/reports") -> dict:
    """Write JSON + markdown artifacts. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"evidence_date": EVIDENCE_DATE, "matrix": matrix(),
               "our_leads": our_leads(), "gaps": gaps()}
    jp = out_dir / "competition_study.json"
    mp = out_dir / "competition_study.md"
    jp.write_text(json.dumps(payload, indent=2))
    mp.write_text(report())
    return {"json": str(jp), "markdown": str(mp)}


# ---------- Self-test ----------
if __name__ == "__main__":
    vendors = ("newmeta", "lean", "mt5")
    for dim, vs in MATRIX.items():
        assert set(vs) == set(vendors), dim
        for v, (s, e) in vs.items():
            assert isinstance(s, int) and 0 <= s <= 5, (dim, v, s)
            assert len(e) > 20, (dim, v)  # evidence, not vibes
    assert set(GAPS[i].closability for i in range(len(GAPS))) <= \
        {"code", "money", "time", "structural"}
    for g in GAPS:
        assert g.leader in vendors + ("both",)
        assert g.next_action and g.our_status
    leads = our_leads()
    assert "statistical_rigor" in leads and "validation_walkforward" in leads
    code = code_closable()
    assert all(g["closability"] == "code" for g in code) and len(code) >= 2
    md = report()
    assert "survivorship" in md and all(d in md for d in MATRIX)
    paths = save()
    print("dimensions:", len(MATRIX), "| leads:", leads)
    print("gaps:", len(GAPS), "| code-closable:", [g["area"][:40] for g in code])
    print("saved:", paths)
    print("SELF-TEST PASS")
