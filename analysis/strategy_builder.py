"""R023: NL strategy builder — plain-English briefs become strategy files.

No LLM needed: a deterministic mini-DSL turns descriptions like

    "Long when rsi(14) below 30, exit when rsi(14) above 70.
     Short when rsi(14) above 70, exit when rsi(14) below 30."

into a real strategies/generated_<slug>.py BaseStrategy subclass that the
pipeline, walk-forward, and MCP tools can run immediately.

Supported indicators (v1): rsi, adx (+diplus/diminus), macd (+signal),
bollinger (+upper/lower), dem.
Conditions: above | below | crosses above | crosses below, against a number
or a companion series (signal, upper, lower, diplus, diminus).
Sides: long | short | exit | cover. Direction both/long/short inferred from
which sides are present (default: both).

Usage:
    from analysis.strategy_builder import build_from_brief, SUPPORTED_HELP
    spec = build_from_brief("Long when rsi(14) below 30, exit when rsi(14) above 70.")
    # spec.path -> strategies/generated_rsi_mean_reversion.py (import-checked)
"""
from __future__ import annotations

import py_compile
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SUPPORTED_HELP = """Supported: rsi(period) | adx(period)+diplus/diminus | macd(fast,slow,signal)+signal
| bollinger(period,dev)+upper/lower | dem(period) | close|open|high|low.
Conditions: above|below|crosses above|crosses below <number|companion>.
Sides: long|short|exit|cover. Bare multi-output indicators default to their
primary series (adx line, macd line, bollinger middle) — name a companion
(signal, upper, lower, diplus, diminus) when you mean something else."""

# indicator -> (arg_names_with_defaults, companion_series)
INDICATORS = {
    "rsi": (["period=14"], {}),
    "adx": (["period=14"], {"diplus": 1, "diminus": 2, "di+": 1, "di-": 2}),
    "macd": (["fast=12", "slow=26", "signal=9"], {"signal": 1, "hist": 2}),
    "bollinger": (["period=20", "dev=2.0"], {"upper": 0, "middle": 1, "lower": 2}),
    "dem": (["period=14"], {}),
}

COMPANION_CALLS = {
    # companion -> template computing it from the base call + ohlc cols
    "adx": {
        "diplus": "ind.adx(h, l, c, period={p})[1]",
        "diminus": "ind.adx(h, l, c, period={p})[2]",
        "di+": "ind.adx(h, l, c, period={p})[1]",
        "di-": "ind.adx(h, l, c, period={p})[2]",
    },
    "macd": {
        "signal": "ind.macd(c, fast={f}, slow={s}, signal={g})[1]",
        "hist": "ind.macd(c, fast={f}, slow={s}, signal={g})[2]",
    },
    "bollinger": {
        "upper": "ind.bollinger(c, period={p}, dev={d})[0]",
        "lower": "ind.bollinger(c, period={p}, dev={d})[2]",
    },
}

BASE_CALLS = {
    "rsi": "ind.rsi(c, period={period})",
    "adx": "ind.adx(h, l, c, period={period})[0]",
    "macd": "ind.macd(c, fast={fast}, slow={slow}, signal={signal})[0]",
    "bollinger": "ind.bollinger(c, period={period}, dev={dev})[1]",
    "dem": "ind.dem(h, l, period={period})",
}


@dataclass
class Condition:
    """One parsed condition: <left_expr> <op> <right_expr>."""
    left: str
    op: str  # above | below | cross_up | cross_down
    right: str

    def _is_number(self, expr: str) -> bool:
        try:
            float(expr)
            return True
        except ValueError:
            return False

    def to_code(self, var: str) -> str:
        if self.op == "above":
            return f"({self.left} > {self.right})"
        if self.op == "below":
            return f"({self.left} < {self.right})"
        # crosses: numbers have no .shift — compare against the scalar directly.
        r_shift = self.right if self._is_number(self.right) else f"({self.right}).shift(1)"
        if self.op == "cross_up":
            return f"((({self.left} > {self.right}) & (({self.left}).shift(1) <= {r_shift})))"
        return f"((({self.left} < {self.right}) & (({self.left}).shift(1) >= {r_shift})))"


@dataclass
class Clause:
    side: str  # long | short | exit | cover
    conditions: list = field(default_factory=list)


@dataclass
class BriefSpec:
    name: str
    class_name: str
    path: Path
    clauses: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    smoke: dict = field(default_factory=dict)


_IND_RE = re.compile(r"^(rsi|adx|macd|bollinger|dem)\s*(?:\(([^)]*)\))?$", re.I)
_OP_RE = re.compile(r"\b(crosses\s+above|crosses\s+below|above|below)\b", re.I)


def _parse_indicator(token: str):
    """Parse 'rsi(14)' -> (canonical_call, params_dict, base_key)."""
    m = _IND_RE.match(token.strip())
    if not m:
        return None
    key = m.group(1).lower()
    raw_args = (m.group(2) or "").strip()
    arg_names = [a.split("=")[0] for a in INDICATORS[key][0]]
    vals = {}
    if raw_args:
        parts = [p.strip() for p in raw_args.split(",")]
        for i, part in enumerate(parts):
            if "=" in part:
                k, v = part.split("=", 1)
                vals[k.strip()] = v.strip()
            elif i < len(arg_names):
                vals[arg_names[i]] = part
    return key, vals


_OHLC = {"close": "c", "high": "h", "low": "l", "open": 'df["open"]'}


def _series_expr(token: str):
    """Turn 'rsi(14)' or 'signal' or '70' or 'close' into (code, params)."""
    token = token.strip()
    # plain number?
    try:
        float(token)
        return token, {}
    except ValueError:
        pass
    # OHLC passthrough (generated code defines c/h/l; open via df).
    if token.lower() in _OHLC:
        return _OHLC[token.lower()], {}
    # companion of a previous indicator? handled by caller via context.
    parsed = _parse_indicator(token)
    if parsed is None:
        raise ValueError(f"cannot parse series expression: {token!r}")
    key, vals = parsed
    defaults = {a.split("=")[0]: a.split("=")[1] for a in INDICATORS[key][0]}
    merged = {**defaults, **vals}
    if key == "rsi":
        code = BASE_CALLS["rsi"].format(period=merged["period"])
    elif key == "adx":
        code = BASE_CALLS["adx"].format(period=merged["period"])
    elif key == "macd":
        code = BASE_CALLS["macd"].format(**merged)
    elif key == "bollinger":
        code = BASE_CALLS["bollinger"].format(**merged)
    else:
        code = BASE_CALLS["dem"].format(period=merged["period"])
    return code, merged


def _companion_expr(comp: str, base_key: str, base_vals: dict):
    """Resolve 'signal'/'upper'/'diplus' against the base indicator call."""
    comp = comp.strip().lower()
    table = COMPANION_CALLS.get(base_key, {})
    if comp not in table:
        raise ValueError(f"{comp!r} is not a companion of {base_key}")
    tmpl = table[comp]
    defaults = {a.split("=")[0]: a.split("=")[1] for a in INDICATORS[base_key][0]}
    merged = {**defaults, **base_vals}
    if base_key == "adx":
        return tmpl.format(p=merged["period"])
    if base_key == "macd":
        return tmpl.format(f=merged["fast"], s=merged["slow"], g=merged["signal"])
    if base_key == "bollinger":
        return tmpl.format(p=merged["period"], d=merged["dev"])
    raise ValueError(f"no companion support for {base_key}")


def parse_brief(text: str):
    """Parse a brief into Clause list. Raises ValueError on bad input."""
    clauses = []
    # Split into sentences on ; newlines, periods (but never the decimal
    # point inside numbers like 2.0), plus commas that start a new
    # side-clause ("..., exit when ..."). Plain commas inside indicator
    # args like macd(12,26,9) are preserved by the lookahead.
    sentences = [s.strip() for s in
                 re.split(r"[;\n]+|(?<!\d)\.|\.(?!\d)"
                          r"|,\s*(?=(?:long|short|exit|cover)\s+when\b)",
                          text, flags=re.I) if s.strip()]
    if not sentences:
        raise ValueError("empty brief")
    for sent in sentences:
        m = re.match(r"^(long|short|exit|cover)\s+when\s+(.+)$", sent, re.I)
        if not m:
            raise ValueError(f"clause must look like '<side> when <cond> [and <cond>...]': {sent!r}")
        side = m.group(1).lower()
        cond_strs = [c.strip() for c in re.split(r"\band\b", m.group(2), flags=re.I)]
        conds = []
        last_base = (None, {})
        for cs in cond_strs:
            om = _OP_RE.search(cs)
            if not om:
                raise ValueError(f"need above|below|crosses above|crosses below in: {cs!r}")
            op_raw = re.sub(r"\s+", " ", om.group(1).lower())
            op = {"above": "above", "below": "below",
                  "crosses above": "cross_up", "crosses below": "cross_down"}[op_raw]
            left_tok = cs[:om.start()].strip()
            right_tok = cs[om.end():].strip()
            left_code, left_vals = _series_expr(left_tok)
            # remember base indicator for companion resolution on the right
            parsed_left = _parse_indicator(left_tok)
            if parsed_left:
                lk, lv = parsed_left
                last_base = (lk, lv)
            try:
                float(right_tok)
                right_code = right_tok
            except ValueError:
                # companion series?
                if last_base[0] and right_tok.lower() in COMPANION_CALLS.get(last_base[0], {}):
                    right_code = _companion_expr(right_tok, last_base[0], last_base[1])
                else:
                    right_code, _ = _series_expr(right_tok)
            conds.append(Condition(left_code, op, right_code))
        clauses.append(Clause(side, conds))
    return clauses


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    words = [w for w in s.split("_") if w not in
             ("long", "short", "when", "exit", "cover", "and", "above", "below",
              "crosses", "the", "a", "an", "is")]
    return "_".join(words[:4]) or "custom"


def generate_code(class_name: str, clauses: list, params: dict) -> str:
    """Emit the strategy module source."""
    lines = []
    lines.append('"""Auto-generated by analysis.strategy_builder — review before live use."""')
    lines.append("from __future__ import annotations")
    lines.append("import pandas as pd")
    lines.append("from . import indicators as ind")
    lines.append("from ._base import BaseStrategy, Signals")
    lines.append("")
    lines.append("")
    lines.append(f"class {class_name}(BaseStrategy):")
    lines.append(f'    name = "{class_name.lower()}"')
    lines.append("")

    var_defs = []   # (varname, code) in encounter order, deduped
    seen = set()

    def _var_for(code):
        if code not in seen:
            seen.add(code)
            var_defs.append((f"_v{len(var_defs)}", code))
        return next(v for v, c in var_defs if c == code)

    clause_masks = []  # (side, mask_expr)
    for ci, cl in enumerate(clauses):
        parts = []
        for cond in cl.conditions:
            lv = _var_for(cond.left)
            rv = _var_for(cond.right)
            c2 = Condition(lv, cond.op, rv)
            parts.append(c2.to_code(f"m{ci}"))
        clause_masks.append((cl.side, " & ".join(f"({p})" for p in parts)))

    lines.append("    def generate(self, df: pd.DataFrame) -> Signals:")
    lines.append("        p = self._resolve_params()")
    lines.append("        c, h, l = df[\"close\"], df[\"high\"], df[\"low\"]")
    for v, code in var_defs:
        lines.append(f"        {v} = {code}")
    lines.append("        z = pd.Series(False, index=df.index)")
    for ci, (side, mask) in enumerate(clause_masks):
        lines.append(f"        m{ci} = {mask}.fillna(False)")
    long_m = " | ".join(f"m{ci}" for ci, (s, _) in enumerate(clause_masks) if s == "long") or "z"
    short_m = " | ".join(f"m{ci}" for ci, (s, _) in enumerate(clause_masks) if s == "short") or "z"
    exit_m = " | ".join(f"m{ci}" for ci, (s, _) in enumerate(clause_masks) if s in ("exit", "cover")) or "z"
    lines.append(f"        entries_long = ({long_m}).fillna(False)")
    lines.append(f"        entries_short = ({short_m}).fillna(False)")
    lines.append(f"        exits = ({exit_m}).fillna(False)")
    lines.append("        direction = pd.Series(0, index=df.index, dtype=int)")
    lines.append("        direction[entries_long] = 1")
    lines.append("        direction[entries_short] = -1")
    lines.append("        entries = (entries_long | entries_short).fillna(False)")
    lines.append("        return Signals(entries=entries, exits=exits, direction=direction)")
    lines.append("")
    return "\n".join(lines)


def build_from_brief(text: str, name: str | None = None,
                     out_dir: str | Path = "strategies") -> BriefSpec:
    """Parse -> generate -> write -> import-check. Returns BriefSpec with smoke report."""
    import importlib as _il
    _il.import_module("strategies.indicators")  # fail fast if indicator lib is broken
    clauses = parse_brief(text)
    slug = re.sub(r"[^a-z0-9_]", "", _slug(name or text)) or "custom"
    class_name = "".join(w.capitalize() for w in slug.split("_")) + "_Strategy"
    code = generate_code(class_name, clauses, {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"generated_{slug}.py"
    path.write_text(code)
    py_compile.compile(str(path), doraise=True)

    # Import-check + structural smoke on synthetic data.
    import importlib

    import numpy as np
    mod = importlib.import_module(f"strategies.generated_{slug}")
    cls = getattr(mod, class_name)
    rng = np.random.default_rng(7)
    n = 300
    close = pd.Series(1.1 + np.cumsum(rng.normal(0, 0.0008, n)))
    synth = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * (1 + rng.uniform(0, 0.001, n)),
        "low": close * (1 - rng.uniform(0, 0.001, n)),
        "close": close,
        "volume": np.full(n, 100.0),
    })
    sig = cls(name=slug, params={}).generate(synth)
    smoke = {
        "entries": int(sig.entries.sum()),
        "exits": int(sig.exits.sum()),
        "len_match": len(sig.entries) == n and len(sig.exits) == n,
        "bool_dtype": bool(sig.entries.dtype == bool and sig.exits.dtype == bool),
    }
    if not (smoke["len_match"] and smoke["bool_dtype"]):
        raise RuntimeError(f"generated strategy failed structural smoke: {smoke}")
    return BriefSpec(name=slug, class_name=class_name, path=path,
                     clauses=clauses, params={}, smoke=smoke)


# ---------- Self-test ----------
if __name__ == "__main__":
    s1 = build_from_brief(
        "Long when rsi(14) below 30, exit when rsi(14) above 70. "
        "Short when rsi(14) above 70, exit when rsi(14) below 30.",
        name="rsi_mean_reversion")
    print("built:", s1.path, s1.class_name, s1.smoke)

    s2 = build_from_brief("Long when macd(12,26,9) crosses above signal.")
    print("built:", s2.path, s2.class_name, s2.smoke)

    s3 = build_from_brief("Long when adx(14) above 25 and close crosses above bollinger(20,2.0).")
    print("built:", s3.path, s3.class_name, s3.smoke)

    try:
        build_from_brief("buy when vibes are good")
        raise SystemExit("should have raised")
    except ValueError as e:
        print("bad brief correctly rejected:", str(e)[:80])

    # cleanup self-test artifacts (keep strategies/ clean; real builds use explicit names)
    for s in (s1, s2, s3):
        s.path.unlink(missing_ok=True)
    import shutil
    shutil.rmtree("strategies/__pycache__", ignore_errors=True)
    print("SELF-TEST PASS")
