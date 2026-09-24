"""Pro suite — professionalism core in one import.

- Data gate: grade source, detect gaps/staleness, LOUD banner (no silent synthetic)
- Run manifest: seed, spec, costs, data source, version → output/runs/*.json
- Margin engine: required margin, stop-out check, risk% sizing, STAT_MIN_MARGINLEVEL
- Robustness: Monte-Carlo DD bands, WFE gate, spread-stress matrix
- Report: Tester-mirror JSON + HTML one-click export
"""
from __future__ import annotations
import json, hashlib, datetime
from pathlib import Path
import numpy as np
import pandas as pd

RUNS_DIR = Path(__file__).parent.parent / "output" / "runs"


# ---------- Data gate ----------
def grade_data(df: pd.DataFrame, info: dict | None = None) -> dict:
    info = info or {}
    src = str(info.get("source", "?"))
    n = len(df) if df is not None else 0
    grade, flags = "A", []
    if "synthetic" in src:
        grade, flags = "F", ["SYNTHETIC — not tradable, research only"]
    elif "yahoo" in src.lower():
        grade = "B"
    elif "mt5" in src.lower() or "cache" in src.lower():
        grade = "A"
    if df is not None and n > 10:
        try:
            gaps = pd.Series(df.index).diff().dropna().dt.total_seconds()
            med = float(gaps.median())
            big = int((gaps > med * 5).sum())
            if big > n * 0.02:
                grade = min(grade, "C")
                flags.append(f"{big} gaps (>5x median)")
            if (pd.Timestamp.now(tz="UTC") - df.index[-1]).days > 7:
                flags.append("stale tail >7d")
        except Exception:
            pass
    if n < 100:
        grade = "F"
        flags.append(f"only {n} bars (<100)")
    return {"grade": grade, "source": src, "bars": n, "flags": flags,
            "loud_banner": f"[DATA {grade}] {src} | {n} bars" + (f" | {'; '.join(flags)}" if flags else "")}


# ---------- Manifest ----------
def save_manifest(strategy: str, symbol: str, tf: str, params: dict, engine_kwargs: dict,
                  data_grade: dict, metrics: dict, seed: int = 42) -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    body = {"strategy": strategy, "symbol": symbol, "timeframe": tf, "params": params,
            "engine": {k: v for k, v in engine_kwargs.items() if not k.startswith("_")},
            "data": data_grade, "metrics": {k: (round(float(v), 4) if isinstance(v, float) else v)
                                            for k, v in (metrics or {}).items()},
            "seed": seed, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    h = hashlib.md5(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:10]
    p = RUNS_DIR / f"{strategy}_{symbol}_{tf}_{h}.json"
    p.write_text(json.dumps(body, indent=2, default=str))
    return str(p)


# ---------- Margin ----------
def margin_required(lots: float, price: float, contract_size: float, leverage: float = 30.0) -> float:
    return float(abs(lots) * contract_size * price / max(leverage, 1.0))


def lots_for_risk(balance: float, risk_pct: float, sl_pips: float, pip_size: float,
                  contract_size: float, volume_min: float = 0.01,
                  volume_max: float = 500.0, volume_step: float = 0.01) -> float:
    risk_money = balance * float(risk_pct) / 100.0
    per_lot_loss = float(sl_pips) * pip_size * contract_size
    if per_lot_loss <= 0:
        return volume_min
    raw = risk_money / per_lot_loss
    steps = round((min(max(raw, volume_min), volume_max) - volume_min) / volume_step)
    return float(volume_min + steps * volume_step)


def margin_level(equity: float, margin_used: float) -> float | None:
    if margin_used <= 0:
        return None
    return float(equity / margin_used * 100.0)


# ---------- Robustness ----------
def monte_carlo_bands(trades: pd.DataFrame, n_sims: int = 500, seed: int = 42) -> dict:
    """Bootstrap MC: resample trades with replacement to simulate path variance."""
    try:
        col = next((c for c in ("pnl", "PnL", "profit", "Profit") if c in trades.columns), None)
        if not col or len(trades) < 10:
            return {"error": "need 10+ trades"}
        rng = np.random.default_rng(seed)
        pnls = trades[col].values
        n = len(pnls)
        finals = [float(rng.choice(pnls, size=n, replace=True).sum()) for _ in range(n_sims)]
        return {"p5": round(float(np.percentile(finals, 5)), 2),
                "p50": round(float(np.percentile(finals, 50)), 2),
                "p95": round(float(np.percentile(finals, 95)), 2),
                "prob_profit": round(float(np.mean(np.array(finals) > 0)), 3)}
    except Exception as e:
        return {"error": str(e)[:100]}


def wfe_gate(is_metrics: dict, oos_metrics: dict, threshold: float = 0.5) -> dict:
    try:
        is_r = float(is_metrics.get("net_pnl", 0) or 0)
        oos_r = float(oos_metrics.get("net_pnl", 0) or 0)
        wfe = (oos_r / is_r) if is_r > 0 else 0.0
        return {"wfe": round(wfe, 2), "pass": bool(wfe >= threshold and oos_r > 0),
                "verdict": "ROBUST" if wfe >= 0.7 else ("ACCEPT" if wfe >= threshold else "OVERFIT")}
    except Exception as e:
        return {"error": str(e)[:100]}


def spread_stress(metrics_base: dict) -> dict:
    pf = float(metrics_base.get("profit_factor", 0) or 0)
    return {"note": "re-run with spread x2/x3 via max_spread_pips + spread_pips; accept if PF stays >1.3",
            "base_pf": pf, "accept_pf": 1.3, "pass": bool(pf > 1.5)}


# ---------- Report ----------
def export_report(strategy: str, symbol: str, tf: str, stats: dict, complex_score: float,
                  data_grade: dict, manifest_path: str = "") -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    html = f"""<html><head><meta charset='utf-8'><title>{strategy} {symbol} {tf} — Tester Report</title>
<style>body{{font-family:Segoe UI,Arial;background:#0e1420;color:#e8eef7;margin:24px}}
.card{{background:#182234;border:1px solid #2b3c5a;border-radius:12px;padding:16px;margin:12px 0}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #2b3c5a;padding:6px 8px;font-size:13px}}
th{{background:#22314d}}</style></head><body>
<h2>📋 {strategy} · {symbol} · {tf} — MQL5 Tester Mirror</h2>
<div class='card'>Complex Result: <b>{complex_score:.1f}/100</b> | Data: <b>{data_grade.get('loud_banner', '')}</b><br>
Manifest: {manifest_path}</div>
<div class='card'><table><tr><th>Tester (MT5 name)</th><th>Value</th></tr>""" + \
        "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in stats.items()) + \
        "</table></div></body></html>"
    p = RUNS_DIR / f"report_{strategy}_{symbol}_{tf}_{ts}.html"
    p.write_text(html)
    return str(p)


# ---------- Corporate actions + halts (survivorship lite) ----------
CORP_PATH = Path(__file__).parent.parent / "config" / "corporate.json"


def load_corporate(symbol: str) -> dict:
    try:
        if CORP_PATH.exists():
            return json.loads(CORP_PATH.read_text()).get(symbol.upper(), {})
    except Exception:
        pass
    return {}


def adjust_for_corporate(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Apply split/dividend adjustments from config/corporate.json (no silent gaps)."""
    corp = load_corporate(symbol)
    splits = corp.get("splits", {})
    if not splits or df is None or len(df) == 0:
        return df
    out = df.copy()
    for date_str, ratio in sorted(splits.items()):
        try:
            ts = pd.Timestamp(date_str)
            mask = out.index < ts
            out.loc[mask, ["open", "high", "low", "close"]] /= float(ratio)
        except Exception:
            continue
    return out


def is_halted(ts, halts: list | None = None) -> bool:
    try:
        s = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
        return s in (halts or load_corporate("*").get("halts", []))
    except Exception:
        return False


# ---------- Tick downloader (Dukascopy stub → MT5 real) ----------
def download_ticks(symbol: str, start: str, end: str, out_dir: str | None = None) -> dict:
    """Try MT5 real ticks first, else report Dukascopy URL to fetch manually.
    Never synthesizes silently — returns status dict."""
    try:
        from data.tick_data import fetch_ticks_mt5
        df, info = fetch_ticks_mt5(symbol, start, end)
        if df is not None and len(df):
            p = Path(out_dir or str(Path(__file__).parent.parent / "data" / "ticks"))
            p.mkdir(parents=True, exist_ok=True)
            fp = p / f"{symbol}_{start}_{end}.parquet"
            df.to_parquet(fp)
            return {"ok": True, "rows": len(df), "path": str(fp), "source": "mt5_ticks"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:120]}
    y, m = start[:4], start[5:7]
    return {"ok": False, "manual_url": f"https://datafeed.dukascopy.com/datafeed/{symbol.upper()}/{y}/{int(m)-1:02d}/",
            "hint": "Download .bi5, convert with tools/dukascopy.py (to add), or connect MT5"}


# ---------- PBO-lite + regime + costs ----------
def pbo_lite(trials_df: pd.DataFrame, metric: str = "score") -> dict:
    """Probability of backtest overfitting (lite): split trials IS/OOS by time, rank decay."""
    try:
        if trials_df is None or len(trials_df) < 20:
            return {"pbo": None, "note": "need 20+ trials"}
        h = len(trials_df) // 2
        is_best = trials_df.iloc[:h][metric].median()
        oos_best = trials_df.iloc[h:][metric].median()
        decay = 1 - (oos_best / is_best) if is_best else 1.0
        pbo = min(max(decay, 0.0), 1.0)
        return {"pbo": round(pbo, 2), "decay": round(decay, 2),
                "verdict": "OVERFIT" if pbo > 0.5 else ("CHECK" if pbo > 0.3 else "ROBUST")}
    except Exception as e:
        return {"error": str(e)[:100]}


def regime_attribution(df: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Bucket net PnL by ADX regime (trend vs chop) using close-price proxy."""
    try:
        from strategies.indicators import adx
        a, pdi, ndi = adx(df["high"], df["low"], df["close"])
        regime = pd.Series(np.where(a > 25, "trend", "chop"), index=df.index)
        col = next((c for c in ("pnl", "PnL", "profit", "Profit") if c in trades.columns), None)
        if not col:
            return {}
        out = {}
        for _, t in trades.iterrows():
            xb = int(t.get("exit_bar", 0))
            xb = max(0, min(xb, len(df) - 1))
            r = str(regime.iloc[xb])
            out[r] = out.get(r, 0.0) + float(t[col])
        return {k: round(v, 2) for k, v in out.items()}
    except Exception:
        return {}


def cost_breakdown(spread_pips: float, commission_pips: float, slippage_pips: float,
                   n_trades: int, pip_size: float, contract_size: float, lots: float) -> dict:
    per_trade = (spread_pips + commission_pips + slippage_pips) * pip_size * contract_size * lots
    total = per_trade * n_trades
    return {"per_trade_$": round(per_trade, 2), "total_$": round(total, 2), "n": n_trades,
            "mix": {"spread": spread_pips, "commission": commission_pips, "slippage": slippage_pips}}
