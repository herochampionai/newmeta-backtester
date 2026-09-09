"""Newmeta Backtester HTTP API — bridges the Python backtester to the NewMeta
Terminal widget. Exposes REST endpoints that the React frontend can call.

Usage:
    pip install fastapi uvicorn
    python -m backtester.api

Endpoints:
    POST /backtest            Run a backtest with strategy params
    POST /optimize            Run Optuna optimization
    POST /monte_carlo         Run Monte Carlo bootstrap
    GET  /strategies           List available built-in strategies
    GET  /health               Health check
"""
from __future__ import annotations
import json
import sys
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.live_fetcher import fetch_with_priority
from data.mt5_export import resolve_terminal
from backtester.engine_full import run_full, GRID_NONE
from backtester.grid_recovery import GRID_LOSS_AND_PROFIT
from strategies import STRATEGY_REGISTRY
from core.regime import RegimeAwareStrategy
from core.loader import load_any_strategy

app = FastAPI(title="Newmeta Backtester API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to the NewMeta terminal origin
    allow_methods=["*"],
    allow_headers=["*"],
)


class BacktestRequest(BaseModel):
    strategy: Optional[str] = None  # built-in name (e.g. "fbb")
    strategy_path: Optional[str] = None  # path to .mq5/.py/.pine/.txt
    symbol: str = "EURUSD"
    timeframe: str = "H1"
    start: str = "2022-01-01"
    end: Optional[str] = None
    init_cash: float = 10000.0
    commission_pips: float = 0.7
    slippage_pips: float = 0.3
    grid_mode: int = 0  # 0=None, 1=Loss, 2=Profit, 3=Both
    base_lot: float = 0.1
    params_override: Optional[dict] = None
    # Regime-aware
    regime_aware: bool = False
    regime_map: Optional[dict] = None
    # MT5 terminal override
    mt5_terminal: Optional[str] = None


def _resolve_strategy(name: Optional[str], path: Optional[str], params_override: Optional[dict]):
    if path:
        cls, default_params, _ = load_any_strategy(path)
        if cls is None:
            raise HTTPException(400, f"Could not load strategy from {path}")
        params = {**(default_params or {}), **(params_override or {})}
    elif name and name in STRATEGY_REGISTRY:
        cls = STRATEGY_REGISTRY[name]
        params = params_override or {}
    else:
        raise HTTPException(400, "Provide strategy (name) or strategy_path")
    return cls, params


def _fetch_data(symbol, timeframe, start, end, mt5_terminal):
    settings_path = ROOT / "config" / "settings.yaml"
    terminal = mt5_terminal or (
        settings_path.read_text().split("mt5_terminal:")[-1].split("\n")[0].strip().strip('"')
        if settings_path.exists() else None)
    df, info = fetch_with_priority(symbol, timeframe, start, end,
                                    terminal, allow_synthetic=True)
    return df, info


@app.get("/health")
def health():
    return {"status": "ok", "strategies": list(STRATEGY_REGISTRY.keys())}


@app.get("/strategies")
def list_strategies():
    out = {}
    for name, cls in STRATEGY_REGISTRY.items():
        out[name] = {
            "name": cls.__name__,
            "doc": (cls.__doc__ or "").strip()[:200],
        }
    return out


@app.post("/backtest")
def run_backtest(req: BacktestRequest):
    try:
        df, info = _fetch_data(req.symbol, req.timeframe, req.start, req.end, req.mt5_terminal)
        cls, params = _resolve_strategy(req.strategy, req.strategy_path, req.params_override)
        if req.regime_aware:
            strat = RegimeAwareStrategy(df, strategy_map=req.regime_map or {
                "trending_up":   ["ms", "adx", "ac_ao"],
                "trending_down": ["ms", "adx", "ac_ao"],
                "ranging":       ["fbb", "dem"],
                "overextended":  ["mfi", "dem"],
                "volatile":      ["fbb"],
                "choppy":        ["fbb", "dem"],
            })
            sig = strat.generate()
            strat_name = "regime_aware"
        else:
            strat = cls(params=params)
            sig = strat.generate(df)
            strat_name = req.strategy or "custom"
        entries = sig.entries.fillna(False).astype(bool)
        direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
        result = run_full(df, {strat_name: (entries, direction)},
                           grid_mode=req.grid_mode, base_lot=req.base_lot,
                           init_cash=req.init_cash,
                           commission_pips=req.commission_pips,
                           slippage_pips=req.slippage_pips)
        # Convert equity + trades to JSON-safe form
        equity = result["equity"]
        equity_dict = {str(k): float(v) for k, v in equity.items()}
        trades_json = result["trades"].fillna("").to_dict(orient="records") if not result["trades"].empty else []
        return {
            "data_source": info.get("source"),
            "n_bars": len(df),
            "metrics": {k: float(v) if isinstance(v, (int, float)) else v
                          for k, v in result["metrics"].items()},
            "equity": equity_dict,
            "trades": trades_json[:500],  # cap
            "n_trades": len(trades_json),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/optimize")
def optimize(req: BacktestRequest, n_trials: int = 50):
    try:
        df, info = _fetch_data(req.symbol, req.timeframe, req.start, req.end, req.mt5_terminal)
        from analysis.optuna_optimizer import optimize_strategy, best_params
        spec_path = ROOT / "config" / "strategies.yaml"
        import yaml
        all_specs = yaml.safe_load(spec_path.read_text())
        spec = all_specs.get(req.strategy or "")
        if not spec:
            raise HTTPException(400, f"No spec in strategies.yaml for {req.strategy}")
        study = optimize_strategy(req.strategy, df, spec, n_trials=n_trials)
        best = best_params(study, metric="sharpe")
        return {"best_params": best, "best_value": float(study.best_value)}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/monte_carlo")
def monte_carlo(req: BacktestRequest, n_sims: int = 1000, block: int = 24):
    try:
        df, info = _fetch_data(req.symbol, req.timeframe, req.start, req.end, req.mt5_terminal)
        cls, params = _resolve_strategy(req.strategy, req.strategy_path, req.params_override)
        strat = cls(params=params)
        sig = strat.generate(df)
        entries = sig.entries.fillna(False).astype(bool)
        direction = pd.Series(sig.direction, index=df.index).fillna(0).astype(int)
        result = run_full(df, {req.strategy or "primary": (entries, direction)},
                           grid_mode=req.grid_mode, base_lot=req.base_lot)
        from analysis.montecarlo import ci_metrics
        ci = ci_metrics(result["equity"].pct_change().dropna(),
                          n_sims=n_sims, block=block)
        return {"confidence_intervals": ci.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    print("[Newmeta Backtester API] Starting on http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")