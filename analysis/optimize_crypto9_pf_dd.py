"""Optimize crypto_9 with a criterion focused on profit factor and max drawdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna

from backtester.engine import run_direction
from data.cache import load as load_cache
from strategies.crypto_9 import CryptoNineStrategy


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"


SPEC = {
    "length": (5, 30),
    "deviations": (1.2, 3.5),
    "breakout_lookback": (10, 80),
    "breakout_confirm_bars": (1, 4),
    "rsi_oversold": (20, 40),
    "rsi_overbought": (60, 80),
    "cooldown_bars": (0, 8),
    "adx_period": (10, 28),
    "trend_adx": (18, 32),
    "meanrev_adx": (14, 24),
    "volume_threshold": (1.0, 2.5),
    "enable_macd_filter": [True, False],
    "enable_volume_filter": [True, False],
    "enable_trend_filter": [True, False],
    "use_breakout": [True, False],
    "use_reversal": [True, False],
    "use_ma_cross": [True, False],
    "use_fake_breakout": [True, False],
    "use_bounce": [True, False],
    "use_breakout_upper": [True, False],
    "use_rejection": [True, False],
    "use_breakout_lower": [True, False],
    "use_midline_reversal": [True, False],
}


def sample(trial: optuna.Trial) -> dict:
    params = {}
    for key, value in SPEC.items():
        if isinstance(value, tuple):
            lo, hi = value
            if isinstance(lo, int) and isinstance(hi, int):
                params[key] = trial.suggest_int(key, lo, hi)
            else:
                params[key] = trial.suggest_float(key, lo, hi)
        else:
            params[key] = trial.suggest_categorical(key, value)
    return params


def profit_factor(pf) -> float:
    trades = pf.trades.records_readable
    if trades.empty or "PnL" not in trades:
        return 0.0
    closed = trades[trades.get("Status", "Closed") == "Closed"] if "Status" in trades else trades
    pnl = closed["PnL"].astype(float)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    if gross_loss <= 0:
        return 999.0 if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def score(metrics: dict, pf_value: float) -> float:
    dd = abs(float(metrics.get("max_drawdown", 0.0)))
    total_return = float(metrics.get("total_return", 0.0))
    trades = int(metrics.get("trades", 0))
    pf_n = min(max(pf_value / 2.5, 0.0), 1.5)
    dd_n = min(max(1.0 - dd / 0.15, 0.0), 1.0)
    ret_n = min(max(total_return / 0.30, -1.0), 1.0)
    trade_penalty = min(trades / 30.0, 1.0)
    return float((0.45 * pf_n + 0.45 * dd_n + 0.10 * ret_n) * trade_penalty * 100.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()

    df, meta = load_cache(args.symbol, args.timeframe)
    best_payload = {"score": -1e9}

    def objective(trial: optuna.Trial) -> float:
        params = sample(trial)
        if not any(params[k] for k in params if k.startswith("use_")):
            return -100.0
        try:
            sig = CryptoNineStrategy(params=params).generate(df)
            pf, metrics = run_direction(df, sig.entries, sig.direction)
            pf_value = profit_factor(pf)
            value = score(metrics, pf_value)
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("profit_factor", pf_value)
            trial.set_user_attr("entries", int(sig.entries.sum()))
            nonlocal best_payload
            if value > best_payload["score"]:
                best_payload = {
                    "score": value,
                    "params": params,
                    "metrics": metrics,
                    "profit_factor": pf_value,
                    "entries": int(sig.entries.sum()),
                    "symbol": args.symbol,
                    "timeframe": args.timeframe,
                    "bars": len(df),
                    "data_meta": meta,
                }
            return value
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            return -100.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"crypto_9_pf_dd_best_{args.symbol}_{args.timeframe}.json"
    out_path.write_text(json.dumps(best_payload, indent=2, default=str))
    print(json.dumps(best_payload, indent=2, default=str))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()

