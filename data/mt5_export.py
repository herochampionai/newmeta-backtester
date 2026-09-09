"""MT5 → Parquet export. Picks first available terminal, initializes it,
downloads bars (or ticks if requested), writes Parquet with SHA256 fingerprint.

Resolution order for terminal path:
  1. CLI flag --mt5-terminal
  2. Env var MT5_TERMINAL_PATH
  3. config/settings.yaml `mt5_terminal`
  4. TERMINAL_CANDIDATES list (D:\MT5_EuroPrinter first by user preference)
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5

CACHE_DIR = Path(__file__).parent / "cache"
ROOT = Path(__file__).parent.parent
SETTINGS_FILE = ROOT / "config" / "settings.yaml"

# User-preferred order: EuroPrinter first (most scripts are EuroPrinter MQL5),
# then Bybit, then MCP mirror.
TERMINAL_CANDIDATES = [
    r"D:\MT5_EuroPrinter\terminal64.exe",
    r"D:\MT5_Bybit\terminal64.exe",
    r"D:\work D\mt5-mcp\MT5_EuroPrinter\terminal64.exe",
]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def resolve_terminal(cli_value: str | None = None) -> str | None:
    """Resolution order: CLI > env > settings.yaml > candidates."""
    if cli_value:
        p = Path(cli_value)
        return str(p) if p.exists() else None
    env = os.environ.get("MT5_TERMINAL_PATH")
    if env and Path(env).exists():
        return env
    s = _load_yaml(SETTINGS_FILE).get("mt5_terminal")
    if s and Path(s).exists():
        return s
    for c in TERMINAL_CANDIDATES:
        if Path(c).exists():
            return c
    return None

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def find_terminal() -> str | None:
    return resolve_terminal()


def init_mt5(terminal: str | None = None) -> bool:
    init_kwargs = {}
    if terminal:
        init_kwargs["path"] = terminal
    ok = mt5.initialize(**init_kwargs)
    if not ok:
        err = mt5.last_error()
        print(f"[MT5] init failed: {err}", file=sys.stderr)
        return False
    info = mt5.terminal_info()
    print(f"[MT5] connected: {info.name} build={info.build}", file=sys.stderr)
    return True


def fetch_bars(symbol: str, timeframe: str, start: str, end: str | None = None) -> pd.DataFrame:
    tf = TF_MAP[timeframe.upper()]
    date_from = datetime.fromisoformat(start)
    date_to = datetime.fromisoformat(end) if end else datetime.now()
    rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"[MT5] no data for {symbol} {timeframe} {start}..{date_to}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def fingerprint(df: pd.DataFrame, meta: dict) -> str:
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes() + json.dumps(meta, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--timeframe", default="H1", choices=list(TF_MAP))
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mt5-terminal", default=None,
                    help="Override MT5 terminal path (e.g. D:\\MT5_EuroPrinter\\terminal64.exe)")
    args = ap.parse_args()

    terminal = resolve_terminal(args.mt5_terminal)
    if not terminal:
        print("[MT5] no terminal found. Set MT5_TERMINAL_PATH or pass --mt5-terminal",
              file=sys.stderr)
        sys.exit(2)
    print(f"[MT5] using terminal: {terminal}", file=sys.stderr)
    if not init_mt5(terminal):
        sys.exit(2)

    df = fetch_bars(args.symbol, args.timeframe, args.start, args.end)
    meta = {"symbol": args.symbol, "timeframe": args.timeframe,
            "start": args.start, "end": args.end or "now",
            "rows": len(df), "first": str(df.index[0]), "last": str(df.index[-1])}
    meta["sha"] = fingerprint(df, meta)

    out = Path(args.out) if args.out else CACHE_DIR / f"{args.symbol}_{args.timeframe}_{meta['sha']}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"[MT5] wrote {out}  rows={len(df)}  sha={meta['sha']}")
    mt5.shutdown()


if __name__ == "__main__":
    main()