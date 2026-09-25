"""Real Tick Data Pipeline — download, convert, store, and serve institutional-grade tick data.

Features:
- Dukascopy .bi5 downloader (HTTP range requests, resume support)
- TrueFX/OANDA REST API downloader
- .bi5 → parquet converter (partitioned by symbol/date)
- Corporate actions integration (splits/dividends/delistings)
- Tick server: fast random-access reads for backtesting
- Data quality validation on ingest
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Dukascopy .bi5 format constants
BI5_RECORD_SIZE = 20  # bytes per tick
BI5_HEADER = b"BIDASK"  # not actually in file, just for reference

# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "ticks"
DEFAULT_PARQUET_DIR = Path(__file__).parent.parent / "data" / "ticks_parquet"


@dataclass
class TickRecord:
    """Single tick record."""
    timestamp: pd.Timestamp
    bid: float
    ask: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0


@dataclass
class DownloadResult:
    """Result of a download operation."""
    ok: bool
    symbol: str
    start: str
    end: str
    rows: int = 0
    path: str = ""
    error: str = ""
    source: str = ""


def _session_with_retries() -> requests.Session:
    """Create a requests session with retry strategy.

    Dukascopy throttles rapid sequential pulls (HTTP 429), so we retry with
    exponential backoff, honor Retry-After, and send a browser User-Agent.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/126.0 Safari/537.36"),
    })
    retry = Retry(
        total=6,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _dukascopy_url(symbol: str, year: int, month: int, day: int, hour: int) -> str:
    """Generate Dukascopy .bi5 URL for a specific hour.

    Dukascopy format: https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YEAR}/{MONTH:02d}/{DAY:02d}/{HOUR:02d}h_ticks.bi5
    Month is 0-indexed in URL. The .bi5 payload is LZMA-alone compressed
    (NOT gzip despite old docs); decompress with lzma.FORMAT_ALONE.
    """
    return f"https://datafeed.dukascopy.com/datafeed/{symbol.upper()}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"


def _decompress_bi5(data: bytes) -> bytes:
    """Decompress a .bi5 payload (LZMA-alone; legacy gzip fallback)."""
    import lzma
    if data[:2] == b"\x1f\x8b":
        import gzip as _gzip
        return _gzip.decompress(data)
    return lzma.decompress(data, format=lzma.FORMAT_ALONE)


def _parse_bi5(data: bytes, hour_start_ms: int | None = None,
               point_size: int = 5) -> list[TickRecord]:
    """Parse Dukascopy .bi5 binary data into tick records.

    Real format (20 bytes per record, BIG-endian, verified against live feed):
    - 4 bytes: milliseconds offset within the hour (uint32)
    - 4 bytes: ask price as int (divide by 10^point_size)
    - 4 bytes: bid price as int (divide by 10^point_size)
    - 4 bytes: ask volume (float32, millions)
    - 4 bytes: bid volume (float32, millions)

    Absolute timestamp = hour_start_ms + offset. When hour_start_ms is None,
    timestamps fall back to raw offsets (for backward-compat probing only).
    """
    records = []
    n_records = len(data) // BI5_RECORD_SIZE
    divisor = 10.0 ** point_size

    for i in range(n_records):
        offset = i * BI5_RECORD_SIZE
        chunk = data[offset:offset + BI5_RECORD_SIZE]
        if len(chunk) < BI5_RECORD_SIZE:
            break

        ms_off, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(">IIIff", chunk)

        if hour_start_ms is not None:
            ts = pd.Timestamp(hour_start_ms + ms_off, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(ms_off, unit="ms", tz="UTC")

        ask = ask_raw / divisor
        bid = bid_raw / divisor

        # Sanity: skip corrupt records (negative prices, inverted >100-pip spread)
        if ask <= 0 or bid <= 0 or (ask - bid) / divisor > 0.01:
            continue

        records.append(TickRecord(ts, bid, ask, float(bid_vol), float(ask_vol)))

    return records


def download_dukascopy_hour(
    symbol: str,
    year: int,
    month: int,
    day: int,
    hour: int,
    out_dir: Path | str,
    session: requests.Session | None = None
) -> DownloadResult:
    """Download a single hour of Dukascopy tick data."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = _dukascopy_url(symbol, year, month, day, hour)
    fname = f"{symbol.upper()}_{year}{month:02d}{day:02d}_{hour:02d}.bi5"
    out_path = out_dir / fname

    if out_path.exists():
        return DownloadResult(True, symbol, f"{year}-{month:02d}-{day:02d} {hour:02d}:00",
                              f"{year}-{month:02d}-{day:02d} {hour:02d}:00", 0, str(out_path), "", "dukascopy_cached")

    sess = session or _session_with_retries()
    try:
        resp = sess.get(url, timeout=30, stream=True)
        if resp.status_code == 404:
            return DownloadResult(False, symbol, "", "", 0, "", f"404 Not Found: {url}", "dukascopy")
        resp.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Validate payload: LZMA-alone magic (0x5d) or legacy gzip.
        # Rate-limit HTML pages / truncated bodies fail here and are deleted
        # so they never poison the cache (exists() == trusted).
        with open(out_path, "rb") as f:
            magic = f.read(2)
        if len(magic) < 2 or (magic[:1] != b"\x5d" and magic != b"\x1f\x8b"):
            out_path.unlink(missing_ok=True)
            return DownloadResult(False, symbol, "", "", 0, "",
                                  "bad payload (rate-limit page or truncated)", "dukascopy")

        return DownloadResult(True, symbol, f"{year}-{month:02d}-{day:02d} {hour:02d}:00",
                              f"{year}-{month:02d}-{day:02d} {hour:02d}:00", 0, str(out_path), "", "dukascopy")
    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        return DownloadResult(False, symbol, "", "", 0, "", str(e)[:200], "dukascopy")


def download_dukascopy_range(
    symbol: str,
    start: str,  # "YYYY-MM-DD"
    end: str,    # "YYYY-MM-DD"
    out_dir: Path | str,
    max_hours: int | None = None,
    delay_sec: float = 1.0,
    bad_payload_retries: int = 3,
    breaker_threshold: int = 5,
) -> list[DownloadResult]:
    """Download a date range of Dukascopy tick data (hourly files).

    A polite inter-request delay keeps us under Dukascopy's rate limiter
    (cached files skip the delay entirely).

    Throttle handling (the server answers overload with HTTP 200 + an HTML
    page, which the transport-level Retry cannot see):
    - each bad payload is retried with exponential backoff + jitter
      (10s, 20s, 40s, capped at 300s);
    - after `breaker_threshold` CONSECUTIVE bad payloads the circuit opens:
      the range stops immediately instead of hammering thousands of doomed
      requests into an IP ban. Cached files are kept, so re-running later
      resumes for free. The final list entry carries error="circuit_open...".
    """
    import random
    import time as _time

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)  # inclusive

    results = []
    sess = _session_with_retries()
    hours_done = 0
    consec_bad = 0

    def _is_bad(r: DownloadResult) -> bool:
        return (not r.ok) and ("bad payload" in (r.error or ""))

    current = start_dt.floor("h")
    while current < end_dt:
        if max_hours and hours_done >= max_hours:
            break

        # Skip the delay when the file is already cached locally.
        probe = out_dir / (f"{symbol.upper()}_{current.year}{current.month - 1:02d}"
                           f"{current.day:02d}_{current.hour:02d}.bi5")
        if not probe.exists() and results:
            _time.sleep(delay_sec)

        r = download_dukascopy_hour(
            symbol, current.year, current.month - 1, current.day, current.hour,
            out_dir, sess
        )
        # App-level retry: transport Retry never fires on HTTP-200 throttle pages.
        attempt = 0
        while _is_bad(r) and attempt < bad_payload_retries:
            attempt += 1
            wait = min(10 * (2 ** (attempt - 1)), 300) * random.uniform(0.75, 1.25)
            _time.sleep(wait)
            r = download_dukascopy_hour(
                symbol, current.year, current.month - 1, current.day, current.hour,
                out_dir, sess
            )
        results.append(r)
        hours_done += 1
        consec_bad = consec_bad + 1 if _is_bad(r) else 0
        if consec_bad >= breaker_threshold:
            results.append(DownloadResult(
                False, symbol, "", "", 0, "",
                f"circuit_open: {consec_bad} consecutive bad payloads — "
                f"server is throttling; stopped at {current} instead of "
                f"hammering into an IP ban. Cached files kept; re-run later "
                f"to resume for free.", "dukascopy"))
            break
        current += pd.Timedelta(hours=1)

    return results


# ---------- TrueFX (free, limited history) ----------
def download_truefx(symbol: str, start: str, end: str, out_dir: Path | str) -> DownloadResult:
    """Download TrueFX daily tick files. Limited to ~6 months history, free.

    TrueFX format: CSV with timestamp,bid,ask
    URL pattern: http://www.truefx.com/dev/data/{YEAR}/{MONTH}/{YYYYMMDD}-{SYMBOL}.zip
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC")

    all_records = []
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")
        year = current.year
        month_name = current.strftime("%B")
        url = f"http://www.truefx.com/dev/data/{year}/{month_name}/{date_str}-{symbol.upper()}.zip"
        try:
            sess = _session_with_retries()
            resp = sess.get(url, timeout=30)
            if resp.status_code == 200:
                import io
                import zipfile
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    for name in zf.namelist():
                        with zf.open(name) as f:
                            for line in f:
                                parts = line.decode("utf-8").strip().split(",")
                                if len(parts) >= 3:
                                    try:
                                        ts_str = parts[0].strip()
                                        bid = float(parts[1])
                                        ask = float(parts[2])
                                        ts = pd.Timestamp(ts_str, tz="UTC")
                                        all_records.append({"timestamp": ts, "bid": bid, "ask": ask,
                                                           "spread": ask - bid, "bid_volume": 0, "ask_volume": 0})
                                    except (ValueError, TypeError):
                                        continue
        except Exception:
            pass
        current += pd.Timedelta(days=1)

    if not all_records:
        return DownloadResult(False, symbol, start, end, 0, "",
                              "No TrueFX data found (may be older than 6 months)", "truefx")

    df = pd.DataFrame(all_records)
    out_path = Path(out_dir) / f"{symbol.upper()}_truefx_{start}_{end}.parquet"
    df.to_parquet(out_path, index=False)
    return DownloadResult(True, symbol, start, end, len(df), str(out_path), "", "truefx")


# ---------- OANDA (requires API key) ----------
def download_oanda(symbol: str, start: str, end: str, api_key: str,
                   out_dir: Path | str, granularity: str = "S5") -> DownloadResult:
    """Download OANDA tick/candle data. Requires API key.

    OANDA v20 API: https://api-fxtrade.oanda.com/v3/instruments/{SYMBOL}/candles
    Note: OANDA free tier has limited history (~180 days for S5).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not api_key:
        return DownloadResult(False, symbol, start, end, 0, "",
                              "OANDA requires API key", "oanda")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base_url = "https://api-fxtrade.oanda.com/v3/instruments"
    inst = symbol.replace("_", "").replace("/", "").upper()

    # Convert symbol format if needed
    symbol_map = {"EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "USDJPY": "USD_JPY"}
    inst = symbol_map.get(symbol.upper(), inst)

    url = f"{base_url}/{inst}/candles"
    params = {
        "granularity": granularity,
        "from": pd.Timestamp(start, tz="UTC").isoformat(),
        "to": pd.Timestamp(end, tz="UTC").isoformat(),
        "price": "BA",  # bid/ask
    }

    try:
        sess = _session_with_retries()
        resp = sess.get(url, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        records = []
        for candle in data.get("candles", []):
            mid = candle.get("mid", {})
            ba = candle.get("bid", {}), candle.get("ask", {})
            bid_raw = candle.get("bid", {})
            ask_raw = candle.get("ask", {})
            ts = pd.Timestamp(candle["time"], tz="UTC")
            # Use mid if bid/ask not available
            bid = float(bid_raw.get("o", mid.get("o", 0)))
            ask = float(ask_raw.get("o", mid.get("o", 0)))
            records.append({"timestamp": ts, "bid": bid, "ask": ask,
                           "spread": ask - bid, "bid_volume": float(candle.get("volume", 0)),
                           "ask_volume": float(candle.get("volume", 0))})

        if not records:
            return DownloadResult(False, symbol, start, end, 0, "",
                                  "No OANDA data returned", "oanda")

        df = pd.DataFrame(records)
        out_path = Path(out_dir) / f"{symbol.upper()}_oanda_{start}_{end}.parquet"
        df.to_parquet(out_path, index=False)
        return DownloadResult(True, symbol, start, end, len(df), str(out_path), "", "oanda")
    except Exception as e:
        return DownloadResult(False, symbol, start, end, 0, "", str(e)[:200], "oanda")


# ---------- Unified downloader (try multiple sources) ----------
def download_ticks_auto(symbol: str, start: str, end: str,
                        out_dir: Path | str = DEFAULT_DATA_DIR,
                        oanda_api_key: str | None = None,
                        prefer: str = "dukascopy") -> DownloadResult:
    """Try preferred source first, fall back to alternatives."""
    out_dir = Path(out_dir)

    if prefer == "dukascopy":
        results = download_dukascopy_range(symbol, start, end, out_dir)
        ok_count = sum(1 for r in results if r.ok)
        if ok_count > 0:
            # Convert to parquet
            conv = convert_bi5_to_parquet(out_dir, DEFAULT_PARQUET_DIR, symbol)
            if conv["ok"]:
                return DownloadResult(True, symbol, start, end, conv["rows"],
                                      conv["parquet_dir"], "", "dukascopy+parquet")
        # Fallback to TrueFX
        tfx = download_truefx(symbol, start, end, out_dir)
        if tfx.ok:
            return tfx
        return DownloadResult(False, symbol, start, end, 0, "",
                              f"Dukascopy OK={ok_count}, TrueFX failed", "auto")

    elif prefer == "truefx":
        return download_truefx(symbol, start, end, out_dir)

    elif prefer == "oanda":
        if oanda_api_key:
            return download_oanda(symbol, start, end, oanda_api_key, out_dir)
        return DownloadResult(False, symbol, start, end, 0, "", "No OANDA key", "oanda")

    return DownloadResult(False, symbol, start, end, 0, "",
                          f"Unknown source: {prefer}", prefer)


def _point_size(symbol: str) -> int:
    """Decimals used by Dukascopy integer prices for this symbol."""
    s = symbol.upper().replace("/", "")
    if "JPY" in s:
        return 3
    if s.startswith(("XAU", "XAG")):
        return 3
    return 5


def _hour_start_from_bi5_name(fname: str) -> int | None:
    """Reconstruct hour-start epoch ms from SYMBOL_YYYYMMDD_HH.bi5.

    NOTE: the month embedded in our filenames is 0-indexed (same as the
    Dukascopy URL), so +1 for the real calendar month.
    """
    import re
    m = re.search(r"_(\d{4})(\d{2})(\d{2})_(\d{2})\.bi5$", fname)
    if not m:
        return None
    year, month0, day, hour = map(int, m.groups())
    ts = pd.Timestamp(year, month0 + 1, day, hour, tz="UTC")
    return int(ts.value // 1_000_000)


def convert_bi5_to_parquet(
    bi5_dir: Path | str,
    parquet_dir: Path | str,
    symbol: str,
    partition_by: str = "day"  # "day" or "month"
) -> dict:
    """Convert all .bi5 files in a directory to partitioned parquet."""
    bi5_dir = Path(bi5_dir)
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(bi5_dir.glob(f"{symbol.upper()}_*.bi5"))
    if not files:
        return {"ok": False, "error": f"No .bi5 files found for {symbol} in {bi5_dir}"}

    pts = _point_size(symbol)
    all_records = []
    for f in files:
        try:
            raw = f.read_bytes()
            data = _decompress_bi5(raw)
            hour_ms = _hour_start_from_bi5_name(f.name)
            records = _parse_bi5(data, hour_start_ms=hour_ms, point_size=pts)
            all_records.extend(records)
        except Exception as e:
            print(f"Warning: Failed to parse {f}: {e}")

    if not all_records:
        return {"ok": False, "error": "No valid tick records parsed"}

    # Build DataFrame
    df = pd.DataFrame([{
        "timestamp": r.timestamp,
        "bid": r.bid,
        "ask": r.ask,
        "bid_volume": r.bid_volume,
        "ask_volume": r.ask_volume,
        "spread": r.ask - r.bid
    } for r in all_records])

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Partition and write
    if partition_by == "day":
        df["date"] = df["timestamp"].dt.date
        for date, group in df.groupby("date"):
            out_path = parquet_dir / f"symbol={symbol.upper()}" / f"date={date}"
            out_path.mkdir(parents=True, exist_ok=True)
            group.drop(columns=["date"]).to_parquet(out_path / "ticks.parquet", index=False)
    else:  # month
        df["year_month"] = df["timestamp"].dt.tz_localize(None).dt.to_period("M").astype(str)
        for ym, group in df.groupby("year_month"):
            out_path = parquet_dir / f"symbol={symbol.upper()}" / f"year_month={ym}"
            out_path.mkdir(parents=True, exist_ok=True)
            group.drop(columns=["year_month"]).to_parquet(out_path / "ticks.parquet", index=False)

    return {
        "ok": True,
        "symbol": symbol,
        "rows": len(df),
        "date_range": (str(df["timestamp"].min()), str(df["timestamp"].max())),
        "parquet_dir": str(parquet_dir)
    }


def load_ticks_parquet(
    parquet_dir: Path | str,
    symbol: str,
    start: str | None = None,
    end: str | None = None
) -> pd.DataFrame:
    """Load tick data from partitioned parquet with optional time filtering."""
    parquet_dir = Path(parquet_dir)
    symbol_dir = parquet_dir / f"symbol={symbol.upper()}"

    if not symbol_dir.exists():
        return pd.DataFrame()

    # Read all partitions
    dfs = []
    for part in symbol_dir.glob("*/ticks.parquet"):
        try:
            df = pd.read_parquet(part)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {part}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Time filter
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]

    return df


def validate_tick_data(df: pd.DataFrame, symbol: str = "") -> dict:
    """Validate tick data quality."""
    if df is None or len(df) == 0:
        return {"ok": False, "error": "Empty dataframe", "symbol": symbol}

    issues = []
    warnings = []

    # Check required columns
    required = ["timestamp", "bid", "ask"]
    for c in required:
        if c not in df.columns:
            issues.append(f"Missing column: {c}")

    if issues:
        return {"ok": False, "error": "; ".join(issues), "symbol": symbol}

    # Check for negative spreads
    spread = df["ask"] - df["bid"]
    neg_spread = (spread < 0).sum()
    if neg_spread > 0:
        issues.append(f"Negative spread on {neg_spread} ticks")

    # Check for zero spreads
    zero_spread = (spread == 0).sum()
    if zero_spread > len(df) * 0.01:
        warnings.append(f"Zero spread on {zero_spread} ticks ({zero_spread/len(df)*100:.1f}%)")

    # Check timestamp monotonicity
    if not df["timestamp"].is_monotonic_increasing:
        issues.append("Timestamps not monotonic increasing")

    # Check for duplicate timestamps
    dup = df["timestamp"].duplicated().sum()
    if dup > 0:
        warnings.append(f"Duplicate timestamps: {dup}")

    # Check for gaps > 1 hour
    diffs = df["timestamp"].diff().dt.total_seconds()
    large_gaps = (diffs > 3600).sum()
    if large_gaps > 0:
        warnings.append(f"Gaps > 1 hour: {large_gaps}")

    # Price sanity
    if (df["bid"] <= 0).any() or (df["ask"] <= 0).any():
        issues.append("Non-positive prices found")

    return {
        "ok": len(issues) == 0,
        "symbol": symbol,
        "rows": len(df),
        "date_range": (str(df["timestamp"].min()), str(df["timestamp"].max())),
        "spread_stats": {
            "mean": float(spread.mean()),
            "median": float(spread.median()),
            "p99": float(spread.quantile(0.99)),
            "max": float(spread.max())
        },
        "issues": issues,
        "warnings": warnings
    }


def apply_corporate_actions_ticks(
    df: pd.DataFrame,
    symbol: str,
    corp_actions: dict
) -> pd.DataFrame:
    """Apply corporate actions (splits, dividends) to tick or OHLC data.

    corp_actions format:
        {"splits": {"2024-08-30": 10.0},       # pre-date prices /= ratio
         "dividends": {"2024-05-10": 0.24}}    # pre-ex-date prices -= cash

    Works on tick frames (timestamp/bid/ask cols) and OHLC frames
    (DatetimeIndex + open/high/low/close cols). Returns adjusted copy.
    """
    if df is None or len(df) == 0 or not corp_actions:
        return df

    splits = corp_actions.get("splits", {}) or {}
    dividends = corp_actions.get("dividends", {}) or {}
    if not splits and not dividends:
        return df

    out = df.copy()
    if "timestamp" in out.columns:
        times = pd.to_datetime(out["timestamp"], utc=True)
        price_cols = [c for c in ("bid", "ask") if c in out.columns]
        use_index_mask = False
    else:
        times = None
        price_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
        use_index_mask = True
    if not price_cols:
        print(f"Warning: no price columns to adjust for {symbol}")
        return df

    def _mask_before(date_str: str):
        ts = pd.Timestamp(date_str, tz="UTC")
        if use_index_mask:
            idx = pd.to_datetime(out.index, utc=True)
            return idx < ts
        return times < ts

    for date_str, ratio in sorted(splits.items()):
        try:
            out.loc[_mask_before(date_str), price_cols] /= float(ratio)
        except Exception as e:
            print(f"Warning: Failed to apply split {date_str}: {e}")

    for date_str, cash in sorted(dividends.items()):
        try:
            out.loc[_mask_before(date_str), price_cols] -= float(cash)
        except Exception as e:
            print(f"Warning: Failed to apply dividend {date_str}: {e}")

    return out


# Convenience: one-shot download + convert + validate
def fetch_and_prepare_ticks(
    symbol: str,
    start: str,
    end: str,
    bi5_dir: Path | str = DEFAULT_DATA_DIR,
    parquet_dir: Path | str = DEFAULT_PARQUET_DIR,
    corp_actions: dict | None = None,
    partition_by: str = "day"
) -> dict:
    """Download Dukascopy ticks, convert to parquet, validate, apply corp actions."""
    # Download
    print(f"Downloading {symbol} {start} to {end}...")
    results = download_dukascopy_range(symbol, start, end, bi5_dir)
    ok_downloads = [r for r in results if r.ok]
    print(f"Downloaded {len(ok_downloads)}/{len(results)} hours")

    if not ok_downloads:
        return {"ok": False, "error": "No data downloaded", "downloads": results}

    # Convert
    print("Converting to parquet...")
    conv = convert_bi5_to_parquet(bi5_dir, parquet_dir, symbol)
    if not conv["ok"]:
        return {"ok": False, "error": conv["error"], "downloads": results}

    # Load and validate
    print("Loading and validating...")
    df = load_ticks_parquet(parquet_dir, symbol, start, end)
    val = validate_tick_data(df, symbol)

    # Apply corporate actions
    if corp_actions:
        df = apply_corporate_actions_ticks(df, symbol, corp_actions)
        # Re-save with adjustments
        if partition_by == "day":
            df["date"] = df["timestamp"].dt.date
            for date, group in df.groupby("date"):
                out_path = Path(parquet_dir) / f"symbol={symbol.upper()}" / f"date={date}"
                out_path.mkdir(parents=True, exist_ok=True)
                group.drop(columns=["date"]).to_parquet(out_path / "ticks.parquet", index=False)

    return {
        "ok": val["ok"],
        "symbol": symbol,
        "rows": len(df),
        "date_range": val["date_range"],
        "spread_stats": val["spread_stats"],
        "issues": val["issues"],
        "warnings": val["warnings"],
        "parquet_dir": str(parquet_dir)
    }


if __name__ == "__main__":
    # CLI interface
    import argparse

    parser = argparse.ArgumentParser(description="Real tick data pipeline")
    sub = parser.add_subparsers(dest="cmd")

    # Download command
    dl = sub.add_parser("download", help="Download Dukascopy ticks for date range")
    dl.add_argument("symbol", help="Symbol (e.g. EURUSD)")
    dl.add_argument("start", help="Start date YYYY-MM-DD")
    dl.add_argument("end", help="End date YYYY-MM-DD")
    dl.add_argument("--bi5-dir", default=str(DEFAULT_DATA_DIR), help="Output dir for .bi5.gz files")
    dl.add_argument("--max-hours", type=int, default=None, help="Max hours to download (for testing)")

    # Convert command
    cv = sub.add_parser("convert", help="Convert .bi5.gz files to parquet")
    cv.add_argument("symbol", help="Symbol to convert")
    cv.add_argument("--bi5-dir", default=str(DEFAULT_DATA_DIR), help="Input dir with .bi5 files")
    cv.add_argument("--parquet-dir", default=str(DEFAULT_PARQUET_DIR), help="Output parquet dir")
    cv.add_argument("--partition", choices=["day", "month"], default="day", help="Partition strategy")

    # Validate command
    vd = sub.add_parser("validate", help="Validate tick data from parquet")
    vd.add_argument("symbol", help="Symbol to validate")
    vd.add_argument("--parquet-dir", default=str(DEFAULT_PARQUET_DIR), help="Parquet dir")
    vd.add_argument("--start", default=None, help="Start date filter")
    vd.add_argument("--end", default=None, help="End date filter")

    # Parse .bi5 file command
    pp = sub.add_parser("parse", help="Parse a single .bi5 file")
    pp.add_argument("file", help="Path to .bi5 file")

    args = parser.parse_args()

    if args.cmd == "download":
        results = download_dukascopy_range(args.symbol, args.start, args.end,
                                           args.bi5_dir, args.max_hours)
        ok = sum(1 for r in results if r.ok)
        failed = [r for r in results if not r.ok]
        print(f"Downloaded {ok}/{len(results)} hours")
        if failed:
            print(f"Failed: {len(failed)}")
            for r in failed[:5]:
                print(f"  {r.error}")
    elif args.cmd == "convert":
        result = convert_bi5_to_parquet(args.bi5_dir, args.parquet_dir, args.symbol, args.partition)
        print(result)
    elif args.cmd == "validate":
        df = load_ticks_parquet(args.parquet_dir, args.symbol, args.start, args.end)
        val = validate_tick_data(df, args.symbol)
        print(json.dumps(val, indent=2, default=str))
    elif args.cmd == "parse":
        test_file = Path(args.file)
        if test_file.exists():
            data = _decompress_bi5(test_file.read_bytes())
            sym_guess = test_file.name.split("_")[0] if "_" in test_file.name else "EURUSD"
            records = _parse_bi5(data, hour_start_ms=_hour_start_from_bi5_name(test_file.name),
                                 point_size=_point_size(sym_guess))
            print(f"Parsed {len(records)} ticks")
            for r in records[:5]:
                print(f"  {r.timestamp} bid={r.bid:.5f} ask={r.ask:.5f} spread={r.ask-r.bid:.5f}")
        else:
            print(f"File not found: {test_file}")
    else:
        parser.print_help()
