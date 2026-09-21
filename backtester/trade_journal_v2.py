"""R015: Trade journal integration — auto-log every backtest trade, reconcile PnL, tax reports.

Features:
- Auto-log: every backtest trade recorded to JSONL journal
- Reconciliation: compare backtest vs paper vs live PnL
- Tax reporting: per-jurisdiction summaries (US short/long-term, EU flat, etc.)
- Per-symbol/per-day aggregates
- Export: CSV for accounting software

Usage:
    journal = TradeJournal(run_id="backtest_2026_09_20", base_currency="USD")
    journal.log_trades(trades_df, run_meta={"strategy": "adx", "params": {...}})
    journal.reconcile_with_broker(broker_trades)
    report = journal.tax_report(jurisdiction="US")
    journal.export_csv("output/journal.csv")
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
import pandas as pd
import numpy as np


# Tax rules per jurisdiction (simplified — real tax is much more complex)
TAX_RULES = {
    "US": {
        "name": "United States",
        "short_term_days": 365,
        "short_term_rate": 0.37,  # ordinary income (max federal)
        "long_term_rate": 0.20,
        "currency": "USD",
    },
    "EU": {
        "name": "European Union",
        "flat_rate": 0.25,  # varies by country
        "currency": "EUR",
    },
    "UK": {
        "name": "United Kingdom",
        "flat_rate": 0.20,  # basic rate
        "currency": "GBP",
    },
    "AE": {
        "name": "UAE",
        "flat_rate": 0.0,  # no personal income tax
        "currency": "AED",
    },
}


@dataclass
class JournalEntry:
    """Single trade journal entry."""
    entry_id: str
    run_id: str
    timestamp: str
    symbol: str
    direction: int
    entry_price: float
    exit_price: float
    lots: float
    pnl: float
    pnl_pct: float
    commission: float
    swap: float
    hold_bars: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ReconciliationReport:
    """Backtest vs broker reconciliation."""
    n_backtest: int
    n_broker: int
    n_matched: int
    n_unmatched_bt: int
    n_unmatched_broker: int
    pnl_backtest: float
    pnl_broker: float
    pnl_diff: float
    pnl_diff_pct: float
    drift_pct: float


class TradeJournal:
    """Persistent trade journal with reconciliation + tax reporting."""

    def __init__(self, run_id: str | None = None, base_currency: str = "USD",
                 journal_dir: str | None = None):
        self.run_id = run_id or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.base_currency = base_currency
        default_dir = Path(__file__).parent.parent / "output" / "journals"
        self.journal_dir = Path(journal_dir) if journal_dir else default_dir
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.journal_dir / f"{self.run_id}.jsonl"
        self.entries: list[JournalEntry] = []
        self.run_meta: dict = {}

    def log_run_meta(self, **meta):
        """Record run-level metadata."""
        self.run_meta.update({
            "ts": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            **meta,
        })

    def log_trades(self, trades: pd.DataFrame, source: str = "backtest",
                   extra_meta: dict | None = None) -> list[JournalEntry]:
        """Auto-log all trades from a backtest result DataFrame.

        Args:
            trades: DataFrame with columns: symbol, direction, entry_price, exit_price,
                    pnl (or similar), and optional entry_bar/exit_bar
            source: "backtest" | "paper" | "live"
            extra_meta: additional metadata to attach
        """
        new_entries = []
        # Find PnL column
        pnl_col = next((c for c in ("pnl", "PnL", "profit", "Profit", "net_pnl") if c in trades.columns), None)
        if not pnl_col:
            raise ValueError(f"No PnL column found. Columns: {list(trades.columns)}")

        meta_base = {"source": source, **(extra_meta or {})}

        for _, row in trades.iterrows():
            symbol = str(row.get("symbol", "?"))
            direction = int(row.get("direction", row.get("side", 0)))
            entry_price = float(row.get("entry_price", row.get("entry", 0)))
            exit_price = float(row.get("exit_price", row.get("exit", 0)))
            lots = float(row.get("lots", row.get("size", 0)))
            pnl = float(row.get(pnl_col, 0))
            pnl_pct = float(row.get("pnl_pct", 0))
            commission = float(row.get("commission", 0))
            swap = float(row.get("swap", 0))
            entry_bar = int(row.get("entry_bar", 0))
            exit_bar = int(row.get("exit_bar", 0))

            # Generate entry ID from content hash
            entry_id = hashlib.md5(
                f"{self.run_id}_{symbol}_{entry_bar}_{exit_bar}".encode()
            ).hexdigest()[:12]

            entry = JournalEntry(
                entry_id=entry_id,
                run_id=self.run_id,
                timestamp=str(row.get("exit_time", row.get("timestamp", ""))),
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                lots=lots,
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=commission,
                swap=swap,
                hold_bars=exit_bar - entry_bar,
                metadata={**meta_base, "entry_bar": entry_bar, "exit_bar": exit_bar},
            )
            new_entries.append(entry)
            self.entries.append(entry)
            self._write_entry(entry)

        return new_entries

    def _write_entry(self, entry: JournalEntry):
        """Append entry to JSONL journal."""
        try:
            with open(self.journal_path, "a") as f:
                f.write(json.dumps(asdict(entry), default=str) + "\n")
        except Exception:
            pass

    def reconcile_with_broker(self, broker_trades: pd.DataFrame,
                              tolerance_pct: float = 0.05) -> ReconciliationReport:
        """Compare backtest entries vs broker-reported trades.

        Matching: by (symbol, entry_bar) tuple. Counts matched/unmatched
        and computes PnL drift.
        """
        if not self.entries:
            return ReconciliationReport(
                n_backtest=0, n_broker=len(broker_trades),
                n_matched=0, n_unmatched_bt=0,
                n_unmatched_broker=len(broker_trades),
                pnl_backtest=0.0, pnl_broker=float(broker_trades.get("pnl", pd.Series([0])).sum()),
                pnl_diff=0.0, pnl_diff_pct=0.0, drift_pct=0.0,
            )

        # Build backtest lookup
        bt_lookup = {}
        for e in self.entries:
            key = (e.symbol, e.metadata.get("entry_bar", 0))
            bt_lookup[key] = e

        # Match broker trades
        pnl_bt = sum(e.pnl for e in self.entries)
        pnl_br = 0.0
        n_matched = 0
        n_unmatched_bt = len(self.entries)
        matched_bt_ids = set()

        if "entry_bar" in broker_trades.columns and "symbol" in broker_trades.columns:
            for _, row in broker_trades.iterrows():
                sym = str(row["symbol"])
                bar = int(row["entry_bar"])
                key = (sym, bar)
                pnl_br += float(row.get("pnl", 0))
                if key in bt_lookup:
                    n_matched += 1
                    matched_bt_ids.add(bt_lookup[key].entry_id)
                    n_unmatched_bt -= 1
        else:
            # Fallback: just sum and report
            pnl_br = float(broker_trades.get("pnl", pd.Series([0])).sum())

        pnl_diff = pnl_bt - pnl_br
        pnl_diff_pct = (pnl_diff / abs(pnl_br) * 100) if pnl_br != 0 else 0
        n_broker = len(broker_trades)
        drift_pct = (n_unmatched_bt / max(n_broker, 1)) * 100

        return ReconciliationReport(
            n_backtest=len(self.entries),
            n_broker=n_broker,
            n_matched=n_matched,
            n_unmatched_bt=n_unmatched_bt,
            n_unmatched_broker=n_broker - n_matched,
            pnl_backtest=round(pnl_bt, 2),
            pnl_broker=round(pnl_br, 2),
            pnl_diff=round(pnl_diff, 2),
            pnl_diff_pct=round(pnl_diff_pct, 2),
            drift_pct=round(drift_pct, 2),
        )

    def tax_report(self, jurisdiction: str = "US",
                   trade_hold_days: list[int] | None = None) -> dict:
        """Generate tax report for jurisdiction.

        Args:
            jurisdiction: "US" | "EU" | "UK" | "AE"
            trade_hold_days: list of hold days per trade (for US short/long-term split)
        """
        if jurisdiction not in TAX_RULES:
            return {"error": f"Unknown jurisdiction: {jurisdiction}"}
        rules = TAX_RULES[jurisdiction]
        report = {
            "jurisdiction": jurisdiction,
            "name": rules["name"],
            "currency": rules["currency"],
            "rules": rules,
            "totals": {},
            "trades": [],
        }

        if not self.entries:
            return report

        if jurisdiction == "US":
            # Short-term (< 365 days) vs long-term (>= 365 days)
            short_pnl = 0.0
            long_pnl = 0.0
            for i, e in enumerate(self.entries):
                # Use provided hold_days or default to hold_bars (proxy)
                days = trade_hold_days[i] if trade_hold_days and i < len(trade_hold_days) else e.hold_bars
                is_short = days < rules["short_term_days"]
                if is_short:
                    short_pnl += e.pnl
                else:
                    long_pnl += e.pnl
                report["trades"].append({
                    "entry_id": e.entry_id,
                    "symbol": e.symbol,
                    "pnl": e.pnl,
                    "hold_days": days,
                    "term": "SHORT" if is_short else "LONG",
                })
            short_tax = short_pnl * rules["short_term_rate"] if short_pnl > 0 else 0
            long_tax = long_pnl * rules["long_term_rate"] if long_pnl > 0 else 0
            report["totals"] = {
                "n_trades": len(self.entries),
                "gross_pnl": round(sum(e.pnl for e in self.entries), 2),
                "short_term_pnl": round(short_pnl, 2),
                "long_term_pnl": round(long_pnl, 2),
                "estimated_tax": round(short_tax + long_tax, 2),
                "net_after_tax": round(sum(e.pnl for e in self.entries) - short_tax - long_tax, 2),
            }
        else:
            # Flat-rate jurisdictions
            gross = sum(e.pnl for e in self.entries)
            tax = gross * rules.get("flat_rate", 0) if gross > 0 else 0
            report["totals"] = {
                "n_trades": len(self.entries),
                "gross_pnl": round(gross, 2),
                "flat_rate": rules.get("flat_rate", 0),
                "estimated_tax": round(tax, 2),
                "net_after_tax": round(gross - tax, 2),
            }

        return report

    def daily_summary(self) -> pd.DataFrame:
        """Aggregate per-day PnL."""
        if not self.entries:
            return pd.DataFrame()
        rows = []
        for e in self.entries:
            date = e.timestamp[:10] if e.timestamp else "?"
            rows.append({"date": date, "symbol": e.symbol, "pnl": e.pnl, "lots": e.lots})
        df = pd.DataFrame(rows)
        return df.groupby("date").agg(
            n_trades=("pnl", "count"),
            pnl=("pnl", "sum"),
            symbols=("symbol", lambda x: list(x.unique())[:5]),
        ).reset_index()

    def symbol_summary(self) -> pd.DataFrame:
        """Aggregate per-symbol PnL."""
        if not self.entries:
            return pd.DataFrame()
        rows = [{"symbol": e.symbol, "pnl": e.pnl, "lots": e.lots} for e in self.entries]
        df = pd.DataFrame(rows)
        return df.groupby("symbol").agg(
            n_trades=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
        ).reset_index().sort_values("total_pnl", ascending=False)

    def export_csv(self, out_path: str | Path) -> str:
        """Export journal to CSV for accounting software."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.entries:
            out_path.write_text("entry_id,run_id,timestamp,symbol,direction,entry_price,exit_price,lots,pnl,pnl_pct,commission,swap,hold_bars\n")
            return str(out_path)
        rows = []
        for e in self.entries:
            rows.append({
                "entry_id": e.entry_id,
                "run_id": e.run_id,
                "timestamp": e.timestamp,
                "symbol": e.symbol,
                "direction": e.direction,
                "entry_price": e.entry_price,
                "exit_price": e.exit_price,
                "lots": e.lots,
                "pnl": e.pnl,
                "pnl_pct": e.pnl_pct,
                "commission": e.commission,
                "swap": e.swap,
                "hold_bars": e.hold_bars,
            })
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return str(out_path)


# ---------- Self-test ----------
if __name__ == "__main__":
    journal = TradeJournal(run_id="test_run_001")
    journal.log_run_meta(strategy="adx", symbol="EURUSD", timeframe="H1")

    # Sample trades
    trades = pd.DataFrame({
        "symbol": ["EURUSD", "EURUSD", "GBPUSD", "USDJPY"],
        "direction": [1, -1, 1, 1],
        "entry_price": [1.1000, 1.1050, 1.2700, 150.00],
        "exit_price": [1.1050, 1.1030, 1.2720, 150.50],
        "lots": [0.1, 0.1, 0.05, 0.1],
        "pnl": [50, -20, 10, 50],
        "pnl_pct": [0.005, -0.002, 0.001, 0.003],
        "commission": [0.7, 0.7, 0.35, 0.7],
        "swap": [0, 0, 0, 0],
        "entry_bar": [10, 20, 30, 40],
        "exit_bar": [15, 25, 35, 45],
        "timestamp": ["2024-01-01T10:00", "2024-01-02T11:00", "2024-01-03T12:00", "2024-01-04T13:00"],
    })
    entries = journal.log_trades(trades)
    print(f"Logged {len(entries)} trades")

    # Reconciliation
    broker_trades = trades.copy()  # assume perfect match
    recon = journal.reconcile_with_broker(broker_trades)
    print(f"Reconciliation: matched={recon.n_matched}, drift={recon.drift_pct}%")

    # Tax reports
    print()
    print("=== US Tax Report ===")
    us_tax = journal.tax_report("US")
    print(json.dumps(us_tax["totals"], indent=2))

    print()
    print("=== EU Tax Report ===")
    eu_tax = journal.tax_report("EU")
    print(json.dumps(eu_tax["totals"], indent=2))

    print()
    print("=== Symbol Summary ===")
    print(journal.symbol_summary().to_string(index=False))

    # Export
    csv_path = journal.export_csv("output/journal_test.csv")
    print(f"\nCSV exported: {csv_path}")