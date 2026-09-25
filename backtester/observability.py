"""R014: Infrastructure — structured logging, metrics, alerting, lightweight API.

Features:
- StructuredLogger: JSONL logs with levels, timestamps, context fields
- MetricsRegistry: in-memory Prometheus-style counters/gauges/histograms
- AlertManager: rule-based alerts on metrics (DD breach, margin call, etc.)
- MiniAPI: lightweight HTTP endpoints for status/metrics (no FastAPI dep)

Usage:
    logger = StructuredLogger("backtest", log_path="output/log.jsonl")
    metrics = MetricsRegistry()
    alerts = AlertManager()

    logger.info("backtest_complete", symbol="EURUSD", sharpe=1.5)
    metrics.counter("backtest_runs_total", {"symbol": "EURUSD"}).inc()
    metrics.gauge("equity").set(10500)

    alerts.add_rule("dd_breach", lambda m: m.gauge("drawdown_pct").value() > 0.25, "critical", "Drawdown > 25%")
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# ---------- Structured Logger ----------
class StructuredLogger:
    """JSONL logger with context fields."""

    def __init__(self, name: str = "backtester", log_path: Optional[str] = None,
                 level: str = "INFO", console: bool = True):
        self.name = name
        self.log_path = Path(log_path) if log_path else (
            Path(__file__).parent.parent / "output" / f"{name}.jsonl"
        )
        self.level = level.upper()
        self.console = console
        self._lock = threading.Lock()
        self._levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}

    def _log(self, level: str, event: str, **fields):
        if self._levels.get(level, 0) < self._levels.get(self.level, 0):
            return
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "logger": self.name,
            "event": event,
            **fields,
        }
        line = json.dumps(entry, default=str)
        with self._lock:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass
            if self.console:
                print(line)

    def debug(self, event: str, **fields): self._log("DEBUG", event, **fields)
    def info(self, event: str, **fields): self._log("INFO", event, **fields)
    def warning(self, event: str, **fields): self._log("WARNING", event, **fields)
    def error(self, event: str, **fields): self._log("ERROR", event, **fields)
    def critical(self, event: str, **fields): self._log("CRITICAL", event, **fields)


# ---------- Metrics Registry ----------
class _Metric:
    """Base metric."""
    def __init__(self, name: str, help: str, labels: dict | None = None):
        self.name = name
        self.help = help
        self.labels = labels or {}
        self._lock = threading.Lock()

    def _label_key(self) -> str:
        return ",".join(f"{k}={v}" for k, v in sorted(self.labels.items()))


class _Counter(_Metric):
    """Monotonic counter."""
    def __init__(self, name: str, help: str = "", labels: dict | None = None):
        super().__init__(name, help, labels)
        self._value = 0.0

    def inc(self, amount: float = 1.0):
        with self._lock:
            self._value += amount

    def value(self) -> float:
        with self._lock:
            return self._value


class _Gauge(_Metric):
    """Arbitrary value that can go up/down."""
    def __init__(self, name: str, help: str = "", labels: dict | None = None):
        super().__init__(name, help, labels)
        self._value = 0.0

    def set(self, value: float):
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0):
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0):
        with self._lock:
            self._value -= amount

    def value(self) -> float:
        with self._lock:
            return self._value


class _Histogram(_Metric):
    """Distribution of values."""
    def __init__(self, name: str, help: str = "", labels: dict | None = None,
                 buckets: list[float] | None = None):
        super().__init__(name, help, labels)
        self.buckets = sorted(buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
        self._values: list[float] = []
        self._counts = {b: 0 for b in self.buckets}
        self._counts[float("inf")] = 0
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float):
        with self._lock:
            self._values.append(value)
            if len(self._values) > 1000:
                self._values = self._values[-1000:]
            self._sum += value
            self._count += 1
            for b in self.buckets:
                if value <= b:
                    self._counts[b] += 1
            self._counts[float("inf")] += 1

    def snapshot(self) -> dict:
        with self._lock:
            if not self._values:
                return {"count": 0, "sum": 0.0, "mean": 0.0,
                        "p50": 0.0, "p95": 0.0, "p99": 0.0}
            arr = np.array(self._values)
            return {
                "count": self._count,
                "sum": round(self._sum, 4),
                "mean": round(float(arr.mean()), 4),
                "p50": round(float(np.percentile(arr, 50)), 4),
                "p95": round(float(np.percentile(arr, 95)), 4),
                "p99": round(float(np.percentile(arr, 99)), 4),
            }


class MetricsRegistry:
    """In-memory Prometheus-style metrics."""

    def __init__(self):
        self._counters: dict[str, _Counter] = {}
        self._gauges: dict[str, _Gauge] = {}
        self._histograms: dict[str, _Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, labels: dict | None = None) -> _Counter:
        key = self._key(name, labels)
        with self._lock:
            if key not in self._counters:
                self._counters[key] = _Counter(name, labels=labels)
            return self._counters[key]

    def gauge(self, name: str, labels: dict | None = None) -> _Gauge:
        key = self._key(name, labels)
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = _Gauge(name, labels=labels)
            return self._gauges[key]

    def histogram(self, name: str, labels: dict | None = None,
                  buckets: list[float] | None = None) -> _Histogram:
        key = self._key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = _Histogram(name, labels=labels, buckets=buckets)
            return self._histograms[key]

    def _key(self, name: str, labels: dict | None) -> str:
        if not labels:
            return name
        return f"{name}#{','.join(f'{k}={v}' for k, v in sorted(labels.items()))}"

    def export_prometheus(self) -> str:
        """Export as Prometheus text format."""
        lines = []
        for c in self._counters.values():
            lines.append(f"# TYPE {c.name} counter")
            if c.labels:
                label_str = "{" + ",".join(f'{k}="{v}"' for k, v in c.labels.items()) + "}"
            else:
                label_str = ""
            lines.append(f"{c.name}{label_str} {c.value()}")
        for g in self._gauges.values():
            lines.append(f"# TYPE {g.name} gauge")
            if g.labels:
                label_str = "{" + ",".join(f'{k}="{v}"' for k, v in g.labels.items()) + "}"
            else:
                label_str = ""
            lines.append(f"{g.name}{label_str} {g.value()}")
        for h in self._histograms.values():
            lines.append(f"# TYPE {h.name} histogram")
            snap = h.snapshot()
            lines.append(f"{h.name}_count {snap['count']}")
            lines.append(f"{h.name}_sum {snap['sum']}")
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """JSON snapshot of all metrics."""
        return {
            "counters": {k: v.value() for k, v in self._counters.items()},
            "gauges": {k: v.value() for k, v in self._gauges.items()},
            "histograms": {k: v.snapshot() for k, v in self._histograms.items()},
        }


# ---------- Alert Manager ----------
@dataclass
class AlertRule:
    """Single alert rule."""
    name: str
    condition: Callable[[MetricsRegistry], bool]
    severity: str  # "info" | "warning" | "critical"
    message: str
    cooldown_sec: float = 60.0
    last_fired: float = 0.0


@dataclass
class AlertEvent:
    """Fired alert."""
    timestamp: float
    rule_name: str
    severity: str
    message: str


class AlertManager:
    """Rule-based alerting on metrics."""

    def __init__(self):
        self.rules: list[AlertRule] = []
        self.fired: list[AlertEvent] = []
        self._lock = threading.Lock()

    def add_rule(self, name: str, condition: Callable, severity: str, message: str,
                 cooldown_sec: float = 60.0):
        self.rules.append(AlertRule(
            name=name, condition=condition, severity=severity,
            message=message, cooldown_sec=cooldown_sec,
        ))

    def check(self, registry: MetricsRegistry) -> list[AlertEvent]:
        """Evaluate all rules, return fired alerts."""
        now = time.time()
        fired_now = []
        with self._lock:
            for rule in self.rules:
                if now - rule.last_fired < rule.cooldown_sec:
                    continue
                try:
                    if rule.condition(registry):
                        alert = AlertEvent(
                            timestamp=now,
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=rule.message,
                        )
                        fired_now.append(alert)
                        self.fired.append(alert)
                        rule.last_fired = now
                except Exception:
                    pass
            if len(self.fired) > 1000:
                self.fired = self.fired[-1000:]
        return fired_now

    def get_fired(self, since: float = 0.0) -> list[AlertEvent]:
        with self._lock:
            return [a for a in self.fired if a.timestamp >= since]


# ---------- Mini API ----------
def _make_handler(registry: MetricsRegistry, alert_manager: AlertManager,
                  extra_status: dict | None = None):
    """Factory that creates a handler with closure-bound dependencies.

    Avoids class-level mutable state so multiple MiniAPI instances can coexist.
    """
    es = extra_status or {}

    class _APIHandler(BaseHTTPRequestHandler):
        """Simple HTTP handler for /status, /metrics, /alerts."""

        def do_GET(self):
            if self.path == "/status":
                self._json(es)
            elif self.path == "/metrics":
                body = registry.export_prometheus()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body.encode())
            elif self.path == "/metrics.json":
                self._json(registry.snapshot())
            elif self.path == "/alerts":
                fired = alert_manager.get_fired() if alert_manager else []
                self._json([{
                    "ts": a.timestamp, "rule": a.rule_name,
                    "severity": a.severity, "message": a.message,
                } for a in fired])
            elif self.path == "/":
                self._json({
                    "service": "newmeta-backtester",
                    "endpoints": ["/status", "/metrics", "/metrics.json", "/alerts"],
                })
            else:
                self.send_response(404)
                self.end_headers()

        def _json(self, obj):
            body = json.dumps(obj, default=str)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, format, *args):
            pass  # silence

    return _APIHandler


class MiniAPI:
    """Lightweight HTTP server for metrics + alerts (no FastAPI dep)."""

    def __init__(self, registry: MetricsRegistry, alert_manager: AlertManager,
                 host: str = "127.0.0.1", port: int = 8765):
        self.registry = registry
        self.alert_manager = alert_manager
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, background: bool = True, extra_status: dict | None = None):
        """Start API server."""
        handler_cls = _make_handler(self.registry, self.alert_manager, extra_status)
        self._server = HTTPServer((self.host, self.port), handler_cls)
        if background:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            return f"http://{self.host}:{self.port}"
        else:
            self._server.serve_forever()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()


# ---------- Default alert rules ----------
def setup_default_alerts(alert_manager: AlertManager, registry: MetricsRegistry,
                         dd_threshold: float = 0.25, margin_threshold: float = 50.0):
    """Setup default alert rules for trading."""
    alert_manager.add_rule(
        "drawdown_breach",
        lambda r: r.gauge("drawdown_pct").value() > dd_threshold,
        "critical",
        f"Drawdown exceeds {dd_threshold:.0%}",
        cooldown_sec=300.0,
    )
    alert_manager.add_rule(
        "margin_call",
        lambda r: r.gauge("margin_level_pct").value() < margin_threshold,
        "critical",
        f"Margin level below {margin_threshold:.0f}%",
        cooldown_sec=60.0,
    )
    alert_manager.add_rule(
        "equity_drawdown",
        lambda r: r.gauge("equity").value() < r.gauge("initial_equity").value() * 0.5,
        "critical",
        "Equity dropped below 50% of initial",
        cooldown_sec=600.0,
    )
    alert_manager.add_rule(
        "high_latency",
        lambda r: r.histogram("tick_latency_ms").snapshot().get("p99", 0) > 1000,
        "warning",
        "Tick P99 latency exceeds 1 second",
        cooldown_sec=120.0,
    )


# ---------- Self-test ----------
if __name__ == "__main__":
    # Logger
    logger = StructuredLogger("test", log_path="output/test_logger.jsonl", level="DEBUG")
    logger.info("test_start", version="1.0")
    logger.warning("slow_tick", latency_ms=600, symbol="EURUSD")

    # Metrics
    registry = MetricsRegistry()
    bt_runs = registry.counter("backtest_runs_total", {"symbol": "EURUSD"})
    bt_runs.inc()
    bt_runs.inc()

    equity = registry.gauge("equity")
    equity.set(10500)
    dd = registry.gauge("drawdown_pct")
    dd.set(0.03)

    initial_eq = registry.gauge("initial_equity")
    initial_eq.set(10000)

    margin = registry.gauge("margin_level_pct")
    margin.set(250)

    latency = registry.histogram("tick_latency_ms")
    for v in [50, 80, 120, 200, 150, 90]:
        latency.observe(v)

    # Alerts
    alerts = AlertManager()
    setup_default_alerts(alerts, registry)
    fired = alerts.check(registry)
    print(f"Fired alerts: {len(fired)}")

    # Trigger DD breach
    dd.set(0.30)
    fired = alerts.check(registry)
    print(f"After DD breach: {fired}")

    # Print snapshot
    print("\n=== Metrics Snapshot ===")
    print(json.dumps(registry.snapshot(), indent=2))

    print("\n=== Prometheus Format (first 10 lines) ===")
    prom = registry.export_prometheus()
    for line in prom.split("\n")[:10]:
        print(line)

    # Mini API test (start in background, stop after)
    print("\n=== Mini API ===")
    api = MiniAPI(registry, alerts, port=8765)
    url = api.start(background=True)
    print(f"API started at {url}")
    import urllib.request
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:8765/").read()
        print(f"GET /: {resp.decode()[:100]}")
        resp = urllib.request.urlopen("http://127.0.0.1:8765/metrics.json").read()
        print(f"GET /metrics.json: {resp.decode()[:100]}")
    finally:
        api.stop()
        print("API stopped")
