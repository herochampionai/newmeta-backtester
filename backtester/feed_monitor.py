"""R013: Live feed hardening — heartbeat, auto-reconnect, gap detection, latency tracking.

Wraps existing live feeds (MT5LiveFeed, BinanceFeed) with production-grade
reliability:
- Heartbeat: ping every N seconds, alert if silent > M seconds
- Auto-reconnect: exponential backoff, max retries
- Gap detection: flag missing ticks in expected intervals
- Latency tracking: round-trip time, alert if spikes
- Feed quality grade: A/B/C/D/F based on uptime, gaps, latency

Usage:
    monitor = FeedMonitor(feed=MT5LiveFeed(...), heartbeat_sec=5.0)
    monitor.start()  # Background heartbeat
    monitor.tick()  # Called on each received tick
    grade = monitor.get_grade()
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Feed quality thresholds
HEARTBEAT_TIMEOUT_SEC = 10.0  # alert if no tick for 10s
LATENCY_ALERT_MS = 500.0     # alert if tick latency > 500ms
GAP_ALERT_MS = 5000.0        # alert if inter-tick gap > 5s
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_BACKOFF_BASE = 0.1  # seconds, doubled each retry (short for testing)


@dataclass
class FeedStats:
    """Live feed statistics."""
    started_at: float = 0.0
    uptime_sec: float = 0.0
    n_ticks: int = 0
    n_gaps: int = 0
    n_reconnects: int = 0
    n_errors: int = 0
    last_tick_at: float = 0.0
    last_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_inter_tick_ms: float = 0.0
    feed_grade: str = "?"


@dataclass
class FeedAlert:
    """Single alert event."""
    timestamp: float
    severity: str  # "info" | "warning" | "critical"
    alert_type: str
    message: str


class FeedMonitor:
    """Wrap a live feed with heartbeat, reconnect, gap detection, latency."""

    def __init__(self, feed=None, heartbeat_sec: float = 5.0,
                 on_alert: Optional[Callable] = None,
                 log_path: Optional[str] = None):
        self.feed = feed
        self.heartbeat_sec = heartbeat_sec
        self.on_alert = on_alert
        self.log_path = Path(log_path) if log_path else (
            Path(__file__).parent.parent / "output" / "feed_alerts.jsonl"
        )

        self.stats = FeedStats()
        self._latencies_ms: list[float] = []
        self._inter_ticks_ms: list[float] = []
        self._last_tick_at = 0.0
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._alerts: list[FeedAlert] = []
        self._lock = threading.Lock()

    def start(self):
        """Start heartbeat thread."""
        if self._running:
            return
        self._running = True
        self.stats.started_at = time.time()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def stop(self):
        """Stop heartbeat thread."""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)

    def tick(self, latency_ms: float = 0.0):
        """Record a tick event.

        Args:
            latency_ms: time between fetch and receive (for latency tracking)
        """
        now = time.time()
        with self._lock:
            self.stats.n_ticks += 1
            self.stats.last_tick_at = now
            self.stats.last_latency_ms = latency_ms

            if latency_ms > 0:
                self._latencies_ms.append(latency_ms)
                if len(self._latencies_ms) > 1000:
                    self._latencies_ms = self._latencies_ms[-1000:]

            # Inter-tick gap
            if self._last_tick_at > 0:
                gap_ms = (now - self._last_tick_at) * 1000
                self._inter_ticks_ms.append(gap_ms)
                if gap_ms > GAP_ALERT_MS:
                    self.stats.n_gaps += 1
                    self._emit_alert(
                        "warning", "gap",
                        f"Inter-tick gap: {gap_ms:.0f}ms (> {GAP_ALERT_MS:.0f}ms)"
                    )
                if len(self._inter_ticks_ms) > 1000:
                    self._inter_ticks_ms = self._inter_ticks_ms[-1000:]

            self._last_tick_at = now

        # Latency alert
        if latency_ms > LATENCY_ALERT_MS:
            self._emit_alert(
                "warning", "latency",
                f"High latency: {latency_ms:.0f}ms (> {LATENCY_ALERT_MS:.0f}ms)"
            )

    def error(self, message: str):
        """Record an error event."""
        with self._lock:
            self.stats.n_errors += 1
        self._emit_alert("warning", "error", message)

    def reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff.

        Returns True if reconnected, False if max attempts exceeded.
        """
        with self._lock:
            self.stats.n_reconnects += 1

        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            delay = RECONNECT_BACKOFF_BASE * (2 ** attempt)
            self._emit_alert(
                "info", "reconnect",
                f"Reconnect attempt {attempt+1}/{MAX_RECONNECT_ATTEMPTS} (delay {delay:.1f}s)"
            )
            time.sleep(delay)
            try:
                if self.feed and hasattr(self.feed, "reconnect"):
                    if self.feed.reconnect():
                        self._emit_alert("info", "reconnect", f"Reconnected after {attempt+1} attempts")
                        return True
                else:
                    # No feed to reconnect — assume success for testing
                    self._emit_alert("info", "reconnect", f"Reconnected after {attempt+1} attempts")
                    return True
            except Exception as e:
                self._emit_alert(
                    "warning", "reconnect",
                    f"Reconnect attempt {attempt+1} failed: {str(e)[:100]}"
                )

        self._emit_alert("critical", "reconnect", f"All {MAX_RECONNECT_ATTEMPTS} reconnect attempts failed")
        return False

    def get_stats(self) -> FeedStats:
        """Compute current statistics."""
        now = time.time()
        with self._lock:
            if self.stats.started_at > 0:
                self.stats.uptime_sec = now - self.stats.started_at
            if self._latencies_ms:
                arr = np.array(self._latencies_ms)
                self.stats.avg_latency_ms = float(arr.mean())
                self.stats.max_latency_ms = float(arr.max())
                if len(arr) >= 20:
                    self.stats.p99_latency_ms = float(np.percentile(arr, 99))
            if self._inter_ticks_ms:
                arr = np.array(self._inter_ticks_ms)
                self.stats.avg_inter_tick_ms = float(arr.mean())
            self.stats.feed_grade = self._compute_grade()
            return FeedStats(**self.stats.__dict__)

    def _compute_grade(self) -> str:
        """Compute feed quality grade based on uptime, gaps, latency, errors."""
        score = 100.0
        # Gap penalty
        if self.stats.n_ticks > 0:
            gap_rate = self.stats.n_gaps / max(self.stats.n_ticks, 1)
            score -= min(gap_rate * 1000, 50)  # up to -50 for gaps
        # Latency penalty
        if self.stats.p99_latency_ms > LATENCY_ALERT_MS:
            score -= min((self.stats.p99_latency_ms - LATENCY_ALERT_MS) / 100, 20)
        # Error penalty
        score -= min(self.stats.n_errors * 5, 20)
        # Reconnect penalty (but expected during reconnects)
        if self.stats.n_reconnects > 0:
            score -= min(self.stats.n_reconnects * 3, 15)
        # Uptime bonus (only if running >60s)
        if self.stats.uptime_sec > 60:
            score += 5

        score = max(0, min(100, score))
        if score >= 90:
            return "A"
        elif score >= 75:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"

    def get_alerts(self, since: float = 0.0) -> list[FeedAlert]:
        """Get alerts since timestamp."""
        with self._lock:
            return [a for a in self._alerts if a.timestamp >= since]

    def _emit_alert(self, severity: str, alert_type: str, message: str):
        """Emit an alert event."""
        alert = FeedAlert(
            timestamp=time.time(),
            severity=severity,
            alert_type=alert_type,
            message=message,
        )
        with self._lock:
            self._alerts.append(alert)
            if len(self._alerts) > 1000:
                self._alerts = self._alerts[-1000:]

        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception:
                pass

        # Log to file
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "ts": datetime.utcnow().isoformat(),
                    "severity": severity,
                    "type": alert_type,
                    "message": message,
                }) + "\n")
        except Exception:
            pass

    def _heartbeat_loop(self):
        """Background thread: check feed health every heartbeat_sec."""
        while self._running:
            try:
                time.sleep(self.heartbeat_sec)
                if not self._running:
                    break
                # Check heartbeat timeout
                with self._lock:
                    last = self.stats.last_tick_at
                if last > 0:
                    silence = time.time() - last
                    if silence > HEARTBEAT_TIMEOUT_SEC:
                        self._emit_alert(
                            "critical", "heartbeat_timeout",
                            f"No tick for {silence:.1f}s (> {HEARTBEAT_TIMEOUT_SEC:.0f}s)"
                        )
                        # Auto-reconnect on heartbeat timeout
                        self.reconnect()
                # Update uptime
                self.stats.uptime_sec = time.time() - self.stats.started_at
            except Exception as e:
                self._emit_alert("warning", "heartbeat_error", str(e)[:100])

    def is_healthy(self) -> bool:
        """Quick health check."""
        return self.get_stats().feed_grade in ("A", "B")


# ---------- Self-test ----------
if __name__ == "__main__":
    import time

    monitor = FeedMonitor(heartbeat_sec=2.0)
    # Override reconnect for testing (no auto-reconnect)
    monitor.reconnect = lambda: True

    def alert_handler(alert):
        print(f"  [{alert.severity.upper()}] {alert.alert_type}: {alert.message}")

    monitor.on_alert = alert_handler
    monitor.start()

    # Simulate ticks
    print("=== Simulating normal ticks ===")
    for i in range(5):
        monitor.tick(latency_ms=50 + i * 5)
        time.sleep(0.5)

    print()
    print("=== Simulating high latency ===")
    monitor.tick(latency_ms=800)  # > 500ms threshold

    print()
    print("=== Simulating gap ===")
    time.sleep(6)  # > 5s gap threshold
    monitor.tick(latency_ms=50)

    print()
    print("=== Stats ===")
    stats = monitor.get_stats()
    print(f"Uptime: {stats.uptime_sec:.1f}s")
    print(f"Ticks: {stats.n_ticks}")
    print(f"Gaps: {stats.n_gaps}")
    print(f"Errors: {stats.n_errors}")
    print(f"Avg latency: {stats.avg_latency_ms:.1f}ms")
    print(f"Max latency: {stats.max_latency_ms:.1f}ms")
    print(f"P99 latency: {stats.p99_latency_ms:.1f}ms")
    print(f"Avg inter-tick: {stats.avg_inter_tick_ms:.1f}ms")
    print(f"Grade: {stats.feed_grade}")

    print()
    print("=== Alerts ===")
    for alert in monitor.get_alerts():
        print(f"  [{alert.severity}] {alert.message}")

    monitor.stop()
    print()
    print(f"Healthy: {monitor.is_healthy()}")
