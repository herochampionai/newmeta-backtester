"""Real-time tick feed — websocket + MT5 live for paper/live trading.

Unified interface: same signal→order path as backtest, just live data.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tick:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    time: float  # unix utc


class MT5LiveFeed:
    """MT5 live tick stream — runs in background thread, pushes to callback."""
    def __init__(self, symbol: str, callback: Callable[[Tick], None], terminal: str | None = None):
        self.symbol = symbol.upper()
        self.callback = callback
        self.terminal = terminal
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        try:
            import MetaTrader5 as mt5

            from data.mt5_export import init_mt5, resolve_terminal
            t = self.terminal or resolve_terminal()
            if not t or not init_mt5(t):
                return
            mt5.symbol_select(self.symbol, True)
            while self._running:
                tick = mt5.symbol_info_tick(self.symbol)
                if tick and tick.bid > 0 and tick.ask > 0:
                    self.callback(Tick(
                        symbol=self.symbol,
                        bid=float(tick.bid),
                        ask=float(tick.ask),
                        last=float(tick.last),
                        volume=float(tick.volume),
                        time=float(tick.time),
                    ))
                time.sleep(0.05)  # ~20 Hz
            mt5.shutdown()
        except Exception:
            pass


class WebsocketFeed:
    """Generic websocket tick feed (Binance, Bybit, etc.) — override _connect/_parse."""
    def __init__(self, url: str, symbol: str, callback: Callable[[Tick], None]):
        self.url = url
        self.symbol = symbol.upper()
        self.callback = callback
        self._running = False
        self._task = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        import websockets
        while self._running:
            try:
                async with websockets.connect(self.url) as ws:
                    await self._subscribe(ws)
                    async for msg in ws:
                        if not self._running:
                            break
                        tick = self._parse(msg)
                        if tick:
                            self.callback(tick)
            except Exception:
                await asyncio.sleep(5)  # reconnect

    async def _subscribe(self, ws):
        pass  # override

    def _parse(self, msg: str) -> Tick | None:
        return None  # override


class BinanceFeed(WebsocketFeed):
    def __init__(self, symbol: str, callback: Callable[[Tick], None]):
        super().__init__(f"wss://stream.binance.com:9443/ws/{symbol.lower()}@bookTicker", symbol, callback)

    async def _subscribe(self, ws):
        pass  # bookTicker auto-subscribes

    def _parse(self, msg: str) -> Tick | None:
        try:
            d = json.loads(msg)
            return Tick(symbol=self.symbol, bid=float(d["b"]), ask=float(d["a"]),
                        last=float(d["b"]), volume=0.0, time=time.time())
        except Exception:
            return None


class TickBuffer:
    """Thread-safe tick buffer with time-window queries for live indicators."""
    def __init__(self, maxlen: int = 10000):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, tick: Tick):
        with self._lock:
            self._buf.append(tick)

    def get_window(self, seconds: float) -> list[Tick]:
        cutoff = time.time() - seconds
        with self._lock:
            return [t for t in self._buf if t.time >= cutoff]

    def last(self) -> Tick | None:
        with self._lock:
            return self._buf[-1] if self._buf else None
