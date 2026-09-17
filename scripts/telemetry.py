#!/usr/bin/env python3
"""Unified telemetry for Yellow Phoenix.

Import: from scripts.telemetry import TELEMETRY
Usage:  TELEMETRY.emit("query", {"q": "...", "route": "keyword_fast", "latency_ms": 0.07})
"""
import json
import time
import threading
from pathlib import Path
from collections import deque
from datetime import datetime

TELEMETRY_DIR = Path("logs/telemetry")
TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)


class Telemetry:
    """Thread-safe, multi-stream JSONL telemetry logger."""

    def __init__(self, buffer_size=500):
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=buffer_size)
        self._streams = {}
        for name in ["query", "system", "decision", "error", "ingest", "tune"]:
            p = TELEMETRY_DIR / f"{name}.jsonl"
            fh = open(p, "a")
            if fh.tell() == 0:
                fh.write(json.dumps({"_schema": "telemetry_v1", "stream": name, "ts": time.time()}) + "\n")
            self._streams[name] = fh

    def emit(self, stream: str, event: dict):
        """Emit one structured event to a stream."""
        if stream not in self._streams:
            stream = "system"
        event["_ts"] = time.time()
        event["_dt"] = datetime.now().isoformat()
        event["_stream"] = stream
        line = json.dumps(event, default=str)
        with self._lock:
            self._streams[stream].write(line + "\n")
            self._streams[stream].flush()
            self._buffer.append(event)

    def tail(self, stream: str, n: int = 20):
        """Return the last n events from a stream."""
        path = TELEMETRY_DIR / f"{stream}.jsonl"
        if not path.exists():
            return []
        lines = deque(maxlen=n)
        with open(path) as f:
            for line in f:
                if line.strip() and not line.startswith('{"_schema"'):
                    try:
                        lines.append(json.loads(line))
                    except Exception:
                        pass
        return list(lines)

    def stats(self):
        """Return per-stream record counts."""
        out = {}
        for name in self._streams:
            p = TELEMETRY_DIR / f"{name}.jsonl"
            if p.exists():
                out[name] = sum(1 for _ in open(p)) - 1
        return out

    def close(self):
        for fh in self._streams.values():
            fh.close()


TELEMETRY = Telemetry()
