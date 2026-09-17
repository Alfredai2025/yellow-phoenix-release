#!/usr/bin/env python3
"""Live telemetry dashboard — every event, every stream, real-time."""
import json
import time
import os
from pathlib import Path
from collections import deque, defaultdict

TELEMETRY_DIR = Path("logs/telemetry")
STREAMS = ["query", "system", "decision", "error", "ingest", "tune"]
REFRESH = 1


def clear():
    os.system("clear" if os.name != "nt" else "cls")


def tail_jsonl(path, n=15):
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


def draw():
    clear()
    print("=" * 70)
    print("  YELLOW PHOENIX — LIVE TELEMETRY DASHBOARD")
    print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    queries = tail_jsonl(TELEMETRY_DIR / "query.jsonl", 100)
    if queries:
        latencies = [q.get("latency_ms", 0) for q in queries]
        routes = defaultdict(int)
        for q in queries:
            routes[q.get("route", "unknown")] += 1
        avg_lat = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        print(f"\n  [QUERIES]  Last 100: count={len(queries)}  avg={avg_lat:.2f}ms  P50={p50:.2f}ms")
        print("            Routes: " + "  ".join(f"{k}={v}" for k, v in routes.items()))
    else:
        print("\n  [QUERIES]  No data yet")

    errors = tail_jsonl(TELEMETRY_DIR / "error.jsonl", 5)
    if errors:
        print(f"\n  [ERRORS]   Last {len(errors)}:")
        for e in errors:
            print(f"    {e.get('_dt', '?')} | {str(e.get('msg', e))[:60]}")
    else:
        print("\n  [ERRORS]   None")

    systems = tail_jsonl(TELEMETRY_DIR / "system.jsonl", 5)
    if systems:
        print(f"\n  [SYSTEM]   Last {len(systems)}:")
        for s in systems:
            print(f"    {s.get('event', '?')} | {s.get('_dt', '?')}")
    else:
        print("\n  [SYSTEM]   None")

    tunes = tail_jsonl(TELEMETRY_DIR / "tune.jsonl", 3)
    if tunes:
        print(f"\n  [TUNING]   Last {len(tunes)}:")
        for t in tunes:
            print(f"    {t.get('param', '?')} {t.get('old', '?')} -> {t.get('new', '?')} (conf={t.get('confidence', 0):.2f})")
    else:
        print("\n  [TUNING]   None")

    ingests = tail_jsonl(TELEMETRY_DIR / "ingest.jsonl", 3)
    if ingests:
        print(f"\n  [INGEST]   Last {len(ingests)}:")
        for i in ingests:
            print(f"    {str(i.get('msg', i))[:60]}")
    else:
        print("\n  [INGEST]   None")

    print("\n  [STREAM SIZES]")
    for name in STREAMS:
        p = TELEMETRY_DIR / f"{name}.jsonl"
        size = p.stat().st_size if p.exists() else 0
        print(f"    {name:12s}: {size:>10,} bytes")

    print("\n" + "=" * 70)
    print("  Ctrl+C to exit  |  Logs: logs/telemetry/*.jsonl")
    print("=" * 70)


if __name__ == "__main__":
    print("Live telemetry starting... (Ctrl+C to stop)")
    try:
        while True:
            draw()
            time.sleep(REFRESH)
    except KeyboardInterrupt:
        print("\nLive telemetry stopped.")
