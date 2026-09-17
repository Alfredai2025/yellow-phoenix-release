#!/usr/bin/env python3
"""Narrative translator — reads Yellow Phoenix telemetry and tells the story."""
from pathlib import Path
from collections import deque
import json
import time

TELEMETRY_DIR = Path("logs/telemetry")


def tail(path, n=10):
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


def explain_queries(events):
    if not events:
        return "No queries yet."
    lines = []
    for e in events:
        q = e.get("q", "?")
        route = e.get("route", "?")
        source = e.get("source", "?")
        ms = e.get("latency_ms", 0)

        if route == "keyword":
            lines.append(
                f"  • User asked '{q}' → Yellow found it instantly in keyword mesh ({ms:.2f} ms)"
            )
        elif route == "beacon":
            lines.append(
                f"  • User asked '{q}' → Yellow used a semantic beacon shortcut ({ms:.2f} ms)"
            )
        elif route == "cache":
            lines.append(
                f"  • User asked '{q}' → Yellow served from memory cache ({ms:.2f} ms)"
            )
        elif route == "deep":
            lines.append(
                f"  • User asked '{q}' → Yellow did full deep search with encoder + HNSW ({ms:.2f} ms)"
            )
        else:
            lines.append(f"  • User asked '{q}' → Yellow routed via {source} ({ms:.2f} ms)")
    return "\n".join(lines)


def explain_system(events):
    if not events:
        return "No system events."
    lines = []
    for e in events:
        evt = e.get("event", "?")
        if evt == "hnsw_load":
            lines.append("  • Yellow loaded its HNSW index from disk.")
        elif evt == "hnsw_save":
            lines.append("  • Yellow saved its HNSW index to disk.")
        elif evt == "drift_alert":
            lines.append(
                f"  • ⚠️ DRIFT ALERT: Yellow detected its encoder may be getting stale "
                f"(score: {e.get('score', '?')})"
            )
        elif evt == "test_event":
            lines.append("  • Yellow ran a self-test.")
        else:
            lines.append(f"  • System event: {evt}")
    return "\n".join(lines)


def explain_tuning(events):
    if not events:
        return "No tuning changes."
    lines = []
    for e in events:
        param = e.get("param", "?")
        old = e.get("old", "?")
        new = e.get("new", "?")
        conf = e.get("confidence", 0)
        lines.append(
            f"  • Yellow auto-tuned '{param}': {old} → {new} (confidence: {conf:.0%})"
        )
    return "\n".join(lines)


def explain_errors(events):
    if not events:
        return "No errors. Yellow is healthy."
    lines = []
    for e in events:
        msg = str(e)[:80]
        lines.append(f"  • ⚠️ ERROR: {msg}")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("  YELLOW PHOENIX — NARRATIVE TRANSLATOR")
    print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    queries = tail(TELEMETRY_DIR / "query.jsonl", 10)
    systems = tail(TELEMETRY_DIR / "system.jsonl", 5)
    tunes = tail(TELEMETRY_DIR / "tune.jsonl", 5)
    errors = tail(TELEMETRY_DIR / "error.jsonl", 5)

    print(f"\n[QUERIES] Last {len(queries)} searches:")
    print(explain_queries(queries))

    print(f"\n[SYSTEM] Last {len(systems)} events:")
    print(explain_system(systems))

    print(f"\n[TUNING] Last {len(tunes)} changes:")
    print(explain_tuning(tunes))

    print("\n[HEALTH] Errors:")
    print(explain_errors(errors))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
