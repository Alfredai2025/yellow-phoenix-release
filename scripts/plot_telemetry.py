#!/usr/bin/env python3
"""Plot telemetry streams — reads JSONL, outputs stats + CSV for external plotting."""
import json
import sys
from pathlib import Path

TELEMETRY_DIR = Path("logs/telemetry")


def export_csv(stream, out_path):
    """Export a telemetry stream to CSV for Excel/Matplotlib/Tableau."""
    p = TELEMETRY_DIR / f"{stream}.jsonl"
    if not p.exists():
        print(f"No data for stream: {stream}")
        return
    with open(p) as f, open(out_path, "w") as out:
        out.write("ts,dt,event,latency_ms,route,source,q\n")
        for line in f:
            if line.strip() and not line.startswith('{"_schema"'):
                try:
                    e = json.loads(line)
                    q = str(e.get("q", "")).replace(",", " ")
                    out.write(
                        f"{e.get('_ts', '')},{e.get('_dt', '')},"
                        f"{e.get('event', '')},{e.get('latency_ms', '')},"
                        f"{e.get('route', '')},{e.get('source', '')},{q}\n"
                    )
                except Exception:
                    pass
    print(f"Exported {stream} -> {out_path}")


def latency_stats(stream="query"):
    """Compute latency stats for a stream."""
    p = TELEMETRY_DIR / f"{stream}.jsonl"
    if not p.exists():
        return {}
    times = []
    for line in open(p):
        if line.strip() and not line.startswith('{"_schema"'):
            try:
                e = json.loads(line)
                if "latency_ms" in e:
                    times.append(e["latency_ms"])
            except Exception:
                pass
    if not times:
        return {}
    times.sort()
    n = len(times)
    return {
        "count": n,
        "avg": sum(times) / n,
        "min": times[0],
        "max": times[-1],
        "p50": times[n // 2],
        "p95": times[int(n * 0.95)],
        "p99": times[int(n * 0.99)] if n >= 100 else times[-1],
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        for s in ["query", "system", "error", "tune"]:
            export_csv(s, f"logs/telemetry_{s}.csv")
    else:
        print("Latency stats (query stream):")
        for k, v in latency_stats().items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        print("\nRun: python3 scripts/plot_telemetry.py export  -> generates CSVs for plotting")
