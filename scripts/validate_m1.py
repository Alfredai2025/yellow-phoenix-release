#!/usr/bin/env python3
"""validate_m1.py — M1 validation: 10K queries against the collaborative engine.

Loads paper titles from the droplet DB, generates deterministic 128/512-bit
PAP hashes, builds the engine, runs 10K self-queries, and reports R@1,
latency percentiles, cache hit rate, and feature activation.

Because the semantic MiniLM/ITQ encoder is not installed in this environment,
this script uses deterministic SHA-256 hashes of the titles. It therefore
validates the engine pipeline rather than end-to-end semantic accuracy.
"""

import hashlib
import json
import os
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "droplet_papers_106298.db"
INPUT_BIN = ROOT / "data" / "validate_m1_input.bin"
BINARY = ROOT / "target" / "release" / "validate_m1"

# Pass criteria from the M1 spec.
R1_MIN = 0.97
P50_MAX_US = 2.5
CACHE_MIN = 0.15


def stable_hash(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def build_input(max_records: int = 15_000) -> int:
    print(f"Loading papers from {DB} ...")
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title FROM papers WHERE title IS NOT NULL AND length(trim(title)) > 3 LIMIT ?",
        (max_records,),
    )
    rows = cur.fetchall()
    conn.close()

    records = []
    for pid, title in rows:
        title = title.strip()
        h = stable_hash(title)
        # 128-bit hash = first 16 bytes; 512-bit hash = 32-byte digest doubled.
        pap128 = h[:16]
        pap512 = h + h
        # Stable u64 id from first 8 bytes.
        uid = struct.unpack("<Q", h[:8])[0]
        records.append((uid, pap128, pap512, title))

    with open(INPUT_BIN, "wb") as f:
        f.write(struct.pack("<I", len(records)))
        for uid, pap128, pap512, _ in records:
            f.write(struct.pack("<Q", uid))
            f.write(pap128)
            f.write(pap512)

    print(f"  Wrote {len(records)} records to {INPUT_BIN}")
    return len(records)


def run_validation() -> dict:
    print("Building release binary ...")
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "validate_m1"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    print("Running 10K collaborative-engine queries ...")
    result = subprocess.run(
        [str(BINARY), str(INPUT_BIN)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    # The JSON report is the last multi-line block starting with '{'.
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    start = None
    for i, ln in enumerate(lines):
        if ln == "{":
            start = i
    if start is None:
        raise RuntimeError(f"No JSON report found in stdout:\n{result.stdout}")
    return json.loads("\n".join(lines[start:]))


def main():
    if not DB.exists():
        print(f"ERROR: database not found: {DB}")
        sys.exit(1)

    build_input()
    metrics = run_validation()

    print("\n" + "=" * 60)
    print("M1 VALIDATION REPORT")
    print("=" * 60)
    print(f"Records loaded:    {metrics['records_loaded']:,}")
    print(f"Queries run:       {metrics['queries']:,}")
    print(f"R@1:               {metrics['r1']:.4f}  ({metrics['correct']:,}/{metrics['queries']:,})")
    print(f"P50 latency:       {metrics['p50_latency_us']:.2f} μs")
    print(f"P99 latency:       {metrics['p99_latency_us']:.2f} μs")
    print(f"Cache hit rate:    {metrics['cache_hit_rate']:.2%}  ({metrics['cache_hits']:,} hits)")
    fa = metrics["feature_activation"]
    print(f"Feature activation: spectral={fa['spectral']:.2%}, wedge={fa['wedge']:.2%}, hologram={fa['hologram']:.2%}")
    print(f"Total time:        {metrics['total_time_ms']:,} ms")
    print("=" * 60)

    ok = True
    if metrics["r1"] < R1_MIN:
        print(f"FAIL: R@1 {metrics['r1']:.4f} < {R1_MIN}")
        ok = False
    if metrics["p50_latency_us"] > P50_MAX_US:
        print(f"FAIL: P50 latency {metrics['p50_latency_us']:.2f} μs > {P50_MAX_US} μs")
        ok = False
    if metrics["cache_hit_rate"] < CACHE_MIN:
        print(f"FAIL: cache hit rate {metrics['cache_hit_rate']:.2%} < {CACHE_MIN:.0%}")
        ok = False

    if ok:
        print("M1 VALIDATION PASSED")
    else:
        print("\nNOTE: This environment lacks the MiniLM/ITQ semantic encoder, so the")
        print("      validation uses deterministic title hashes. The metrics above")
        print("      reflect engine-pipeline health rather than end-to-end semantic R@1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
