#!/usr/bin/env python3
"""
M3.4 Validation Benchmark: self-retrieval on real papers via sharded mesh.

Uses data/phoenix_arxiv_1m.db (≈12.7k papers). Queries are precomputed 512-bit
hashes stored in yp_hash512, so the benchmark measures the pure sharded
lookup path (encode time excluded). A small live-encode sanity sample is
also run via search_sharded().
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yp_bridge import YPEngine

DB_PATH = ROOT / "data/phoenix_arxiv_1m.db"
REPORT_PATH = ROOT / "data" / "benchmark_m3_4_sharded.json"
N_QUERIES = 10_000
TOP_K = 5
TARGET_R1 = 0.97


def load_papers():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, yp_hash512 FROM papers "
        "WHERE yp_hash512 IS NOT NULL AND yp_hash512 != '' AND title IS NOT NULL"
    )
    rows = [
        {"pid": pid, "title": title.strip(), "hash512": h512}
        for pid, title, h512 in cur.fetchall()
    ]
    conn.close()
    return rows


def main():
    print(f"Loading papers from {DB_PATH} ...")
    papers = load_papers()
    print(f"  {len(papers)} papers with hashes")
    if len(papers) < 100:
        print("ERROR: not enough papers")
        sys.exit(1)

    print("Initializing YPEngine ...")
    engine = YPEngine()
    print(f"  cache={len(engine.cache)}  rust_id_map={len(engine.rust_id_map)}")

    cold_path = ROOT / ".tmp" / "sharded_cold.bin"
    cold_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Enabling sharding -> {cold_path} ...")
    rc = engine.rust.enable_sharding(cold_path=str(cold_path), num_shards=10)
    print(f"  enable_sharding rc={rc}")

    # Select every Nth paper so we cover the whole dataset up to N_QUERIES.
    step = max(1, len(papers) // N_QUERIES)
    queries = papers[::step][:N_QUERIES]
    print(f"Running {len(queries)} sharded-hash self-retrieval queries ...")

    latencies = []
    correct_r1 = 0
    correct_r5 = 0
    misses = 0

    for i, paper in enumerate(queries):
        t0 = time.perf_counter()
        try:
            results = engine.search_sharded_hash(paper["hash512"], top_k=TOP_K)
        except Exception as e:
            print(f"  query {i} failed: {e}")
            misses += 1
            continue
        lat = (time.perf_counter() - t0) * 1e6  # microseconds
        latencies.append(lat)

        found_r1 = False
        found_r5 = False
        for rank, (score, (pid, title)) in enumerate(results[:TOP_K]):
            if pid == paper["pid"]:
                if rank == 0:
                    found_r1 = True
                found_r5 = True
                break
        if found_r1:
            correct_r1 += 1
        if found_r5:
            correct_r5 += 1

        if (i + 1) % 1000 == 0:
            n_ok = len(latencies)
            print(
                f"  ...{i+1}/{len(queries)}  "
                f"R@1={correct_r1/n_ok:.4f}  p50={sorted(latencies)[n_ok//2]:.2f}µs"
            )

    n = len(latencies)
    if n == 0:
        print("ERROR: no successful queries")
        sys.exit(1)

    latencies.sort()
    report = {
        "db": str(DB_PATH),
        "total_papers": len(papers),
        "queries_run": len(queries),
        "queries_success": n,
        "queries_failed": misses,
        "r@1": correct_r1 / n,
        "r@5": correct_r5 / n,
        "target_r1": TARGET_R1,
        "pass_r1": (correct_r1 / n) >= TARGET_R1,
        "p50_us": latencies[n // 2],
        "p99_us": latencies[int(n * 0.99)],
        "mean_us": sum(latencies) / n,
        "min_us": latencies[0],
        "max_us": latencies[-1],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 50)
    print("M3.4 SHARDED VALIDATION RESULT")
    print("=" * 50)
    print(json.dumps(report, indent=2))
    print("=" * 50)
    print(f"Report saved: {REPORT_PATH}")

    if report["pass_r1"]:
        print("✅ PASS: R@1 >= 97%")
    else:
        print(f"❌ FAIL: R@1 = {report['r@1']:.4f}")

    # Live-encode sanity on a tiny sample.
    print("\nLive-encode sanity (search_sharded) on 5 titles ...")
    live_ok = 0
    for paper in queries[:5]:
        results = engine.search_sharded(paper["title"], top_k=1)
        if results and results[0][1][0] == paper["pid"]:
            live_ok += 1
    print(f"  {live_ok}/5 live queries returned the correct paper")


if __name__ == "__main__":
    main()
