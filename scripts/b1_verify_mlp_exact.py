#!/usr/bin/env python3
"""B1 verification: exact self-retrieval on global mesh with fresh MLP hashes."""
import json
import os
import random
import sqlite3
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from yp_bridge import RustBridge

DB_PATH = os.path.join(ROOT, "data/phoenix_arxiv_1m.db")
HASH_PATH = os.path.join(ROOT, "hashes_12702_mlp.npy")
COLD_PATH = os.path.join(ROOT, ".tmp", "b1_sharded_cold.bin")

def load_rows():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title FROM papers WHERE title IS NOT NULL AND title != '' ORDER BY rowid"
    )
    rows = [(pid, title.strip()) for pid, title in cur.fetchall() if title and len(title.strip()) > 3]
    conn.close()
    return rows

def main():
    print("Loading rows from DB...")
    rows = load_rows()
    N = len(rows)
    print(f"Rows: {N}")

    print(f"Loading hashes from {HASH_PATH}...")
    hash_bytes = np.load(HASH_PATH)
    assert hash_bytes.shape[0] >= N, f"{hash_bytes.shape[0]} hashes < {N} rows"
    hash_bytes = hash_bytes[:N]
    print(f"Hashes: {hash_bytes.shape}")

    print("Inserting into global HYBRID_MESH...")
    bridge = RustBridge()
    t0 = time.time()
    for i in range(N):
        h = hash_bytes[i]
        coarse = h[:16].tobytes()
        fine = h[:64].tobytes()
        bridge.lib.yp_insert_to_mesh_bytes(i + 1, coarse, len(coarse), fine, len(fine))
        if i % 2000 == 0 and i > 0:
            print(f"  {i // 1000}K inserted ({time.time() - t0:.1f}s)")
    print(f"  Insert done in {time.time() - t0:.1f}s")
    print(f"  mesh_count={bridge.lib.yp_mesh_count()}")

    print("Enabling sharding...")
    os.makedirs(os.path.dirname(COLD_PATH), exist_ok=True)
    rc = bridge.enable_sharding(cold_path=COLD_PATH, num_shards=10, pre_fault=False)
    print(f"  enable_sharding rc={rc}")

    print("Testing exact self-retrieval on 1000 queries...")
    random.seed(42)
    queries = list(range(N))
    random.shuffle(queries)
    queries = queries[:1000]

    latencies = []
    correct = 0
    misses = 0
    for i, qi in enumerate(queries):
        hhex = hash_bytes[qi].tobytes().hex()
        t0 = time.perf_counter()
        raw = bridge.query_sharded(hhex)
        lat = (time.perf_counter() - t0) * 1e6
        latencies.append(lat)

        results = raw.get("results", [])
        if results and int(results[0]["id"]) == qi + 1:
            correct += 1
        else:
            misses += 1
        if (i + 1) % 250 == 0:
            print(
                f"  {i + 1}/{len(queries)}  "
                f"R@1={correct / (i + 1):.4f}  p50={sorted(latencies)[len(latencies) // 2]:.2f}µs"
            )

    n = len(queries)
    print(f"\nExact self-retrieval: {correct}/{n} = {100.0 * correct / n:.2f}%")
    print(f"Misses: {misses}")
    print(f"P50 latency: {sorted(latencies)[n // 2]:.2f} µs")
    print(f"P95 latency: {np.percentile(latencies, 95):.2f} µs")

if __name__ == "__main__":
    main()
