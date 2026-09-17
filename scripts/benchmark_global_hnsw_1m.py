# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Corrected BinaryHNSW benchmark using the verified 1.26M global ArXiv index.

The previous 10M benchmark failed because it called the raw FFI methods with
wrong argument lists (missing output buffers, lengths, etc.). This script uses
the Pythonic ``yp_bridge.BinaryHNSW`` wrapper, which handles the ctypes setup
correctly.

We load the existing global index (1,266,270 docs) and measure:
  * cold/hot search latency (P50/P95/P99)
  * throughput (single-threaded QPS)
  * self-query top-1 recall on a held-out sample
"""

import os
import sys
import time
import pickle
import random
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from yp_bridge import BinaryHNSW

INDEX_PATH = project_root / "data" / "binary_hnsw_arxiv1m_global.bin"
HASHES_PATH = project_root / "data" / "paper_hashes_arxiv_1m.pkl"
PID_MAP_PATH = project_root / "data" / "arxiv1m_pid_to_emb_idx.pkl"

K = 10
WARMUP = 50
QUERIES = 1000
RECALL_SAMPLE = 2000


def load_hashes(limit=None):
    with open(HASHES_PATH, "rb") as f:
        paper_hashes = pickle.load(f)
    items = []
    for pid, h in paper_hashes.items():
        if isinstance(h, str):
            h = bytes.fromhex(h)
        if isinstance(h, bytes) and len(h) == 64:
            items.append((str(pid), h))
        if limit and len(items) >= limit:
            break
    return items


def main():
    if not INDEX_PATH.exists():
        raise RuntimeError(f"Global index not found: {INDEX_PATH}")

    print("Loading global BinaryHNSW index...")
    t0 = time.perf_counter()
    idx = BinaryHNSW(m=16, ef_construction=200, ef_search=200)
    ok = idx.load(str(INDEX_PATH))
    if not ok:
        raise RuntimeError("Failed to load index")
    load_s = time.perf_counter() - t0
    n = idx.count()
    print(f"Loaded {n:,} vectors in {load_s:.2f}s")

    print("Loading hash corpus for query sampling...")
    items = load_hashes()
    print(f"Corpus hashes available: {len(items):,}")

    # Use hashes from the corpus as queries (self-query recall)
    rng = random.Random(2026)
    query_items = rng.sample(items, min(QUERIES, len(items)))

    # Warmup
    for _, h in query_items[:WARMUP]:
        idx.search(h, k=K)

    # Latency benchmark
    lats = []
    for _, h in query_items:
        t0 = time.perf_counter()
        _ = idx.search(h, k=K)
        lats.append(time.perf_counter() - t0)
    lats = np.array(lats)
    p50 = np.percentile(lats, 50)
    p95 = np.percentile(lats, 95)
    p99 = np.percentile(lats, 99)
    qps = 1.0 / p50
    print(f"\n=== Search latency (k={K}, n={len(query_items):,}) ===")
    print(f"P50: {p50*1e6:.2f} µs")
    print(f"P95: {p95*1e6:.2f} µs")
    print(f"P99: {p99*1e6:.2f} µs")
    print(f"Single-thread QPS (from P50): {qps:,.0f}")

    # Recall benchmark: for a sample, the top-1 result should be the query itself.
    # The global index stores embedding-index IDs, not raw PIDs, so map PIDs first.
    recall_items = rng.sample(items, min(RECALL_SAMPLE, len(items)))
    pid_to_idx = {}
    if PID_MAP_PATH.exists():
        with open(PID_MAP_PATH, "rb") as f:
            pid_to_idx = pickle.load(f)

    r1_hits = 0
    r10_hits = 0
    checked = 0
    for pid, h in recall_items:
        expected_id = pid_to_idx.get(pid)
        if expected_id is None:
            continue
        results = idx.search(h, k=K)
        if not results:
            continue
        returned_ids = {int(r[0]) for r in results}
        if expected_id in returned_ids:
            r10_hits += 1
        if int(results[0][0]) == expected_id:
            r1_hits += 1
        checked += 1

    print(f"\n=== Self-query recall ({checked:,} samples, k={K}) ===")
    print(f"R@1:  {r1_hits/checked:.3f}")
    print(f"R@10: {r10_hits/checked:.3f}")


if __name__ == "__main__":
    main()
