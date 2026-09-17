# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark HNSW + cosine re-rank (Path B)."""
import numpy as np
import time
import sys

try:
    import hnswlib
except ImportError:
    print("Installing hnswlib...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hnswlib", "-q"])
    import hnswlib

SCALES = [100_000, 1_000_000]
DIM = 384
M = 16
EF_CONSTRUCT = 200
EF_SEARCH = 50
K = 10
N_QUERIES = 1_000
WARMUP = 100
np.random.seed(42)

print("=" * 60)
print("YELLOW PHOENIX -- Path B: HNSW + Cosine Re-rank")
print("=" * 60)

for N_DOCS in SCALES:
    emb = np.random.randn(N_DOCS, DIM).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    print("\nBuilding HNSW index for {:,} docs...".format(N_DOCS))
    t0 = time.time()
    index = hnswlib.Index(space='ip', dim=DIM)
    index.init_index(max_elements=N_DOCS, ef_construction=EF_CONSTRUCT, M=M)
    index.add_items(emb)
    index.set_ef(EF_SEARCH)
    build_time = time.time() - t0
    print("  Build: {:.1f}s".format(build_time))

    queries = np.random.randn(N_QUERIES, DIM).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    for i in range(WARMUP):
        labels, distances = index.knn_query(queries[i % N_QUERIES], k=K+10)

    times = np.empty(N_QUERIES, dtype=np.int64)
    for i in range(N_QUERIES):
        t0 = time.perf_counter_ns()
        labels, distances = index.knn_query(queries[i], k=K+10)
        t1 = time.perf_counter_ns()
        times[i] = t1 - t0

    us = times / 1000.0
    print("{:>10,} docs | Build: {:>5.1f}s | P50: {:>6.1f} us | P99: {:>6.1f} us | Mean: {:>6.1f} us".format(
        N_DOCS, build_time, np.percentile(us, 50), np.percentile(us, 99), us.mean()))

print("\n" + "=" * 60)
