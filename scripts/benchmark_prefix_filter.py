# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark dynamic prefix filter: top-5% bucket selection."""
import numpy as np
import time

N_DOCS = 1_000_000
HASH_BITS = 384
PREFIX_BITS = 64
N_BUCKETS = 2**20
N_QUERIES = 1_000
TOP_K_PERCENT = 0.05
np.random.seed(42)

print("=" * 60)
print("YELLOW PHOENIX -- Dynamic Prefix Filter")
print("=" * 60)

hashes = np.random.randint(0, 2**64, size=N_DOCS, dtype=np.uint64)

buckets = {}
for i, h in enumerate(hashes):
    key = int(h & (N_BUCKETS - 1))
    buckets.setdefault(key, []).append(i)

q_hashes = np.random.randint(0, 2**64, size=N_QUERIES, dtype=np.uint64)

print("\nBenchmarking {:,} queries, top-{}% filter...".format(N_QUERIES, int(TOP_K_PERCENT*100)))
times = np.empty(N_QUERIES, dtype=np.int64)
recalls = []

for i in range(N_QUERIES):
    q = q_hashes[i]

    t0 = time.perf_counter_ns()
    scores = {}
    for b_key, b_docs in buckets.items():
        score = bin(int(q) ^ int(b_key)).count('1')
        scores[b_key] = score

    sorted_buckets = sorted(scores.items(), key=lambda x: x[1])
    n_select = max(1, int(len(sorted_buckets) * TOP_K_PERCENT))
    selected = [k for k, _ in sorted_buckets[:n_select]]
    t1 = time.perf_counter_ns()
    times[i] = t1 - t0

    true_nn = min(buckets.keys(), key=lambda k: bin(int(q) ^ int(k)).count('1'))
    recalls.append(1.0 if true_nn in selected else 0.0)

us = times / 1000.0
print("\nP50: {:>6.1f} us | P99: {:>6.1f} us | Mean: {:>6.1f} us".format(
    np.percentile(us, 50), np.percentile(us, 99), us.mean()))
print("Recall: {:.1f}% (target: 98-99.8%)".format(np.mean(recalls) * 100))
print("Candidates: ~{:,} per query".format(int(N_BUCKETS * TOP_K_PERCENT)))

print("\n" + "=" * 60)
