# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark O(1) direct hash lookup (Path A)."""
import numpy as np
import time

SCALES = [100_000, 1_000_000, 10_000_000, 20_000_000]
DIM = 384
HASH_BITS = 384
BUCKET_BITS = 20
N_QUERIES = 10_000
WARMUP = 1_000
np.random.seed(42)

print("=" * 60)
print("YELLOW PHOENIX -- Path A: Direct Hash Lookup")
print("=" * 60)

for N_DOCS in SCALES:
    emb = np.random.randn(N_DOCS, DIM).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    mean = emb.mean(axis=0)
    X = emb - mean
    _, S, Vt = np.linalg.svd(X.T @ X / N_DOCS)
    W_pca = Vt[:HASH_BITS].T / np.sqrt(S[:HASH_BITS] + 1e-10)
    R, _ = np.linalg.qr(np.random.randn(HASH_BITS, HASH_BITS))
    packed = np.packbits((X @ W_pca @ R >= 0).astype(np.uint8), axis=1).view(np.uint64)

    bucket_mask = (1 << BUCKET_BITS) - 1
    buckets = {}
    for i in range(N_DOCS):
        buckets.setdefault(int(packed[i, 0] & bucket_mask), []).append(i)

    q = np.random.randn(N_QUERIES, DIM).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    Q = np.packbits(((q - mean) @ W_pca @ R >= 0).astype(np.uint8), axis=1).view(np.uint64)

    for i in range(WARMUP):
        _ = buckets.get(int(Q[i % N_QUERIES, 0] & bucket_mask), [])

    times = np.empty(N_QUERIES, dtype=np.int64)
    for i in range(N_QUERIES):
        t0 = time.perf_counter_ns()
        _ = buckets.get(int(Q[i, 0] & bucket_mask), [])
        times[i] = time.perf_counter_ns() - t0

    us = times / 1000.0
    print("\n{:>10,} docs | P50: {:>6.3f} us | P99: {:>6.3f} us | Mean: {:>6.3f} us | QPS: {:>10,.0f}".format(
        N_DOCS, np.percentile(us, 50), np.percentile(us, 99), us.mean(), 1e6/us.mean()))

print("\n" + "=" * 60)
