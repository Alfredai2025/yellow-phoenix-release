# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark ITQ hash generation from embeddings."""
import numpy as np
import time

DIM = 384
HASH_BITS = 384
BATCH_SIZES = [1, 10, 100, 1_000, 10_000]
N_TRIALS = 100
np.random.seed(42)

print("=" * 60)
print("YELLOW PHOENIX -- ITQ Hash Generation")
print("=" * 60)

emb = np.random.randn(100_000, DIM).astype(np.float32)
emb /= np.linalg.norm(emb, axis=1, keepdims=True)
mean = emb.mean(axis=0)
X = emb - mean
_, S, Vt = np.linalg.svd(X.T @ X / len(emb))
W_pca = Vt[:HASH_BITS].T / np.sqrt(S[:HASH_BITS] + 1e-10)
R, _ = np.linalg.qr(np.random.randn(HASH_BITS, HASH_BITS))

for batch_size in BATCH_SIZES:
    queries = np.random.randn(batch_size, DIM).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    _ = np.packbits(((queries - mean) @ W_pca @ R >= 0).astype(np.uint8), axis=1).view(np.uint64)

    times = np.empty(N_TRIALS, dtype=np.int64)
    for i in range(N_TRIALS):
        t0 = time.perf_counter_ns()
        hashes = np.packbits(((queries - mean) @ W_pca @ R >= 0).astype(np.uint8), axis=1).view(np.uint64)
        t1 = time.perf_counter_ns()
        times[i] = t1 - t0

    us = times / 1000.0
    per_query = us.mean() / batch_size
    print("Batch {:>5} | Total: {:>7.1f} us | Per-query: {:>7.3f} us | QPS: {:>10,.0f}".format(
        batch_size, us.mean(), per_query, 1e6/per_query))

print("\n" + "=" * 60)
