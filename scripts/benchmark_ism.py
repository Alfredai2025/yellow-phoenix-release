# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark ISM: Inverted Slot Map."""
import numpy as np
import time

SCALES = [1_000_000, 10_000_000, 100_000_000, 200_000_000]
HASH_BITS = 384
SLOT_COUNT = 1_048_576
N_QUERIES = 10_000
WARMUP = 1_000
np.random.seed(42)

print("=" * 60)
print("YELLOW PHOENIX -- ISM: Inverted Slot Map")
print("=" * 60)

for N_DOCS in SCALES:
    hashes = np.random.randint(0, 2**HASH_BITS, size=N_DOCS, dtype=np.uint64)

    t0 = time.time()
    slots = [[] for _ in range(SLOT_COUNT)]
    for i, h in enumerate(hashes):
        slots[h % SLOT_COUNT].append(i)
    slot_arrays = [np.array(s, dtype=np.int32) if s else np.array([], dtype=np.int32) for s in slots]
    build_time = time.time() - t0

    q_hashes = np.random.randint(0, 2**HASH_BITS, size=N_QUERIES, dtype=np.uint64)

    for i in range(WARMUP):
        slot_idx = int(q_hashes[i % N_QUERIES] % SLOT_COUNT)
        _ = slot_arrays[slot_idx]

    times = np.empty(N_QUERIES, dtype=np.int64)
    for i in range(N_QUERIES):
        t0 = time.perf_counter_ns()
        slot_idx = int(q_hashes[i] % SLOT_COUNT)
        candidates = slot_arrays[slot_idx]
        t1 = time.perf_counter_ns()
        times[i] = t1 - t0

    us = times / 1000.0
    print("\n{:>10,} docs | Build: {:>5.1f}s | P50: {:>6.3f} us | P99: {:>6.3f} us | Mean: {:>6.3f} us | QPS: {:>10,.0f}".format(
        N_DOCS, build_time, np.percentile(us, 50), np.percentile(us, 99), us.mean(), 1e6/us.mean()))

print("\n" + "=" * 60)
