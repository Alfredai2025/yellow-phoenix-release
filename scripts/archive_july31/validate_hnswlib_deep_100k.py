#!/usr/bin/env python3
"""
Quick 100k validation of exact-hash fast path + hnswlib deep path.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.scale_engine_pid import ScaleEnginePID

N = 100_000
D = 384
N_Q = 1_000


def main():
    print("Generating 100k normalized synthetic embeddings...")
    rng = np.random.default_rng(2026)
    emb = rng.standard_normal((N, D), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    ids = np.arange(N, dtype=np.uint64)

    print("Building ScaleEnginePID...")
    e = ScaleEnginePID(scale_engine=None)
    t0 = time.time()
    e._build_hybrid_mesh384(emb, ids, hashes=None, n_probes=1)
    print(f"Build: {time.time() - t0:.1f}s")

    # Exact-match query
    q_exact = emb[42].copy()
    t0 = time.perf_counter()
    r_exact = e.search_production(q_exact, k=1)
    t_exact = (time.perf_counter() - t0) * 1000
    print(f"\nExact match: {t_exact:.2f} ms -> {r_exact}")

    # Warm hnswlib
    q_warm = rng.standard_normal(D).astype(np.float32)
    q_warm /= np.linalg.norm(q_warm) + 1e-10
    t0 = time.perf_counter()
    e.search_production(q_warm, k=10)
    t_warm = (time.perf_counter() - t0) * 1000
    print(f"First deep query (includes index build): {t_warm:.1f} ms")

    queries = rng.standard_normal((N_Q, D)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10
    lats = []
    for q in queries:
        t0 = time.perf_counter()
        e.search_production(q, k=10)
        lats.append(time.perf_counter() - t0)
    lats = np.array(lats)
    print(f"\nNovel queries (n={N_Q}):")
    print(f"  P50 = {np.median(lats)*1000:.2f} ms")
    print(f"  P99 = {np.percentile(lats, 99)*1000:.2f} ms")
    print(f"  mean = {np.mean(lats)*1000:.2f} ms")


if __name__ == "__main__":
    main()
