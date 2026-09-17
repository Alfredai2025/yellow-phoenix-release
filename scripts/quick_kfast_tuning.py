#!/usr/bin/env python3
"""
Quick microbenchmark: single 100k HybridMesh384 shard at k_fast=1000 vs 500.
No recall computation unless fast; focuses on latency.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.scale_engine_pid import ScaleEnginePID

EMB_PATH = "data/paper_embeddings_arxiv_1m.npy"
N_DB = 100_000
N_Q = 1_000


def main():
    if not os.path.exists(EMB_PATH):
        print(f"[!] {EMB_PATH} not found")
        return

    print("Loading embeddings...")
    emb = np.load(EMB_PATH).astype(np.float32)
    db = emb[:N_DB]
    queries = emb[N_DB:N_DB + N_Q]
    ids = np.arange(N_DB, dtype=np.uint64)
    print(f"DB: {len(db)}, queries: {len(queries)}, dim: {db.shape[1]}")

    pid = ScaleEnginePID(scale_engine=None)
    print("Building 100k HybridMesh384 shard...")
    t0 = time.time()
    pid._build_hybrid_mesh384(db, ids, n_probes=1)
    print(f"Build done in {time.time() - t0:.1f}s")

    for k_fast in (1000, 500):
        lats = []
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            pid.search_production(q, k=10, k_fast=k_fast)
            lats.append(time.perf_counter() - t0)
        lats = np.array(lats)
        print(f"\nk_fast={k_fast}:")
        print(f"  P50 = {np.median(lats)*1000:.2f} ms")
        print(f"  P99 = {np.percentile(lats, 99)*1000:.2f} ms")
        print(f"  mean = {np.mean(lats)*1000:.2f} ms")


if __name__ == "__main__":
    main()
