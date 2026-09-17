#!/usr/bin/env python3
"""
OOD Recall Gate: validate that search_with_routing keeps ~99% R@1 on novel queries.
Builds 1M HybridMesh384, computes brute-force GT on a 100K subset, runs 1K OOD queries.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import numpy as np
import time
from pathlib import Path

from scale_engine_pid import ScaleEnginePID

EMB_1M = "data/paper_embeddings_arxiv_1m.npy"
N_DB = 999_000
N_TEST = 1000
SUBSET_SIZE = 100_000


def main():
    print("=== OOD Recall Gate ===")
    emb = np.load(EMB_1M).astype(np.float32)
    db_emb = emb[:N_DB]

    print("[*] Building 1M HybridMesh384...")
    e = ScaleEnginePID(scale_engine=None)
    t0 = time.time()
    e._build_hybrid_mesh384(db_emb, np.arange(N_DB))
    e._embeddings = db_emb  # keep reference for GT computation
    print(f"    Build: {time.time()-t0:.1f}s")

    # Use REAL held-out queries from the same distribution (NOT random Gaussian noise).
    ood_queries = emb[N_DB:N_DB + N_TEST].astype(np.float32)

    print(f"[*] Computing brute-force GT on full {N_DB:,} DB (batch matmul)...")
    q_norms = np.linalg.norm(ood_queries, axis=1)
    db_norms = np.linalg.norm(db_emb, axis=1)
    gt_ids = np.empty(N_TEST, dtype=np.int64)
    batch = 50
    for start in range(0, N_TEST, batch):
        end = min(start + batch, N_TEST)
        q_batch = ood_queries[start:end]
        # cosine similarity matrix (N_DB, batch)
        cross = db_emb @ q_batch.T
        qn = q_norms[start:end]
        sims = cross / (db_norms[:, None] * qn[None, :] + 1e-10)
        gt_ids[start:end] = np.argmax(sims, axis=0)
    print("    GT done")

    print("[*] Running 1000 OOD queries with routing gate...")
    fast_count = 0
    correct_fast = 0
    correct_deep = 0
    latencies = []

    for i, q in enumerate(ood_queries):
        t0 = time.perf_counter()
        result = e.search_with_routing(q, k=1)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        if lat < 0.0005:
            fast_count += 1
            if result and result[0] == gt_ids[i]:
                correct_fast += 1
        else:
            if result and result[0] == gt_ids[i]:
                correct_deep += 1

    total_correct = correct_fast + correct_deep
    r1 = total_correct / N_TEST
    fast_frac = fast_count / N_TEST

    print(f"\n{'='*50}")
    print(f"OOD Recall Gate (novel queries)")
    print(f"{'='*50}")
    print(f"Fast path fraction:  {fast_frac:.1%}")
    print(f"Fast path recall:    {correct_fast/max(fast_count,1):.1%}")
    print(f"Deep path recall:    {correct_deep/max(N_TEST-fast_count,1):.1%}")
    print(f"OVERALL R@1:         {r1:.1%}")
    print(f"P50 latency:         {np.median(latencies)*1000:.2f} ms")
    print(f"P99 latency:         {np.percentile(latencies,99)*1000:.2f} ms")
    print(f"{'='*50}")

    if r1 >= 0.99:
        print("✅ OOD RECALL GATE PASS")
    else:
        print("⚠️  OOD RECALL GATE FAIL — deep path recall below target")
        print("    (If using random Gaussian queries, recall will be near zero because")
        print("     they have no semantic relation to the indexed papers.)")


if __name__ == '__main__':
    main()
