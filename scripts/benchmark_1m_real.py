#!/usr/bin/env python3
"""
1M Real Benchmark: single index with M=32, ef=200 (production sweet spot).
"""

import numpy as np
import hnswlib
import time
import json
from pathlib import Path

EMB_1M = "data/paper_embeddings_arxiv_1m.npy"
OUT_PATH = "logs/benchmark_1m_real.json"


def main():
    if not Path(EMB_1M).exists():
        print(f"[!] {EMB_1M} not found.")
        return

    print("[*] Loading 1M embeddings...")
    emb = np.load(EMB_1M).astype(np.float32)
    n_db = 999_000
    n_q = 1000
    db = emb[:n_db]
    queries = emb[n_db:n_db + n_q]

    print(f"    DB: {n_db}, Queries: {n_q}, Dim: {db.shape[1]}")

    # Build single index (production params)
    print("[*] Building hnswlib M=32, ef_construction=200...")
    t0 = time.time()
    index = hnswlib.Index(space='l2', dim=db.shape[1])
    index.init_index(max_elements=n_db, ef_construction=200, M=32)
    index.add_items(db, np.arange(n_db), num_threads=-1)
    index.set_ef(200)
    build_s = time.time() - t0
    print(f"    Build: {build_s:.1f}s")

    # Ground truth (vectorized, batch 100)
    print("[*] Ground truth...")
    gt = np.empty(n_q, dtype=np.int64)
    db_norm = np.sum(db ** 2, axis=1)
    for start in range(0, n_q, 100):
        end = min(start + 100, n_q)
        q = queries[start:end]
        q_norm = np.sum(q ** 2, axis=1)
        cross = db @ q.T
        dists = db_norm[:, None] + q_norm[None, :] - 2.0 * cross
        gt[start:end] = np.argmin(dists, axis=0)
    print("    Done.")

    # Query
    print("[*] Querying...")
    latencies = []
    correct = 0
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        labels, distances = index.knn_query(q, k=1)
        lat = time.perf_counter() - t0
        latencies.append(lat)
        if labels[0][0] == gt[i]:
            correct += 1

    r1 = correct / n_q
    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50) * 1e6)
    p95 = float(np.percentile(latencies, 95) * 1e6)
    qps = n_q / np.sum(latencies)

    print(f"\n=== 1M Real Benchmark ===")
    print(f"R@1:  {r1:.4f}")
    print(f"P50:  {p50:.0f} µs")
    print(f"P95:  {p95:.0f} µs")
    print(f"QPS:  {qps:.0f}")
    print(f"Build: {build_s:.1f}s")

    Path(OUT_PATH).parent.mkdir(exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump({
            "scale": "1M", "M": 32, "ef": 200,
            "r1": round(r1, 4), "p50_us": round(p50, 1),
            "p95_us": round(p95, 1), "qps": round(qps, 0),
            "build_s": round(build_s, 1),
            "mode": "single", "note": "Production sweet spot"
        }, f, indent=2)
    print(f"[+] Saved to {OUT_PATH}")


if __name__ == '__main__':
    main()
