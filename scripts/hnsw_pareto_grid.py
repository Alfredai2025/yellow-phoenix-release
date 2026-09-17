#!/usr/bin/env python3
"""
HNSW Phase 3: Overnight Pareto grid search.
Tests M x ef_search on 100K embeddings.
Saves frontier to logs/hnsw_pareto_frontier.json
"""

import numpy as np
import hnswlib
import time
import json
from pathlib import Path

EMB_PATH = "data/paper_embeddings_100k.npy"
OUT_PATH = "logs/hnsw_pareto_frontier.json"
N_QUERIES = 1000


def brute_force_nn(db, queries):
    """Vectorized exact L2 nearest neighbor."""
    db_norm = np.sum(db.astype(np.float32) ** 2, axis=1)
    gt = np.empty(queries.shape[0], dtype=np.int64)
    batch_q = 500
    for start in range(0, queries.shape[0], batch_q):
        end = min(start + batch_q, queries.shape[0])
        q = queries[start:end].astype(np.float32)
        q_norm = np.sum(q ** 2, axis=1)
        cross = db @ q.T
        dists = db_norm[:, None] + q_norm[None, :] - 2.0 * cross
        gt[start:end] = np.argmin(dists, axis=0)
    return gt


def benchmark(M, ef_search, db, queries, gt_nn):
    n_db, dim = db.shape
    n_q = queries.shape[0]

    # Build HNSW
    t0 = time.time()
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=n_db, ef_construction=200, M=M)
    index.add_items(db, np.arange(n_db))
    index.set_ef(ef_search)
    build_time = time.time() - t0

    # Query
    latencies = []
    correct = 0
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        labels, distances = index.knn_query(q, k=1)
        lat = time.perf_counter() - t0
        latencies.append(lat)
        if labels[0][0] == gt_nn[i]:
            correct += 1

    r1 = correct / n_q
    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    qps = n_q / np.sum(latencies)

    return {
        "M": M,
        "ef_search": ef_search,
        "build_s": round(build_time, 2),
        "r1": round(r1, 4),
        "p50_us": round(p50 * 1e6, 1),
        "p95_us": round(p95 * 1e6, 1),
        "qps": round(qps, 0),
    }


def main():
    print("[*] Loading embeddings...")
    emb = np.load(EMB_PATH)
    n_total = emb.shape[0]
    n_db = n_total - N_QUERIES

    db = emb[:n_db].astype(np.float32)
    queries = emb[n_db:].astype(np.float32)

    print(f"    DB: {n_db}, Queries: {N_QUERIES}, Dim: {emb.shape[1]}")

    print("[*] Computing ground truth (vectorized L2)...")
    t0 = time.time()
    gt_nn = brute_force_nn(db, queries)
    print(f"    Done in {time.time()-t0:.2f}s")

    Ms = [8, 16, 32, 64]
    efs = [50, 100, 200, 400]
    results = []

    total = len(Ms) * len(efs)
    done = 0

    for M in Ms:
        for ef in efs:
            done += 1
            print(f"\n[{done}/{total}] M={M}, ef_search={ef}")
            try:
                res = benchmark(M, ef, db, queries, gt_nn)
                results.append(res)
                print(f"    R@1={res['r1']:.4f}, P50={res['p50_us']:.1f}µs, QPS={res['qps']:.0f}, Build={res['build_s']:.1f}s")
            except Exception as e:
                print(f"    FAILED: {e}")
                results.append({"M": M, "ef_search": ef, "error": str(e)})

    Path(OUT_PATH).parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved to {OUT_PATH}")

    print("\n=== Pareto Frontier (R@1 vs P50 latency) ===")
    frontier = []
    valid = [r for r in results if "r1" in r and "p50_us" in r]
    for r in sorted(valid, key=lambda x: x["p50_us"]):
        dominated = False
        for other in valid:
            if other["p50_us"] <= r["p50_us"] and other["r1"] >= r["r1"] and (other["p50_us"] < r["p50_us"] or other["r1"] > r["r1"]):
                dominated = True
                break
        if not dominated:
            frontier.append(r)
            print(f"  M={r['M']:2d}, ef={r['ef_search']:3d}: R@1={r['r1']:.4f}, P50={r['p50_us']:7.1f}µs, Build={r['build_s']:.1f}s")

    frontier_path = Path(OUT_PATH).with_suffix('.frontier.json')
    with open(frontier_path, "w") as f:
        json.dump(frontier, f, indent=2)
    print(f"\n[+] Frontier saved to {frontier_path}")


if __name__ == "__main__":
    main()
