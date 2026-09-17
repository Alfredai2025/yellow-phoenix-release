#!/usr/bin/env python3
"""
Phase 5: End-to-end production benchmark.
Tests ScaleEngine (production API): hnswlib fast ANN + exact cosine re-rank.
"""

import numpy as np
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phoenix_bench_api import ScaleEngine

EMB_100K = "data/paper_embeddings_100k.npy"
EMB_1M = "data/paper_embeddings_arxiv_1m.npy"
OUT_PATH = "logs/benchmark_phase5.json"


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-10)


def brute_force_nn(db, queries):
    """Exact nearest neighbor on normalized vectors (matches cosine)."""
    db_norm = normalize(db)
    q_norm = normalize(queries)
    gt = np.empty(queries.shape[0], dtype=np.int64)
    batch_q = 500
    for start in range(0, queries.shape[0], batch_q):
        end = min(start + batch_q, queries.shape[0])
        q = q_norm[start:end]
        cross = db_norm @ q.T
        # L2^2 on normalized = 2 - 2*cosine, so argmin L2 == argmax cosine
        dists = 2.0 - 2.0 * cross
        gt[start:end] = np.argmin(dists, axis=0)
    return gt


def extract_ids(results, exclude_id=None):
    ids = []
    for r in results:
        if isinstance(r, tuple) and len(r) == 2:
            payload = r[1]
            if isinstance(payload, tuple) and len(payload) >= 1:
                try:
                    doc_id = int(payload[0])
                    if exclude_id is None or doc_id != exclude_id:
                        ids.append(doc_id)
                except Exception:
                    pass
    return ids


def benchmark(engine, db, queries, gt_nn, k=1, scale="100k"):
    n_q = queries.shape[0]
    latencies = []
    correct = 0

    offset = db.shape[0]  # query indices in full file start here

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        # Ask for k+1 so we can drop the query's own ID when it appears in index
        results = engine.search_hybrid_by_embedding(q, k=k + 1, scale=scale)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        result_ids = extract_ids(results, exclude_id=offset + i)
        if gt_nn[i] in result_ids[:k]:
            correct += 1

    latencies = np.array(latencies)
    return {
        "scale": scale,
        "n_db": int(db.shape[0]),
        "n_queries": int(n_q),
        "k": int(k),
        "r1": float(correct / n_q),
        "p50_us": float(np.percentile(latencies, 50) * 1e6),
        "p95_us": float(np.percentile(latencies, 95) * 1e6),
        "qps": float(n_q / np.sum(latencies)),
        "total_query_s": float(np.sum(latencies)),
    }


def run_scale(path, n_db, n_q, scale):
    print(f"\n=== Phase 5: {scale} End-to-End Benchmark ===")
    emb = np.load(path).astype(np.float32)
    db = emb[:n_db]
    queries = emb[n_db:n_db + n_q]
    print(f"    DB: {n_db}, Queries: {n_q}, Dim: {db.shape[1]}")

    engine = ScaleEngine()

    # Warmup / index build
    print("[*] Warming up (building hnswlib index)...")
    t0 = time.time()
    _ = engine.search_hybrid_by_embedding(db[0], k=1, scale=scale)
    warmup = time.time() - t0
    print(f"    Warmup/build: {warmup:.1f}s")

    # Ground truth (normalized cosine)
    print("[*] Computing ground truth...")
    t0 = time.time()
    gt_nn = brute_force_nn(db, queries)
    print(f"    GT computed in {time.time()-t0:.2f}s")

    # Benchmark
    print("[*] Benchmarking engine queries...")
    result = benchmark(engine, db, queries, gt_nn, k=1, scale=scale)

    print(f"\n  R@1:        {result['r1']:.4f}")
    print(f"  P50:        {result['p50_us']:.1f} µs")
    print(f"  P95:        {result['p95_us']:.1f} µs")
    print(f"  QPS:        {result['qps']:.0f}")
    print(f"  Query time: {result['total_query_s']:.2f}s")
    return result


def main():
    results = {}

    if Path(EMB_100K).exists():
        results["100k"] = run_scale(EMB_100K, 99000, 1000, "100k")

    if Path(EMB_1M).exists():
        results["1m"] = run_scale(EMB_1M, 999000, 100, "1m")

    Path(OUT_PATH).parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
