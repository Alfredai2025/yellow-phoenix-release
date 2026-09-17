#!/usr/bin/env python3
"""
End-to-End Production Benchmark: ScaleEnginePID.search_production()
Tests: 100K and 1M real embeddings via HybridMesh384 (HNSW384 + cosine rerank)
Metrics: R@1, P50/P95 latency, QPS, build time
"""

import numpy as np
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from scale_engine_pid import ScaleEnginePID

EMB_100K = "data/paper_embeddings_100k.npy"
EMB_1M = "data/paper_embeddings_arxiv_1m.npy"
ITQ_PATH = "data/itq_model_384.npz"
OUT_PATH = "logs/benchmark_production_e2e.json"


def brute_force_l2_nn(db, queries):
    """Vectorized exact L2 nearest neighbor."""
    gt = np.empty(len(queries), dtype=np.int64)
    batch = 100
    db_norm = np.sum(db.astype(np.float32) ** 2, axis=1)
    for start in range(0, len(queries), batch):
        end = min(start + batch, len(queries))
        q = queries[start:end].astype(np.float32)
        q_norm = np.sum(q ** 2, axis=1)
        cross = db @ q.T
        dists = db_norm[:, None] + q_norm[None, :] - 2.0 * cross
        gt[start:end] = np.argmin(dists, axis=0)
    return gt


def benchmark_scale(engine, db_emb, queries, gt_nn, k_final=1, k_fast=None):
    n_q = len(queries)
    latencies = []
    correct = 0

    for i in range(n_q):
        t0 = time.perf_counter()
        if k_fast is None:
            result = engine.search_production(queries[i], k=k_final)
        else:
            result = engine.search_production(queries[i], k=k_final, k_fast=k_fast)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        if result and result[0] == gt_nn[i]:
            correct += 1

    r1 = correct / n_q
    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50) * 1e6)
    p95 = float(np.percentile(latencies, 95) * 1e6)
    qps = n_q / np.sum(latencies)

    return {
        "r1": round(r1, 4),
        "p50_us": round(p50, 1),
        "p95_us": round(p95, 1),
        "qps": round(qps, 0),
        "n_queries": n_q,
        "n_db": len(db_emb),
    }


def run_benchmark(name, emb_path, n_db, n_q):
    print(f"\n{'='*60}")
    print(f"=== {name}: {n_db:,} DB + {n_q:,} Queries ===")
    print(f"{'='*60}")

    if not Path(emb_path).exists():
        print(f"[!] {emb_path} not found. Skipping.")
        return None

    # Load
    print("[*] Loading embeddings...")
    emb = np.load(emb_path).astype(np.float32)
    db_emb = emb[:n_db]
    q_emb = emb[n_db:n_db + n_q]

    # Ground truth
    print("[*] Computing L2 ground truth...")
    t0 = time.time()
    gt_nn = brute_force_l2_nn(db_emb, q_emb)
    print(f"    GT done in {time.time()-t0:.1f}s")

    # Build production engine
    print("[*] Building HybridMesh384...")
    engine = ScaleEnginePID(scale_engine=None)  # standalone
    t0 = time.time()
    engine._build_hybrid_mesh384(db_emb, np.arange(n_db))
    build_s = time.time() - t0
    print(f"    Build: {build_s:.1f}s")

    # Benchmark
    print("[*] Running production queries (default k_fast)...")
    result = benchmark_scale(engine, db_emb, q_emb, gt_nn, k_final=1)
    result["build_s"] = round(build_s, 1)
    result["scale"] = name

    print(f"\n=== {name} Results ===")
    print(f"R@1:  {result['r1']:.4f}")
    print(f"P50:  {result['p50_us']:.0f} µs")
    print(f"P95:  {result['p95_us']:.0f} µs")
    print(f"QPS:  {result['qps']:.0f}")
    print(f"Build: {result['build_s']:.1f}s")

    return result


def main():
    results = {}

    # 100K
    res_100k = run_benchmark("100K", EMB_100K, 99_000, 1000)
    if res_100k:
        results["100k"] = res_100k

    # 1M (if available)
    res_1m = run_benchmark("1M", EMB_1M, 999_000, 1000)
    if res_1m:
        results["1m"] = res_1m

    # Save
    Path(OUT_PATH).parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Saved to {OUT_PATH}")
    print("\n=== Summary ===")
    for scale, res in results.items():
        print(f"{scale:6s}: R@1={res['r1']:.4f}  P50={res['p50_us']:7.0f}µs  Build={res['build_s']:5.1f}s")


if __name__ == '__main__':
    main()
