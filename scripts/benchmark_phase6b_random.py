#!/usr/bin/env python3
"""
Phase 6b Benchmark: RandomShardedIndex on 1M embeddings.
Target: 99% R@1, build <60s, query <2ms P50.
"""

import numpy as np
import time
import json
from pathlib import Path
from sharded_index_v2 import RandomShardedIndex

EMB_1M = "data/paper_embeddings_arxiv_1m.npy"
OUT_PATH = "logs/benchmark_phase6b_random.json"


def brute_force_nn(db, queries):
    """Vectorized exact L2 nearest neighbor."""
    db_norm = np.sum(db.astype(np.float32) ** 2, axis=1)
    gt = np.empty(queries.shape[0], dtype=np.int64)
    batch_q = 100
    for start in range(0, queries.shape[0], batch_q):
        end = min(start + batch_q, queries.shape[0])
        q = queries[start:end].astype(np.float32)
        q_norm = np.sum(q ** 2, axis=1)
        cross = db @ q.T
        dists = db_norm[:, None] + q_norm[None, :] - 2.0 * cross
        gt[start:end] = np.argmin(dists, axis=0)
    return gt


def benchmark(idx, db, queries, gt_nn, k_per_shard=200, k_final=1):
    n_q = len(queries)
    latencies = []
    correct = 0

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        ids, dists = idx.search(q, k_per_shard=k_per_shard, k_final=k_final)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        if len(ids) > 0 and int(ids[0]) == int(gt_nn[i]):
            correct += 1

    r1 = correct / n_q
    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50) * 1e6)
    p95 = float(np.percentile(latencies, 95) * 1e6)
    qps = n_q / np.sum(latencies)

    return {
        "n_shards": idx.n_shards,
        "k_per_shard": k_per_shard,
        "M": idx.M,
        "ef_search": idx.ef_search,
        "r1": round(r1, 4),
        "p50_us": round(p50, 1),
        "p95_us": round(p95, 1),
        "qps": round(qps, 0),
    }


def main():
    if not Path(EMB_1M).exists():
        print(f"[!] {EMB_1M} not found. Skipping.")
        return

    print("=== Phase 6b: 1M Random Sharding Benchmark ===")
    emb = np.load(EMB_1M).astype(np.float32)
    n_total = len(emb)
    n_db = 999_000
    n_q = 1000
    db = emb[:n_db]
    queries = emb[n_db : n_db + n_q]

    print(f"[*] DB: {n_db}, Queries: {n_q}, Dim: {emb.shape[1]}")

    # Ground truth
    print("[*] Computing ground truth (vectorized L2)...")
    t0 = time.time()
    gt_nn = brute_force_nn(db, queries)
    print(f"    GT computed in {time.time()-t0:.2f}s")

    # Configs: random sharding with various k_per_shard
    configs = [
        {"n_shards": 4, "k_per_shard": 200, "M": 16, "ef": 50, "threads": 8},
        {"n_shards": 4, "k_per_shard": 500, "M": 16, "ef": 50, "threads": 8},
        {"n_shards": 8, "k_per_shard": 200, "M": 16, "ef": 50, "threads": 8},
        {"n_shards": 4, "k_per_shard": 200, "M": 32, "ef": 50, "threads": 8},
    ]

    results = []
    for cfg in configs:
        print(f"\n[*] Config: shards={cfg['n_shards']}, k_per_shard={cfg['k_per_shard']}, M={cfg['M']}, ef={cfg['ef']}")
        idx = RandomShardedIndex(
            dim=emb.shape[1],
            n_shards=cfg["n_shards"],
            M=cfg["M"],
            ef_construction=200,
            ef_search=cfg["ef"],
        )
        build_t = idx.build(db, num_threads=cfg["threads"])
        print(f"    Build: {build_t:.2f}s")

        res = benchmark(idx, db, queries, gt_nn, k_per_shard=cfg["k_per_shard"], k_final=1)
        res["build_s"] = round(build_t, 2)
        results.append(res)

        print(f"    R@1: {res['r1']:.4f}")
        print(f"    P50: {res['p50_us']:.1f} µs")
        print(f"    P95: {res['p95_us']:.1f} µs")
        print(f"    QPS: {res['qps']:.0f}")

    # Save
    Path(OUT_PATH).parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved to {OUT_PATH}")

    best = max(results, key=lambda x: x["r1"])
    print(f"\nBest config: shards={best['n_shards']}, k_per_shard={best['k_per_shard']}, M={best['M']}")
    print(f"  R@1: {best['r1']:.4f}, P50: {best['p50_us']:.1f} µs, Build: {best['build_s']:.1f}s")


if __name__ == "__main__":
    main()
