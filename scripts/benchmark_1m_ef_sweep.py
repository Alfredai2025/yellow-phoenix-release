#!/usr/bin/env python3
"""
1M ef_search × k_fast sweep to recover R@1.
Builds one HybridMesh384, then varies ef_search and k_fast without rebuild.
"""

import numpy as np
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from scale_engine_pid import ScaleEnginePID

EMB_1M = "data/paper_embeddings_arxiv_1m.npy"
OUT_PATH = "logs/benchmark_1m_ef_sweep.json"


def brute_force_l2_nn(db, queries):
    gt = np.empty(len(queries), dtype=np.int64)
    batch = 50
    db_norm = np.sum(db.astype(np.float32) ** 2, axis=1)
    for start in range(0, len(queries), batch):
        end = min(start + batch, len(queries))
        q = queries[start:end].astype(np.float32)
        q_norm = np.sum(q ** 2, axis=1)
        cross = db @ q.T
        dists = db_norm[:, None] + q_norm[None, :] - 2.0 * cross
        gt[start:end] = np.argmin(dists, axis=0)
    return gt


def benchmark_config(engine, queries, gt_nn, k_fast, ef_search, n_q=1000):
    engine.set_hnsw_ef_search(ef_search)
    latencies = []
    correct = 0

    for i in range(n_q):
        t0 = time.perf_counter()
        result = engine.search_production(queries[i], k=1, k_fast=k_fast)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        if result and result[0] == gt_nn[i]:
            correct += 1

    r1 = correct / n_q
    lat_arr = np.array(latencies) * 1e6
    return {
        "k_fast": k_fast,
        "ef_search": ef_search,
        "r1": round(r1, 4),
        "p50_us": round(float(np.percentile(lat_arr, 50)), 1),
        "p95_us": round(float(np.percentile(lat_arr, 95)), 1),
        "p99_us": round(float(np.percentile(lat_arr, 99)), 1),
        "qps": round(n_q / sum(latencies), 0),
    }


def main():
    if not Path(EMB_1M).exists():
        print(f"[!] {EMB_1M} not found.")
        return

    print("=== 1M ef_search × k_fast sweep ===")
    emb = np.load(EMB_1M).astype(np.float32)
    n_db = 999_000
    n_q = 1000
    db_emb = emb[:n_db]
    q_emb = emb[n_db:n_db + n_q]

    print("[*] Computing L2 ground truth...")
    t0 = time.time()
    gt_nn = brute_force_l2_nn(db_emb, q_emb)
    print(f"    GT done in {time.time()-t0:.1f}s")

    print("[*] Building HybridMesh384 (single build)...")
    engine = ScaleEnginePID(scale_engine=None)
    t0 = time.time()
    engine._build_hybrid_mesh384(db_emb, np.arange(n_db))
    build_s = time.time() - t0
    print(f"    Build: {build_s:.1f}s")

    ef_values = [50, 100, 200, 400]
    k_fast_values = [500, 1000, 2000]
    results = []

    for ef in ef_values:
        for k_fast in k_fast_values:
            print(f"\n[*] ef_search={ef}, k_fast={k_fast}...")
            res = benchmark_config(engine, q_emb, gt_nn, k_fast, ef, n_q=n_q)
            res["build_s"] = round(build_s, 1)
            results.append(res)
            print(f"    R@1: {res['r1']:.4f}, P50: {res['p50_us']:.0f} µs, P95: {res['p95_us']:.0f} µs, QPS: {res['qps']:.0f}")

    Path(OUT_PATH).parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Saved to {OUT_PATH}")

    # Table
    print("\n=== Trade-off Table ===")
    for r in results:
        status = "✅" if r["r1"] >= 0.99 else "⚠️" if r["r1"] >= 0.95 else "❌"
        print(f"{status} ef={r['ef_search']:3d} k={r['k_fast']:4d}: R@1={r['r1']:.4f}  P50={r['p50_us']:6.0f}µs  P95={r['p95_us']:6.0f}µs  QPS={r['qps']:5.0f}")

    best_99 = [r for r in results if r["r1"] >= 0.99]
    if best_99:
        fastest = min(best_99, key=lambda x: x["p50_us"])
        print(f"\n=== Fastest ≥99% config ===")
        print(f"ef_search={fastest['ef_search']}, k_fast={fastest['k_fast']}: R@1={fastest['r1']:.4f}, P50={fastest['p50_us']:.0f} µs")


if __name__ == '__main__':
    main()
