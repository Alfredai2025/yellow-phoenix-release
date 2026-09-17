#!/usr/bin/env python3
"""
1M Parameter Sweep: Find k_fast and ef_search that recover 99% R@1.
Tests k_fast in {50, 100, 200, 500} and ef_search in {50, 100, 200}.
Reuses single built index, only varies query parameters.
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
ITQ_PATH = "data/itq_model_384.npz"
OUT_PATH = "logs/benchmark_1m_param_sweep.json"


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


def benchmark_with_params(engine, queries, gt_nn, k_fast, n_q=1000):
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
    p50 = float(np.percentile(np.array(latencies) * 1e6, 50))
    p95 = float(np.percentile(np.array(latencies) * 1e6, 95))
    qps = n_q / sum(latencies)

    return {
        "k_fast": k_fast,
        "r1": round(r1, 4),
        "p50_us": round(p50, 1),
        "p95_us": round(p95, 1),
        "qps": round(qps, 0),
    }


def main():
    if not Path(EMB_1M).exists():
        print(f"[!] {EMB_1M} not found.")
        return

    print("=== 1M Parameter Sweep ===")
    emb = np.load(EMB_1M).astype(np.float32)
    n_db = 999_000
    n_q = 1000
    db_emb = emb[:n_db]
    q_emb = emb[n_db:n_db + n_q]

    print("[*] Computing L2 ground truth for 1000 queries...")
    t0 = time.time()
    gt_nn = brute_force_l2_nn(db_emb, q_emb)
    print(f"    GT done in {time.time()-t0:.1f}s")

    print("[*] Building HybridMesh384...")
    engine = ScaleEnginePID(scale_engine=None)
    t0 = time.time()
    engine._build_hybrid_mesh384(db_emb, np.arange(n_db))
    build_s = time.time() - t0
    print(f"    Build: {build_s:.1f}s")

    k_fast_values = [50, 100, 200, 500]
    results = []
    for k_fast in k_fast_values:
        print(f"\n[*] Testing k_fast={k_fast}...")
        res = benchmark_with_params(engine, q_emb, gt_nn, k_fast, n_q=n_q)
        res["build_s"] = round(build_s, 1)
        results.append(res)

        print(f"    R@1: {res['r1']:.4f}")
        print(f"    P50: {res['p50_us']:.0f} µs")
        print(f"    P95: {res['p95_us']:.0f} µs")
        print(f"    QPS: {res['qps']:.0f}")

    Path(OUT_PATH).parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Saved to {OUT_PATH}")

    best = max(results, key=lambda x: x["r1"])
    print(f"\n=== Best Config ===")
    print(f"k_fast={best['k_fast']}: R@1={best['r1']:.4f}, P50={best['p50_us']:.0f} µs")

    print(f"\n=== Trade-off Table ===")
    for r in results:
        status = "✅" if r["r1"] >= 0.99 else "⚠️" if r["r1"] >= 0.95 else "❌"
        print(f"{status} k_fast={r['k_fast']:3d}: R@1={r['r1']:.4f}  P50={r['p50_us']:6.0f}µs  QPS={r['qps']:5.0f}")


if __name__ == '__main__':
    main()
