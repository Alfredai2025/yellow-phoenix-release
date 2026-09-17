#!/usr/bin/env python3
"""Benchmark HNSW + re-rank hybrid search (spectral and exact cosine)."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from yp_bridge import HybridSpectralHNSW

EMB_PATH = "data/paper_embeddings_100k.npy"
BASIS_PATH = "data/spectral_basis_100k_64.npz"


def brute_force_topk(embs, idx, k=10):
    q = embs[idx]
    sims = embs @ q
    sims[idx] = -np.inf
    topk = np.argpartition(-sims, k)[:k]
    return topk[np.argsort(-sims[topk])]


def recall_at_k(pred, gt, k):
    return len(set(pred[:k]) & set(gt[:k])) / k


def run_method(name, engine, embs, gt_list, searcher, k, candidates):
    times = []
    r1, r5, r10 = [], [], []
    for i in range(len(gt_list)):
        q = embs[i]
        gt = gt_list[i]

        t0 = time.perf_counter()
        pred = searcher(q, k=k + 1, candidates=candidates) if candidates else searcher(q, k=k + 1)
        t1 = time.perf_counter()

        pred = [int(x) for x in pred if x != i][:k]
        times.append((t1 - t0) * 1e6)
        r1.append(recall_at_k(pred, gt, 1))
        r5.append(recall_at_k(pred, gt, 5))
        r10.append(recall_at_k(pred, gt, 10))

    return {
        "p50_us": float(np.percentile(times, 50)),
        "p95_us": float(np.percentile(times, 95)),
        "p99_us": float(np.percentile(times, 99)),
        "r1_pct": float(np.mean(r1)) * 100,
        "r5_pct": float(np.mean(r5)) * 100,
        "r10_pct": float(np.mean(r10)) * 100,
    }


def main():
    print("Loading data and building hybrid index...")
    t0 = time.perf_counter()
    engine = HybridSpectralHNSW(
        EMB_PATH,
        BASIS_PATH,
        hnsw_m=16,
        hnsw_ef_construction=200,
        hnsw_ef=200,
    )
    build_sec = time.perf_counter() - t0
    print(f"  Built in {build_sec:.2f}s")

    embs = engine._embs
    nq = 200
    k = 10
    print(f"Vectors: {len(embs)}, Queries: {nq}, Dim: {embs.shape[1]}")

    print("\nComputing brute-force ground truth...")
    gt_list = [brute_force_topk(embs, i, k=k) for i in range(nq)]
    print("GT done")

    results = {}

    # Tune HNSW ef. Lower ef = faster, slightly lower recall.
    for ef in [20, 50]:
        engine.set_ef(ef)

        # Warm-up
        for i in range(5):
            _ = engine.search_hnsw(embs[i], k=k)
            _ = engine.search_hybrid(embs[i], k=k, candidates=200)
            _ = engine.search_hybrid_cosine(embs[i], k=k, candidates=100)

        label = f"hnsw_top10_ef{ef}"
        results[label] = run_method(
            label, engine, embs, gt_list,
            lambda q, k, candidates=None: engine.search_hnsw(q, k=k),
            k, None,
        )

        label = f"hybrid_spectral_ef{ef}_c200"
        results[label] = run_method(
            label, engine, embs, gt_list, engine.search_hybrid, k, 200,
        )

        label = f"hybrid_cosine_ef{ef}_c100"
        results[label] = run_method(
            label, engine, embs, gt_list, engine.search_hybrid_cosine, k, 100,
        )

        label = f"hybrid_cosine_ef{ef}_c200"
        results[label] = run_method(
            label, engine, embs, gt_list, engine.search_hybrid_cosine, k, 200,
        )

    print("\n" + "=" * 110)
    print(f"{'Method':<35} {'Build(s)':>8} {'P50(µs)':>9} {'R@1%':>7} {'R@5%':>7} {'R@10%':>7}")
    print("=" * 110)
    for label, r in results.items():
        name = label.replace("_", " ")
        print(
            f"{name:<35} {build_sec:>8.2f} {r['p50_us']:>9.1f} "
            f"{r['r1_pct']:>7.1f} {r['r5_pct']:>7.1f} {r['r10_pct']:>7.1f}"
        )
    print("=" * 110)

    os.makedirs("logs", exist_ok=True)
    out = "logs/bench_hybrid_spectral.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
