#!/usr/bin/env python3
"""Tune hnswlib ef for the hybrid HNSW + exact-cosine re-rank pipeline."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from yp_bridge import HybridSpectralHNSW

EMB_PATH = "data/paper_embeddings_100k.npy"
BASIS_PATH = "data/spectral_basis_100k_64.npz"
NQ = 200
K = 10


def brute_force_topk(embs, idx, k=10):
    q = embs[idx]
    sims = embs @ q
    sims[idx] = -np.inf
    topk = np.argpartition(-sims, k)[:k]
    return topk[np.argsort(-sims[topk])]


def recall_at_k(pred, gt, k):
    return len(set(pred[:k]) & set(gt[:k])) / k


def benchmark(engine, embs, gt_list, ef, candidates):
    engine.set_ef(ef)

    # Warm-up
    for i in range(5):
        _ = engine.search_hnsw(embs[i], k=K)
        _ = engine.search_hybrid(embs[i], k=K, candidates=candidates)

    hnsw_times = []
    hnsw_r10 = []
    hybrid_times = []
    hybrid_r10 = []

    for i in range(NQ):
        q = embs[i]
        gt = gt_list[i]

        t0 = time.perf_counter()
        pred = engine.search_hnsw(q, k=K + 1)
        t1 = time.perf_counter()
        pred = [int(x) for x in pred if x != i][:K]
        hnsw_times.append((t1 - t0) * 1e6)
        hnsw_r10.append(recall_at_k(pred, gt, K))

        t0 = time.perf_counter()
        pred = engine.search_hybrid(q, k=K + 1, candidates=candidates)
        t1 = time.perf_counter()
        pred = [int(x) for x in pred if x != i][:K]
        hybrid_times.append((t1 - t0) * 1e6)
        hybrid_r10.append(recall_at_k(pred, gt, K))

    return {
        "ef": ef,
        "candidates": candidates,
        "hnsw_p50": float(np.percentile(hnsw_times, 50)),
        "hnsw_r10": float(np.mean(hnsw_r10)) * 100,
        "hybrid_p50": float(np.percentile(hybrid_times, 50)),
        "hybrid_r10": float(np.mean(hybrid_r10)) * 100,
    }


def main():
    print("Building hybrid index...")
    engine = HybridSpectralHNSW(
        EMB_PATH,
        BASIS_PATH,
        hnsw_m=16,
        hnsw_ef_construction=200,
        hnsw_ef=200,
    )
    embs = engine._embs

    print("Computing ground truth...")
    gt_list = [brute_force_topk(embs, i, k=K) for i in range(NQ)]

    print("\n" + "=" * 80)
    print(f"{'ef':>4} {'cands':>6} {'HNSW P50 µs':>12} {'HNSW R@10':>10} {'Hybrid P50 µs':>14} {'Hybrid R@10':>12}")
    print("=" * 80)

    for ef in [20, 50, 100, 200]:
        for candidates in [100, 200]:
            r = benchmark(engine, embs, gt_list, ef, candidates)
            print(
                f"{r['ef']:>4} {r['candidates']:>6} {r['hnsw_p50']:>12.1f} "
                f"{r['hnsw_r10']:>10.1f} {r['hybrid_p50']:>14.1f} {r['hybrid_r10']:>12.1f}"
            )


if __name__ == "__main__":
    main()
