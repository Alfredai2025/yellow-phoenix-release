#!/usr/bin/env python3
"""Find the hnswlib ef that gives 100% R@10 under 250 µs."""

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


def main():
    print("Building index...")
    engine = HybridSpectralHNSW(
        EMB_PATH,
        BASIS_PATH,
        hnsw_m=16,
        hnsw_ef_construction=200,
        hnsw_ef=200,
    )
    embs = engine._embs

    print("Ground truth...")
    gt_list = [brute_force_topk(embs, i, k=K) for i in range(NQ)]

    print("\n" + "=" * 60)
    print(f"{'ef':>4} {'P50 µs':>10} {'P95 µs':>10} {'R@1%':>8} {'R@5%':>8} {'R@10%':>8}")
    print("=" * 60)

    for ef in [20, 25, 30, 35, 40, 45, 50]:
        engine.set_ef(ef)
        # Warm-up
        for i in range(10):
            _ = engine.search_hnsw(embs[i], k=K)

        times = []
        r1, r5, r10 = [], [], []
        for i in range(NQ):
            q = embs[i]
            gt = gt_list[i]
            t0 = time.perf_counter()
            pred = engine.search_hnsw(q, k=K + 1)
            t1 = time.perf_counter()
            pred = [int(x) for x in pred if x != i][:K]
            times.append((t1 - t0) * 1e6)
            r1.append(recall_at_k(pred, gt, 1))
            r5.append(recall_at_k(pred, gt, 5))
            r10.append(recall_at_k(pred, gt, 10))

        print(
            f"{ef:>4} {np.percentile(times, 50):>10.1f} {np.percentile(times, 95):>10.1f} "
            f"{np.mean(r1)*100:>8.1f} {np.mean(r5)*100:>8.1f} {np.mean(r10)*100:>8.1f}"
        )


if __name__ == "__main__":
    main()
