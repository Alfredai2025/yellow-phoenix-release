#!/usr/bin/env python3
"""Prototype Spectral Diffusion Hologram in pure NumPy.

Phase 2 sanity check before Rust implementation.
"""

import os, time, json
import numpy as np

EMB_PATH = "data/paper_embeddings_100k.npy"
BASIS_PATH = "data/spectral_basis_100k_64.npz"

def cosine_sim(query, candidates):
    qn = query / (np.linalg.norm(query) + 1e-10)
    cn = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-10)
    return qn @ cn.T

def brute_force_topk(query_emb, embs, k=10):
    sims = cosine_sim(query_emb, embs)
    topk = np.argpartition(-sims, k)[:k]
    return topk[np.argsort(-sims[topk])]

def recall_at_k(pred, gt, k):
    return len(set(pred[:k]) & set(gt[:k])) / k

def main():
    print("Loading embeddings and spectral basis...")
    embs = np.load(EMB_PATH).astype(np.float32)
    data = np.load(BASIS_PATH)
    V = data['V'].astype(np.float32)      # (384, 64)
    C = data['projections'].astype(np.float32)  # (100000, 64)
    print(f"Embeddings: {embs.shape}, V: {V.shape}, C: {C.shape}")

    # Holographic field: H = C^T @ C  (64 x 64)
    print("Building holographic field H = C^T @ C...")
    H = C.T @ C
    print(f"H shape: {H.shape}")

    nq = 200
    print(f"\nRunning {nq} queries...")
    direct_r1, direct_r5, direct_r10 = [], [], []
    diff_r1, diff_r5, diff_r10 = [], [], []
    diff_times = []

    for i in range(nq):
        q_emb = embs[i]
        gt = list(brute_force_topk(q_emb, embs, k=10))

        # Query projection
        q = q_emb @ V  # (64,)

        # Direct spectral similarity: q @ C^T
        direct_scores = C @ q
        direct_top = np.argpartition(-direct_scores, 10)[:10]
        direct_top = direct_top[np.argsort(-direct_scores[direct_top])]
        direct_r1.append(recall_at_k(direct_top, gt, 1))
        direct_r5.append(recall_at_k(direct_top, gt, 5))
        direct_r10.append(recall_at_k(direct_top, gt, 10))

        # Diffusion hologram: v = q^T @ H, scores = v @ C^T
        t0 = time.perf_counter()
        v = q @ H  # (64,)
        diff_scores = C @ v
        t1 = time.perf_counter()
        diff_times.append((t1 - t0) * 1e6)

        diff_top = np.argpartition(-diff_scores, 10)[:10]
        diff_top = diff_top[np.argsort(-diff_scores[diff_top])]
        diff_r1.append(recall_at_k(diff_top, gt, 1))
        diff_r5.append(recall_at_k(diff_top, gt, 5))
        diff_r10.append(recall_at_k(diff_top, gt, 10))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{nq} done")

    def avg(arr): return float(np.mean(arr))

    print("\n" + "="*70)
    print(f"{'Method':<35} {'P50(µs)':>10} {'R@1%':>7} {'R@5%':>7} {'R@10%':>7}")
    print("="*70)
    print(f"{'Direct spectral (q·C^T)':<35} {np.percentile(diff_times,50):>10.2f} {avg(direct_r1)*100:>7.1f} {avg(direct_r5)*100:>7.1f} {avg(direct_r10)*100:>7.1f}")
    print(f"{'Spectral diffusion (q^T H C^T)':<35} {np.percentile(diff_times,50):>10.2f} {avg(diff_r1)*100:>7.1f} {avg(diff_r5)*100:>7.1f} {avg(diff_r10)*100:>7.1f}")
    print("="*70)

    print("\nNote: brute-force cosine R@10 baseline would be 100% by definition.")
    print("Compare these numbers to Binary HNSW R@10 ~84% to see if diffusion helps.")

    results = {
        "direct": {"r1": round(avg(direct_r1)*100,2), "r5": round(avg(direct_r5)*100,2), "r10": round(avg(direct_r10)*100,2)},
        "diffusion": {"r1": round(avg(diff_r1)*100,2), "r5": round(avg(diff_r5)*100,2), "r10": round(avg(diff_r10)*100,2), "p50_us": round(float(np.percentile(diff_times,50)),2)},
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/prototype_spectral_diffusion.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to logs/prototype_spectral_diffusion.json")

if __name__ == "__main__":
    main()
