#!/usr/bin/env python3
"""Two-tier benchmark: Binary HNSW (Tier 1) + cosine re-rank (Tier 2)."""

import os, sys, time, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from yp_bridge import BinaryHNSW

HASH_PATHS = [
    "data/paper_hashes_100k.npy",
    "paper_hashes_100k.npy",
]
EMB_PATHS = [
    "data/paper_embeddings_100k.npy",
    "paper_embeddings_100k.npy",
]

def load_data():
    hash_path = next((p for p in HASH_PATHS if os.path.exists(p)), None)
    emb_path = next((p for p in EMB_PATHS if os.path.exists(p)), None)
    if not hash_path or not emb_path:
        raise FileNotFoundError("Need paper_hashes_100k.npy and paper_embeddings_100k.npy")

    hashes = np.load(hash_path)
    if hashes.dtype != np.uint8:
        hashes = hashes.astype(np.uint8)
    if hashes.ndim == 1:
        hashes = hashes.reshape(-1, 64)

    embs = np.load(emb_path)
    if embs.dtype != np.float32:
        embs = embs.astype(np.float32)

    assert len(hashes) == len(embs), f"Hash/emb count mismatch: {len(hashes)} vs {len(embs)}"
    return hashes, embs

def build_hnsw(hashes):
    hnsw = BinaryHNSW()
    t0 = time.perf_counter()
    for i in range(len(hashes)):
        hnsw.insert(i, bytes(hashes[i]))
    build_s = time.perf_counter() - t0
    return hnsw, build_s

def cosine_sim(query, candidates):
    """query: (d,), candidates: (n, d) -> sims: (n,)"""
    q_norm = query / (np.linalg.norm(query) + 1e-10)
    c_norm = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-10)
    return q_norm @ c_norm.T

def brute_force_topk(query_emb, embs, k):
    sims = cosine_sim(query_emb, embs)
    topk = np.argpartition(-sims, k)[:k]
    topk = topk[np.argsort(-sims[topk])]
    return list(topk)

def two_tier_search(hnsw, query_hash, query_emb, embs, hnsw_k=100, final_k=10):
    # Tier 1: HNSW
    t1 = time.perf_counter()
    candidates = hnsw.search(bytes(query_hash), k=hnsw_k)
    t1_end = time.perf_counter()

    if not candidates:
        return [], (t1_end - t1) * 1e6, 0.0

    candidate_idxs = [int(c[0]) for c in candidates]
    candidate_embs = embs[candidate_idxs]

    # Tier 2: cosine re-rank
    t2 = time.perf_counter()
    sims = cosine_sim(query_emb, candidate_embs)
    t2_end = time.perf_counter()

    ranked = sorted(zip(candidate_idxs, sims), key=lambda x: -x[1])
    topk = [idx for idx, _ in ranked[:final_k]]

    return topk, (t1_end - t1) * 1e6, (t2_end - t2) * 1e6

def recall_at_k(pred, gt, k):
    return len(set(pred[:k]) & set(gt[:k])) / k

def main():
    print("Loading data...")
    hashes, embs = load_data()
    n, dim = len(hashes), embs.shape[1]
    print(f"Loaded {n} vectors, dim={dim}")

    print("Building Binary HNSW...")
    hnsw, build_s = build_hnsw(hashes)
    print(f"HNSW built in {build_s:.2f}s  ({n/build_s:,.0f} vec/s)")

    nq = 500
    print(f"Running {nq} two-tier queries (self-queries)...")

    tier1_times, tier2_times, total_times = [], [], []
    r1, r5, r10 = [], [], []

    for i in range(nq):
        q_hash, q_emb = hashes[i], embs[i]

        t0 = time.perf_counter()
        pred, t1_us, t2_us = two_tier_search(hnsw, q_hash, q_emb, embs, hnsw_k=100, final_k=10)
        total_us = (time.perf_counter() - t0) * 1e6

        tier1_times.append(t1_us)
        tier2_times.append(t2_us)
        total_times.append(total_us)

        gt = brute_force_topk(q_emb, embs, k=10)
        r1.append(recall_at_k(pred, gt, 1))
        r5.append(recall_at_k(pred, gt, 5))
        r10.append(recall_at_k(pred, gt, 10))

    def pct(arr, p):
        return float(np.percentile(arr, p))

    results = {
        "n_vectors": int(n),
        "n_queries": int(nq),
        "hnsw_build_sec": round(build_s, 2),
        "tier1_p50_us": round(pct(tier1_times, 50), 2),
        "tier1_p95_us": round(pct(tier1_times, 95), 2),
        "tier2_p50_us": round(pct(tier2_times, 50), 2),
        "tier2_p95_us": round(pct(tier2_times, 95), 2),
        "total_p50_us": round(pct(total_times, 50), 2),
        "total_p95_us": round(pct(total_times, 95), 2),
        "total_p99_us": round(pct(total_times, 99), 2),
        "recall_at_1_pct": round(float(np.mean(r1)) * 100, 2),
        "recall_at_5_pct": round(float(np.mean(r5)) * 100, 2),
        "recall_at_10_pct": round(float(np.mean(r10)) * 100, 2),
    }

    print("\n" + "="*55)
    print("  TWO-TIER BENCHMARK RESULTS")
    print("="*55)
    for k, v in results.items():
        print(f"  {k:30s}: {v}")
    print("="*55)

    os.makedirs("logs", exist_ok=True)
    out_path = "logs/bench_two_tier.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
