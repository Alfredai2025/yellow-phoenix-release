#!/usr/bin/env python3
"""
End-to-end benchmark: 384-bit ITQ + top-50 Hamming + cosine rerank.
Measures R@1, latency P50/P95, throughput QPS on 100K and 1M.
Optimized with packed Hamming popcount and vectorized ground truth.
"""

import numpy as np
import time
import json
from pathlib import Path

EMB_100K = "data/paper_embeddings_100k.npy"
EMB_1M = "data/paper_embeddings_arxiv_1m.npy"

# lookup table for uint8 popcount
_POPCOUNT = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


def load_itq_384():
    m = np.load("data/itq_model_384.npz")
    return m["R"], m["mean"]


def encode_itq(embs, R, mean):
    Z = (embs - mean) @ R
    return (np.sign(Z) > 0).astype(np.uint8)


def ground_truth_l2(db, queries, batch_q=500):
    """Vectorized exact L2 nearest neighbor."""
    nq = queries.shape[0]
    nn = np.empty(nq, dtype=np.int64)
    db_norm = np.sum(db.astype(np.float32) ** 2, axis=1)
    for start in range(0, nq, batch_q):
        end = min(start + batch_q, nq)
        q = queries[start:end].astype(np.float32)
        q_norm = np.sum(q ** 2, axis=1)
        cross = db @ q.T
        dists = db_norm[:, None] + q_norm[None, :] - 2.0 * cross
        nn[start:end] = np.argmin(dists, axis=0)
    return nn


def benchmark(db_embs, query_embs, R, mean, k_coarse=50):
    n_db, dim = db_embs.shape
    n_q = query_embs.shape[0]

    # Encode
    t0 = time.time()
    db_hash = encode_itq(db_embs, R, mean)
    q_hash = encode_itq(query_embs, R, mean)
    encode_time = time.time() - t0
    print(f"  Encode: {encode_time:.2f}s ({n_db} db + {n_q} queries)")

    db_packed = np.packbits(db_hash, axis=1)  # (N, 64) uint8
    q_packed = np.packbits(q_hash, axis=1)    # (Q, 64) uint8

    # Ground truth (kept feasible by limiting query count for 1M)
    t0 = time.time()
    gt_nn = ground_truth_l2(db_embs, query_embs)
    gt_time = time.time() - t0
    print(f"  Ground truth L2: {gt_time:.2f}s")

    latencies = []
    correct = 0

    for i in range(n_q):
        q = q_packed[i]
        q_emb = query_embs[i]

        t0 = time.perf_counter()

        # Packed Hamming distance via popcount lookup
        xor = db_packed ^ q
        hamming = _POPCOUNT[xor].sum(axis=1)

        # Top-k coarse candidates
        topk = np.argpartition(hamming, k_coarse - 1)[:k_coarse]

        # Cosine rerank
        scores = db_embs[topk].astype(np.float32) @ q_emb.astype(np.float32)
        best_idx = int(topk[np.argmax(scores)])

        lat = time.perf_counter() - t0
        latencies.append(lat)

        if gt_nn is not None and best_idx == gt_nn[i]:
            correct += 1

    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50) * 1e6)
    p95 = float(np.percentile(latencies, 95) * 1e6)
    qps = float(n_q / latencies.sum())
    r1 = float(correct / n_q) if gt_nn is not None else 0.0

    print(f"  R@1: {r1:.4f} ({correct}/{n_q})")
    print(f"  Latency P50: {p50:.0f}µs  P95: {p95:.0f}µs")
    print(f"  Throughput: {qps:.0f} QPS")

    return {
        "n_db": int(n_db),
        "n_queries": int(n_q),
        "k_coarse": int(k_coarse),
        "r1": r1,
        "p50_us": p50,
        "p95_us": p95,
        "qps": qps,
        "encode_s": float(encode_time),
    }


def main():
    R, mean = load_itq_384()
    print(f"[+] ITQ 384-bit model loaded: R={R.shape}")

    results = {}

    if Path(EMB_100K).exists():
        print("\n=== 100K Benchmark (99K db / 1K queries) ===")
        emb = np.load(EMB_100K)
        db = emb[:99000]
        queries = emb[99000:]
        results["100k"] = benchmark(db, queries, R, mean, k_coarse=50)

    if Path(EMB_1M).exists():
        print("\n=== 1M Benchmark (999K db / 100 queries) ===")
        emb = np.load(EMB_1M)
        db = emb[:999000]
        queries = emb[999000:999100]  # keep GT feasible
        results["1m"] = benchmark(db, queries, R, mean, k_coarse=50)

    out = Path("logs/benchmark_99r1.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved to {out}")


if __name__ == "__main__":
    main()
