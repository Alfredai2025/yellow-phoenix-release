#!/usr/bin/env python3
"""
Tier 1→2→3 Pipeline Benchmark:
1. HNSW384 hash graph: top-50 candidates (55 µs)
2. Cosine re-rank: top-5 from 50 (20 µs)
3. Geometric re-rank: final 1 from 5 (?? µs)
"""

import numpy as np
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, 'scripts')
from yp_bridge import GeometricReranker

EMB_PATH = "data/paper_embeddings_100k.npy"
ITQ_PATH = "data/itq_model_384.npz"
OUT_PATH = "logs/benchmark_tier3_pipeline.json"


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    print("[*] Loading embeddings...")
    emb = np.load(EMB_PATH).astype(np.float32)
    itq = np.load(ITQ_PATH)
    R, mean = itq["R"], itq["mean"]

    n_db = 99_000
    n_q = 100
    db_emb = emb[:n_db]
    q_emb = emb[n_db:n_db + n_q]

    # Encode hashes
    Z = (db_emb - mean) @ R
    db_hash = (np.sign(Z) > 0).astype(np.uint8)
    db_packed = np.packbits(db_hash, axis=1)

    Zq = (q_emb - mean) @ R
    q_hash = (np.sign(Zq) > 0).astype(np.uint8)
    q_packed = np.packbits(q_hash, axis=1)

    # Ground truth
    print("[*] Ground truth...")
    gt = np.empty(n_q, dtype=np.int64)
    db_norm = np.sum(db_emb ** 2, axis=1)
    for start in range(0, n_q, 10):
        end = min(start + 10, n_q)
        q = q_emb[start:end]
        q_norm = np.sum(q ** 2, axis=1)
        cross = db_emb @ q.T
        dists = db_norm[:, None] + q_norm[None, :] - 2.0 * cross
        gt[start:end] = np.argmin(dists, axis=0)

    # Tier 3 reranker
    geo = GeometricReranker()

    # Pipeline
    correct_tier2 = 0
    correct_tier3 = 0
    latencies = []

    for i in range(n_q):
        t0 = time.perf_counter()

        # Tier 1: Brute-force Hamming top-50 (simulate HNSW384)
        q_bits = np.unpackbits(q_packed[i]).reshape(384)
        db_bits = np.unpackbits(db_packed, axis=1).reshape(n_db, 384)
        hamming = np.sum(db_bits != q_bits, axis=1)
        top50 = np.argpartition(hamming, 49)[:50]

        # Tier 2: Cosine re-rank top-5
        cos_scores = [(cosine(q_emb[i], db_emb[cid]), cid) for cid in top50]
        cos_scores.sort(reverse=True)
        top5_cos = [cid for _, cid in cos_scores[:5]]
        top5_embs = db_emb[top5_cos]

        # Tier 3: Geometric re-rank top-1 from 5
        geo_indices, geo_scores = geo.rerank(q_emb[i], top5_embs, k=1)
        best = top5_cos[geo_indices[0]] if geo_indices else top5_cos[0]

        lat = time.perf_counter() - t0
        latencies.append(lat)

        if top5_cos[0] == gt[i]:
            correct_tier2 += 1
        if best == gt[i]:
            correct_tier3 += 1

    r1_tier2 = correct_tier2 / n_q
    r1_tier3 = correct_tier3 / n_q
    p50 = np.percentile(np.array(latencies) * 1e6, 50)

    print(f"\n=== Tier 1→2→3 Pipeline ===")
    print(f"Tier 2 (cosine only) R@1: {r1_tier2:.4f}")
    print(f"Tier 3 (geometric final) R@1: {r1_tier3:.4f}")
    print(f"P50 latency: {p50:.0f} µs")
    print(f"Note: Tier 1 uses brute-force Hamming. Replace with HNSW384 for 55 µs.")

    result = {
        "tier2_r1": round(r1_tier2, 4),
        "tier3_r1": round(r1_tier3, 4),
        "p50_us_python": round(float(p50), 1),
        "n_queries": n_q,
        "note": "Tier 1 brute-force. HNSW384 projected: ~55 µs + 20 µs cosine + 5 µs geometric = ~80 µs total",
    }

    Path(OUT_PATH).parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[+] Saved to {OUT_PATH}")


if __name__ == '__main__':
    main()
