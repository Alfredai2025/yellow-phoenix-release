#!/usr/bin/env python3
"""
Real-hash recall benchmark for native Rust HNSW384.
Compares HNSW Hamming search vs brute-force L2 on embeddings.
"""

import numpy as np
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, 'scripts')

EMB_PATH = "data/paper_embeddings_100k.npy"
ITQ_PATH = "data/itq_model_384.npz"
OUT_PATH = "logs/benchmark_hnsw384_recall.json"


def encode_itq(embs, R, mean):
    Z = (embs - mean) @ R
    return (np.sign(Z) > 0).astype(np.uint8)


def brute_force_l2_nn(db, queries):
    gt = np.empty(len(queries), dtype=np.int64)
    batch = 100
    db_norm = np.sum(db.astype(np.float32) ** 2, axis=1)
    for start in range(0, len(queries), batch):
        end = min(start + batch, len(queries))
        q = queries[start:end].astype(np.float32)
        q_norm = np.sum(q ** 2, axis=1)
        cross = db @ q.T
        dists = db_norm[:, None] + q_norm[None, :] - 2.0 * cross
        gt[start:end] = np.argmin(dists, axis=0)
    return gt


def main():
    print("[*] Loading 100K embeddings + ITQ model...")
    emb = np.load(EMB_PATH).astype(np.float32)
    itq = np.load(ITQ_PATH)
    R, mean = itq["R"], itq["mean"]

    n_db = 99_000
    n_q = 1000
    db_emb = emb[:n_db]
    q_emb = emb[n_db:n_db + n_q]

    # Encode to 384-bit hashes
    db_hash = encode_itq(db_emb, R, mean)
    q_hash = encode_itq(q_emb, R, mean)

    # Pack to 48 bytes (C-contiguous)
    db_packed = np.packbits(db_hash, axis=1)
    q_packed = np.packbits(q_hash, axis=1)

    # Ground truth: L2 nearest neighbor in embedding space
    print("[*] Computing L2 ground truth...")
    gt_nn = brute_force_l2_nn(db_emb, q_emb)

    # Save hashes to binary file for Rust benchmark
    hash_bin = "data/paper_hashes_100k_384.bin"
    with open(hash_bin, "wb") as f:
        f.write(np.array([n_db], dtype=np.uint32).tobytes())
        for i in range(n_db):
            f.write(np.array([i], dtype=np.uint64).tobytes())
            f.write(db_packed[i].tobytes())

    print(f"[+] Saved {hash_bin} ({n_db} hashes, 48 bytes each)")

    # Brute-force Hamming baseline (Python)
    print("[*] Brute-force Hamming baseline...")
    db_unpacked = np.unpackbits(db_packed, axis=1)
    correct_hamming = 0
    for i in range(n_q):
        q_u = np.unpackbits(q_packed[i])
        hamming = np.sum(db_unpacked != q_u, axis=1)
        nn = np.argmin(hamming)
        if nn == gt_nn[i]:
            correct_hamming += 1

    r1_hamming = correct_hamming / n_q
    print(f"  Hamming R@1 (brute-force): {r1_hamming:.4f}")

    # Save for Rust benchmark
    with open("data/query_hashes_1k_384.bin", "wb") as f:
        f.write(np.array([n_q], dtype=np.uint32).tobytes())
        for i in range(n_q):
            f.write(np.array([gt_nn[i]], dtype=np.uint64).tobytes())
            f.write(q_packed[i].tobytes())

    print("[+] Saved query_hashes_1k_384.bin")

    result = {
        "scale": "100K",
        "hash_dim": 384,
        "hamming_r1_brute_force": round(r1_hamming, 4),
        "note": "Run Rust benchmark binary to get HNSW R@1",
    }

    Path(OUT_PATH).parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[+] Saved to {OUT_PATH}")
    print(f"\nNext: Run Rust benchmark with these hash files to measure HNSW recall + latency")


if __name__ == '__main__':
    main()
