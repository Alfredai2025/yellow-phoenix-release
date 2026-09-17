#!/usr/bin/env python3
"""
ITQ Grid Search — find iteration count maximizing true recall@1 vs L2 ground truth.
Tests n_iter from 10 to 200, step 10.
Uses 384-d embeddings -> 384-bit hashes.
Recall is measured on held-out queries against the full DB (excluding self).
"""

import numpy as np
import json
from pathlib import Path

EMB_PATH = "data/paper_embeddings_100k.npy"
OUT_PATH = "logs/itq_grid_search.json"
N_QUERY = 1000


def train_itq(X: np.ndarray, n_iter: int, seed: int = 42) -> tuple:
    """Return binary codes (n, bits) and rotation matrix R."""
    n, dim = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean

    rng = np.random.default_rng(seed)
    R = rng.standard_normal((dim, dim))
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    for _ in range(n_iter):
        Z = Xc @ R
        B = np.sign(Z)
        C = B.T @ Xc
        U2, _, Vt2 = np.linalg.svd(C)
        R = Vt2.T @ U2.T

    B = np.sign(Xc @ R)
    B = (B >= 0).astype(np.uint8)
    return B, mean, R


def hamming_nn_recall(B_db: np.ndarray, B_q: np.ndarray, true_nn: np.ndarray, k: int = 1) -> float:
    """For each query, check if true L2 NN is in top-k Hamming neighbors."""
    n_db, n_bytes = B_db.shape
    n_q = B_q.shape[0]
    correct = 0
    for i in range(n_q):
        q = B_q[i]
        xor = B_db ^ q  # broadcast over rows
        dists = np.bitwise_count(xor).sum(axis=1)
        topk = np.argpartition(dists, k - 1)[:k]
        if true_nn[i] in topk:
            correct += 1
    return correct / n_q


def l2_true_nn(queries: np.ndarray, db: np.ndarray) -> np.ndarray:
    """For each query, index of nearest DB vector (excluding self)."""
    n_q = queries.shape[0]
    nn = np.empty(n_q, dtype=np.int64)
    for i in range(n_q):
        dists = np.linalg.norm(db - queries[i], axis=1)
        dists[i] = np.inf  # exclude self
        nn[i] = int(np.argmin(dists))
    return nn


def main():
    print("[*] Loading embeddings...")
    emb = np.load(EMB_PATH).astype(np.float32)
    print(f"    Shape: {emb.shape}")

    queries = emb[:N_QUERY]
    db = emb  # full DB; self excluded in ground truth

    print("[*] Computing L2 ground truth (this may take a minute)...")
    true_nn = l2_true_nn(queries, db)
    print(f"    Done.")

    results = []
    for n_iter in range(10, 201, 10):
        print(f"\n[*] Training ITQ n_iter={n_iter}...")
        B_full, _, _ = train_itq(emb, n_iter)
        B_db = B_full
        B_q = B_full[:N_QUERY]

        r1 = hamming_nn_recall(B_db, B_q, true_nn, k=1)
        r10 = hamming_nn_recall(B_db, B_q, true_nn, k=10)
        print(f"    Recall@1 = {r1:.4f}, Recall@10 = {r10:.4f}")
        results.append({"n_iter": n_iter, "r1": float(r1), "r10": float(r10)})

    Path(OUT_PATH).parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda x: x["r1"])
    print(f"\n[+] Best: n_iter={best['n_iter']}, Recall@1={best['r1']:.4f}, Recall@10={best['r10']:.4f}")
    print(f"    Full results: {OUT_PATH}")


if __name__ == "__main__":
    main()
