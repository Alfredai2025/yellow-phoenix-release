#!/usr/bin/env python3
"""ITQ v2 Trainer — Yellow Phoenix
PCA whitening + ITQ rotation with grid search.
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def load_embeddings(path: str = "data/paper_embeddings.npy") -> np.ndarray:
    return np.load(path).astype(np.float32)


def load_queries(path: str = "data/queries.jsonl") -> list:
    queries = []
    with open(path) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def itq_train(X: np.ndarray, nbits: int = 512, n_iter: int = 50):
    """Train ITQ: PCA whitening + orthogonal rotation."""
    nbits = min(nbits, X.shape[1])
    pca = PCA(n_components=nbits, whiten=True)
    X_proj = pca.fit_transform(X)

    R = np.random.randn(nbits, nbits)
    R, _ = np.linalg.qr(R)

    for i in range(n_iter):
        Z = X_proj @ R
        B = np.sign(Z)
        B[B == 0] = 1
        C = B.T @ X_proj
        U, _, Vt = np.linalg.svd(C)
        R = Vt.T @ U.T

    def encode(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        proj = pca.transform(v)
        bits = np.sign(proj @ R)
        return (bits > 0).astype(np.uint8)

    return encode, pca, R


def evaluate(encoder, queries: list, embeddings: np.ndarray, top_k: int = 1) -> float:
    correct = 0
    for q in queries:
        q_vec = np.array(q["embedding"], dtype=np.float32)
        true_pid = q["paper_id"]

        q_hash = encoder(q_vec)

        # Brute-force Hamming search over the small evaluation set
        hashes = encoder(embeddings)
        dists = np.sum(q_hash != hashes, axis=1)
        ranked = np.argsort(dists)
        if true_pid in ranked[:top_k]:
            correct += 1

    return correct / len(queries)


def grid_search(embeddings: np.ndarray, queries: list):
    best = {"r1": 0.0, "nbits": 0, "n_iter": 0}
    best_model = None

    max_bits = embeddings.shape[1]
    for nbits in [256, 384, 512]:
        effective_nbits = min(nbits, max_bits)
        for n_iter in [30, 50, 75]:
            print(f"\n[grid] nbits={effective_nbits} (requested {nbits}), n_iter={n_iter}")
            t0 = time.time()
            enc, pca, R = itq_train(embeddings, nbits=nbits, n_iter=n_iter)
            train_t = time.time() - t0

            t0 = time.time()
            r1 = evaluate(enc, queries, embeddings, top_k=1)
            eval_t = time.time() - t0

            print(f"  R@1={r1:.3f}  train={train_t:.1f}s  eval={eval_t:.1f}s")

            if r1 > best["r1"]:
                best = {
                    "r1": r1,
                    "nbits": effective_nbits,
                    "n_iter": n_iter,
                }
                best_model = (pca, R)

    return best, best_model


def save_model(best: dict, model, out_path: str = "data/itq_model_v2.npz"):
    pca, R = model
    np.savez(
        out_path,
        R=R,
        nbits=best["nbits"],
        n_iter=best["n_iter"],
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_,
    )
    print(f"\n[save] Model saved to {out_path}")
    print(
        f"       Best config: nbits={best['nbits']}, n_iter={best['n_iter']}, R@1={best['r1']:.3f}"
    )


def main():
    print("[load] Loading embeddings...")
    embeddings = load_embeddings()
    print(f"       Shape: {embeddings.shape}")

    print("[load] Loading queries...")
    queries = load_queries()
    print(f"       Count: {len(queries)}")

    print("\n[train] Starting grid search...")
    best, model = grid_search(embeddings, queries)

    if model is not None:
        save_model(best, model)
    else:
        print("[train] No model trained.")


if __name__ == "__main__":
    main()
