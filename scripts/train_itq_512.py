#!/usr/bin/env python3
"""
Train 512-bit ITQ encoder on 384-dim paper embeddings.
Projects 384 -> 512 via random orthonormal projection, then ITQ rotation.
Outputs: itq_model_512.npz, data/paper_hashes_512.npy
"""
import numpy as np
import json
import os
import sys

EMBEDDINGS_PATH = "data/paper_embeddings.npy"
HARD_NEGATIVES_PATH = "data/hard_negatives.jsonl"
OUTPUT_MODEL = "itq_model_512.npz"
N_BITS = 512


def load_embeddings(path=EMBEDDINGS_PATH):
    if not os.path.exists(path):
        print(f"[train] Embeddings not found at {path}")
        sys.exit(1)
    X = np.load(path).astype(np.float32)
    print(f"[train] Loaded embeddings: {X.shape}")
    return X


def random_projection_384_512(X, seed=42):
    """Random orthonormal projection from 384 dims up to 512 dims."""
    d_in = X.shape[1]
    print(f"[train] Random orthonormal projection {d_in} -> {N_BITS}...")
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d_in, N_BITS))
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    W = U @ Vt  # (d_in, N_BITS), rows orthonormal
    return W


def itq_rotation(X_proj, n_iter=50):
    """Iterative Quantization: rotate projected space to minimize quantization error."""
    print(f"[train] ITQ rotation ({n_iter} iterations)...")
    R = np.random.randn(X_proj.shape[1], X_proj.shape[1])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt  # orthogonal init

    for i in range(n_iter):
        V = X_proj @ R
        B = np.sign(V)
        B[B == 0] = 1
        U, _, Vt = np.linalg.svd(X_proj.T @ B)
        R = U @ Vt
        if i % 10 == 0:
            loss = np.linalg.norm(V - B, 'fro')
            print(f"  iter {i}: quantization loss = {loss:.4f}")

    return R


def apply_hard_negatives(X, W, R, hard_negatives_path=HARD_NEGATIVES_PATH):
    """Placeholder for triplet-aware ITQ."""
    if not os.path.exists(hard_negatives_path):
        return W, R

    with open(hard_negatives_path) as f:
        hard_negs = [json.loads(l) for l in f if l.strip()]

    if not hard_negs:
        return W, R

    print(f"[train] Found {len(hard_negs)} hard negatives (not applied — full triplet ITQ is research-grade)")
    return W, R


def evaluate_r1(X_hash, ground_truth_neighbors, k=1):
    """X_hash: (N, N_BITS) binary {-1, +1}."""
    correct = 0
    total = 0
    for q_idx, neighbors in ground_truth_neighbors:
        query_hash = X_hash[q_idx]
        dists = np.sum(X_hash != query_hash, axis=1)
        dists[q_idx] = 999999
        top_k = np.argsort(dists)[:k]
        if any(n in top_k for n in neighbors):
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def build_ground_truth(X, n_neighbors=5):
    """Use cosine similarity on original embeddings as ground truth."""
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(X)
    neighbors = []
    for i in range(len(X)):
        idx = np.argsort(sim[i])[::-1][1:n_neighbors+1]
        neighbors.append((i, idx.tolist()))
    return neighbors


def train():
    X = load_embeddings()

    if X.shape[1] >= N_BITS:
        print("[train] Embedding dimension already >= 512; adjust N_BITS if needed")

    mean = X.mean(axis=0)
    Xc = X - mean

    W = random_projection_384_512(X)
    X_proj = Xc @ W

    R = itq_rotation(X_proj, n_iter=50)
    W, R = apply_hard_negatives(X, W, R)

    # Generate binary hashes
    V = Xc @ W @ R
    hashes = np.sign(V)
    hashes[hashes == 0] = 1

    # Evaluate
    gt = build_ground_truth(X, n_neighbors=5)
    r1 = evaluate_r1(hashes, gt, k=1)
    r5 = evaluate_r1(hashes, gt, k=5)
    print(f"\n[train] R@1 = {r1:.3%}")
    print(f"[train] R@5 = {r5:.3%}")

    # Engine-compatible projection: (emb - mean) @ proj -> 512-bit scores
    proj = W @ R
    np.savez(OUTPUT_MODEL,
             mean=mean.astype(np.float32),
             W=W.astype(np.float32),
             R=R.astype(np.float32),
             proj=proj.astype(np.float32),
             n_bits=N_BITS)
    print(f"[train] Model saved to {OUTPUT_MODEL}")

    # Save hashes for Yellow Phoenix
    np.save("data/paper_hashes_512.npy", (hashes > 0).astype(np.uint8))
    print("[train] Hashes saved to data/paper_hashes_512.npy")


if __name__ == "__main__":
    train()
