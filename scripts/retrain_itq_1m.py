# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Retrain ITQ on 1M arXiv embeddings."""
import numpy as np, time

print("Loading embeddings...")
embs = np.load("data/paper_embeddings_arxiv_1m.npy").astype(np.float32)
n, d = embs.shape
print(f"Loaded: {n:,} papers x {d} dims")

# L2 normalize
embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

# PCA to at most 512 dims (never exceed input dimension)
target_dim = min(512, d)
mean = embs.mean(axis=0)
X = embs - mean
cov = X.T @ X / n
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
W_pca = eigvecs[:, idx[:target_dim]]
X_pca = X @ W_pca
print(f"PCA done (target_dim={target_dim})")

# ITQ: iterative quantization
R = np.eye(target_dim)
for i in range(50):
    Z = X_pca @ R
    B = np.sign(Z)
    U, _, Vt = np.linalg.svd(X_pca.T @ B)
    R = U @ Vt
    if i % 10 == 0:
        loss = np.mean((Z - B) ** 2)
        print(f"  Iter {i}: quant loss = {loss:.4f}")
print("ITQ converged")

# Save
np.savez("data/itq_model_512_retrained.npz",
         mean=mean.astype(np.float32),
         W=W_pca.astype(np.float32),
         R=R.astype(np.float32),
         proj=(W_pca @ R).astype(np.float32))
print("Saved: data/itq_model_512_retrained.npz")
