#!/usr/bin/env python3
"""
Train 384-bit ITQ model on 384-d MiniLM embeddings.
Saves data/itq_model_384.npz with R (384,384) and mean (384,).
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

EMB_PATH = "data/paper_embeddings_100k.npy"
OUT_PATH = "data/itq_model_384.npz"

print("[*] Loading embeddings...")
emb = np.load(EMB_PATH)
n, dim = emb.shape
print(f"    {n} vectors, dim={dim}")

# Center
mean = emb.mean(axis=0)
Xc = emb - mean

# PCA to 384 (full rank)
pca = PCA(n_components=dim)
Z = pca.fit_transform(Xc)

# Random rotation
R = np.random.randn(dim, dim)
U, _, Vt = np.linalg.svd(R)
R = U @ Vt

# ITQ (10 iterations — optimal from grid search)
n_iter = 10
for i in range(n_iter):
    B = np.sign(Z @ R)
    C = B.T @ Z
    U2, _, Vt2 = np.linalg.svd(C)
    R = Vt2.T @ U2.T
    if (i + 1) % 5 == 0:
        bits = (np.sign(Z @ R) > 0).astype(np.uint8)
        print(f"    Iter {i+1}: bit balance = {bits.mean():.3f}")

print("[*] Saving model...")
np.savez(OUT_PATH, R=R, mean=mean, pca_components=pca.components_, pca_mean=pca.mean_)
print(f"[+] Saved to {OUT_PATH}")
print(f"    R shape: {R.shape}, mean shape: {mean.shape}")
