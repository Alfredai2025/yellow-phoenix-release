#!/usr/bin/env python3
"""Build low-rank spectral basis for SpectralHologram.

Usage:
    python scripts/build_spectral_basis.py \
        --embeddings data/paper_embeddings_100k.npy \
        --k 64 \
        --output data/spectral_basis_100k_64.npz
"""

import argparse
import numpy as np
from sklearn.utils.extmath import randomized_svd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="data/paper_embeddings_100k.npy")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--output", default="data/spectral_basis_100k_64.npz")
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings}...")
    X = np.load(args.embeddings).astype(np.float32)
    print(f"Shape: {X.shape}")

    print(f"Computing top-{args.k} randomized SVD...")
    U, S, Vt = randomized_svd(X, n_components=args.k, random_state=42)

    total_var = np.sum(X ** 2)
    explained = np.sum(S ** 2)
    print(f"Variance captured by top {args.k}: {explained / total_var * 100:.2f}%")

    projections = (X @ Vt.T).astype(np.float32)
    V = Vt.T.astype(np.float32)
    eigenvalues = S.astype(np.float32)

    np.savez(args.output, V=V, eigenvalues=eigenvalues, projections=projections)
    print(f"Saved basis to {args.output}")


if __name__ == "__main__":
    main()
