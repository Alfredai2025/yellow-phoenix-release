#!/usr/bin/env python3
"""Check how often Binary HNSW (Hamming) top-m contains the true embedding top-10."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from yp_bridge import BinaryHNSW

EMB_PATH = "data/paper_embeddings_100k.npy"
HASH_PATH = "data/paper_hashes_100k.npy"
NQ = 200
CANDIDATES = 200


def main():
    print("Loading data...")
    embs = np.load(EMB_PATH).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    hashes = np.load(HASH_PATH)
    if hashes.dtype != np.uint8:
        hashes = hashes.astype(np.uint8)
    if hashes.ndim == 1:
        hashes = hashes.reshape(-1, 64)

    print("Building Binary HNSW...")
    t0 = time.perf_counter()
    idx = BinaryHNSW()
    for i in range(len(hashes)):
        idx.insert(i, bytes(hashes[i]))
    print(f"  Built in {time.perf_counter() - t0:.2f}s")

    coverage = []
    for i in range(NQ):
        q = embs[i]
        sims = embs @ q
        sims[i] = -np.inf
        true_top10 = set(np.argpartition(-sims, 10)[:10])

        cands = idx.search(bytes(hashes[i]), k=CANDIDATES)
        cand_set = set(int(c[0]) for c in cands)

        missing = len(true_top10 - cand_set)
        coverage.append(1.0 if missing == 0 else 0.0)

    cov = float(np.mean(coverage))
    print(f"\nBinary HNSW top-{CANDIDATES} coverage: {cov * 100:.2f}%")
    out = {
        "candidates": CANDIDATES,
        "coverage": cov,
        "n_queries": NQ,
    }
    with open("logs/binary_hnsw_coverage_200.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
