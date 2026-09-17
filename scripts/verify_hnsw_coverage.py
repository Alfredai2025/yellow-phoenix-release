#!/usr/bin/env python3
"""Verify that HNSW top-m candidates contain the true embedding top-10."""

import argparse
import json
import numpy as np
import hnswlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="data/paper_embeddings_100k.npy")
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--n_queries", type=int, default=1000)
    parser.add_argument("--output", default="logs/hnsw_coverage.json")
    parser.add_argument("--ef", type=int, default=200)
    parser.add_argument("--M", type=int, default=16)
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings}...")
    embs = np.load(args.embeddings).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    N, DIM = embs.shape
    print(f"Shape: {N} x {DIM}")

    print("Building HNSW index...")
    index = hnswlib.Index(space='cosine', dim=DIM)
    index.init_index(max_elements=N, ef_construction=args.ef, M=args.M)
    index.add_items(embs)
    index.set_ef(args.ef)

    coverage = []
    for i in range(args.n_queries):
        q = embs[i]
        sims = embs @ q
        sims[i] = -999.0
        true_top10 = set(np.argpartition(-sims, 10)[:10])

        labels, _ = index.knn_query(q.reshape(1, -1), k=args.candidates)
        hnsw_set = set(labels[0])

        missing = len(true_top10 - hnsw_set)
        coverage.append(1.0 if missing == 0 else 0.0)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n_queries} done")

    cov = float(np.mean(coverage))
    print(f"\nCoverage: {cov * 100:.2f}% of queries have all true top-10 inside HNSW top-{args.candidates}")

    out = {
        "candidates": args.candidates,
        "coverage": cov,
        "n_queries": args.n_queries,
    }
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
