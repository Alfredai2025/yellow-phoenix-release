#!/usr/bin/env python3
"""Scale test: embedding-space HNSW on 1M synthetic papers."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def recall_at_k(pred, gt, k):
    return len(set(pred[:k]) & set(gt[:k])) / k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="data/paper_embeddings_1m_synthetic.npy")
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--index_path", default="")
    parser.add_argument("--output", default="logs/bench_1m_embedding_hnsw.json")
    args = parser.parse_args()

    try:
        import hnswlib
    except ImportError as exc:
        raise RuntimeError("hnswlib required: pip install hnswlib") from exc

    print(f"Loading {args.embeddings}...")
    embs = np.load(args.embeddings).astype(np.float32)
    n, dim = embs.shape
    print(f"  Shape: {n:,} x {dim}")

    print("L2-normalizing...")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-10)

    index = hnswlib.Index(space="cosine", dim=dim)
    if args.index_path and os.path.exists(args.index_path):
        print(f"Loading hnswlib index from {args.index_path}...")
        index.init_index(max_elements=n, ef_construction=args.ef_construction, M=args.M)
        index.load_index(args.index_path, max_elements=n)
        build_sec = 0.0
    else:
        print(f"Building hnswlib index (M={args.M}, ef_construction={args.ef_construction})...")
        t0 = time.perf_counter()
        index.init_index(max_elements=n, ef_construction=args.ef_construction, M=args.M)
        index.add_items(embs)
        build_sec = time.perf_counter() - t0
        print(f"  Built in {build_sec:.1f}s")
        if args.index_path:
            os.makedirs(os.path.dirname(args.index_path) or ".", exist_ok=True)
            index.save_index(args.index_path)
            print(f"  Saved to {args.index_path}")

    print(f"Computing exact ground truth for {args.n_queries} queries...")
    queries = embs[: args.n_queries]
    t0 = time.perf_counter()
    sims = queries @ embs.T
    # Exclude self for each query
    for i in range(args.n_queries):
        sims[i, i] = -np.inf
    gt_topk = np.argpartition(-sims, args.k, axis=1)[:, : args.k]
    # Sort each row
    gt_sorted = np.array(
        [row[np.argsort(-sims[i, row])] for i, row in enumerate(gt_topk)]
    )
    gt_time = time.perf_counter() - t0
    print(f"  GT computed in {gt_time:.1f}s")

    results = {}
    for ef in [20, 30, 40, 50, 100, 200]:
        index.set_ef(ef)

        # Warm-up
        for i in range(5):
            _ = index.knn_query(queries[i].reshape(1, -1), k=args.k + 1)

        times = []
        r1, r5, r10 = [], [], []
        for i in range(args.n_queries):
            q = queries[i]
            gt = gt_sorted[i]

            t0 = time.perf_counter()
            labels, _ = index.knn_query(q.reshape(1, -1), k=args.k + 1)
            t1 = time.perf_counter()

            pred = [int(x) for x in labels[0] if x != i][: args.k]
            times.append((t1 - t0) * 1e6)
            r1.append(recall_at_k(pred, gt, 1))
            r5.append(recall_at_k(pred, gt, 5))
            r10.append(recall_at_k(pred, gt, 10))

        results[f"ef{ef}"] = {
            "build_sec": round(build_sec, 2),
            "p50_us": round(float(np.percentile(times, 50)), 2),
            "p95_us": round(float(np.percentile(times, 95)), 2),
            "p99_us": round(float(np.percentile(times, 99)), 2),
            "r1_pct": round(float(np.mean(r1)) * 100, 2),
            "r5_pct": round(float(np.mean(r5)) * 100, 2),
            "r10_pct": round(float(np.mean(r10)) * 100, 2),
        }

    print("\n" + "=" * 85)
    print(f"{'ef':>4} {'Build(s)':>10} {'P50(µs)':>10} {'P95(µs)':>10} {'R@1%':>8} {'R@5%':>8} {'R@10%':>8}")
    print("=" * 85)
    for ef in [20, 30, 40, 50, 100, 200]:
        r = results[f"ef{ef}"]
        print(
            f"{ef:>4} {r['build_sec']:>10.1f} {r['p50_us']:>10.1f} {r['p95_us']:>10.1f} "
            f"{r['r1_pct']:>8.1f} {r['r5_pct']:>8.1f} {r['r10_pct']:>8.1f}"
        )
    print("=" * 85)

    os.makedirs("logs", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
