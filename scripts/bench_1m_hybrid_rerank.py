#!/usr/bin/env python3
"""1M scale: HNSW candidate fetch + exact cosine re-rank."""

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


def build_or_load_index(embs, index_path, M, ef_construction):
    import hnswlib

    n, dim = embs.shape
    index = hnswlib.Index(space="cosine", dim=dim)

    if index_path and os.path.exists(index_path):
        print(f"Loading index from {index_path}...")
        index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
        index.load_index(index_path, max_elements=n)
        return index, 0.0

    print(f"Building index (M={M}, ef_construction={ef_construction})...")
    t0 = time.perf_counter()
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(embs)
    build_sec = time.perf_counter() - t0
    print(f"  Built in {build_sec:.1f}s")

    if index_path:
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        index.save_index(index_path)
        print(f"  Saved to {index_path}")

    return index, build_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="data/paper_embeddings_1m_synthetic.npy")
    parser.add_argument("--index_path", default="data/hnsw_1m.index")
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--output", default="logs/bench_1m_hybrid_rerank.json")
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

    index, build_sec = build_or_load_index(embs, args.index_path, args.M, args.ef_construction)

    print(f"Computing exact ground truth for {args.n_queries} queries...")
    queries = embs[: args.n_queries]
    t0 = time.perf_counter()
    sims = queries @ embs.T
    for i in range(args.n_queries):
        sims[i, i] = -np.inf
    gt_topk = np.argpartition(-sims, args.k, axis=1)[:, : args.k]
    gt_sorted = np.array(
        [row[np.argsort(-sims[i, row])] for i, row in enumerate(gt_topk)]
    )
    print(f"  GT computed in {time.perf_counter() - t0:.1f}s")

    results = {}
    configs = [
        (20, [100, 200, 500, 1000]),
        (50, [100, 200, 500, 1000]),
        (100, [200, 500, 1000]),
        (200, [500, 1000]),
    ]

    for ef, cand_list in configs:
        index.set_ef(ef)
        for candidates in cand_list:
            # Warm-up
            for i in range(3):
                _ = index.knn_query(queries[i].reshape(1, -1), k=candidates)

            times = []
            r1, r5, r10 = [], [], []
            for i in range(args.n_queries):
                q = queries[i]
                gt = gt_sorted[i]

                t0 = time.perf_counter()
                labels, _ = index.knn_query(q.reshape(1, -1), k=candidates)
                cand_ids = np.asarray(labels[0], dtype=np.int64)
                sims_local = embs[cand_ids] @ q
                # Select k+1 so that after excluding self we still have k results
                select = min(args.k + 1, len(cand_ids))
                top_local = np.argpartition(-sims_local, select - 1)[:select]
                top_local = top_local[np.argsort(-sims_local[top_local])]
                pred_ids = cand_ids[top_local]
                t1 = time.perf_counter()

                pred = [int(x) for x in pred_ids if x != i][: args.k]
                times.append((t1 - t0) * 1e6)
                r1.append(recall_at_k(pred, gt, 1))
                r5.append(recall_at_k(pred, gt, 5))
                r10.append(recall_at_k(pred, gt, 10))

            key = f"ef{ef}_c{candidates}"
            results[key] = {
                "build_sec": round(build_sec, 2),
                "ef": ef,
                "candidates": candidates,
                "p50_us": round(float(np.percentile(times, 50)), 2),
                "p95_us": round(float(np.percentile(times, 95)), 2),
                "p99_us": round(float(np.percentile(times, 99)), 2),
                "r1_pct": round(float(np.mean(r1)) * 100, 2),
                "r5_pct": round(float(np.mean(r5)) * 100, 2),
                "r10_pct": round(float(np.mean(r10)) * 100, 2),
            }

    print("\n" + "=" * 100)
    print(f"{'Config':>12} {'Build(s)':>10} {'P50(µs)':>10} {'P95(µs)':>10} {'R@1%':>8} {'R@5%':>8} {'R@10%':>8}")
    print("=" * 100)
    for key in sorted(results.keys(), key=lambda s: (int(s.split('_')[0][2:]), int(s.split('_c')[1]))):
        r = results[key]
        print(
            f"{key:>12} {r['build_sec']:>10.1f} {r['p50_us']:>10.1f} {r['p95_us']:>10.1f} "
            f"{r['r1_pct']:>8.1f} {r['r5_pct']:>8.1f} {r['r10_pct']:>8.1f}"
        )
    print("=" * 100)

    os.makedirs("logs", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
