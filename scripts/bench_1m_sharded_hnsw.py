#!/usr/bin/env python3
"""1M scale via sharded 100K embedding-HNSW indexes + exact re-rank."""

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


def build_shard_index(embs, M, ef_construction):
    import hnswlib

    n, dim = embs.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(embs)
    index.set_ef(ef_construction)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="data/paper_embeddings_1m_synthetic.npy")
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--shard_k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--output", default="logs/bench_1m_sharded_hnsw.json")
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

    # Random shard assignment
    rng = np.random.default_rng(42)
    shard_id = rng.integers(0, args.shards, size=n)
    shard_map = [np.where(shard_id == s)[0] for s in range(args.shards)]

    print(f"Building {args.shards} shard indexes...")
    t0 = time.perf_counter()
    indexes = []
    for s, idx in enumerate(shard_map):
        print(f"  Shard {s+1}/{args.shards}: {len(idx):,} vectors")
        indexes.append(build_shard_index(embs[idx], args.M, args.ef_construction))
    build_sec = time.perf_counter() - t0
    print(f"  Total build: {build_sec:.1f}s")

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
    for ef in [20, 30, 40, 50, 100]:
        for index in indexes:
            index.set_ef(ef)

        # Warm-up
        for i in range(3):
            q = queries[i]
            for s, idx in enumerate(shard_map):
                _ = indexes[s].knn_query(q.reshape(1, -1), k=args.shard_k)

        times = []
        r1, r5, r10 = [], [], []
        for i in range(args.n_queries):
            q = queries[i]
            gt = gt_sorted[i]

            t0 = time.perf_counter()
            all_cands = []
            for s, idx in enumerate(shard_map):
                labels, _ = indexes[s].knn_query(q.reshape(1, -1), k=args.shard_k)
                all_cands.extend(int(idx[x]) for x in labels[0])
            all_cands = np.array(all_cands, dtype=np.int64)
            sims_local = embs[all_cands] @ q
            top_local = np.argpartition(-sims_local, args.k - 1)[: args.k]
            top_local = top_local[np.argsort(-sims_local[top_local])]
            pred_ids = all_cands[top_local]
            t1 = time.perf_counter()

            pred = [int(x) for x in pred_ids if x != i][: args.k]
            times.append((t1 - t0) * 1e6)
            r1.append(recall_at_k(pred, gt, 1))
            r5.append(recall_at_k(pred, gt, 5))
            r10.append(recall_at_k(pred, gt, 10))

        key = f"ef{ef}_k{args.shard_k}"
        results[key] = {
            "build_sec": round(build_sec, 2),
            "shards": args.shards,
            "shard_k": args.shard_k,
            "ef": ef,
            "p50_us": round(float(np.percentile(times, 50)), 2),
            "p95_us": round(float(np.percentile(times, 95)), 2),
            "p99_us": round(float(np.percentile(times, 99)), 2),
            "r1_pct": round(float(np.mean(r1)) * 100, 2),
            "r5_pct": round(float(np.mean(r5)) * 100, 2),
            "r10_pct": round(float(np.mean(r10)) * 100, 2),
        }

    print("\n" + "=" * 90)
    print(f"{'Config':>14} {'Build(s)':>10} {'P50(µs)':>10} {'P95(µs)':>10} {'R@1%':>8} {'R@5%':>8} {'R@10%':>8}")
    print("=" * 90)
    for key in sorted(results.keys(), key=lambda s: int(s.split('_')[0][2:])):
        r = results[key]
        print(
            f"{key:>14} {r['build_sec']:>10.1f} {r['p50_us']:>10.1f} {r['p95_us']:>10.1f} "
            f"{r['r1_pct']:>8.1f} {r['r5_pct']:>8.1f} {r['r10_pct']:>8.1f}"
        )
    print("=" * 90)

    os.makedirs("logs", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
