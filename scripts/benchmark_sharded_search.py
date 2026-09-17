# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark category-sharded BinaryHNSW vs brute-force Hamming on 1M arXiv.

Measures candidate-recall and latency of the Phase 2 sharded search path in
yp_bridge.py against both global and within-category numpy brute-force scans.
"""
import argparse
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

YP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(YP_ROOT))

import yp_bridge


def encode_query(text):
    """MiniLM + ITQ -> 64-byte hash and normalized embedding."""
    model = yp_bridge._get_minilm()
    emb = model.encode(
        [text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )[0].astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    X = emb - yp_bridge._ITQ_MEAN
    q_hash = np.packbits((X @ yp_bridge._ITQ_PROJ) >= 0)
    return q_hash, emb


def brute_force_candidates(q_hash, candidates):
    return yp_bridge._brute_force_candidates(q_hash, candidates)


def brute_force_candidates_category(q_hash, category, cat_indices, candidates):
    idx_arr = cat_indices[category]
    ham = np.unpackbits(q_hash ^ yp_bridge._ITQ_HASHES[idx_arr], axis=1).sum(axis=1)
    k = min(candidates, len(ham))
    local = np.argpartition(ham, k - 1)[:k]
    return idx_arr[local]


def hnsw_single_candidates(cat, q_hash, candidates):
    idx = yp_bridge._hnsw_search_category(cat, q_hash, candidates)
    return idx if idx is not None else np.array([], dtype=np.int64)


def hnsw_multi_candidates(emb, q_hash, candidates):
    cats = yp_bridge._route_categories_from_embedding(emb)
    idx = yp_bridge._hnsw_search_multi(cats, q_hash, candidates)
    return idx if idx is not None else np.array([], dtype=np.int64), cats


def build_category_indices(db_path):
    """Map primary category -> numpy array of row indices in yp_bridge order."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, categories FROM papers")
    cat_lists = {}
    for pid, cats in cur:
        idx = yp_bridge._PID_TO_IDX.get(pid)
        if idx is None:
            continue
        primary = cats.split()[0] if cats else "misc"
        cat_lists.setdefault(primary, []).append(idx)
    conn.close()
    return {cat: np.array(idx_list, dtype=np.int64) for cat, idx_list in cat_lists.items()}


def sample_queries(db_path, cat_indices, n, seed):
    """Sample title/category rows from categories that have a built shard."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, categories FROM papers
        WHERE title IS NOT NULL AND LENGTH(title) > 15
          AND categories IS NOT NULL AND categories != ''
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n * 4,),
    )
    rows = cur.fetchall()
    conn.close()

    shard_dir = Path(yp_bridge._HNSW_SHARD_DIR)
    out = []
    rng = random.Random(seed)
    rng.shuffle(rows)
    for pid, title, cats in rows:
        if len(out) >= n:
            break
        primary = cats.split()[0]
        safe = yp_bridge._safe_cat_name(primary)
        shard_path = shard_dir / f"binary_hnsw_arxiv1m_{safe}.bin"
        if not shard_path.exists():
            continue
        idx = yp_bridge._PID_TO_IDX.get(pid)
        if idx is None:
            continue
        out.append({"pid": pid, "title": title, "category": primary, "idx": idx})
    return out


def recall(set_a, set_b, q_idx):
    a = set(int(x) for x in set_a if int(x) != q_idx)
    b = set(int(x) for x in set_b if int(x) != q_idx)
    return len(a & b) / max(1, len(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    db_path = YP_ROOT / "data" / "papers_arxiv_1m.db"
    print("Loading ITQ hashes / embeddings...")
    yp_bridge._load_itq_embeddings()
    print(f"  hashes: {yp_bridge._ITQ_HASHES.shape}")
    print(f"  embeddings: {yp_bridge._ITQ_EMBS.shape}")

    print("Building category index maps...")
    cat_indices = build_category_indices(db_path)
    print(f"  {len(cat_indices)} categories")

    print(f"Sampling {args.n_queries} queries from sharded categories...")
    queries = sample_queries(db_path, cat_indices, args.n_queries, args.seed)
    if len(queries) < args.n_queries:
        print(f"  WARNING: only found {len(queries)} usable queries")
    if not queries:
        print("No usable queries; exiting.")
        return

    # Pre-warm all shards so latency measurements reflect steady-state queries.
    shard_dir = Path(yp_bridge._HNSW_SHARD_DIR)
    all_shards = sorted(shard_dir.glob("binary_hnsw_arxiv1m_*.bin"))
    print(f"Pre-warming {len(all_shards)} category shards...")
    for f in all_shards:
        safe = f.name[len("binary_hnsw_arxiv1m_"):-len(".bin")]
        cat = safe.replace("_", ".")
        yp_bridge._load_hnsw_shard(cat)

    # Pre-warm the global fallback index too so first-query load doesn't skew latency.
    yp_bridge._load_global_hnsw()

    cand_times_bf_global = []
    cand_times_bf_cat = []
    cand_times_hnsw_single = []
    cand_times_hnsw_multi = []
    full_times_bf = []
    full_times_oracle = []
    full_times_router = []
    recalls_vs_global_single = []
    recalls_vs_global_multi = []
    recalls_vs_category = []
    true_paper_recalls_single = []
    true_paper_recalls_multi = []
    final_overlaps_oracle = []
    final_overlaps_router = []
    multi_fallbacks = 0

    print("Running benchmark...")
    for q in queries:
        title = q["title"]
        cat = q["category"]
        q_idx = q["idx"]

        q_hash, emb = encode_query(title)

        # Global brute-force candidate retrieval.
        t0 = time.perf_counter()
        bf_all = brute_force_candidates(q_hash, args.candidates)
        cand_times_bf_global.append(time.perf_counter() - t0)

        # Within-category brute-force candidate retrieval.
        t0 = time.perf_counter()
        bf_cat = brute_force_candidates_category(q_hash, cat, cat_indices, args.candidates)
        cand_times_bf_cat.append(time.perf_counter() - t0)

        # Single-category HNSW candidate retrieval (oracle category).
        t0 = time.perf_counter()
        h_single = hnsw_single_candidates(cat, q_hash, args.candidates)
        cand_times_hnsw_single.append(time.perf_counter() - t0)

        # Multi-category HNSW candidate retrieval (production router).
        t0 = time.perf_counter()
        h_multi, chosen_cats = hnsw_multi_candidates(emb, q_hash, args.candidates)
        cand_times_hnsw_multi.append(time.perf_counter() - t0)

        recalls_vs_global_single.append(recall(h_single, bf_all, q_idx))
        recalls_vs_global_multi.append(recall(h_multi, bf_all, q_idx))
        recalls_vs_category.append(recall(h_single, bf_cat, q_idx))
        true_paper_recalls_single.append(
            1.0 if int(q_idx) in set(int(x) for x in h_single) else 0.0
        )
        true_paper_recalls_multi.append(
            1.0 if int(q_idx) in set(int(x) for x in h_multi) else 0.0
        )

        # Full end-to-end latency: brute-force global.
        t0 = time.perf_counter()
        res_bf = yp_bridge.search(title, top_k=args.top_k, candidates=args.candidates, use_hnsw=False)
        full_times_bf.append(time.perf_counter() - t0)

        # Full end-to-end latency: oracle sharded (category supplied).
        t0 = time.perf_counter()
        res_oracle = yp_bridge.search(title, top_k=args.top_k, candidates=args.candidates, use_hnsw=True, category=cat)
        full_times_oracle.append(time.perf_counter() - t0)

        # Full end-to-end latency: production router (multi-shard).
        t0 = time.perf_counter()
        res_router = yp_bridge.search(title, top_k=args.top_k, candidates=args.candidates, use_hnsw=True)
        full_times_router.append(time.perf_counter() - t0)

        bf_topk = {pid for _, (pid, _) in res_bf[:args.top_k]}
        final_overlaps_oracle.append(
            len(bf_topk & {pid for _, (pid, _) in res_oracle[:args.top_k]}) / args.top_k
        )
        final_overlaps_router.append(
            len(bf_topk & {pid for _, (pid, _) in res_router[:args.top_k]}) / args.top_k
        )

        if len(h_multi) < args.candidates // 2:
            multi_fallbacks += 1

    def stats(name, arr):
        arr_ms = np.array(arr) * 1000.0
        return (
            f"{name}: mean={arr_ms.mean():.3f}ms median={np.median(arr_ms):.3f}ms "
            f"p95={np.percentile(arr_ms, 95):.3f}ms"
        )

    print("\n=== Candidate retrieval latency ===")
    print(stats("Brute-force (global)", cand_times_bf_global))
    print(stats("Brute-force (within category)", cand_times_bf_cat))
    print(stats("Sharded HNSW (oracle single cat)", cand_times_hnsw_single))
    print(stats("Sharded HNSW (router multi-cat)", cand_times_hnsw_multi))
    print(f"HNSW single vs global BF: mean={(np.array(cand_times_bf_global)/np.array(cand_times_hnsw_single)).mean():.2f}x")
    print(f"HNSW multi  vs global BF: mean={(np.array(cand_times_bf_global)/np.array(cand_times_hnsw_multi)).mean():.2f}x")

    print("\n=== Full search latency (incl. MiniLM encode) ===")
    print(stats("Brute-force (global)", full_times_bf))
    print(stats("Sharded HNSW (oracle cat)", full_times_oracle))
    print(stats("Sharded HNSW (router multi-cat)", full_times_router))

    print("\n=== Recall vs brute-force candidates ===")
    print(f"Single-cat HNSW vs global top-{args.candidates}: mean={np.mean(recalls_vs_global_single):.3%} "
          f"median={np.median(recalls_vs_global_single):.3%} p5={np.percentile(recalls_vs_global_single, 5):.3%}")
    print(f"Multi-cat HNSW  vs global top-{args.candidates}: mean={np.mean(recalls_vs_global_multi):.3%} "
          f"median={np.median(recalls_vs_global_multi):.3%} p5={np.percentile(recalls_vs_global_multi, 5):.3%}")
    print(f"Single-cat HNSW vs within-category top-{args.candidates}: mean={np.mean(recalls_vs_category):.3%} "
          f"median={np.median(recalls_vs_category):.3%} p5={np.percentile(recalls_vs_category, 5):.3%}")
    print(f"True paper in single-cat HNSW candidates: {np.mean(true_paper_recalls_single):.3%}")
    print(f"True paper in multi-cat HNSW candidates:  {np.mean(true_paper_recalls_multi):.3%}")

    print("\n=== Final top-k overlap with global brute-force ===")
    print(f"Oracle sharded top-{args.top_k}: mean={np.mean(final_overlaps_oracle):.3%} "
          f"median={np.median(final_overlaps_oracle):.3%}")
    print(f"Router multi-cat top-{args.top_k}: mean={np.mean(final_overlaps_router):.3%} "
          f"median={np.median(final_overlaps_router):.3%}")
    print(f"Multi-cat candidate shortfall fallbacks: {multi_fallbacks}/{len(queries)}")


if __name__ == "__main__":
    main()
