#!/usr/bin/env python3
"""512D local-geometry test for the learned linear hash encoder v2.

Tests whether the fast linear hash places true ITQ nearest neighbors near the
top of the fast-hash ranking. This is the right question for high-dimensional
Hamming spaces: not "do distances correlate globally?" but "do the right docs
end up near the top?"
"""
import ctypes
import os
import sqlite3
import time

import numpy as np
from sentence_transformers import SentenceTransformer

LIB = ctypes.CDLL(os.path.abspath("target/release/libpams.dylib"))
LIB.yp_linear_hash_load.argtypes = [ctypes.c_char_p]
LIB.yp_linear_hash_load.restype = ctypes.c_int
LIB.yp_linear_hash_encode.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
]
LIB.yp_linear_hash_encode.restype = ctypes.c_int

DIM_BYTES = 64


def encode_text(text: str):
    out = (ctypes.c_uint8 * DIM_BYTES)()
    rc = LIB.yp_linear_hash_encode(text.encode(), out, DIM_BYTES)
    return out, rc


def compute_fast_hashes(rows):
    valid = np.ones(len(rows), dtype=bool)
    hashes = np.zeros((len(rows), DIM_BYTES), dtype=np.uint8)
    for i, (_, t, a) in enumerate(rows):
        text = f"{t or ''} {a or ''}".strip()
        if not text:
            valid[i] = False
            continue
        out, rc = encode_text(text)
        if rc != 0:
            valid[i] = False
            continue
        hashes[i] = list(out)
    return hashes, valid


def main(n_index: int = 10000, n_query: int = 2000):
    rc = LIB.yp_linear_hash_load(
        os.path.abspath("data/linear_hash_weights_v2.bin").encode()
    )
    print(f"Load: {rc}")
    if rc != 0:
        return

    conn = sqlite3.connect("data/phoenix_arxiv_1m.db")
    cur = conn.cursor()
    cur.execute(
        f"SELECT id, title, abstract FROM papers ORDER BY RANDOM() LIMIT {n_index + n_query}"
    )
    rows = cur.fetchall()
    conn.close()

    index_rows = rows[:n_index]
    query_rows = rows[n_index:]

    print("[+] True ITQ hashes (local MiniLM)...")
    model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)
    itq = np.load("data/itq_model_512.npz")
    mean = itq["mean"]
    M = itq["proj"]

    index_texts = [f"{t or ''} {a or ''}".strip() for _, t, a in index_rows]
    query_texts = [f"{t or ''} {a or ''}".strip() for _, t, a in query_rows]

    t0 = time.perf_counter()
    index_embs = model.encode(index_texts, convert_to_numpy=True, show_progress_bar=True)
    query_embs = model.encode(query_texts, convert_to_numpy=True, show_progress_bar=True)
    print(f"    Embedded in {time.perf_counter()-t0:.2f}s")

    index_proj = (index_embs - mean) @ M
    query_proj = (query_embs - mean) @ M
    index_hashes_true = (index_proj > 0).astype(np.uint8)
    query_hashes_true = (query_proj > 0).astype(np.uint8)

    print("[+] Fast hashes...")
    fast_index, index_valid = compute_fast_hashes(index_rows)
    fast_query, query_valid = compute_fast_hashes(query_rows)

    valid_index_idx = np.where(index_valid)[0]
    valid_query_idx = np.where(query_valid)[0]
    print(f"    Valid index: {len(valid_index_idx)}/{len(index_rows)}")
    print(f"    Valid queries: {len(valid_query_idx)}/{len(query_rows)}")

    index_hashes_true_v = index_hashes_true[valid_index_idx]
    fast_index_v = fast_index[valid_index_idx]
    query_hashes_true_v = query_hashes_true[valid_query_idx]
    fast_query_v = fast_query[valid_query_idx]
    n_idx = len(valid_index_idx)
    n_q = len(valid_query_idx)

    print("\n[+] 512D local geometry test...")
    ranks_of_true_neighbors = []
    recall_at_100 = recall_at_500 = recall_at_1000 = 0

    t0 = time.perf_counter()
    for qi in range(n_q):
        true_dists = np.sum(query_hashes_true_v[qi:qi+1] != index_hashes_true_v, axis=1)
        true_top10_local = np.argpartition(true_dists, 10)[:10]
        true_top10_idx = valid_index_idx[true_top10_local]

        fast_dists = np.sum(fast_query_v[qi:qi+1] != fast_index_v, axis=1)
        fast_ranks = valid_index_idx[np.argsort(fast_dists)]

        true_ranks_in_fast = [
            int(np.where(fast_ranks == idx)[0][0]) for idx in true_top10_idx
        ]
        ranks_of_true_neighbors.extend(true_ranks_in_fast)

        true_best_local = true_top10_local[np.argmin(true_dists[true_top10_local])]
        true_best_idx = valid_index_idx[true_best_local]
        best_fast_rank = int(np.where(fast_ranks == true_best_idx)[0][0])

        if best_fast_rank < 100:
            recall_at_100 += 1
        if best_fast_rank < 500:
            recall_at_500 += 1
        if best_fast_rank < 1000:
            recall_at_1000 += 1

    print(f"    Computed in {time.perf_counter()-t0:.2f}s")

    ranks = np.array(ranks_of_true_neighbors)
    random_median = n_idx / 2
    med = float(np.median(ranks))

    print(f"\n{'='*60}")
    print(f"512D LOCAL GEOMETRY RESULTS (n_index={n_idx}, n_query={n_q})")
    print(f"{'='*60}")
    print(f"True neighbors' rank in fast-hash space (out of {n_idx}):")
    print(f"  Mean rank:   {np.mean(ranks):.0f}  (random ≈ {random_median:.0f})")
    print(f"  Median rank: {med:.0f}  (random ≈ {random_median:.0f})")
    print(f"  P10 rank:    {np.percentile(ranks, 10):.0f}")
    print(f"  P25 rank:    {np.percentile(ranks, 25):.0f}")
    print(f"  P50 rank:    {np.percentile(ranks, 50):.0f}")
    print(f"  P75 rank:    {np.percentile(ranks, 75):.0f}")
    print(f"  P90 rank:    {np.percentile(ranks, 90):.0f}")
    print(f"  In top-100:  {np.sum(ranks < 100)}/{len(ranks)} = {np.sum(ranks < 100)/len(ranks)*100:.1f}%")
    print(f"  In top-500:  {np.sum(ranks < 500)}/{len(ranks)} = {np.sum(ranks < 500)/len(ranks)*100:.1f}%")
    print(f"  In top-1000: {np.sum(ranks < 1000)}/{len(ranks)} = {np.sum(ranks < 1000)/len(ranks)*100:.1f}%")
    print(f"\nPer-query true #1 neighbor:")
    print(f"  Recall@100:  {recall_at_100/n_q*100:.1f}%")
    print(f"  Recall@500:  {recall_at_500/n_q*100:.1f}%")
    print(f"  Recall@1000: {recall_at_1000/n_q*100:.1f}%")

    if med < 500:
        print("\nSIGNAL: STRONG — true neighbors rank in top 5%")
    elif med < n_idx * 0.2:
        print("\nSIGNAL: MODERATE — true neighbors in top 20%, cascade may be viable")
    elif med < n_idx * 0.4:
        print("\nSIGNAL: WEAK — some local structure, but scattered")
    else:
        print(f"\nSIGNAL: NONE — true neighbors randomly distributed (~{random_median:.0f})")


if __name__ == "__main__":
    main()
