#!/usr/bin/env python3
"""Benchmark the learned linear hash encoder v2 (binary BOW) in Rust.

Loads data/linear_hash_weights_v2.bin via the Rust FFI, builds a binary HNSW
index over ArXiv papers, and measures:
  - self-match retrieval (exact duplicate recall)
  - semantic overlap / recall@K vs MiniLM+ITQ cosine ground truth
  - end-to-end query latency
"""
import ctypes
import hashlib
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
LIB.yp_binary_hnsw_new_with_params.argtypes = [
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
LIB.yp_binary_hnsw_new_with_params.restype = ctypes.c_void_p
LIB.yp_binary_hnsw_insert.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
]
LIB.yp_binary_hnsw_insert.restype = ctypes.c_int
LIB.yp_binary_hnsw_search.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
]
LIB.yp_binary_hnsw_search.restype = ctypes.c_size_t
LIB.yp_binary_hnsw_free.argtypes = [ctypes.c_void_p]

DIM_BYTES = 64


def id_to_u64(pid: str) -> int:
    return int(hashlib.sha256(pid.encode()).hexdigest()[:16], 16)


def encode_text(text: str):
    out = (ctypes.c_uint8 * DIM_BYTES)()
    rc = LIB.yp_linear_hash_encode(text.encode(), out, DIM_BYTES)
    return out, rc


def build_index(rows, hnsw):
    inserted = 0
    oov = 0
    t0 = time.perf_counter()
    for pid, title, abstract in rows:
        text = f"{title or ''} {abstract or ''}".strip()
        if not text:
            continue
        out, rc = encode_text(text)
        if rc == -10:
            oov += 1
            continue
        if rc != 0:
            print(f"insert encode rc={rc} for {pid}")
            continue
        LIB.yp_binary_hnsw_insert(hnsw, id_to_u64(pid), out, DIM_BYTES)
        inserted += 1
    print(f"    Inserted {inserted} docs in {time.perf_counter()-t0:.2f}s (OOV: {oov})")
    return inserted


def self_match(rows, hnsw):
    correct = total = skipped = 0
    latencies = []
    for pid, title, abstract in rows:
        text = f"{title or ''} {abstract or ''}".strip()
        if not text:
            continue
        t0 = time.perf_counter()
        out, rc = encode_text(text)
        if rc == -10:
            skipped += 1
            continue
        if rc != 0:
            print(f"self encode rc={rc} for {pid}")
            continue
        ids = (ctypes.c_uint64 * 5)()
        dists = (ctypes.c_uint32 * 5)()
        n = LIB.yp_binary_hnsw_search(hnsw, out, DIM_BYTES, 5, ids, dists, 5)
        latencies.append((time.perf_counter() - t0) * 1000)
        if n >= 1 and ids[0] == id_to_u64(pid):
            correct += 1
        total += 1
    print(f"Self-match top-1: {correct/total*100:.1f}% ({correct}/{total}), skipped={skipped}")
    print(f"Self P50: {np.median(latencies):.3f} ms")


def recall_at_k(index_rows, query_rows, hnsw, k_list=(1, 5, 10, 50, 100, 500)):
    model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)
    itq = np.load("data/itq_model_512.npz")
    mean = itq["mean"]
    M = itq["proj"]

    index_texts = [f"{t or ''} {a or ''}".strip() for _, t, a in index_rows]
    query_texts = [f"{t or ''} {a or ''}".strip() for _, t, a in query_rows]
    index_embs = model.encode(index_texts, convert_to_numpy=True, show_progress_bar=True)
    query_embs = model.encode(query_texts, convert_to_numpy=True, show_progress_bar=True)
    index_proj = (index_embs - mean) @ M
    query_proj = (query_embs - mean) @ M

    u64_to_pid = {id_to_u64(pid): pid for pid, _, _ in index_rows}
    recall_counts = {k: 0 for k in k_list}
    latencies = []
    max_k = max(k_list)

    for qi, (pid, title, abstract) in enumerate(query_rows):
        text = f"{title or ''} {abstract or ''}".strip()
        if not text:
            continue
        t0 = time.perf_counter()
        out, rc = encode_text(text)
        if rc != 0:
            continue
        ids = (ctypes.c_uint64 * max_k)()
        dists = (ctypes.c_uint32 * max_k)()
        n = LIB.yp_binary_hnsw_search(hnsw, out, DIM_BYTES, max_k, ids, dists, max_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        fast_pids = [
            u64_to_pid[ids[i]] for i in range(min(n, max_k)) if ids[i] in u64_to_pid
        ]

        sims = index_proj @ query_proj[qi]
        gt_idx = int(np.argmax(sims))
        gt_pid = index_rows[gt_idx][0]

        for k in k_list:
            if gt_pid in fast_pids[:k]:
                recall_counts[k] += 1

    total = len(latencies)
    print(f"Processed queries: {total}")
    print(f"Query latency P50: {np.median(latencies):.3f} ms\n")
    for k in k_list:
        pct = recall_counts[k] / total * 100
        print(f"  Recall@{k}: {pct:.1f}% ({recall_counts[k]}/{total})")


def main():
    rc = LIB.yp_linear_hash_load(
        os.path.abspath("data/linear_hash_weights_v2.bin").encode()
    )
    print(f"Load: {rc}")
    if rc != 0:
        return

    hnsw = LIB.yp_binary_hnsw_new_with_params(16, 200, 200)

    conn = sqlite3.connect("data/phoenix_arxiv_1m.db")
    cur = conn.cursor()
    cur.execute("SELECT id, title, abstract FROM papers ORDER BY RANDOM() LIMIT 16000")
    rows = cur.fetchall()
    conn.close()

    index_rows = rows[:10000]
    self_rows = rows[10000:11000]
    query_rows = rows[11000:11500]

    print("[+] Building index...")
    build_index(index_rows, hnsw)

    # Insert self rows too so self-match is a meaningful exact-duplicate test.
    print("[+] Inserting self-match docs into index...")
    build_index(self_rows, hnsw)

    print("[+] Self-match test...")
    self_match(self_rows, hnsw)

    print("[+] Recall@K vs MiniLM+ITQ...")
    recall_at_k(index_rows, query_rows, hnsw)

    LIB.yp_binary_hnsw_free(hnsw)


if __name__ == "__main__":
    main()
