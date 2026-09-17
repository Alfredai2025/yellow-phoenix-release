#!/usr/bin/env python3
"""Multi-probe geometric search test for the learned linear hash encoder v2.

Builds a binary HNSW index via the Rust FFI, then for each query computes the
raw z-vector in Python (using binary-BOW tokenization matching Rust), generates
probe hashes by flipping the most uncertain bits, and measures whether the true
MiniLM+ITQ nearest neighbor appears in the merged candidate set.
"""
import ctypes
import hashlib
import os
import sqlite3

import numpy as np
from sentence_transformers import SentenceTransformer

DIM_BYTES = 64


def load_rust_ffi():
    lib = ctypes.CDLL(os.path.abspath("target/release/libpams.dylib"))
    lib.yp_linear_hash_load.argtypes = [ctypes.c_char_p]
    lib.yp_linear_hash_load.restype = ctypes.c_int
    lib.yp_linear_hash_encode.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.yp_linear_hash_encode.restype = ctypes.c_int
    lib.yp_binary_hnsw_new_with_params.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.yp_binary_hnsw_new_with_params.restype = ctypes.c_void_p
    lib.yp_binary_hnsw_insert.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.yp_binary_hnsw_insert.restype = ctypes.c_int
    lib.yp_binary_hnsw_search.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.yp_binary_hnsw_search.restype = ctypes.c_size_t
    return lib


def pack_hash(h):
    out = np.zeros(DIM_BYTES, dtype=np.uint8)
    for byte_idx in range(DIM_BYTES):
        dim_start = (byte_idx // 2) * 16 + (byte_idx % 2) * 8
        for d in range(8):
            if h[dim_start + d]:
                out[byte_idx] |= 1 << d
    return out


def text_to_hash(text, W, b, vocab_map):
    tokens = text.lower().split()
    indices = [vocab_map[t] for t in tokens if t in vocab_map]
    if not indices:
        return None, None
    x = np.zeros(len(vocab_map), dtype=np.float32)
    x[indices] = 1.0
    z = x @ W + b
    h = (z > 0).astype(np.uint8)
    return pack_hash(h), z


def id_to_u64(pid: str) -> int:
    return int(hashlib.sha256(pid.encode()).hexdigest()[:16], 16)


def hnsw_search_bytes(lib, hnsw, hash_bytes, k, u64_to_pid):
    ids = (ctypes.c_uint64 * k)()
    dists = (ctypes.c_uint32 * k)()
    n = lib.yp_binary_hnsw_search(
        hnsw,
        hash_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        DIM_BYTES,
        k,
        ids,
        dists,
        k,
    )
    return {u64_to_pid[ids[i]] for i in range(min(n, k)) if ids[i] in u64_to_pid}


def main(n_index=10000, n_query=500):
    print("[+] Loading Python model weights...")
    m = np.load("data/linear_hash_model_v2.npz", allow_pickle=True)
    W = m["W"].astype(np.float32)
    b = m["b"].astype(np.float32)
    vocab = m["vocab"]
    vocab_map = {t: i for i, t in enumerate(vocab)}
    print(f"    W: {W.shape}, vocab: {len(vocab)}")

    lib = load_rust_ffi()
    rc = lib.yp_linear_hash_load(
        os.path.abspath("data/linear_hash_weights_v2.bin").encode()
    )
    print(f"Rust load: {rc}")
    hnsw = lib.yp_binary_hnsw_new_with_params(16, 200, 200)

    conn = sqlite3.connect("data/phoenix_arxiv_1m.db")
    cur = conn.cursor()
    cur.execute(
        f"SELECT id, title, abstract FROM papers ORDER BY RANDOM() LIMIT {n_index + n_query}"
    )
    rows = cur.fetchall()
    conn.close()

    index_rows = rows[:n_index]
    query_rows = rows[n_index : n_index + n_query]

    print("[+] Building HNSW index...")
    for pid, t, a in index_rows:
        text = f"{t or ''} {a or ''}".strip()
        if not text:
            continue
        out = (ctypes.c_uint8 * DIM_BYTES)()
        if lib.yp_linear_hash_encode(text.encode(), out, DIM_BYTES) != 0:
            continue
        lib.yp_binary_hnsw_insert(hnsw, id_to_u64(pid), out, DIM_BYTES)

    print("[+] Computing MiniLM+ITQ ground truth...")
    model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)
    itq = np.load("data/itq_model_512.npz")
    mean = itq["mean"]
    M = itq["proj"]

    index_texts = [f"{t or ''} {a or ''}".strip() for _, t, a in index_rows]
    query_texts = [f"{t or ''} {a or ''}".strip() for _, t, a in query_rows]
    index_embs = model.encode(index_texts, convert_to_numpy=True, show_progress_bar=True)
    query_embs = model.encode(
        query_texts, convert_to_numpy=True, show_progress_bar=True
    )
    index_proj = (index_embs - mean) @ M
    query_proj = (query_embs - mean) @ M

    u64_to_pid = {id_to_u64(pid): pid for pid, _, _ in index_rows}

    print("[+] Multi-probe geometric search...")
    recall_single_100 = recall_single_500 = 0
    recall_multi_8_100 = recall_multi_8_500 = 0
    recall_multi_16_100 = recall_multi_16_500 = 0
    recall_multi_comb_500 = 0
    skipped = 0

    for qi, text in enumerate(query_texts):
        if not text:
            skipped += 1
            continue
        base_bytes, z = text_to_hash(text, W, b, vocab_map)
        if base_bytes is None:
            skipped += 1
            continue
        base_hash = (z > 0).astype(np.uint8)

        single_pids_500 = hnsw_search_bytes(lib, hnsw, base_bytes, 500, u64_to_pid)
        single_pids_100 = hnsw_search_bytes(lib, hnsw, base_bytes, 100, u64_to_pid)

        uncertainty = np.abs(z)
        uncertain_bits = np.argsort(uncertainty)

        multi8_pids_500 = set(single_pids_500)
        multi8_pids_100 = set(single_pids_100)
        multi16_pids_500 = set(single_pids_500)
        multi16_pids_100 = set(single_pids_100)

        for rank, bit_idx in enumerate(uncertain_bits[:16]):
            probe = base_hash.copy()
            probe[bit_idx] = 1 - probe[bit_idx]
            probe_bytes = pack_hash(probe)
            pids_500 = hnsw_search_bytes(lib, hnsw, probe_bytes, 500, u64_to_pid)
            pids_100 = hnsw_search_bytes(lib, hnsw, probe_bytes, 100, u64_to_pid)
            multi16_pids_500 |= pids_500
            multi16_pids_100 |= pids_100
            if rank < 8:
                multi8_pids_500 |= pids_500
                multi8_pids_100 |= pids_100

        comb_pids_500 = set(single_pids_500)
        top4 = uncertain_bits[:4]
        for mask in range(1, 16):
            probe = base_hash.copy()
            for j in range(4):
                if mask & (1 << j):
                    probe[top4[j]] = 1 - probe[top4[j]]
            probe_bytes = pack_hash(probe)
            comb_pids_500 |= hnsw_search_bytes(lib, hnsw, probe_bytes, 500, u64_to_pid)

        sims = index_proj @ query_proj[qi]
        gt_idx = int(np.argmax(sims))
        gt_pid = index_rows[gt_idx][0]

        if gt_pid in single_pids_100:
            recall_single_100 += 1
        if gt_pid in single_pids_500:
            recall_single_500 += 1
        if gt_pid in multi8_pids_100:
            recall_multi_8_100 += 1
        if gt_pid in multi8_pids_500:
            recall_multi_8_500 += 1
        if gt_pid in multi16_pids_100:
            recall_multi_16_100 += 1
        if gt_pid in multi16_pids_500:
            recall_multi_16_500 += 1
        if gt_pid in comb_pids_500:
            recall_multi_comb_500 += 1

    n_valid = n_query - skipped
    print(f"Skipped: {skipped}")
    print(f"\nMULTI-PROBE GEOMETRIC SEARCH (n={n_valid})")
    print(f"Single probe Recall@100:  {recall_single_100/n_valid*100:.1f}%")
    print(f"Single probe Recall@500:  {recall_single_500/n_valid*100:.1f}%")
    print(
        f"Multi 8 flips Recall@100: {recall_multi_8_100/n_valid*100:.1f}%  "
        f"(+{recall_multi_8_100-recall_single_100})"
    )
    print(
        f"Multi 8 flips Recall@500: {recall_multi_8_500/n_valid*100:.1f}%  "
        f"(+{recall_multi_8_500-recall_single_500})"
    )
    print(
        f"Multi 16 flips Recall@500: {recall_multi_16_500/n_valid*100:.1f}%  "
        f"(+{recall_multi_16_500-recall_single_500})"
    )
    print(
        f"Top-4 combinations Recall@500: {recall_multi_comb_500/n_valid*100:.1f}%  "
        f"(+{recall_multi_comb_500-recall_single_500})"
    )

    single = recall_single_500 / n_valid
    multi = recall_multi_8_500 / n_valid
    if multi > single * 1.2:
        print("\nSIGNAL: STRONG — uncertainty is directional")
    elif multi > single * 1.05:
        print("\nSIGNAL: MODEST — uncertainty is somewhat structured")
    else:
        print("\nSIGNAL: NONE — uncertainty is isotropic noise")


if __name__ == "__main__":
    main()
