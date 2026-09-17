#!/usr/bin/env python3
"""Geometric correlation test for the learned linear hash encoder v2.

Tests whether the fast linear hash preserves the *local topology* of the true
ITQ hash space, independent of absolute bit accuracy.
"""
import ctypes
import os
import sqlite3

import numpy as np
from scipy.stats import kendalltau, pearsonr
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


def main(n_docs: int = 2000, n_pairs: int = 5000, seed: int = 42):
    rc = LIB.yp_linear_hash_load(
        os.path.abspath("data/linear_hash_weights_v2.bin").encode()
    )
    print(f"Load: {rc}")
    if rc != 0:
        return

    conn = sqlite3.connect("data/phoenix_arxiv_1m.db")
    cur = conn.cursor()
    cur.execute(
        f"SELECT id, title, abstract FROM papers ORDER BY RANDOM() LIMIT {n_docs}"
    )
    rows = cur.fetchall()
    conn.close()

    print("[+] Computing true ITQ hashes (local MiniLM)...")
    model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)
    itq = np.load("data/itq_model_512.npz")
    mean = itq["mean"]
    M = itq["proj"]

    texts = [f"{t or ''} {a or ''}".strip() for _, t, a in rows]
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    proj = (embs - mean) @ M
    true_hashes = (proj > 0).astype(np.uint8)

    print("[+] Computing fast linear hashes...")
    fast_hashes = np.zeros((len(rows), DIM_BYTES), dtype=np.uint8)
    valid_mask = np.ones(len(rows), dtype=bool)
    for i, (_, t, a) in enumerate(rows):
        text = f"{t or ''} {a or ''}".strip()
        if not text:
            valid_mask[i] = False
            continue
        out, rc = encode_text(text)
        if rc != 0:
            valid_mask[i] = False
            continue
        fast_hashes[i] = list(out)

    valid_idx = np.where(valid_mask)[0]
    print(f"    Valid docs: {len(valid_idx)}/{len(rows)}")

    print("[+] Pairwise Hamming correlation (random pairs)...")
    rng = np.random.default_rng(seed)
    idx1 = rng.choice(valid_idx, n_pairs)
    idx2 = rng.choice(valid_idx, n_pairs)

    true_dists = []
    fast_dists = []
    for i, j in zip(idx1, idx2):
        if i == j:
            continue
        true_dists.append(np.sum(true_hashes[i] != true_hashes[j]))
        fast_dists.append(np.sum(fast_hashes[i] != fast_hashes[j]))

    true_dists = np.array(true_dists, dtype=np.float64)
    fast_dists = np.array(fast_dists, dtype=np.float64)

    r, p = pearsonr(true_dists, fast_dists)
    print(f"\n  Pearson r = {r:.3f} (p = {p:.2e})")
    print(f"  Mean true Hamming: {true_dists.mean():.1f} bits")
    print(f"  Mean fast Hamming: {fast_dists.mean():.1f} bytes ({fast_dists.mean()*8:.1f} bits)")
    if r > 0.5:
        print("  SIGNAL: STRONG geometric preservation")
    elif r > 0.3:
        print("  SIGNAL: MODERATE")
    elif r > 0.1:
        print("  SIGNAL: WEAK")
    else:
        print("  SIGNAL: NONE — errors are random")

    # Disjoint query/index pools so self-match cannot inflate scores.
    print("\n[+] Local rank correlation (Kendall's tau, disjoint pools)...")
    index_pool = valid_idx[:200]
    query_pool = valid_idx[200:400]

    taus = []
    for q in query_pool:
        t_d = np.sum(true_hashes[q : q + 1] != true_hashes[index_pool], axis=1)
        f_d = np.sum(fast_hashes[q : q + 1] != fast_hashes[index_pool], axis=1)
        tau, _ = kendalltau(np.argsort(t_d), np.argsort(f_d))
        taus.append(tau)

    mean_tau = np.mean(taus)
    print(f"  Mean Kendall's tau: {mean_tau:.3f}")
    print(
        "  Local ranking structure: PRESERVED"
        if mean_tau > 0.2
        else "  Local ranking structure: LOST"
    )

    print("\n[+] Nearest-neighbor preservation (true top-10 vs fast top-10)...")
    nn_preserved = 0
    overlaps = []
    for q in query_pool:
        t_d = np.sum(true_hashes[q : q + 1] != true_hashes[index_pool], axis=1)
        f_d = np.sum(fast_hashes[q : q + 1] != fast_hashes[index_pool], axis=1)
        true_top10 = set(np.argsort(t_d)[:10])
        fast_top10 = set(np.argsort(f_d)[:10])
        overlap = len(true_top10 & fast_top10)
        overlaps.append(overlap)
        if overlap >= 1:
            nn_preserved += 1

    print(
        f"  At least 1 true top-10 neighbor in fast top-10: "
        f"{nn_preserved}/{len(query_pool)} ({nn_preserved/len(query_pool)*100:.1f}%)"
    )
    print(f"  Mean overlap size: {np.mean(overlaps):.2f}")


if __name__ == "__main__":
    main()
