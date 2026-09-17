# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Measure true-paper R@K using 6-byte lattice L1 vs 512-bit Hamming on 1M corpus."""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sentence_transformers import SentenceTransformer

HASH_PATH = "data/itq_hashes_1m.npy"
COORD_PATH = "data/lattice_coords_1m_6byte.npy"
META_PATH = "data/paper_meta_arxiv_1m.pkl"
QUERY_PATH = "data/paraphrase_queries_500.pkl"
ITQ_PATH = "itq_model_512.npz"

print("Loading metadata, hashes, coords ...")
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
pids = meta["pids"]
pid_to_idx = {pid: i for i, pid in enumerate(pids)}

hashes = np.load(HASH_PATH, mmap_mode='r')
coords = np.load(COORD_PATH, mmap_mode='r')
N = hashes.shape[0]

itq = np.load(ITQ_PATH)
mean = itq["mean"].astype(np.float32)
input_dim = mean.shape[0]
for key in ("proj", "W", "R"):
    if key in itq and itq[key].shape[0] == input_dim:
        W = itq[key].astype(np.float32)
        break

def emb_to_hash(emb):
    x = emb - mean
    x = x @ W
    bits = (x > 0).astype(np.uint8)
    return np.packbits(bits, axis=1 if emb.ndim > 1 else 0)

model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)

with open(QUERY_PATH, "rb") as f:
    variants = pickle.load(f)
queries = variants[:200]

q_texts = [q for _, _, q in queries]
q_embs = model.encode(q_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
q_hashes = emb_to_hash(q_embs)
q_coords = np.empty((len(queries), 6), dtype=np.uint8)
for i, h in enumerate(q_hashes):
    words = h.view(np.uint64).reshape(8)
    c0 = ((words >> 0) & 1).sum()
    c63 = ((words >> 63) & 1).sum()
    q_coords[i] = [words[0].bit_count(), words[7].bit_count(), c0, c63,
                   (words[0] ^ words[1]).bit_count(), (words[6] ^ words[7]).bit_count()]

K = [1, 10, 50, 100]
lat_hits = {k: 0 for k in K}
ham_hits = {k: 0 for k in K}

for i, (true_pid, title, query_text) in enumerate(queries):
    true_idx = pid_to_idx[true_pid]

    l1 = np.abs(q_coords[i].astype(np.int16) - coords.astype(np.int16)).sum(axis=1)
    diffs = q_hashes[i].astype(np.uint8) ^ hashes.astype(np.uint8)
    ham = np.bitwise_count(diffs).sum(axis=1)

    for k in K:
        lat_top = np.argpartition(l1, k-1)[:k]
        if true_idx in lat_top:
            lat_hits[k] += 1
        ham_top = np.argpartition(ham, k-1)[:k]
        if true_idx in ham_top:
            ham_hits[k] += 1

print("\n=== True-paper R@K on 1M corpus ===")
print(f"{'K':>5} | {'Lattice L1':>12} | {'Hamming':>12}")
print("-" * 35)
for k in K:
    print(f"{k:>5} | {lat_hits[k]/len(queries)*100:>11.1f}% | {ham_hits[k]/len(queries)*100:>11.1f}%")
