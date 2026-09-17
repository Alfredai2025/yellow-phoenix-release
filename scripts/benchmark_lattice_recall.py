# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Measure how well 6-byte lattice L1 preserves 512-bit Hamming neighbors."""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sentence_transformers import SentenceTransformer

HASH_PATH = "data/itq_hashes_1m.npy"
COORD_PATH = "data/lattice_coords_1m_6byte.npy"
QUERY_PATH = "data/paraphrase_queries_500.pkl"
MODEL_PATH = "models/all-MiniLM-L6-v2"
ITQ_PATH = "itq_model_512.npz"

print("Loading hashes and lattice coords ...")
hashes = np.load(HASH_PATH, mmap_mode='r')          # N x 64 uint8
coords = np.load(COORD_PATH, mmap_mode='r')         # N x 6 uint8
N = hashes.shape[0]
print(f"Corpus: {N:,} papers")

print("Loading ITQ model ...")
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

print("Loading MiniLM ...")
model = SentenceTransformer(MODEL_PATH, local_files_only=True)

print("Loading paraphrase queries ...")
with open(QUERY_PATH, "rb") as f:
    variants = pickle.load(f)
# Use up to 200 queries for speed
queries = variants[:200]
print(f"Benchmark queries: {len(queries)}")

# Encode queries
t0 = time.time()
q_texts = [q for _, _, q in queries]
q_embs = model.encode(q_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
q_hashes = emb_to_hash(q_embs)
q_coords = np.empty((len(queries), 6), dtype=np.uint8)
print(f"Queries encoded in {time.time()-t0:.1f}s")

# We already have lattice_batch_convert, but do it manually to avoid import overhead
for i, h in enumerate(q_hashes):
    # quick manual 6-byte extraction (same logic as Rust)
    words = h.view(np.uint64).reshape(8)
    c0 = ((words >> 0) & 1).sum()
    c63 = ((words >> 63) & 1).sum()
    q_coords[i] = [
        words[0].bit_count(),
        words[7].bit_count(),
        c0, c63,
        (words[0] ^ words[1]).bit_count(),
        (words[6] ^ words[7]).bit_count(),
    ]

# Evaluate recall at various K
K_VALUES = [1, 10, 50, 100]
recall = {k: 0 for k in K_VALUES}
lat_times = []
ham_times = []

for i, (true_pid, title, query_text) in enumerate(queries):
    # Lattice L1 distances (fast)
    t0 = time.perf_counter()
    l1 = np.abs(q_coords[i].astype(np.int16) - coords.astype(np.int16)).sum(axis=1)
    lat_top = set(np.argpartition(l1, K_VALUES[-1]-1)[:K_VALUES[-1]].tolist())
    lat_times.append(time.perf_counter() - t0)

    # Hamming distances (ground truth)
    t0 = time.perf_counter()
    diffs = q_hashes[i].astype(np.uint8) ^ hashes.astype(np.uint8)
    ham = np.bitwise_count(diffs).sum(axis=1)
    ham_top = set(np.argpartition(ham, K_VALUES[-1]-1)[:K_VALUES[-1]].tolist())
    ham_times.append(time.perf_counter() - t0)

    for k in K_VALUES:
        lat_k = set(np.argpartition(l1, k-1)[:k].tolist())
        ham_k = set(np.argpartition(ham, k-1)[:k].tolist())
        overlap = len(lat_k & ham_k)
        recall[k] += overlap / k

print("\n=== Lattice vs Hamming neighbor overlap (recall@K) ===")
for k in K_VALUES:
    print(f"  R@{k}: {recall[k]/len(queries)*100:.1f}%")
print(f"\nAvg lattice scan time: {np.mean(lat_times)*1000:.2f} ms")
print(f"Avg Hamming scan time: {np.mean(ham_times)*1000:.2f} ms")
