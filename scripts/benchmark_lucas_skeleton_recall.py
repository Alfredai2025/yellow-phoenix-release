# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Compare 256-bit Lucas skeleton R@K vs full 512-bit ITQ R@K."""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sentence_transformers import SentenceTransformer
from yp_bridge import lucas_skeleton

HASH_PATH = "data/itq_hashes_1m.npy"
META_PATH = "data/paper_meta_arxiv_1m.pkl"
QUERY_PATH = "data/paraphrase_queries_500.pkl"
MASK_PATH = "data/lucas_mask_256.bin"
ITQ_PATH = "itq_model_512.npz"

print("Loading metadata and hashes ...")
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
pids = meta["pids"]
titles = meta["titles"]
pid_to_idx = {pid: i for i, pid in enumerate(pids)}

hashes = np.load(HASH_PATH, mmap_mode='r')  # N x 64
N = hashes.shape[0]

# Lucas mask
mask = np.fromfile(MASK_PATH, dtype=np.uint16)
print(f"Lucas mask: {len(mask)} indices")

def batch_skeleton(hashes_2d):
    bits = np.unpackbits(hashes_2d, axis=1)  # N x 512
    skel_bits = bits[:, mask]                # N x 256
    return np.packbits(skel_bits, axis=1)    # N x 32

# Load ITQ model and MiniLM
itq = np.load(ITQ_PATH)
mean = itq["mean"].astype(np.float32)
W = itq["proj"].astype(np.float32)
model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)

def encode_query(text):
    emb = model.encode(text, convert_to_numpy=True)
    x = (emb - mean) @ W
    bits = (x > 0).astype(np.uint8)
    return np.packbits(bits).tobytes()

# Load queries
with open(QUERY_PATH, "rb") as f:
    queries = pickle.load(f)[:200]

# Candidate pool: 100K including all true papers
true_indices = [pid_to_idx[pid] for pid, _, _ in queries]
true_set = set(true_indices)
other = [i for i in range(N) if i not in true_set]
other = np.random.choice(other, 100000 - len(true_indices), replace=False).tolist()
cand_indices = np.array(list(true_set) + other)
np.random.shuffle(cand_indices)

print(f"Candidate pool: {len(cand_indices):,}")

t0 = time.time()
cand_hashes = hashes[cand_indices]
cand_skels = batch_skeleton(cand_hashes)
print(f"Skeletons computed in {time.time()-t0:.1f}s")

K = [1, 10, 50, 100]
hits_512 = {k: 0 for k in K}
hits_256 = {k: 0 for k in K}

for i, (true_pid, title, query_text) in enumerate(queries):
    true_idx_in_cand = np.where(cand_indices == true_indices[i])[0][0]

    h = encode_query(query_text)
    skel = lucas_skeleton(h)

    # 512-bit Hamming
    diffs = np.frombuffer(h, dtype=np.uint8).astype(np.uint8) ^ cand_hashes.astype(np.uint8)
    d512 = np.bitwise_count(diffs).sum(axis=1)

    # 256-bit Hamming
    diffs2 = np.frombuffer(skel, dtype=np.uint8).astype(np.uint8) ^ cand_skels.astype(np.uint8)
    d256 = np.bitwise_count(diffs2).sum(axis=1)

    for k in K:
        if true_idx_in_cand in np.argpartition(d512, k-1)[:k]:
            hits_512[k] += 1
        if true_idx_in_cand in np.argpartition(d256, k-1)[:k]:
            hits_256[k] += 1

    if (i + 1) % 50 == 0:
        print(f"  processed {i+1}/{len(queries)}")

print("\n=== R@K: 512-bit ITQ vs 256-bit Lucas skeleton ===")
print(f"{'K':>5} | {'512-bit':>10} | {'256-bit':>10}")
print("-" * 30)
for k in K:
    print(f"{k:>5} | {hits_512[k]/len(queries)*100:>9.1f}% | {hits_256[k]/len(queries)*100:>9.1f}%")
