# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Regenerate 512-bit ITQ hashes for full 1.27M corpus."""
import numpy as np
import pickle
import time

print("Loading ITQ model...")
model = np.load("data/itq_model_512.npz")
m = model["mean"].astype(np.float32)
R = None
for key in ("proj", "W", "R"):
    if key in model and model[key].shape[0] == m.shape[0]:
        R = model[key].astype(np.float32)
        break
if R is None:
    raise ValueError("No usable projection in itq_model_512.npz")
print(f"Model: mean {m.shape}, proj {R.shape}")

print("Loading 1.27M embeddings...")
embs = np.load("data/paper_embeddings_arxiv_1m.npy").astype(np.float32)
n, d = embs.shape
print(f"Embeddings: {n:,} x {d}")

print("Projecting to 512-bit hashes...")
t0 = time.time()
batch_size = 50000
all_bits = []
for i in range(0, n, batch_size):
    batch = embs[i:i+batch_size]
    x = batch - m
    x = x @ R
    bits = (x > 0).astype(np.uint8)
    all_bits.append(bits)
    if i % 200000 == 0:
        print(f"  {i:,}/{n:,} done")

all_bits = np.vstack(all_bits)
print(f"Projection done in {time.time()-t0:.1f}s")

print("Packing bits...")
packed = np.packbits(all_bits, axis=1)
print(f"Packed shape: {packed.shape}")

print("Loading metadata for pids...")
with open("data/paper_meta_arxiv_1m.pkl", "rb") as f:
    meta = pickle.load(f)
pids = meta["pids"]
if len(pids) != n:
    raise ValueError(f"pid count {len(pids)} != embedding count {n}")

hash_dict = {pid: packed[i].tobytes() for i, pid in enumerate(pids)}
print(f"Hash dict: {len(hash_dict):,} entries")

out_path = "data/paper_hashes_arxiv_1m.pkl"
with open(out_path, "wb") as f:
    pickle.dump(hash_dict, f)
print(f"Saved: {out_path} ({len(packed)*64/8/1024/1024:.1f} MB)")

with open(out_path, "rb") as f:
    verify = pickle.load(f)
print(f"Verified: {len(verify):,} hashes loaded")
sample_pid = pids[0]
print(f"Sample pid {sample_pid}: {len(verify[sample_pid])} bytes")
print("DONE")
