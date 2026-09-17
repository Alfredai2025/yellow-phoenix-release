# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Generate 512-bit ITQ hashes for the real 1M corpus from cached embeddings."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path

EMB_PATH = Path("data/paper_embeddings_arxiv_1m.npy")
MODEL_PATH = Path("itq_model_512.npz")
OUT_PATH = Path("data/itq_hashes_1m.npy")

if not EMB_PATH.exists():
    raise FileNotFoundError(EMB_PATH)
if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

print(f"Loading ITQ model from {MODEL_PATH} ...")
model = np.load(MODEL_PATH)
mean = model["mean"].astype(np.float32)
input_dim = mean.shape[0]
for key in ("proj", "W", "R"):
    if key in model and model[key].shape[0] == input_dim:
        W = model[key].astype(np.float32)
        print(f"Using projection '{key}' shape {W.shape}")
        break
else:
    raise ValueError("No projection matrix found matching mean dimension")

print(f"Memory-mapping embeddings from {EMB_PATH} ...")
embs = np.load(EMB_PATH, mmap_mode='r')
N, dim = embs.shape
print(f"Embeddings: {N:,} x {dim}")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# Pre-allocate output file as N x 64 uint8
out = np.lib.format.open_memmap(OUT_PATH, mode='w+', dtype=np.uint8, shape=(N, 64))

batch_size = 100_000
batches = (N + batch_size - 1) // batch_size
for b in range(batches):
    start = b * batch_size
    end = min(N, start + batch_size)
    chunk = embs[start:end].astype(np.float32)
    x = (chunk - mean) @ W
    bits = (x > 0).astype(np.uint8)
    h = np.packbits(bits, axis=1)
    out[start:end] = h
    print(f"  batch {b+1}/{batches}: rows {start:,}-{end:,} -> hashes shape {h.shape}")

out.flush()
print(f"\nSaved {N:,} hashes to {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB)")
