#!/usr/bin/env python3
"""Build compact OPQ-PQ index files for the Rust re-ranker.

Usage:
    python build_pq_index.py <embeddings.npy> <opq.npz> <pq.npz> <out_prefix> [ids.npy]

Outputs (all little-endian):
    <out_prefix>_opq.bin       : [384*384 f32 R][384 f32 mean]
    <out_prefix>_codebooks.bin : [8*256*48 f32 centroids]
    <out_prefix>_index.bin     : [u64 N][N*8 bytes PQ codes]
    <out_prefix>_ids.bin       : [N u64 IDs]  (only if ids.npy is supplied)

If <opq.npz> does not contain a 'mean' array, the mean is computed from the
embeddings file in 50k-row chunks.
"""

import sys, struct, numpy as np
from pathlib import Path

embed_path = Path(sys.argv[1])
opq_path = Path(sys.argv[2])
pq_path = Path(sys.argv[3])
out_prefix = Path(sys.argv[4])
ids_path = Path(sys.argv[5]) if len(sys.argv) > 5 else None

opz = np.load(opq_path)
R = opz['R'].astype(np.float32)
if R.shape != (384, 384):
    raise ValueError(f"R shape must be (384, 384), got {R.shape}")

if 'mean' in opz.files:
    mean = opz['mean'].astype(np.float32)
else:
    print("Computing mean from embeddings...")
    embeddings = np.load(embed_path, mmap_mode='r')
    N = len(embeddings)
    accum = np.zeros(384, dtype=np.float64)
    chunk = 50000
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        accum += embeddings[start:end].astype(np.float64).sum(axis=0)
    mean = (accum / N).astype(np.float32)
    print(f"  mean norm = {np.linalg.norm(mean):.4f}")

pqz = np.load(pq_path)
codebooks = pqz['codebooks'].astype(np.float32)
if codebooks.shape != (8, 256, 48):
    raise ValueError(f"codebooks shape must be (8, 256, 48), got {codebooks.shape}")

with open(out_prefix.parent / (out_prefix.name + '_opq.bin'), 'wb') as f:
    f.write(R.tobytes())
    f.write(mean.tobytes())

with open(out_prefix.parent / (out_prefix.name + '_codebooks.bin'), 'wb') as f:
    f.write(codebooks.tobytes())

embeddings = np.load(embed_path, mmap_mode='r')
N = len(embeddings)
with open(out_prefix.parent / (out_prefix.name + '_index.bin'), 'wb') as f:
    f.write(struct.pack('<Q', N))
    for start in range(0, N, 50000):
        end = min(start + 50000, N)
        batch = embeddings[start:end].astype(np.float32) - mean
        rot = batch @ R.T
        codes = np.empty((end - start, 8), dtype=np.uint8)
        for s in range(8):
            sub = rot[:, s * 48:(s + 1) * 48]
            cents = codebooks[s]
            d2 = np.sum(sub ** 2, 1)[:, None] + np.sum(cents ** 2, 1)[None, :] - 2 * (sub @ cents.T)
            codes[:, s] = np.argmin(d2, axis=1).astype(np.uint8)
        f.write(codes.tobytes())
        print(f"  {end}/{N}")

if ids_path is not None:
    ids = np.load(ids_path)
    if len(ids) != N:
        raise ValueError(f"ids length {len(ids)} does not match embeddings {N}")
    with open(out_prefix.parent / (out_prefix.name + '_ids.bin'), 'wb') as f:
        f.write(ids.astype(np.uint64).tobytes())
    print(f"Wrote ids ({ids.dtype})")

print("Done")
