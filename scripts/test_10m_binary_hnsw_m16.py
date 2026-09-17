#!/usr/bin/env python3
"""Memory-efficient 10M BinaryHNSW validation (M=16).

Generates hashes in chunks so the full 15 GB float32 embedding matrix is never
held simultaneously with the HNSW graph. Reports build time and latency.
"""

import os
import sys
import time
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from yp_bridge import BinaryHNSW

N = 10_000_000
D = 384
CHUNK = 500_000

model_path = os.path.join(project_root, "data", "itq_model_512.npz")
if not os.path.exists(model_path):
    raise RuntimeError(f"ITQ-512 model not found: {model_path}")
d = np.load(model_path)
mean = d["mean"].astype(np.float32)
proj = d["proj"].astype(np.float32)

rng = np.random.default_rng(2026)
ids = np.arange(N, dtype=np.uint64)
hashes = np.empty((N, 64), dtype=np.uint8)

print(f"[10M] Encoding {N} embeddings in chunks of {CHUNK} ...")
t0 = time.time()
for start in range(0, N, CHUNK):
    end = min(start + CHUNK, N)
    chunk = rng.standard_normal((end - start, D)).astype(np.float32)
    norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-10
    chunk /= norms
    x = chunk - mean
    bits = (x @ proj) > 0
    hashes[start:end] = np.packbits(bits.astype(np.uint8), axis=1)
    del chunk, x, bits
    print(f"  encoded {end}/{N} ({time.time()-t0:.1f}s)")
print(f"[10M] Hash encoding done: {time.time()-t0:.1f}s")

idx = BinaryHNSW()
print("[10M] Building BinaryHNSW M=16 ...")
t0 = time.time()
idx.insert_batch(ids, hashes)
build_s = time.time() - t0
print(f"[10M] Build: {build_s:.1f}s")
print(f"[10M] Index count: {len(idx)}")

# Free hashes to measure search-only memory; Rust side owns the graph.
del hashes

# Encode a single query.
q_emb = rng.standard_normal(D).astype(np.float32)
q_emb /= np.linalg.norm(q_emb) + 1e-10
x = q_emb - mean
bits = (x @ proj) > 0
q_hash = np.packbits(bits.astype(np.uint8)).tobytes()

lats = []
for _ in range(100):
    t0 = time.perf_counter()
    idx.search(q_hash, k=10)
    lats.append(time.perf_counter() - t0)
print(f"[10M] P50: {np.median(lats)*1000:.2f} ms")
print(f"[10M] P99: {np.percentile(lats,99)*1000:.2f} ms")
