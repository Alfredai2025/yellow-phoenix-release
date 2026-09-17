# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Sanity + speed test for tabulated ITQ.

NOTE: Bit-for-bit identity with MiniLM+ITQ is NOT achievable because MiniLM
is a deep transformer: its sentence embedding depends on context, position
embeddings, LayerNorm, and self-attention. Tabulating input word embeddings
and summing them gives a fast Bag-of-Embeddings approximation, not the exact
MiniLM output. This test verifies the implementation is deterministic and
measures speed + hash divergence from the standard pipeline.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from yp_bridge import encode_tabulated, encode_tabulated_batch

# Force PyTorch MiniLM to match the embeddings used for ITQ training
import yp_engine
yp_engine._ort_model = False
from yp_engine import get_model

print("Loading ITQ model and MiniLM ...")
itq = np.load("itq_model_512.npz")
mean = itq["mean"].astype(np.float32) if "mean" in itq else itq["m"].astype(np.float32)
input_dim = mean.shape[0]
for key in ("proj", "W", "R"):
    if key in itq and itq[key].shape[0] == input_dim:
        W = itq[key].astype(np.float32)
        break
model = get_model()

def standard_hash(text: str) -> bytes:
    emb = model.encode(text, convert_to_numpy=True)
    x = (emb - mean) @ W
    bits = (x > 0).astype(np.uint8)
    return np.packbits(bits).tobytes()

# --- Basic tests ---
h1 = encode_tabulated("neural network attention mechanism")
assert len(h1) == 64, f"Expected 64 bytes, got {len(h1)}"
print(f"[PASS] Single encode: {len(h1)} bytes")

h2 = encode_tabulated("neural network attention mechanism")
assert h1 == h2, "Hash should be deterministic"
print("[PASS] Deterministic")

h3 = encode_tabulated("machine learning for medical imaging")
assert h1 != h3, "Different text -> different hash"
print("[PASS] Different text")

batch = ["neural network", "machine learning", "quantum error correction"]
hashes = encode_tabulated_batch(batch)
assert len(hashes) == 3 and all(len(h) == 64 for h in hashes)
print("[PASS] Batch encode")

# --- Divergence from standard MiniLM+ITQ ---
print("\nDivergence from standard MiniLM+ITQ (Hamming distance / 512):")
test_titles = [
    "neural network attention mechanism",
    "machine learning for medical imaging",
    "quantum error correction with surface codes",
    "galaxy cluster thermal x-ray spectra",
    "transformer",
    "attention is all you need",
]
for title in test_titles:
    tab = encode_tabulated(title)
    std = standard_hash(title)
    dist = sum((a ^ b).bit_count() for a, b in zip(tab, std))
    print(f"  {title[:45]:45s} {dist:3d}/512")

print("\n  > Tabulated is a fast BoE approximation, not MiniLM-equivalent.")

# --- Speed benchmark ---
N = 1000
t0 = time.perf_counter()
for _ in range(N):
    encode_tabulated("neural network attention mechanism")
t1 = time.perf_counter()
print(f"\n[BENCH] {N} tabulated encodes: {(t1-t0)/N*1e6:.2f} us avg")

t0 = time.perf_counter()
for _ in range(100):
    standard_hash("neural network attention mechanism")
t1 = time.perf_counter()
print(f"[BENCH] 100 standard MiniLM+ITQ encodes: {(t1-t0)/100*1e6:.2f} us avg")

print("\nTabulated ITQ sanity tests passed.")
