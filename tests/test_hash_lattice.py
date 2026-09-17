# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from yp_bridge import hash_to_lattice_6byte, lattice_l1_distance, lattice_batch_convert, lattice_bruteforce_topk

# --- Test 1: Extraction produces 6 bytes ---
h = np.random.bytes(64)
c = hash_to_lattice_6byte(h)
assert len(c) == 6, f"Expected 6, got {len(c)}"
print(f"[PASS] Extract: {len(c)} bytes")

# --- Test 2: Self-distance is zero ---
d = lattice_l1_distance(c, c)
assert d == 0, f"Self-dist should be 0, got {d}"
print(f"[PASS] Self L1: {d}")

# --- Test 3: Two random hashes have non-zero distance ---
c2 = hash_to_lattice_6byte(np.random.bytes(64))
d2 = lattice_l1_distance(c, c2)
assert d2 > 0, f"Random dist should be >0, got {d2}"
print(f"[PASS] Random L1: {d2}")

# --- Test 4: Batch conversion shape ---
batch = np.random.randint(0, 256, size=(100, 64), dtype=np.uint8)
batch_coords = lattice_batch_convert(batch)
assert batch_coords.shape == (100, 6)
print(f"[PASS] Batch convert: {batch_coords.shape}")

# --- Test 5: Batch matches single extraction ---
single = hash_to_lattice_6byte(batch[0].tobytes())
assert batch_coords[0].tobytes() == single
print("[PASS] Batch matches single extraction")

# --- Test 6: Speed ---
t0 = time.perf_counter()
for _ in range(200000):
    hash_to_lattice_6byte(h)
t1 = time.perf_counter()
print(f"[INFO] Extract 6-byte: {(t1-t0)/200000*1e9:.1f} ns")

t0 = time.perf_counter()
for _ in range(1000000):
    lattice_l1_distance(c, c2)
t1 = time.perf_counter()
print(f"[INFO] L1 distance: {(t1-t0)/1000000*1e9:.1f} ns")

# --- Test 7: Brute-force 1M (if coords exist) ---
try:
    coords = np.load("data/lattice_coords_1m_6byte.npy")
    q = hash_to_lattice_6byte(np.random.bytes(64))
    t0 = time.perf_counter()
    idx, scores = lattice_bruteforce_topk(coords, q, top_k=100)
    t1 = time.perf_counter()
    print(f"[INFO] Brute-force {len(coords):,}: {(t1-t0)*1000:.2f} ms | top score: {scores[0]}")
except FileNotFoundError:
    print("[SKIP] 1M coords not found — run scripts/convert_lattice_1m.py first")

print("\nAll smoke tests passed.")
