# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from yp_bridge import lucas_skeleton, lucas_reconstruct_512, hamming_256

# Test 1: Extract -> 32 bytes
h = np.random.bytes(64)
skel = lucas_skeleton(h)
assert len(skel) == 32, f"Expected 32, got {len(skel)}"
print(f"[PASS] Extract: {len(skel)} bytes")

# Test 2: Self-distance = 0
d = hamming_256(skel, skel)
assert d == 0, f"Self-dist should be 0, got {d}"
print(f"[PASS] Self Hamming: {d}")

# Test 3: Different -> non-zero
skel2 = lucas_skeleton(np.random.bytes(64))
d2 = hamming_256(skel, skel2)
assert d2 > 0, f"Random dist should be >0, got {d2}"
print(f"[PASS] Random Hamming: {d2}")

# Test 4: Round-trip reconstruction preserves skeleton bits
h_back = lucas_reconstruct_512(skel)
assert len(h_back) == 64
skel_rt = lucas_skeleton(h_back)
assert skel == skel_rt, "Lucas reconstruction must preserve skeleton bits"
print("[PASS] Round-trip reconstruction")

# Test 5: Speed
t0 = time.perf_counter()
for _ in range(10000):
    lucas_skeleton(h)
t1 = time.perf_counter()
print(f"[BENCH] Skeleton extract: {(t1-t0)/10000*1e6:.2f} us")

t0 = time.perf_counter()
for _ in range(10000):
    hamming_256(skel, skel2)
t1 = time.perf_counter()
print(f"[BENCH] Hamming 256: {(t1-t0)/10000*1e6:.2f} us")

print("\nAll Lucas skeleton tests passed.")
