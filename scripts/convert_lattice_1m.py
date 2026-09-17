# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""One-time: convert existing ITQ hashes → 6-byte lattice coordinates."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from yp_bridge import lattice_batch_convert

# Adjust path to your existing hashes
candidates = [
    Path("data/itq_hashes_1m.npy"),
    Path("data/paper_hashes_512.npy"),
    Path("paper_hashes_512.npy"),
]

hash_path = None
for c in candidates:
    if c.exists():
        hash_path = c
        break

if hash_path is None:
    print("ERROR: No pre-exported hash file found.")
    print("Expected one of:", [str(c) for c in candidates])
    print("To build from the real 1M corpus, run scripts/generate_itq_hashes_1m.py first.")
    exit(1)

hashes = np.load(hash_path)
if hashes.ndim == 1:
    hashes = hashes.reshape(-1, 64)

if hashes.shape[1] != 64:
    raise ValueError(f"Expected N x 64 uint8 hashes, got shape {hashes.shape}")

print(f"Converting {len(hashes)} hashes from {hash_path} ...")
coords = lattice_batch_convert(hashes)

out_path = Path("data/lattice_coords_1m_6byte.npy")
out_path.parent.mkdir(parents=True, exist_ok=True)
np.save(out_path, coords)
print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
print(f"Compression: {hashes.nbytes / coords.nbytes:.1f}x")
