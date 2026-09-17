# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark HNSW index build time and size."""
import numpy as np
import time
import sys
import os

try:
    import hnswlib
except ImportError:
    print("Installing hnswlib...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hnswlib", "-q"])
    import hnswlib

SCALES = [100_000, 500_000, 1_000_000]
DIM = 384
M = 16
EF_CONSTRUCT = 200
np.random.seed(42)

print("=" * 60)
print("YELLOW PHOENIX -- HNSW Index Build")
print("=" * 60)

for N_DOCS in SCALES:
    emb = np.random.randn(N_DOCS, DIM).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    t0 = time.time()
    index = hnswlib.Index(space='ip', dim=DIM)
    index.init_index(max_elements=N_DOCS, ef_construction=EF_CONSTRUCT, M=M)
    index.add_items(emb)
    build_time = time.time() - t0

    index.save_index(f"/tmp/hnsw_{N_DOCS}.bin")
    file_size = os.path.getsize(f"/tmp/hnsw_{N_DOCS}.bin")

    print("\n{:>10,} docs | Build: {:>6.1f}s | Index: {:>6.1f} MB | Per-doc: {:>5.2f} bytes".format(
        N_DOCS, build_time, file_size/1e6, file_size/N_DOCS))

print("\n" + "=" * 60)
