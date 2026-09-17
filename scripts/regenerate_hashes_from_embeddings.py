#!/usr/bin/env python3
"""Regenerate ITQ hashes directly from the unified embedding matrix.

This avoids re-encoding 1.27M titles through MiniLM. The embedding matrix is
assumed to be compatible with the current ITQHasher (verified by sampling
titles before running this script).
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from yp_bridge import ITQHasher, YP_ROOT

EMB_PATH = Path(YP_ROOT) / "data" / "paper_embeddings_unified.npy"
MAP_PATH = Path(YP_ROOT) / "data" / "unified_pid_to_emb_idx.pkl"
OUT_UNIFIED = Path(YP_ROOT) / "data" / "paper_hashes_unified.pkl"
OUT_ARXIV = Path(YP_ROOT) / "data" / "paper_hashes_arxiv_1m.pkl"


def main():
    print("Loading ITQ model...")
    hasher = ITQHasher()
    mean = hasher.mean.astype(np.float64)
    proj = hasher.proj.astype(np.float64)
    n_bits = hasher.n_bits
    print(f"  ITQ: {n_bits}-bit, mean={mean.shape}, proj={proj.shape}")

    print(f"Loading pid→row mapping from {MAP_PATH}...")
    with open(MAP_PATH, "rb") as f:
        pid_to_idx = pickle.load(f)
    print(f"  {len(pid_to_idx):,} pids")

    # Verify ordering: pid_to_idx values should be 0..N-1
    idx_to_pid = {v: k for k, v in pid_to_idx.items()}
    n_rows = len(pid_to_idx)
    assert len(idx_to_pid) == n_rows, "pid_to_idx is not a bijection"
    assert set(pid_to_idx.values()) == set(range(n_rows)), "pid_to_idx values are not contiguous"

    print(f"Loading embeddings from {EMB_PATH} (mmap)...")
    embs = np.load(EMB_PATH, mmap_mode="r")
    assert embs.shape[0] == n_rows, f"embedding rows {embs.shape[0]} != pid count {n_rows}"
    print(f"  shape: {embs.shape}")

    # Compute hashes in chunks to control memory
    chunk = 50_000
    n = n_rows
    bytes_per_hash = (n_bits + 7) // 8  # 64 for 512 bits
    packed = np.empty((n, bytes_per_hash), dtype=np.uint8)

    print("Computing ITQ binary hashes in chunks...")
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = np.asarray(embs[start:end], dtype=np.float64)
        V = (block - mean) @ proj
        bits = (V >= 0).astype(np.uint8)
        packed[start:end] = np.packbits(bits, axis=1)
        if (start // chunk) % 5 == 0:
            print(f"  processed {end:,}/{n:,} in {time.time()-t0:.1f}s")

    print(f"Hashed {n:,} rows in {time.time()-t0:.1f}s")

    # Build unified hash dict
    print("Building unified hash dict...")
    unified = {}
    for idx in range(n):
        unified[idx_to_pid[idx]] = packed[idx].tobytes()

    print(f"Saving {OUT_UNIFIED} ({len(unified):,} entries)...")
    with open(OUT_UNIFIED, "wb") as f:
        pickle.dump(unified, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ArXiv-only subset
    print("Building arXiv-only subset...")
    arxiv = {pid: h for pid, h in unified.items() if pid.startswith("arxiv1m:")}
    print(f"  {len(arxiv):,} arxiv entries")
    with open(OUT_ARXIV, "wb") as f:
        pickle.dump(arxiv, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nDone.")
    print(f"  unified: {OUT_UNIFIED} ({OUT_UNIFIED.stat().st_size:,} bytes)")
    print(f"  arxiv:   {OUT_ARXIV} ({OUT_ARXIV.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
