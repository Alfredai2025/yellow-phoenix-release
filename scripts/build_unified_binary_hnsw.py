#!/usr/bin/env python3
"""Build a 512-bit BinaryHNSW index from paper_hashes_unified.pkl.

Stores node IDs as Rust FNV-1a u64 hashes of the string paper IDs,
matching _rust_id(). Covers all papers present in the unified hash file.
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

from yp_bridge import RustBridge, BinaryHNSW, _rust_id

HASH_PKL = ROOT / "data" / "paper_hashes_unified.pkl"
SAVE_PATH = ROOT / "data" / "binary_hnsw_arxiv1m_m16.bin"

# HNSW parameters
M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 128
BATCH = 10_000


def main():
    if not HASH_PKL.exists():
        raise SystemExit(f"Hash pkl not found: {HASH_PKL}")

    print("Loading unified hashes...")
    t0 = time.time()
    with open(HASH_PKL, "rb") as f:
        hash_dict = pickle.load(f)
    print(f"  loaded {len(hash_dict):,} hashes in {time.time()-t0:.1f}s")

    print("Creating BinaryHNSW index...")
    bridge = RustBridge()
    hnsw = BinaryHNSW(bridge, m=M, ef_construction=EF_CONSTRUCTION, ef_search=EF_SEARCH)

    ids = np.empty(BATCH, dtype=np.uint64)
    hashes = np.empty((BATCH, 64), dtype=np.uint8)

    batch = []
    inserted = 0
    t0 = time.time()

    for pid, h in hash_dict.items():
        if len(h) != 64:
            continue
        batch.append((_rust_id(pid), np.frombuffer(h, dtype=np.uint8)))
        if len(batch) >= BATCH:
            for i, (rid, hh) in enumerate(batch):
                ids[i] = rid
                hashes[i] = hh
            hnsw.insert_batch(ids, hashes)
            inserted += len(batch)
            batch.clear()
            if inserted % 100_000 == 0:
                print(f"  inserted {inserted:,} ... {time.time()-t0:.1f}s  count={len(hnsw)}")

    if batch:
        ids = np.empty(len(batch), dtype=np.uint64)
        hashes = np.empty((len(batch), 64), dtype=np.uint8)
        for i, (rid, hh) in enumerate(batch):
            ids[i] = rid
            hashes[i] = hh
        hnsw.insert_batch(ids, hashes)
        inserted += len(batch)

    print(f"\nIndex built: {len(hnsw):,} nodes in {time.time()-t0:.1f}s")

    # Backup existing file if present
    if SAVE_PATH.exists():
        backup = SAVE_PATH.with_suffix(f".bin.bak.{int(time.time())}")
        SAVE_PATH.rename(backup)
        print(f"  backed up old index to {backup}")

    print(f"Saving to {SAVE_PATH}...")
    t0 = time.time()
    if hnsw.save(str(SAVE_PATH)):
        print(f"  saved {SAVE_PATH.stat().st_size:,} bytes in {time.time()-t0:.1f}s")
    else:
        raise RuntimeError("save returned False")


if __name__ == "__main__":
    main()
