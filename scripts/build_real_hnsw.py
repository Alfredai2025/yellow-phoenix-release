#!/usr/bin/env python3
"""
Build REAL 10M and 20M Binary HNSW indexes from synthetic ISM hashes.
Uses hnswlib with L2 distance on 512-bit (64-byte) vectors.

Output format:
  Header: magic(u32)=0x48505331, version(u32)=1, count(u64), dim(u64), M(u32), ef(u32)
  Vectors: count * dim bytes (uint8)
  Graph: hnswlib native serialization
"""

import numpy as np
import struct
import os
from pathlib import Path

try:
    import hnswlib
except ImportError:
    print("Installing hnswlib...")
    os.system("pip3 install hnswlib")
    import hnswlib

DATA_DIR = Path.home() / "yellow_phoenix" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 400
DIM = 64

def load_ism_hashes(path: Path):
    print(f"  Loading hashes from {path.name}...")
    with open(path, "rb") as f:
        count = struct.unpack("<Q", f.read(8))[0]
        hashes = np.frombuffer(f.read(count * DIM), dtype=np.uint8).reshape(count, DIM)
        f.seek(count * 8, 1)  # Skip ids
    print(f"  Loaded {count:,} hashes ({hashes.nbytes / 1024**3:.2f} GB)")
    return hashes

def build_hnsw(hashes: np.ndarray, M: int, ef_construction: int):
    count, dim = hashes.shape
    print(f"  Building HNSW: count={count:,}, M={M}, ef_construction={ef_construction}")
    
    data = hashes.astype(np.float32)
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=count, ef_construction=ef_construction, M=M)
    index.set_ef(EF_SEARCH)
    
    batch_size = 50_000
    for i in range(0, count, batch_size):
        end = min(i + batch_size, count)
        index.add_items(data[i:end], np.arange(i, end, dtype=np.int32))
        if (i // batch_size) % 5 == 0 or end == count:
            print(f"    Progress: {end:,} / {count:,} ({100*end/count:.1f}%)")
    
    print(f"  Index built. Elements: {index.element_count}")
    return index

def save_hnsw_binary(index, path: Path, hashes: np.ndarray, M: int, ef: int):
    count, dim = hashes.shape
    print(f"  Saving to {path.name}...")
    
    graph_bytes = index.save_index_to_bytes()
    
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x48505331))   # magic "HPS1"
        f.write(struct.pack("<I", 1))             # version
        f.write(struct.pack("<Q", count))          # num vectors
        f.write(struct.pack("<Q", dim))            # dim in bytes
        f.write(struct.pack("<I", M))              # M
        f.write(struct.pack("<I", ef))             # ef_construction
        f.write(struct.pack("<I", EF_SEARCH))      # ef_search
        f.write(struct.pack("<I", 0))              # reserved
        f.write(hashes.tobytes())
        f.write(graph_bytes)
    
    size_gb = path.stat().st_size / 1024**3
    print(f"  Saved: {path.name} ({size_gb:.2f} GB)")

def build_scale(scale: str):
    ism_path = DATA_DIR / f"synth_{scale}.ism"
    if not ism_path.exists():
        print(f"\n❌ Missing {ism_path}")
        return
    
    out_path = DATA_DIR / f"synth_{scale}_hnsw.bin"
    if out_path.exists():
        print(f"\n⚠️  {out_path.name} already exists ({out_path.stat().st_size / 1024**3:.2f} GB)")
        print(f"   Delete to rebuild: rm {out_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Building REAL {scale.upper()} HNSW")
    print(f"{'='*60}")
    
    hashes = load_ism_hashes(ism_path)
    index = build_hnsw(hashes, M, EF_CONSTRUCTION)
    save_hnsw_binary(index, out_path, hashes, M, EF_CONSTRUCTION)
    print(f"  ✅ Done: {out_path}")

if __name__ == "__main__":
    print("="*60)
    print("REAL HNSW Index Builder for Yellow Phoenix")
    print("="*60)
    
    build_scale("10m")
    build_scale("20m")
    
    print("\n" + "="*60)
    print("Done. Push to iPhone with:")
    print("  cd ~/yellow_phoenix_mobile && ./scripts/push_now.sh")
    print("="*60)
