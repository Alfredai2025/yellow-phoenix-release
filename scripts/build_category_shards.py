#!/usr/bin/env python3
"""Build per-category BinaryHNSW shards from arXiv 1M DB."""
import sys, os, sqlite3, ctypes, pickle
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

YP_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = YP_ROOT / "data" / "papers_arxiv_1m.db"
HASH_PATH = YP_ROOT / "data" / "itq_hashes_1m.npy"
META_PATH = YP_ROOT / "data" / "paper_meta_arxiv_1m.pkl"
LIB_PATH = YP_ROOT / "target" / "release" / "libpams.dylib"
OUT_DIR = YP_ROOT / "data" / "hnsw_shards"
OUT_DIR.mkdir(exist_ok=True)

# ── Load dylib ──
lib = ctypes.CDLL(str(LIB_PATH))

lib.yp_binary_hnsw_new_with_params.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
]
lib.yp_binary_hnsw_new_with_params.restype = ctypes.c_void_p

lib.yp_binary_hnsw_insert.argtypes = [
    ctypes.c_void_p, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
]
lib.yp_binary_hnsw_insert.restype = ctypes.c_int

lib.yp_binary_hnsw_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.yp_binary_hnsw_save.restype = ctypes.c_int

lib.yp_binary_hnsw_free.argtypes = [ctypes.c_void_p]
lib.yp_binary_hnsw_free.restype = None

# ── Load hashes and PID order ──
import numpy as np
hashes = np.load(HASH_PATH).astype(np.uint8)
N, HASH_DIM = hashes.shape
print(f"Loaded {N} hashes, dim={HASH_DIM}")

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
pids = meta["pids"]
pid_to_idx = {pid: i for i, pid in enumerate(pids)}

# ── Load categories from DB, align by PID ──
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT id, categories FROM papers")
rows = cursor.fetchall()

# Group by primary category using hash index as numeric id
shards = defaultdict(list)  # cat -> [(idx, pid)]
unknown_cat = 0
for doc_id, cats_str in rows:
    idx = pid_to_idx.get(doc_id)
    if idx is None:
        continue
    primary = cats_str.split()[0] if cats_str else "misc"
    shards[primary].append((idx, doc_id))

print(f"\n{len(shards)} categories found. Top 15:")
for cat, items in sorted(shards.items(), key=lambda x: -len(x[1]))[:15]:
    print(f"  {cat:24s} {len(items):>8,} papers")

# ── Build each shard ──
for cat, items in shards.items():
    if len(items) < 50:
        print(f"[{cat}] Skipping (only {len(items)} docs)")
        continue

    safe_cat = cat.replace(".", "_")
    out_path = OUT_DIR / f"binary_hnsw_arxiv1m_{safe_cat}.bin"

    print(f"\n[{cat}] Building HNSW with {len(items)} docs...")
    hnsw = lib.yp_binary_hnsw_new_with_params(16, 200, 200)

    for idx, pid in items:
        h = hashes[idx]
        lib.yp_binary_hnsw_insert(
            hnsw, idx,
            h.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), HASH_DIM
        )

    rc = lib.yp_binary_hnsw_save(hnsw, str(out_path).encode())
    lib.yp_binary_hnsw_free(hnsw)
    print(f"  Saved -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, rc={rc})")

print("\n✅ All shards built.")

# ── BUILD GLOBAL INDEX (all papers, no category filter) ──
print("\n" + "=" * 60)
print("BUILDING GLOBAL HNSW INDEX (all papers)")
print("=" * 60)

global_path = YP_ROOT / "data" / "binary_hnsw_arxiv1m_global.bin"
global_hnsw = lib.yp_binary_hnsw_new_with_params(32, 400, 400)

for idx in range(N):
    h = hashes[idx]
    lib.yp_binary_hnsw_insert(
        global_hnsw, idx,
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), HASH_DIM
    )

rc = lib.yp_binary_hnsw_save(global_hnsw, str(global_path).encode())
lib.yp_binary_hnsw_free(global_hnsw)
print(f"  Global index saved -> {global_path} ({global_path.stat().st_size / 1e6:.1f} MB, rc={rc})")
print(f"  Total papers: {N}")
print("=" * 60)
