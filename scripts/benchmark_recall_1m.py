#!/usr/bin/env python3
"""
benchmark_recall_1m.py — 1M synthetic MiniLM-like embeddings → ITQ hash → YP index → recall benchmark.

Memory-optimized version:
- Deletes raw embeddings as soon as hashes are computed.
- Computes FAISS exact ground truth in chunks.
- Deletes the FAISS index before building the YP index.
"""
import ctypes
import gc
import json
import time
from pathlib import Path

import faiss
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
N_TOTAL = 1_000_000   # total synthetic embeddings
N_INDEX = 990_000     # vectors to index
N_QUERIES = 10_000    # held-out queries
DIM = 384             # MiniLM dimension
K = 10                # top-k for recall
GT_CHUNK = 1_000      # FAISS ground-truth chunk size to limit peak RAM
YP_ROOT = Path.home() / "yellow_phoenix"

# ── 1. Generate synthetic MiniLM-like embeddings ─────────────────────────────
print(f"Generating {N_TOTAL:,} synthetic {DIM}-dim embeddings...")
rng = np.random.default_rng(42)
embeddings = rng.standard_normal((N_TOTAL, DIM), dtype=np.float32)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms

index_vectors = embeddings[:N_INDEX]
query_vectors = embeddings[N_INDEX:]
del embeddings

# ── 2. Load ITQ model ────────────────────────────────────────────────────────
print("Loading ITQ model...")
itq_path = YP_ROOT / "itq_model_512.npz"
if itq_path.exists():
    itq = np.load(itq_path)
    mean = itq["mean"].astype(np.float32)   # (384,)
    R = itq["R"].astype(np.float32)         # (384, 512)
else:
    mean_path = YP_ROOT / "itq_mean.npy"
    r_path = YP_ROOT / "itq_W.npy"
    if not mean_path.exists() or not r_path.exists():
        raise FileNotFoundError(
            f"ITQ model not found. Tried {itq_path}, {mean_path}, {r_path}"
        )
    mean = np.load(mean_path).astype(np.float32)   # (384,)
    R = np.load(r_path).astype(np.float32)         # (384, 512)
    if R.shape[0] != DIM or R.shape[1] != 512:
        raise ValueError(f"Unexpected ITQ rotation shape {R.shape}; expected ({DIM}, 512)")

# ── 3. Hash with ITQ ─────────────────────────────────────────────────────────
def itq_hash(X: np.ndarray) -> np.ndarray:
    bits = (X - mean) @ R >= 0
    return np.packbits(bits, axis=1).astype(np.uint8)

print("Hashing index vectors...")
index_hashes = itq_hash(index_vectors)
print("Hashing query vectors...")
query_hashes = itq_hash(query_vectors)

# ── 4. FAISS exact ground truth (chunked to save RAM) ────────────────────────
print("Building FAISS IndexFlatIP for exact ground truth...")
faiss_index = faiss.IndexFlatIP(DIM)
faiss_index.add(index_vectors)

# Free the raw index vectors now that FAISS has its own copy and hashes are ready.
del index_vectors

gt_ids = np.empty((N_QUERIES, K), dtype=np.int64)
print(f"Searching FAISS for exact top-{K} in chunks of {GT_CHUNK}...")
faiss_start = time.perf_counter()
for start in range(0, N_QUERIES, GT_CHUNK):
    end = min(start + GT_CHUNK, N_QUERIES)
    _, ids = faiss_index.search(query_vectors[start:end], K)
    gt_ids[start:end] = ids
faiss_time = time.perf_counter() - faiss_start
print(f"FAISS ground truth done in {faiss_time:.1f}s")

# Free FAISS index and raw query vectors before YP build.
del faiss_index

gc.collect()

# ── 5. Build YP index via FFI ────────────────────────────────────────────────
print("Loading libpams.dylib...")
lib_path = YP_ROOT / "target" / "release" / "libpams.dylib"
lib = ctypes.CDLL(str(lib_path))

lib.yp_insert_to_mesh_bytes.argtypes = [
    ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.c_size_t,
]
lib.yp_insert_to_mesh_bytes.restype = ctypes.c_int

lib.yp_query_mesh_bytes.argtypes = [
    ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_char_p, ctypes.c_size_t,
]
lib.yp_query_mesh_bytes.restype = ctypes.c_int

lib.yp_build_edges.argtypes = [ctypes.c_int]
lib.yp_build_edges.restype = ctypes.c_int

print(f"Inserting {N_INDEX:,} hashes into YP...")
insert_start = time.perf_counter()
for i in range(N_INDEX):
    h = index_hashes[i]
    coarse = h[:16].tobytes()
    fine = h[:64].tobytes()
    rc = lib.yp_insert_to_mesh_bytes(i, coarse, len(coarse), fine, len(fine))
    if rc < 0:
        raise RuntimeError(f"Insert failed at {i}: {rc}")
    if (i + 1) % 100_000 == 0:
        print(f"  Inserted {i+1:,}/{N_INDEX:,} ({(i+1)/N_INDEX*100:.1f}%)")
insert_time = time.perf_counter() - insert_start
print(f"Insert done in {insert_time:.1f}s ({N_INDEX/insert_time:,.0f} rec/s)")

# Free index hashes before query to lower peak RAM.
del index_hashes

gc.collect()

print("Building YP graph edges...")
lib.yp_build_edges(8)

# ── 6. Query YP ──────────────────────────────────────────────────────────────
print(f"Querying YP with {N_QUERIES:,} vectors...")
out_buf = ctypes.create_string_buffer(16384)
yp_results = []

query_start = time.perf_counter()
for i in range(N_QUERIES):
    h = query_hashes[i]
    coarse = h[:16].tobytes()
    fine = h[:64].tobytes()
    written = lib.yp_query_mesh_bytes(
        coarse, len(coarse), fine, len(fine),
        K, 100, out_buf, len(out_buf),
    )
    if written <= 0:
        yp_results.append(np.array([], dtype=np.int64))
        continue
    try:
        payload = json.loads(out_buf.value.decode("utf-8"))
        ids = [int(item["id"]) for item in payload.get("results", [])]
        yp_results.append(np.array(ids, dtype=np.int64))
    except Exception:
        yp_results.append(np.array([], dtype=np.int64))

    if (i + 1) % 1000 == 0:
        print(f"  Queried {i+1:,}/{N_QUERIES:,}")

query_time = time.perf_counter() - query_start
p50_us = (query_time / N_QUERIES) * 1e6
print(f"Query done in {query_time:.1f}s (mean/P50 ~{p50_us:.1f} µs)")

# ── 7. Compute recall ────────────────────────────────────────────────────────
def compute_recall(yp_results, gt_ids, k):
    hits = 0
    total = 0
    for i, yp_ids in enumerate(yp_results):
        gt_set = set(gt_ids[i, :k])
        yp_set = set(yp_ids[:k]) if len(yp_ids) > 0 else set()
        hits += len(gt_set & yp_set)
        total += k
    return hits / total if total > 0 else 0.0

r1 = compute_recall(yp_results, gt_ids, 1)
r5 = compute_recall(yp_results, gt_ids, 5)
r10 = compute_recall(yp_results, gt_ids, 10)

# ── Report ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1M RECALL BENCHMARK RESULTS")
print("=" * 60)
print(f"Index size:    {N_INDEX:,} vectors")
print(f"Queries:       {N_QUERIES:,}")
print(f"YP insert:     {insert_time:.1f}s ({N_INDEX/insert_time:,.0f} rec/s)")
print(f"YP query P50:  {p50_us:.1f} µs")
print(f"Recall@1:      {r1:.4f}")
print(f"Recall@5:      {r5:.4f}")
print(f"Recall@10:     {r10:.4f}")
print("=" * 60)

results = {
    "n_index": N_INDEX,
    "n_queries": N_QUERIES,
    "insert_s": round(insert_time, 2),
    "insert_rate": round(N_INDEX / insert_time, 0),
    "query_p50_us": round(p50_us, 2),
    "r1": round(r1, 4),
    "r5": round(r5, 4),
    "r10": round(r10, 4),
}
out_path = YP_ROOT / "data" / "benchmark_recall_1m.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out_path}")
