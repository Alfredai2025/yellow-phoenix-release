#!/usr/bin/env python3
"""
YP vs FAISS-HNSW on 100K — FAST PATH benchmark.
YP: build 100K mesh, query via yp_unified_mesh_query_fast (coarse-bucket fast path).
FAISS: HNSW on same embeddings.
"""
import numpy as np
import faiss
import time
import sys
import os
import json
import ctypes

sys.path.insert(0, '.')
from yp_bridge import RustBridge

N_DB = 100_000
N_Q = 10_000
K = 10
D = 384

np.random.seed(42)

print("=" * 70)
print("YP vs FAISS-HNSW (YP FAST PATH)")
print("=" * 70)

# ── Load data ───────────────────────────────────────────────────────
print("\n[1] Loading 100K embeddings + hashes...")
embs = np.load("data/paper_embeddings_100k.npy").astype(np.float32)
hashes = np.load("data/paper_hashes_100k.npy").astype(np.uint8)
assert len(embs) == N_DB and len(hashes) == N_DB

q_idx = np.random.choice(N_DB, N_Q, replace=False)
queries = embs[q_idx]
query_hashes = hashes[q_idx]

# ── Build YP 100K mesh ──────────────────────────────────────────────
print("\n[2] Building YP 100K mesh (bucket index only)...")
rb = RustBridge()
handle = rb.mesh_512
if not handle:
    print("❌ YP 512-bit mesh handle not available")
    sys.exit(1)

t0 = time.time()
for i in range(N_DB):
    rb.yp_unified_mesh_insert_512(handle, str(i), hashes[i].tobytes().hex())
yp_build_total = time.time() - t0
print(f"    Build: {yp_build_total:.1f}s")

# Fast path needs bucket arrays populated. Build edges with max_edges=0 so
# no expensive graph edges are created, but bucket lookup works.
t0 = time.time()
rb.yp_unified_mesh_build_edges(handle, 0)
yp_edge_time = time.time() - t0
yp_build_total += yp_edge_time
print(f"    Bucket build: {yp_edge_time:.1f}s")

# ── YP Fast Query (coarse-bucket fast path) ────────────────────────
print(f"\n[3] YP fast query {N_Q:,}...")
query_func = rb.available.get("yp_unified_mesh_query_fast")
if not query_func:
    print("❌ yp_unified_mesh_query_fast not available")
    sys.exit(1)


def fast_query(hash_bytes):
    hash_hex_b = hash_bytes.hex().encode('utf-8')
    ptr = query_func(handle, hash_hex_b, len(hash_hex_b), K)
    if not ptr:
        return []
    s = ctypes.c_char_p(ptr).value
    if rb.available.get("yp_free_string"):
        rb.lib.yp_free_string(ptr)
    if not s:
        return []
    raw = s.decode('utf-8', errors='ignore') if isinstance(s, bytes) else s
    try:
        data = json.loads(raw)
        return [(r['id'], r.get('score', 0.0)) for r in data.get('results', [])]
    except Exception:
        return []


t0 = time.time()
yp_results = []
for i in range(N_Q):
    yp_results.append(fast_query(query_hashes[i].tobytes()))
yp_query_time = time.time() - t0
yp_p50 = (yp_query_time / N_Q) * 1_000_000
yp_qps = N_Q / yp_query_time
print(f"    P50 (fast path): {yp_p50:.1f} µs")
print(f"    Throughput:      {yp_qps:.0f} qps")

# R@1: mesh IDs start at 1, so ground truth = q_idx[i] + 1
yp_hits = 0
for i, idx in enumerate(q_idx):
    if yp_results[i] and str(yp_results[i][0][0]) == str(idx + 1):
        yp_hits += 1
yp_r1 = yp_hits / N_Q
print(f"    R@1 (self-query): {yp_r1:.1%}")

# ── FAISS HNSW Build ───────────────────────────────────────────────
print("\n[4] FAISS HNSW build...")
index = faiss.IndexHNSWFlat(D, 32)
index.hnsw.efConstruction = 200
t0 = time.time()
index.add(embs)
faiss_build = time.time() - t0
print(f"    Build: {faiss_build:.1f}s")

# ── FAISS HNSW Query ──────────────────────────────────────────────
print(f"\n[5] FAISS HNSW query {N_Q:,}...")
index.hnsw.efSearch = 128
t0 = time.time()
D_faiss, I_faiss = index.search(queries, K)
faiss_total = time.time() - t0
faiss_p50 = (faiss_total / N_Q) * 1_000_000
faiss_qps = N_Q / faiss_total
print(f"    P50:        {faiss_p50:.1f} µs")
print(f"    Throughput: {faiss_qps:.0f} qps")

faiss_r1 = sum(1 for i, idx in enumerate(q_idx) if idx in I_faiss[i][:1]) / N_Q
print(f"    R@1:        {faiss_r1:.1%}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FAST PATH COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Metric':<28} {'FAISS HNSW':>18} {'YP fast path':>18}")
print("-" * 70)
print(f"{'Build time':<28} {faiss_build:>17.1f}s {yp_build_total:>17.1f}s")
print(f"{'P50 latency':<28} {faiss_p50:>17.1f}µs {yp_p50:>17.1f}µs")
print(f"{'Throughput (qps)':<28} {faiss_qps:>18.0f} {yp_qps:>18.0f}")
print(f"{'R@1 (self-query)':<28} {faiss_r1:>18.1%} {yp_r1:>18.1%}")
print("-" * 70)
if faiss_p50 > yp_p50:
    print(f"YP fast path is {faiss_p50 / yp_p50:.1f}x faster than FAISS-HNSW")
else:
    print("WARNING: YP fast path is NOT faster than FAISS-HNSW on this test")
print("=" * 70)
