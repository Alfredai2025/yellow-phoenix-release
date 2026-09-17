#!/usr/bin/env python3
"""
Soft Encoder Tuning Rig — dial parameters, read signal.

DIALS (edit these):
  SELF_MATCH_N      — papers to insert AND query (should hit ~100%)
  SEMANTIC_INDEX_N  — papers to build index from
  SEMANTIC_QUERY_N  — held-out papers to query
  HNSW_M            — graph connectivity (higher = better recall, slower build)
  HNSW_EF_BUILD     — construction quality (higher = better graph)
  HNSW_EF_SEARCH    — search depth (higher = better recall, slower query)
  OVERLAP_TOP_K     — how many neighbors to compare for overlap
"""

# ===================== DIALS =====================
SELF_MATCH_N      = 1000   # Self-match sanity check size
SEMANTIC_INDEX_N  = 10000  # Index size for semantic test
SEMANTIC_QUERY_N  = 500    # Held-out queries for semantic test
HNSW_M            = 16     # Graph degree
HNSW_EF_BUILD     = 200    # Construction ef
HNSW_EF_SEARCH    = 200    # Query-time ef
OVERLAP_TOP_K     = 5      # Compare top-1, top-5, top-10 overlap
# ================================================

import ctypes, os, sys, sqlite3, hashlib, time, numpy as np
from sentence_transformers import SentenceTransformer

lib = ctypes.CDLL(os.path.abspath("target/release/libpams.dylib"))

# FFI setup
lib.yp_hierarchical_load.argtypes = [ctypes.c_char_p]
lib.yp_hierarchical_load.restype = ctypes.c_int
lib.yp_hierarchical_encode.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
lib.yp_hierarchical_encode.restype = ctypes.c_int
lib.yp_binary_hnsw_new_with_params.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
lib.yp_binary_hnsw_new_with_params.restype = ctypes.c_void_p
lib.yp_binary_hnsw_insert.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
lib.yp_binary_hnsw_insert.restype = ctypes.c_int
lib.yp_binary_hnsw_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
lib.yp_binary_hnsw_search.restype = ctypes.c_size_t
lib.yp_binary_hnsw_free.argtypes = [ctypes.c_void_p]

rc = lib.yp_hierarchical_load(os.path.abspath("data").encode())
print(f"[+] yp_hierarchical_load: {rc} (0=OK)")

def id_to_u64(pid):
    return int(hashlib.sha256(pid.encode()).hexdigest()[:16], 16)

# Load corpus
conn = sqlite3.connect("data/phoenix_arxiv_1m.db")
cur = conn.cursor()
cur.execute("SELECT id, title, abstract FROM papers ORDER BY RANDOM() LIMIT ?", 
            (SELF_MATCH_N + SEMANTIC_INDEX_N + SEMANTIC_QUERY_N,))
rows = cur.fetchall()
conn.close()

# Split into three disjoint pools
self_match_rows   = rows[:SELF_MATCH_N]
index_rows        = rows[SELF_MATCH_N : SELF_MATCH_N + SEMANTIC_INDEX_N]
query_rows        = rows[SELF_MATCH_N + SEMANTIC_INDEX_N :]

# ============================================================
# TEST 1: SELF-MATCH (determinism + index integrity)
# Insert N papers. Query the SAME N papers.
# Expected: ~100% R@1. Proves encoder is stable and index works.
# ============================================================
print(f"\n{'='*60}")
print(f"TEST 1: SELF-MATCH (N={SELF_MATCH_N})")
print(f"{'='*60}")

hnsw_self = lib.yp_binary_hnsw_new_with_params(HNSW_M, HNSW_EF_BUILD, HNSW_EF_SEARCH)

for pid, title, abstract in self_match_rows:
    text = f"{title or ''} {abstract or ''}".strip()
    if not text: continue
    out = (ctypes.c_uint8 * 64)()
    if lib.yp_hierarchical_encode(text.encode(), out, 64) != 0: continue
    lib.yp_binary_hnsw_insert(hnsw_self, id_to_u64(pid), out, 64)

correct = total = 0
latencies = []
for pid, title, abstract in self_match_rows:
    text = f"{title or ''} {abstract or ''}".strip()
    if not text: continue
    t0 = time.perf_counter()
    out = (ctypes.c_uint8 * 64)()
    if lib.yp_hierarchical_encode(text.encode(), out, 64) != 0: continue
    ids = (ctypes.c_uint64 * 2)()
    dists = (ctypes.c_uint32 * 2)()
    n = lib.yp_binary_hnsw_search(hnsw_self, out, 64, 2, ids, dists, 2)
    t1 = time.perf_counter()
    latencies.append((t1-t0)*1000)
    
    found = any(ids[i] == id_to_u64(pid) for i in range(min(n, 2)))
    if found: correct += 1
    total += 1

print(f"  R@1: {correct/total*100:.1f}% ({correct}/{total})")
print(f"  P50: {np.median(latencies):.3f} ms")
print(f"  P99: {np.percentile(latencies, 99):.3f} ms")
lib.yp_binary_hnsw_free(hnsw_self)

# ============================================================
# TEST 2: SEMANTIC OVERLAP vs MiniLM+ITQ ground truth
# Build index with fast encoder on INDEX pool.
# For each query in QUERY pool:
#   A) Find top-K neighbors using fast encoder + HNSW
#   B) Find top-K neighbors using MiniLM+ITQ cosine similarity
#   C) Measure overlap between A and B
# Signal = overlap %. Higher = fast encoder preserves semantics.
# ============================================================
print(f"\n{'='*60}")
print(f"TEST 2: SEMANTIC OVERLAP (index={SEMANTIC_INDEX_N}, query={SEMANTIC_QUERY_N})")
print(f"{'='*60}")

hnsw_sem = lib.yp_binary_hnsw_new_with_params(HNSW_M, HNSW_EF_BUILD, HNSW_EF_SEARCH)

# Build fast index
for pid, title, abstract in index_rows:
    text = f"{title or ''} {abstract or ''}".strip()
    if not text: continue
    out = (ctypes.c_uint8 * 64)()
    if lib.yp_hierarchical_encode(text.encode(), out, 64) != 0: continue
    lib.yp_binary_hnsw_insert(hnsw_sem, id_to_u64(pid), out, 64)

# Load MiniLM + ITQ for ground truth
print("  [+] Loading MiniLM + ITQ for ground truth...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
itq = np.load('data/itq_model_512.npz')
mean = itq['mean']
M = itq['proj']

# Embed index pool
index_texts = [f"{t or ''} {a or ''}".strip() for _, t, a in index_rows]
index_embs = model.encode(index_texts, convert_to_numpy=True, show_progress_bar=True)
index_proj = (index_embs - mean) @ M

# Embed query pool
query_texts = [f"{t or ''} {a or ''}".strip() for _, t, a in query_rows]
query_embs = model.encode(query_texts, convert_to_numpy=True, show_progress_bar=True)
query_proj = (query_embs - mean) @ M

# Compute ground truth top-K for each query using cosine in ITQ space
print(f"  [+] Computing ground truth top-{OVERLAP_TOP_K}...")
gt_topk = []
for q_vec in query_proj:
    sims = index_proj @ q_vec
    top_idx = np.argpartition(-sims, OVERLAP_TOP_K)[:OVERLAP_TOP_K]
    gt_topk.append(set(index_rows[i][0] for i in top_idx))

# Build reverse map: u64 -> pid
u64_to_pid = {id_to_u64(pid): pid for pid, _, _ in index_rows}

fast_results = []
fast_latencies = []

for qi, (pid, title, abstract) in enumerate(query_rows):
    text = f"{title or ''} {abstract or ''}".strip()
    if not text: continue
    t0 = time.perf_counter()
    out = (ctypes.c_uint8 * 64)()
    if lib.yp_hierarchical_encode(text.encode(), out, 64) != 0: continue
    ids = (ctypes.c_uint64 * OVERLAP_TOP_K)()
    dists = (ctypes.c_uint32 * OVERLAP_TOP_K)()
    n = lib.yp_binary_hnsw_search(hnsw_sem, out, 64, OVERLAP_TOP_K, ids, dists, OVERLAP_TOP_K)
    t1 = time.perf_counter()
    fast_latencies.append((t1-t0)*1000)
    
    fast_pids = set()
    for i in range(min(n, OVERLAP_TOP_K)):
        u = ids[i]
        if u in u64_to_pid:
            fast_pids.add(u64_to_pid[u])
    fast_results.append(fast_pids)

# Measure overlap
overlaps = []
for fast_set, gt_set in zip(fast_results, gt_topk):
    inter = len(fast_set & gt_set)
    overlaps.append(inter)

for k in range(1, OVERLAP_TOP_K + 1):
    count = sum(1 for o in overlaps if o >= k)
    pct = count / len(overlaps) * 100 if overlaps else 0
    print(f"  R@{k} overlap: {pct:.1f}% ({count}/{len(overlaps)})")

print(f"  P50 latency: {np.median(fast_latencies):.3f} ms")
print(f"  P99 latency: {np.percentile(fast_latencies, 99):.3f} ms")

lib.yp_binary_hnsw_free(hnsw_sem)
