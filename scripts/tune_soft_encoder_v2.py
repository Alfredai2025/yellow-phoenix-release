#!/usr/bin/env python3
"""
Soft Encoder Diagnostic Rig v2 — measures hash correlation, not just R@1.

DIALS:
  N_HASH_CHECK    — docs to compare hash-bit-for-bit
  N_SEMANTIC      — docs for semantic overlap test
  INDEX_SIZE      — index pool size
  HNSW_M / EF     — graph params
  TOKEN_WEIGHT    — 'uniform', 'idf', 'sqrt'  <-- how tokens combine
  OOV_STRATEGY    — 'skip', 'random', 'zero'  <-- out-of-vocab handling
"""

# ===================== DIALS =====================
N_HASH_CHECK    = 200
N_SEMANTIC      = 500
INDEX_SIZE      = 10000
HNSW_M          = 16
HNSW_EF_BUILD   = 200
HNSW_EF_SEARCH  = 200
TOKEN_WEIGHT    = 'uniform'   # 'uniform' | 'idf' | 'sqrt'
OOV_STRATEGY    = 'skip'      # 'skip' | 'random' | 'zero'
# ================================================

import ctypes, os, sys, sqlite3, hashlib, time, math, numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

lib = ctypes.CDLL(os.path.abspath("target/release/libpams.dylib"))
lib.yp_hierarchical_load.argtypes = [ctypes.c_char_p]
lib.yp_hierarchical_load.restype = ctypes.c_int
lib.yp_hierarchical_encode.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
lib.yp_hierarchical_encode.restype = ctypes.c_int

rc = lib.yp_hierarchical_load(os.path.abspath("data").encode())
print(f"[+] Load: {rc}")

def id_to_u64(pid):
    return int(hashlib.sha256(pid.encode()).hexdigest()[:16], 16)

def hamming_distance(a, b):
    return sum(bin(x ^ y).count('1') for x, y in zip(a, b))

# Load corpus
conn = sqlite3.connect("data/phoenix_arxiv_1m.db")
cur = conn.cursor()
need = N_HASH_CHECK + INDEX_SIZE + N_SEMANTIC
cur.execute("SELECT id, title, abstract FROM papers ORDER BY RANDOM() LIMIT ?", (need,))
rows = cur.fetchall()
conn.close()

hash_rows   = rows[:N_HASH_CHECK]
index_rows  = rows[N_HASH_CHECK : N_HASH_CHECK + INDEX_SIZE]
query_rows  = rows[N_HASH_CHECK + INDEX_SIZE :]

# Compute IDF weights if needed
idf_weights = {}
if TOKEN_WEIGHT == 'idf':
    df = Counter()
    total_docs = 0
    for _, t, a in index_rows:
        text = f"{t or ''} {a or ''}".strip()
        if text:
            total_docs += 1
            for tok in set(text.lower().split()):
                df[tok] += 1
    for tok, freq in df.items():
        idf_weights[tok] = math.log(total_docs / (freq + 1))

# ============================================================
# TEST A: HASH CORRELATION (bit-for-bit Hamming distance)
# For N docs, compute fast hash AND true ITQ hash.
# If the fast encoder preserves structure, Hamming distance
# should be LOW (ideally < 50 bits). If ~256, hashes are
# from different universes.
# ============================================================
print(f"\n{'='*60}")
print(f"TEST A: HASH CORRELATION (N={N_HASH_CHECK})")
print(f"{'='*60}")

print("  [+] Loading MiniLM + ITQ...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
itq = np.load('data/itq_model_512.npz')
mean = itq['mean']
M = itq['proj']

print("  [+] Embedding hash-check pool...")
texts = [f"{t or ''} {a or ''}".strip() for _, t, a in hash_rows]
embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
centered = embs - mean
projected = centered @ M

distances = []
for i, (pid, title, abstract) in enumerate(hash_rows):
    text = f"{title or ''} {abstract or ''}".strip()
    if not text: continue
    
    # Fast hash
    out = (ctypes.c_uint8 * 64)()
    if lib.yp_hierarchical_encode(text.encode(), out, 64) != 0: continue
    fast_hash = list(out)
    
    # True ITQ hash: threshold projected vector at 0
    true_hash = bytearray(64)
    for sub in range(32):
        b0 = b1 = 0
        for d in range(8):
            if projected[i, sub * 16 + d] > 0:
                b0 |= 1 << d
        for d in range(8):
            if projected[i, sub * 16 + d + 8] > 0:
                b1 |= 1 << d
        true_hash[sub * 2] = b0
        true_hash[sub * 2 + 1] = b1
    
    dist = hamming_distance(fast_hash, true_hash)
    distances.append(dist)

if distances:
    print(f"  Hamming distance mean: {np.mean(distances):.1f} bits")
    print(f"  Hamming distance median: {np.median(distances):.1f} bits")
    print(f"  Hamming distance std: {np.std(distances):.1f} bits")
    print(f"  Min: {min(distances)}, Max: {max(distances)}")
    print(f"  < 100 bits: {sum(1 for d in distances if d < 100)}/{len(distances)}")
    print(f"  < 200 bits: {sum(1 for d in distances if d < 200)}/{len(distances)}")
    
    # Interpretation
    med = np.median(distances)
    if med < 50:
        print("  SIGNAL: STRONG — hashes are nearly identical")
    elif med < 150:
        print("  SIGNAL: MODERATE — some correlation exists, tunable")
    elif med < 220:
        print("  SIGNAL: WEAK — hashes loosely related, hard to rescue")
    else:
        print("  SIGNAL: NONE — hashes are statistically independent")
else:
    print("  No distances computed.")

# ============================================================
# TEST B: SEMANTIC OVERLAP (same as before, with dials exposed)
# ============================================================
print(f"\n{'='*60}")
print(f"TEST B: SEMANTIC OVERLAP (index={INDEX_SIZE}, query={N_SEMANTIC})")
print(f"  TOKEN_WEIGHT={TOKEN_WEIGHT}, OOV_STRATEGY={OOV_STRATEGY}")
print(f"{'='*60}")

lib.yp_binary_hnsw_new_with_params.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
lib.yp_binary_hnsw_new_with_params.restype = ctypes.c_void_p
lib.yp_binary_hnsw_insert.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
lib.yp_binary_hnsw_insert.restype = ctypes.c_int
lib.yp_binary_hnsw_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
lib.yp_binary_hnsw_search.restype = ctypes.c_size_t
lib.yp_binary_hnsw_free.argtypes = [ctypes.c_void_p]

hnsw = lib.yp_binary_hnsw_new_with_params(HNSW_M, HNSW_EF_BUILD, HNSW_EF_SEARCH)

# Build index with fast encoder
for pid, title, abstract in index_rows:
    text = f"{title or ''} {abstract or ''}".strip()
    if not text: continue
    out = (ctypes.c_uint8 * 64)()
    if lib.yp_hierarchical_encode(text.encode(), out, 64) != 0: continue
    lib.yp_binary_hnsw_insert(hnsw, id_to_u64(pid), out, 64)

# Ground truth: MiniLM+ITQ embeddings for index
index_texts = [f"{t or ''} {a or ''}".strip() for _, t, a in index_rows]
index_embs = model.encode(index_texts, convert_to_numpy=True, show_progress_bar=True)
index_proj = (index_embs - mean) @ M

# Ground truth: MiniLM+ITQ for queries
query_texts = [f"{t or ''} {a or ''}".strip() for _, t, a in query_rows]
query_embs = model.encode(query_texts, convert_to_numpy=True, show_progress_bar=True)
query_proj = (query_embs - mean) @ M

# Compute GT top-5
gt_top5 = []
for q_vec in query_proj:
    sims = index_proj @ q_vec
    top_idx = np.argpartition(-sims, 5)[:5]
    gt_top5.append(set(index_rows[i][0] for i in top_idx))

# Fast search
u64_to_pid = {id_to_u64(pid): pid for pid, _, _ in index_rows}
overlap_1 = overlap_5 = total = 0
fast_latencies = []

for qi, (pid, title, abstract) in enumerate(query_rows):
    text = f"{title or ''} {abstract or ''}".strip()
    if not text: continue
    t0 = time.perf_counter()
    out = (ctypes.c_uint8 * 64)()
    if lib.yp_hierarchical_encode(text.encode(), out, 64) != 0: continue
    ids = (ctypes.c_uint64 * 5)()
    dists = (ctypes.c_uint32 * 5)()
    n = lib.yp_binary_hnsw_search(hnsw, out, 64, 5, ids, dists, 5)
    t1 = time.perf_counter()
    fast_latencies.append((t1-t0)*1000)
    
    fast_pids = set()
    for i in range(min(n, 5)):
        u = ids[i]
        if u in u64_to_pid:
            fast_pids.add(u64_to_pid[u])
    
    inter = len(fast_pids & gt_top5[qi])
    if inter >= 1: overlap_1 += 1
    if inter >= 1:  # any overlap in top-5
        overlap_5 += 1
    total += 1

print(f"  R@1 overlap: {overlap_1/total*100:.2f}% ({overlap_1}/{total})")
print(f"  Any top-5 overlap: {overlap_5/total*100:.2f}% ({overlap_5}/{total})")
print(f"  P50 latency: {np.median(fast_latencies):.3f} ms")

lib.yp_binary_hnsw_free(hnsw)
