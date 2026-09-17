# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
import sys, time, random, numpy as np, ctypes
from pathlib import Path

sys.path.insert(0, str(Path.home() / "yellow_phoenix"))
import yp_bridge

YP_ROOT = Path.home() / "yellow_phoenix"
_lib = ctypes.CDLL(str(YP_ROOT / "target" / "release" / "libpams.dylib"))

# Verify spectral exists
try:
    _lib.yp_spectral_query
    print("✅ yp_spectral_query loaded")
except AttributeError:
    print("❌ yp_spectral_query NOT in dylib")
    sys.exit(1)

# Wire if missing
if not hasattr(yp_bridge, 'spectral_query'):
    _lib.yp_spectral_query.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    _lib.yp_spectral_query.restype = ctypes.c_int

    def spectral_query(query_hash: bytes, candidate_ids: list) -> list:
        n = len(candidate_ids)
        ids_arr = (ctypes.c_uint64 * n)(*candidate_ids)
        scores_arr = (ctypes.c_float * n)()
        rc = _lib.yp_spectral_query(
            ctypes.cast(query_hash, ctypes.POINTER(ctypes.c_uint8)),
            len(query_hash),
            ids_arr, n, scores_arr
        )
        return [scores_arr[i] for i in range(n)] if rc >= 0 else []

    yp_bridge.spectral_query = spectral_query

# Config
N_QUERIES = 50
N_CANDIDATES = 200
TOP_K = 5

# Load ITQ search metadata and embeddings
yp_bridge._load_itq_search()
yp_bridge._load_itq_embeddings()

# Pick random query rows that have titles
rng = random.Random(42)
n_total = len(yp_bridge._ITQ_PIDS)
query_indices = rng.sample(range(n_total), N_QUERIES)

# Benchmark
spectral_times = []
minilm_times = []
overlap_scores = []
errors = 0

for idx, q_row in enumerate(query_indices):
    query_emb = yp_bridge._ITQ_EMBS[q_row].astype(np.float32)
    q_hash = yp_bridge._ITQ_HASHES[q_row].astype(np.uint8)

    # Global HNSW candidate retrieval (returns row indices)
    cand_idx = yp_bridge._search_global_hnsw(q_hash, N_CANDIDATES)
    if cand_idx is None or len(cand_idx) < 10:
        continue

    # Exclude the query itself and deduplicate
    cand_idx = np.array([int(i) for i in cand_idx if i != q_row], dtype=np.int64)
    if len(cand_idx) < 5:
        continue

    cand_ids = [int(i) for i in cand_idx]

    # MiniLM ground truth (using the stored query embedding)
    t1 = time.perf_counter()
    sims = yp_bridge._ITQ_EMBS[cand_idx] @ query_emb
    minilm_top5 = [int(i) for i in np.argpartition(-sims, TOP_K - 1)[:TOP_K]]
    minilm_times.append((time.perf_counter() - t1) * 1000)

    # Spectral shadow
    t2 = time.perf_counter()
    try:
        s_scores = yp_bridge.spectral_query(q_hash.tobytes(), cand_ids)
        if s_scores and len(s_scores) == len(cand_ids):
            spectral_top5 = [cid for cid, _ in sorted(zip(cand_ids, s_scores), key=lambda x: x[1], reverse=True)[:TOP_K]]
            overlap = len(set(minilm_top5) & set(spectral_top5)) / TOP_K
            overlap_scores.append(overlap)
        else:
            errors += 1
    except Exception as e:
        print(f"  Error query {idx}: {e}")
        errors += 1

    spectral_times.append((time.perf_counter() - t2) * 1000)

    if (idx + 1) % 10 == 0 and overlap_scores:
        print(f"  {idx+1}/{N_QUERIES} — overlap: {sum(overlap_scores)/len(overlap_scores):.1%}, "
              f"spectral: {sum(spectral_times)/len(spectral_times):.1f}ms")

# Results
print(f"\n{'='*60}")
if not overlap_scores:
    print("❌ No valid comparisons.")
    sys.exit(1)

print(f"Queries: {len(overlap_scores)} | Errors: {errors}")
print(f"Spectral avg: {sum(spectral_times)/len(spectral_times):.2f}ms")
print(f"MiniLM avg:   {sum(minilm_times)/len(minilm_times):.2f}ms")
print(f"Speedup:      {sum(minilm_times)/sum(spectral_times):.1f}x")
print(f"Overlap mean: {sum(overlap_scores)/len(overlap_scores):.1%}")
print(f"Overlap med:  {sorted(overlap_scores)[len(overlap_scores)//2]:.1%}")
identical = sum(1 for o in overlap_scores if o == 1.0)
close = sum(1 for o in overlap_scores if 0.6 <= o < 1.0)
poor = sum(1 for o in overlap_scores if o < 0.6)
print(f"Identical: {identical} | Close: {close} | Poor: {poor}")
avg = sum(overlap_scores)/len(overlap_scores)
print("✅ PROMOTE" if avg >= 0.90 else "🟡 HYBRID" if avg >= 0.75 else "❌ KEEP MINILM")
print(f"{'='*60}")
