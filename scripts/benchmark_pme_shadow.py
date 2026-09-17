# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
PME Shadow Re-Rank Benchmark.
Runs PME in parallel with MiniLM on N queries.
Logs overlap between PME top-5 and MiniLM top-5.
Does NOT affect production results.
"""

import os
import sys
import time
import random
import numpy as np
import ctypes
from pathlib import Path

sys.path.insert(0, str(Path.home() / "yellow_phoenix"))
import yp_bridge

# ── Load PME FFI ──
YP_ROOT = Path.home() / "yellow_phoenix"
LIB_PATH = YP_ROOT / "target" / "release" / "libpams.dylib"
_lib = ctypes.CDLL(str(LIB_PATH))

_lib.pme_expand.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
_lib.pme_expand.restype = None

_lib.pme_manifold_distance.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
_lib.pme_manifold_distance.restype = ctypes.c_float


def _as_uint8_arr(b: bytes):
    return (ctypes.c_uint8 * len(b))(*b)


def expand_pme(hash512: bytes) -> bytes:
    """Expand 512-bit seed to 4096-bit PME signature."""
    seed = _as_uint8_arr(hash512)
    out = (ctypes.c_uint8 * 512)()
    _lib.pme_expand(seed, out)
    return bytes(out)


def manifold_dist(a: bytes, b: bytes) -> float:
    """PME manifold distance. Higher = more similar."""
    return _lib.pme_manifold_distance(_as_uint8_arr(a), _as_uint8_arr(b))


# ── Load candidate hashes ──
print("Loading ITQ hashes...")
HASHES = np.load(YP_ROOT / "data" / "itq_hashes_1m.npy").astype(np.uint8)

# ── Config ──
N_QUERIES = int(os.environ.get("PME_N_QUERIES", "1000"))
N_CANDIDATES = int(os.environ.get("PME_N_CANDIDATES", "200"))
TOP_K = 5
random.seed(42)

# Sample queries from paper titles
print(f"Sampling {N_QUERIES} queries from paper titles...")
yp_bridge._load_itq_embeddings()
idxs = random.sample(range(len(yp_bridge._ITQ_TITLES)), N_QUERIES)
queries = [str(yp_bridge._ITQ_TITLES[i]) for i in idxs]

# ── Benchmark ──
print(f"\n{'='*60}")
print(f"PME SHADOW BENCHMARK: {N_QUERIES} queries, up to {N_CANDIDATES} candidates each")
print(f"{'='*60}")

overlap_scores = []
pme_times = []
minilm_times = []
processed = 0

for idx, query_text in enumerate(queries):
    # 1. Encode query
    try:
        emb, q_hash = yp_bridge._encode_query_cached(query_text)
    except Exception as e:
        print(f"[skip] encode failed: {e}")
        continue

    # 2. Get global HNSW candidates
    try:
        cand_idx = yp_bridge._search_global_hnsw(q_hash, N_CANDIDATES)
    except Exception as e:
        print(f"[skip] HNSW failed: {e}")
        continue
    if len(cand_idx) < 10:
        continue

    # 3. MINILM RE-RANK
    t1 = time.perf_counter()
    sims = yp_bridge._ITQ_EMBS[cand_idx] @ emb
    minilm_order = np.argsort(-sims)
    minilm_top5 = [str(yp_bridge._ITQ_PIDS[cand_idx[i]]) for i in minilm_order[:TOP_K]]
    minilm_times.append((time.perf_counter() - t1) * 1000)

    # 4. PME RE-RANK (shadow)
    t2 = time.perf_counter()
    q_expanded = expand_pme(q_hash.tobytes())

    pme_scores = []
    for c_idx in cand_idx:
        c_hash = HASHES[c_idx].tobytes()
        c_expanded = expand_pme(c_hash)
        dist = manifold_dist(q_expanded, c_expanded)
        pme_scores.append((str(yp_bridge._ITQ_PIDS[c_idx]), dist))

    pme_ranked = sorted(pme_scores, key=lambda x: x[1], reverse=True)
    pme_top5 = [pid for pid, _ in pme_ranked[:TOP_K]]
    pme_times.append((time.perf_counter() - t2) * 1000)

    # 5. Compare
    overlap = len(set(minilm_top5) & set(pme_top5)) / TOP_K
    overlap_scores.append(overlap)
    processed += 1

    if (idx + 1) % 50 == 0:
        mean_overlap = sum(overlap_scores) / len(overlap_scores)
        mean_pme = sum(pme_times) / len(pme_times)
        print(f"  {idx+1}/{N_QUERIES} done — processed={processed} — avg overlap: {mean_overlap:.1%} — PME {mean_pme:.1f}ms")

# ── Results ──
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
if not overlap_scores:
    print("No queries successfully evaluated.")
    sys.exit(1)

mean_overlap = sum(overlap_scores) / len(overlap_scores)
median_overlap = sorted(overlap_scores)[len(overlap_scores) // 2]
p90_overlap = sorted(overlap_scores)[int(len(overlap_scores) * 0.9)]

print(f"Queries evaluated:        {len(overlap_scores)} / {N_QUERIES}")
print(f"PME avg re-rank time:     {sum(pme_times)/len(pme_times):.2f} ms")
print(f"MiniLM avg re-rank time:  {sum(minilm_times)/len(minilm_times):.2f} ms")
print(f"Speedup vs MiniLM:        {sum(minilm_times)/sum(pme_times):.1f}x")
print("")
print(f"Top-5 overlap with MiniLM:")
print(f"  Mean:   {mean_overlap:.1%}")
print(f"  Median: {median_overlap:.1%}")
print(f"  P90:    {p90_overlap:.1%}")
print("")
identical = sum(1 for o in overlap_scores if o == 1.0)
close = sum(1 for o in overlap_scores if 0.6 <= o < 1.0)
poor = sum(1 for o in overlap_scores if o < 0.6)
print(f"Overlap distribution:")
print(f"  100% (identical):       {identical} ({identical/len(overlap_scores):.1%})")
print(f"  60–80% (close):         {close} ({close/len(overlap_scores):.1%})")
print(f"  <60% (poor):            {poor} ({poor/len(overlap_scores):.1%})")
print("")
if mean_overlap >= 0.90:
    print("✅ VERDICT: PME can replace MiniLM for most queries.")
elif mean_overlap >= 0.80:
    print("⚠️  VERDICT: PME is close — consider hybrid fallback to MiniLM.")
else:
    print("❌ VERDICT: PME quality too low. Keep MiniLM as primary re-rank.")
print(f"{'='*60}")
