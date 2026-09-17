# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Benchmark tabulated Bag-of-Embeddings (BoE) vs MiniLM+ITQ on real paraphrased queries.

Uses:
  - data/paraphrase_queries_500.pkl  (true_pid, title, paraphrase)
  - data/itq_hashes_1m.npy
  - data/paper_meta_arxiv_1m.pkl
  - itq_model_512.npz
  - models/all-MiniLM-L6-v2
  - yp_bridge.encode_tabulated()

Constructs a 100K candidate pool (including all true papers) and computes R@K
for both encoders, plus top-k overlap and per-query Hamming divergence.
"""
import sys, os, pickle, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sentence_transformers import SentenceTransformer
from yp_bridge import encode_tabulated

HASH_PATH = "data/itq_hashes_1m.npy"
META_PATH = "data/paper_meta_arxiv_1m.pkl"
QUERY_PATH = "data/paraphrase_queries_500.pkl"
ITQ_PATH = "itq_model_512.npz"
RESULTS_PATH = "data/benchmark_boe_results.json"

K_LIST = [1, 10, 50, 100]

print("Loading metadata, hashes, and queries ...")
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
pids = meta["pids"]
titles = meta["titles"]
pid_to_idx = {pid: i for i, pid in enumerate(pids)}

hashes = np.load(HASH_PATH, mmap_mode="r")  # N x 64
N = hashes.shape[0]

with open(QUERY_PATH, "rb") as f:
    queries = pickle.load(f)[:200]

# Candidate pool: 100K including all true papers
true_indices = [pid_to_idx[pid] for pid, _, _ in queries]
true_set = set(true_indices)
other = [i for i in range(N) if i not in true_set]
other = np.random.choice(other, 100000 - len(true_indices), replace=False).tolist()
cand_indices = np.array(list(true_set) + other, dtype=np.int64)
np.random.shuffle(cand_indices)
print(f"Candidate pool: {len(cand_indices):,}")

cand_hashes = hashes[cand_indices]

# Load ITQ model and MiniLM
itq = np.load(ITQ_PATH)
mean = itq["mean"].astype(np.float32)
W = itq["proj"].astype(np.float32)
model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)

def encode_minilm_itq(text: str) -> bytes:
    emb = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    x = (emb.astype(np.float32) - mean) @ W
    bits = (x > 0).astype(np.uint8)
    return np.packbits(bits).tobytes()

def hamming_bytes(a: bytes, b: bytes) -> int:
    return int(np.bitwise_count(np.frombuffer(a, dtype=np.uint8) ^ np.frombuffer(b, dtype=np.uint8)).sum())

hits_itq = {k: 0 for k in K_LIST}
hits_boe = {k: 0 for k in K_LIST}
overlap_10 = []
hamming_divs = []
itq_times = []
boe_times = []

print(f"Benchmarking {len(queries)} queries ...")
for i, (true_pid, title, query_text) in enumerate(queries):
    true_idx_in_cand = np.where(cand_indices == true_indices[i])[0][0]

    t0 = time.perf_counter()
    h_itq = encode_minilm_itq(query_text)
    itq_times.append((time.perf_counter() - t0) * 1e6)

    t0 = time.perf_counter()
    h_boe = encode_tabulated(query_text)
    boe_times.append((time.perf_counter() - t0) * 1e6)

    hamming_divs.append(hamming_bytes(h_itq, h_boe))

    # Top-k via 512-bit Hamming on candidate pool
    d_itq = np.bitwise_count(np.frombuffer(h_itq, dtype=np.uint8) ^ cand_hashes).sum(axis=1)
    d_boe = np.bitwise_count(np.frombuffer(h_boe, dtype=np.uint8) ^ cand_hashes).sum(axis=1)

    top_itq_10 = set(np.argpartition(d_itq, 10 - 1)[:10])
    top_boe_10 = set(np.argpartition(d_boe, 10 - 1)[:10])
    overlap_10.append(len(top_itq_10 & top_boe_10))

    for k in K_LIST:
        if true_idx_in_cand in np.argpartition(d_itq, k - 1)[:k]:
            hits_itq[k] += 1
        if true_idx_in_cand in np.argpartition(d_boe, k - 1)[:k]:
            hits_boe[k] += 1

    if (i + 1) % 50 == 0:
        print(f"  processed {i + 1}/{len(queries)}")

n = len(queries)

print("\n" + "=" * 70)
print("BOE vs MiniLM+ITQ BENCHMARK RESULTS")
print("=" * 70)

print(f"\n{'K':>5} | {'ITQ R@K':>10} | {'BoE R@K':>10} | {'Delta':>10}")
print("-" * 45)
for k in K_LIST:
    r_itq = hits_itq[k] / n * 100
    r_boe = hits_boe[k] / n * 100
    print(f"{k:>5} | {r_itq:>9.1f}% | {r_boe:>9.1f}% | {r_boe - r_itq:>+9.1f}%")

print(f"\nHamming divergence (BoE vs ITQ):")
print(f"  Mean:   {np.mean(hamming_divs):.1f} / 512 ({np.mean(hamming_divs)/512*100:.1f}%)")
print(f"  Median: {np.median(hamming_divs):.1f}")
print(f"  Min:    {np.min(hamming_divs)}, Max: {np.max(hamming_divs)}")

print(f"\nTop-10 overlap (same candidate indices):")
print(f"  Avg: {np.mean(overlap_10):.1f} / 10")
print(f"  >0:  {sum(1 for x in overlap_10 if x > 0) / n * 100:.1f}% of queries")

print(f"\nLatency:")
print(f"  BoE encode:  {np.mean(boe_times):.1f} us (P50: {np.percentile(boe_times, 50):.1f})")
print(f"  ITQ encode:  {np.mean(itq_times):.1f} us (P50: {np.percentile(itq_times, 50):.1f})")
print(f"  Speedup:     {np.mean(itq_times) / np.mean(boe_times):.1f}x")

r1_boe = hits_boe[1] / n * 100
r1_itq = hits_itq[1] / n * 100

print("\n" + "=" * 70)
if r1_boe >= 50:
    print("VERDICT: KEEP as Tier-0 fast path.")
    print("         BoE captures strong signal and can be used as a coarse filter.")
elif r1_boe >= 25:
    print("VERDICT: KEEP as optional shadow encoder.")
    print("         Weak but not random. Log hit rate; use when speed > accuracy.")
else:
    print("VERDICT: MUSEUM.")
    print("         BoE is too far from MiniLM+ITQ to be useful for retrieval.")
print("=" * 70)

results = {
    "n_queries": n,
    "candidate_pool": int(len(cand_indices)),
    "k_list": K_LIST,
    "r_at_k_itq": {str(k): hits_itq[k] / n * 100 for k in K_LIST},
    "r_at_k_boe": {str(k): hits_boe[k] / n * 100 for k in K_LIST},
    "hamming_divergence_mean": float(np.mean(hamming_divs)),
    "hamming_divergence_median": float(np.median(hamming_divs)),
    "hamming_divergence_min": int(np.min(hamming_divs)),
    "hamming_divergence_max": int(np.max(hamming_divs)),
    "avg_top10_overlap": float(np.mean(overlap_10)),
    "queries_with_overlap_pct": float(sum(1 for x in overlap_10 if x > 0) / n * 100),
    "boe_encode_us_mean": float(np.mean(boe_times)),
    "boe_encode_us_p50": float(np.percentile(boe_times, 50)),
    "itq_encode_us_mean": float(np.mean(itq_times)),
    "itq_encode_us_p50": float(np.percentile(itq_times, 50)),
    "speedup": float(np.mean(itq_times) / np.mean(boe_times)),
    "verdict": "tier0" if r1_boe >= 50 else ("shadow" if r1_boe >= 25 else "museum"),
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {RESULTS_PATH}")
