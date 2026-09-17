# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark cross-encoder reranker vs current cosine rerank using yp_bridge."""
import sys, os, pickle, time, numpy as np
sys.path.insert(0, os.getcwd())
import yp_bridge
from sentence_transformers import CrossEncoder

print("Loading cross-encoder...")
ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Ready")

with open("data/paraphrase_queries_500.pkl", "rb") as f:
    variants = pickle.load(f)

test = variants[:50]
cos_hits, cross_hits = 0, 0
cos_time, cross_time = 0, 0

for true_pid, title, query in test:
    # Current pipeline: adaptive HNSW + cosine rerank
    t0 = time.time()
    results = yp_bridge.search(query, top_k=10, candidates=500)
    cos_time += time.time() - t0
    if results and results[0][1][0] == true_pid:
        cos_hits += 1

    # Cross-encoder rerank on top-20 candidates
    t0 = time.time()
    candidates = yp_bridge.search(query, top_k=20, candidates=500)
    if candidates:
        pairs = [(query, c[1][1]) for c in candidates]
        scores = ce.predict(pairs)
        best_idx = int(np.argmax(scores))
        best_pid = candidates[best_idx][1][0]
        cross_time += time.time() - t0
        if best_pid == true_pid:
            cross_hits += 1

print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
print(f"Cosine (current):  {cos_hits}/{len(test)} = {cos_hits/len(test)*100:.1f}% | {cos_time/len(test)*1000:.0f}ms/query")
print(f"Cross-encoder:     {cross_hits}/{len(test)} = {cross_hits/len(test)*100:.1f}% | {cross_time/len(test)*1000:.0f}ms/query")
print(f"\nIf cross > cosine, reranker wins. If cross < cosine, retrain ITQ wins.")
