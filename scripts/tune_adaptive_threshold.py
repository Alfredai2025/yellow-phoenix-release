# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Tune YP_MARGIN_THRESHOLD for adaptive shard/global routing.

For each query we compare the candidate set produced by adaptive routing
against a global brute-force Hamming ground truth and report recall@500.
We also measure the shard-vs-global split and latency.
"""
import os
import sys
import time
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yp_bridge as yp

yp._load_itq_embeddings()

CANDIDATES = 500
TOP_K = 5
N_QUERIES = int(os.environ.get("TUNE_N", "200"))
THRESHOLDS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]


def sample_queries(n):
    """Sample random paper titles as queries."""
    idxs = random.sample(range(len(yp._ITQ_TITLES)), n)
    return [(str(yp._ITQ_PIDS[i]), str(yp._ITQ_TITLES[i])) for i in idxs]


def ground_truth_candidates(q_hash):
    return set(yp._brute_force_candidates(q_hash, CANDIDATES).tolist())


def adaptive_candidates_for_threshold(q_hash, emb, threshold):
    """Mirror _adaptive_candidates but with a dynamic threshold."""
    yp.YP_MARGIN_THRESHOLD = threshold
    cand, path, margin = yp._adaptive_candidates(q_hash, emb, CANDIDATES)
    return set(cand.tolist()), path, margin


def main():
    print(f"Tuning adaptive threshold on {N_QUERIES} random queries")
    print(f"Candidates={CANDIDATES}  thresholds={THRESHOLDS}\n")

    queries = sample_queries(N_QUERIES)

    # Pre-compute ground truth for each query
    gt_sets = []
    for pid, title in queries:
        _, q_hash = yp._encode_query_cached(title)
        gt_sets.append(ground_truth_candidates(q_hash))

    print("threshold | shard% | global% | brute% | recall_mean | recall_median | latency_ms")
    print("-" * 80)

    for thr in THRESHOLDS:
        shard_count = global_count = brute_count = 0
        recalls = []
        latencies = []

        for (pid, title), gt in zip(queries, gt_sets):
            emb, q_hash = yp._encode_query_cached(title)

            t0 = time.perf_counter()
            cand, path, margin = adaptive_candidates_for_threshold(q_hash, emb, thr)
            latencies.append((time.perf_counter() - t0) * 1000)

            if path == "shards":
                shard_count += 1
            elif path == "global":
                global_count += 1
            else:
                brute_count += 1

            if gt:
                recalls.append(len(cand & gt) / len(gt))
            else:
                recalls.append(0.0)

        mean_recall = float(np.mean(recalls))
        med_recall = float(np.median(recalls))
        mean_lat = float(np.mean(latencies))
        total = len(queries)

        print(f"{thr:8.2f} | {shard_count/total:5.1%} | {global_count/total:6.1%} | "
              f"{brute_count/total:5.1%} | {mean_recall:10.1%} | {med_recall:12.1%} | "
              f"{mean_lat:9.2f}")

    print("\nPick the threshold where shard% is high enough for your latency target")
    print("and recall_mean exceeds your quality target (e.g. >90%).")


if __name__ == "__main__":
    main()
