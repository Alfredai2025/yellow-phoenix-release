#!/usr/bin/env python3
"""
Fair A/B benchmark on REAL 1M corpus.
Loads ScaleEngine (1,266,270 vectors), wires PID, interleaves baseline vs PID.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.phoenix_bench_api import ScaleEngine
from scripts.pid_search_path import PIDSearchPath


REAL_SCALE = "1m"
TOTAL_QUERIES = 2500
K = 10
WARMUP = 100  # queries before timing starts


class ScaleEngineWrapper:
    """
    Wraps ScaleEngine to expose the interface PIDSearchPath expects.
    ScaleEngine doesn't have _cascade_prefix_search or search_geometric,
    so PID will fall back to search_hybrid_by_embedding for everything.
    """
    def __init__(self, scale_engine):
        self._engine = scale_engine

    def search_hybrid_by_embedding(self, q, k=10, scale=REAL_SCALE):
        return self._engine.search_hybrid_by_embedding(q, k=k, scale=scale)

    # Dummy methods so PIDSearchPath doesn't crash when it checks hasattr
    def _cascade_prefix_search(self, q, k=10, l1_threshold=8, l2_threshold=4, l3_threshold=2):
        # ScaleEngine has no cascade — always return None (fallback to HNSW)
        return None

    def search_geometric(self, q, k=10, scale=REAL_SCALE):
        # ScaleEngine has no geometric re-rank — fallback to HNSW
        return self._engine.search_hybrid_by_embedding(q, k=k, scale=scale)


def main():
    print("=" * 70)
    print("FAIR A/B BENCHMARK: REAL 1M CORPUS (PID vs BASELINE)")
    print("=" * 70)

    # Load 1M engine
    print("\nLoading 1M engine...")
    scale = ScaleEngine()
    scale._get_engine(REAL_SCALE)
    embs = scale._engines[REAL_SCALE][0]
    n = len(embs)
    print(f"Loaded {n:,} vectors.")

    # Wrap for PID
    wrapper = ScaleEngineWrapper(scale)

    # Init PID
    pid_engine = PIDSearchPath(
        wrapper,
        enable_cache=True,
        enable_cascade=True,
        enable_router=True
    )

    np.random.seed(42)

    # Warmup: run queries through both paths to warm CPU cache + HNSW
    print(f"\nWarming up with {WARMUP} queries...")
    warmup_idx = np.random.choice(n, WARMUP, replace=False)
    pid_warmup_count = 0
    for idx in warmup_idx[:WARMUP // 2]:
        wrapper.search_hybrid_by_embedding(embs[idx], k=K, scale=REAL_SCALE)
    for idx in warmup_idx[WARMUP // 2:]:
        pid_engine.search(embs[idx], k=K, scale=REAL_SCALE)
        pid_warmup_count += 1
        if pid_warmup_count % 50 == 0:
            pid_engine.step_pid(n_queries=50)

    # Reset PID metrics after warmup
    pid_engine._reset_metrics()

    # Test queries: interleaved
    print(f"\nRunning {TOTAL_QUERIES} interleaved queries...")
    query_idx = np.random.choice(n, TOTAL_QUERIES, replace=False)
    assignments = np.array([0] * (TOTAL_QUERIES // 2) + [1] * (TOTAL_QUERIES // 2))
    np.random.shuffle(assignments)

    baseline_times = []
    pid_times = []

    for i, idx in enumerate(query_idx):
        q = embs[idx]

        if assignments[i] == 0:
            t0 = time.perf_counter()
            wrapper.search_hybrid_by_embedding(q, k=K, scale=REAL_SCALE)
            t1 = time.perf_counter()
            baseline_times.append((t1 - t0) * 1e6)
        else:
            t0 = time.perf_counter()
            pid_engine.search(q, k=K, scale=REAL_SCALE)
            t1 = time.perf_counter()
            pid_times.append((t1 - t0) * 1e6)

            # Step PID every 50 PID queries
            if (len(pid_times) % 50) == 0:
                pid_engine.step_pid(n_queries=50)

        if (i + 1) % 250 == 0:
            print(f"  ...{i + 1}/{TOTAL_QUERIES} "
                  f"(baseline: {len(baseline_times)}, pid: {len(pid_times)})")

    # ============================================================
    # Results
    # ============================================================
    b_p50 = float(np.percentile(baseline_times, 50)) if baseline_times else 0
    b_mean = float(np.mean(baseline_times)) if baseline_times else 0
    b_p95 = float(np.percentile(baseline_times, 95)) if baseline_times else 0
    p_p50 = float(np.percentile(pid_times, 50)) if pid_times else 0
    p_mean = float(np.mean(pid_times)) if pid_times else 0
    p_p95 = float(np.percentile(pid_times, 95)) if pid_times else 0

    print("\n" + "=" * 70)
    print("RESULTS (real 1M, interleaved, warmup done)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':>12} {'PID':>12} {'Delta':>10}")
    print("-" * 70)
    print(f"{'Queries':<25} {len(baseline_times):>12} {len(pid_times):>12}")
    print(f"{'P50 latency (µs)':<25} {b_p50:>12.1f} {p_p50:>12.1f} "
          f"{((p_p50-b_p50)/b_p50*100) if b_p50 else 0:>+9.1f}%")
    print(f"{'Mean latency (µs)':<25} {b_mean:>12.1f} {p_mean:>12.1f} "
          f"{((p_mean-b_mean)/b_mean*100) if b_mean else 0:>+9.1f}%")
    print(f"{'P95 latency (µs)':<25} {b_p95:>12.1f} {p_p95:>12.1f} "
          f"{((p_p95-b_p95)/b_p95*100) if b_p95 else 0:>+9.1f}%")

    # PID state
    state = pid_engine.state()
    metrics = pid_engine._metrics

    print(f"\n[PID state]")
    print(f"  Cascade thresholds: L1={state['cascade_thresholds']['L1']} "
          f"L2={state['cascade_thresholds']['L2']} "
          f"L3={state['cascade_thresholds']['L3']}")
    print(f"  Router: {state['route_percent']}% geometric")
    print(f"  Cache: TTL={state['cache_ttl']}s")
    print(f"  Cache hits/misses this window: {metrics['cache_hits']}/{metrics['cache_misses']}")
    print(f"  Cascade L1/L2/L3/miss: {metrics['cascade_l1']}/{metrics['cascade_l2']}/"
          f"{metrics['cascade_l3']}/{metrics['cascade_miss']}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/bench_pid_1m_real.json", "w") as f:
        json.dump({
            "baseline": {"count": len(baseline_times), "p50_us": b_p50,
                         "mean_us": b_mean, "p95_us": b_p95},
            "pid": {"count": len(pid_times), "p50_us": p_p50,
                    "mean_us": p_mean, "p95_us": p_p95},
            "pid_state": state,
            "pid_metrics": metrics,
        }, f, indent=2)
    print("\nSaved logs/bench_pid_1m_real.json")

    # Honest verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if p_p50 <= b_p50 * 1.02:
        print("PID is NEUTRAL or FASTER on this workload.")
    else:
        print("PID adds overhead on this workload (random queries, no cascade/geometric).")
        print("Benefit appears with: repeated queries (cache), prefix-structured data (cascade),")
        print("or a real geometric re-rank path (router).")


if __name__ == "__main__":
    main()
