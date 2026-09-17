#!/usr/bin/env python3
"""
Fair A/B benchmark: interleaved baseline vs PID on real yp_engine.py.
Eliminates warmup bias by randomizing which path each query takes.
Uses YPEngine.search_hybrid_by_embedding for baseline and YPEngine.search_pid for PID.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from yp_engine import YPEngine


TOTAL_QUERIES = 2500  # 1250 baseline + 1250 PID, interleaved
K = 10


def main():
    print("=" * 70)
    print("FAIR A/B BENCHMARK: BASELINE vs PID (interleaved)")
    print("=" * 70)

    # Init engine
    print("\nLoading YPEngine...")
    engine = YPEngine()
    engine._pid_enabled = True  # search_pid will use it

    embs = engine.embeddings
    n = len(embs)
    print(f"Loaded {n:,} vectors.")

    np.random.seed(42)
    query_idx = np.random.choice(n, TOTAL_QUERIES, replace=False)

    # Randomly assign each query to baseline (0) or PID (1), 50/50
    assignments = np.array([0] * (TOTAL_QUERIES // 2) + [1] * (TOTAL_QUERIES // 2))
    np.random.shuffle(assignments)

    # Warm up HNSW index and PID wrapper so both paths pay the build cost
    # before timing begins. This keeps the interleaved timing fair.
    print("\nWarming up HNSW index and PID wrapper...")
    _ = engine.search_hybrid_by_embedding(embs[query_idx[0]], k=K)
    _ = engine.search_pid(embs[query_idx[1]], k=K)

    baseline_times = []
    pid_times = []

    print(f"\nRunning {TOTAL_QUERIES} interleaved queries...")
    t_start = time.perf_counter()
    for i, idx in enumerate(query_idx):
        q = embs[idx]

        if assignments[i] == 0:
            t0 = time.perf_counter()
            engine.search_hybrid_by_embedding(q, k=K)
            t1 = time.perf_counter()
            baseline_times.append((t1 - t0) * 1e6)
        else:
            t0 = time.perf_counter()
            engine.search_pid(q, k=K)
            t1 = time.perf_counter()
            pid_times.append((t1 - t0) * 1e6)

        if (i + 1) % 250 == 0:
            print(f"  ...{i + 1}/{TOTAL_QUERIES} "
                  f"(baseline: {len(baseline_times)}, pid: {len(pid_times)})")

    elapsed = time.perf_counter() - t_start

    # ============================================================
    # Results
    # ============================================================
    b_p50 = float(np.percentile(baseline_times, 50)) if baseline_times else 0
    b_mean = float(np.mean(baseline_times)) if baseline_times else 0
    p_p50 = float(np.percentile(pid_times, 50)) if pid_times else 0
    p_mean = float(np.mean(pid_times)) if pid_times else 0

    print("\n" + "=" * 70)
    print("RESULTS (interleaved, warmup done before timing)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':>12} {'PID':>12} {'Delta':>10}")
    print("-" * 70)
    print(f"{'Queries':<25} {len(baseline_times):>12} {len(pid_times):>12}")
    print(f"{'P50 latency (µs)':<25} {b_p50:>12.1f} {p_p50:>12.1f} "
          f"{((p_p50-b_p50)/b_p50*100) if b_p50 else 0:>+9.1f}%")
    print(f"{'Mean latency (µs)':<25} {b_mean:>12.1f} {p_mean:>12.1f} "
          f"{((p_mean-b_mean)/b_mean*100) if b_mean else 0:>+9.1f}%")
    print(f"{'Total elapsed (s)':<25} {elapsed:>12.1f}")

    # PID state
    state = None
    if hasattr(engine, '_pid_wrapper') and engine._pid_wrapper:
        state = engine._pid_wrapper.state()
        print(f"\n[PID state]")
        print(f"  Cascade: L1={state['cascade_thresholds']['L1']} "
              f"L2={state['cascade_thresholds']['L2']} "
              f"L3={state['cascade_thresholds']['L3']}")
        print(f"  Router: {state['route_percent']}% geometric")
        print(f"  Cache: TTL={state['cache_ttl']}s")

        # Cache hit rate
        c = engine._pid_wrapper.cache
        if c:
            print(f"  Cache: {c.hits} hits / {c.misses} misses "
                  f"({c.hit_rate()*100:.1f}%)")

    os.makedirs("logs", exist_ok=True)
    with open("logs/bench_pid_fair.json", "w") as f:
        json.dump({
            "baseline": {"count": len(baseline_times), "p50_us": b_p50, "mean_us": b_mean},
            "pid": {"count": len(pid_times), "p50_us": p_p50, "mean_us": p_mean},
            "elapsed_s": elapsed,
            "pid_state": state,
        }, f, indent=2)
    print("\nSaved logs/bench_pid_fair.json")


if __name__ == "__main__":
    main()
