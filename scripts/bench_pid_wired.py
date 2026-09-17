#!/usr/bin/env python3
"""
Real 1M benchmark: PID-wired search path vs baseline.
Measures actual latency difference with cache, cascade, and router active.
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
STEPS = 50
NQ_PER_STEP = 50
TOTAL_QUERIES = STEPS * NQ_PER_STEP


def main():
    print("=" * 70)
    print("REAL 1M BENCHMARK: PID WIRED INTO SEARCH PATH")
    print("=" * 70)

    # Load engine
    print("\nLoading 1M engine...")
    engine = ScaleEngine()
    engine._get_engine(REAL_SCALE)
    embs = engine._engines[REAL_SCALE][0]
    n = len(embs)
    print(f"Loaded {n:,} vectors.")

    np.random.seed(42)
    query_idx = np.random.choice(n, TOTAL_QUERIES, replace=False)

    # ============================================================
    # BASELINE: plain search_hybrid_by_embedding
    # ============================================================
    print(f"\n[BENCHMARK 1/2] Baseline (no PID) — {TOTAL_QUERIES} queries...")
    baseline_times = []
    for i, idx in enumerate(query_idx):
        q = embs[idx]
        t0 = time.perf_counter()
        engine.search_hybrid_by_embedding(q, k=10, scale=REAL_SCALE)
        t1 = time.perf_counter()
        baseline_times.append((t1 - t0) * 1e6)

        if (i + 1) % 250 == 0:
            print(f"  ...{i + 1}/{TOTAL_QUERIES}")

    # ============================================================
    # PID: full wired path
    # ============================================================
    print(f"\n[BENCHMARK 2/2] PID wired — {TOTAL_QUERIES} queries...")
    pid_engine = PIDSearchPath(
        engine,
        enable_cache=True,
        enable_cascade=True,
        enable_router=True
    )
    pid_times = []

    for i, idx in enumerate(query_idx):
        q = embs[idx]
        results = pid_engine.search(q, k=10, scale=REAL_SCALE)
        # Latency is recorded inside search()
        pid_times.append(pid_engine._metrics['latencies'][-1])

        # Step PID every 50 queries
        if (i + 1) % 50 == 0:
            pid_engine.step_pid(n_queries=50)

        if (i + 1) % 250 == 0:
            print(f"  ...{i + 1}/{TOTAL_QUERIES}")

    # ============================================================
    # Results
    # ============================================================
    b_p50 = float(np.percentile(baseline_times, 50))
    b_mean = float(np.mean(baseline_times))
    p_p50 = float(np.percentile(pid_times, 50))
    p_mean = float(np.mean(pid_times))

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':>12} {'PID Wired':>12} {'Delta':>10}")
    print("-" * 70)
    print(f"{'P50 latency (µs)':<25} {b_p50:>12.1f} {p_p50:>12.1f} {(p_p50-b_p50)/b_p50*100:>+9.1f}%")
    print(f"{'Mean latency (µs)':<25} {b_mean:>12.1f} {p_mean:>12.1f} {(p_mean-b_mean)/b_mean*100:>+9.1f}%")

    # PID final state
    state = pid_engine.state()
    print(f"\n[PID final state]")
    print(f"  Cascade: L1={state['cascade_thresholds']['L1']} "
          f"L2={state['cascade_thresholds']['L2']} "
          f"L3={state['cascade_thresholds']['L3']}")
    print(f"  Router: {state['route_percent']}% geometric")
    print(f"  Cache: TTL={state['cache_ttl']}s")

    # Save
    os.makedirs("logs", exist_ok=True)
    with open("logs/bench_pid_wired.json", "w") as f:
        json.dump({
            "baseline": {"p50_us": b_p50, "mean_us": b_mean, "times": baseline_times},
            "pid": {"p50_us": p_p50, "mean_us": p_mean, "times": pid_times},
            "pid_state": state,
        }, f, indent=2)
    print("\nSaved logs/bench_pid_wired.json")


if __name__ == "__main__":
    main()
