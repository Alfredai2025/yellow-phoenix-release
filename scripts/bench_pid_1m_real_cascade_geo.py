#!/usr/bin/env python3
"""
Fair A/B benchmark: real 1M with cascade + geometric re-rank wired.
Measures latency, recall, cascade hit rates, and geometric routing.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.phoenix_bench_api import ScaleEngine
from scripts.scale_engine_pid import ScaleEnginePID
from scripts.pid_search_path import PIDSearchPath


REAL_SCALE = "1m"
TOTAL_QUERIES = 2500
K = 10
WARMUP = 100


def main():
    print("=" * 70)
    print("FAIR A/B: REAL 1M + CASCADE + GEOMETRIC RE-RANK")
    print("=" * 70)

    # Load 1M
    print("\nLoading 1M engine...")
    scale = ScaleEngine()
    scale._get_engine(REAL_SCALE)
    embs = scale._engines[REAL_SCALE][0]
    n = len(embs)
    print(f"Loaded {n:,} vectors.")

    # Wrap with real cascade + geometric
    print("Wiring real cascade + geometric re-rank...")
    wrapper = ScaleEnginePID(scale)

    # PID on
    pid_engine = PIDSearchPath(
        wrapper,
        enable_cache=True,
        enable_cascade=True,
        enable_router=False
    )

    np.random.seed(42)

    # Warmup
    print(f"\nWarming up with {WARMUP} queries...")
    warmup_idx = np.random.choice(n, WARMUP, replace=False)
    for idx in warmup_idx[:WARMUP // 2]:
        wrapper.search_hybrid_by_embedding(embs[idx], k=K, scale=REAL_SCALE)
    for idx in warmup_idx[WARMUP // 2:]:
        pid_engine.search(embs[idx], k=K, scale=REAL_SCALE)
        if len(pid_engine._metrics.get('latencies', [])) % 50 == 0:
            pid_engine.step_pid(n_queries=50)

    pid_engine._reset_metrics()

    # Interleaved test
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

            if (len(pid_times) % 50) == 0:
                pid_engine.step_pid(n_queries=50)

        if (i + 1) % 250 == 0:
            print(f"  ...{i + 1}/{TOTAL_QUERIES} "
                  f"(baseline: {len(baseline_times)}, pid: {len(pid_times)})")

    # Results
    b_p50 = float(np.percentile(baseline_times, 50))
    b_mean = float(np.mean(baseline_times))
    b_p95 = float(np.percentile(baseline_times, 95))
    p_p50 = float(np.percentile(pid_times, 50))
    p_mean = float(np.mean(pid_times))
    p_p95 = float(np.percentile(pid_times, 95))

    print("\n" + "=" * 70)
    print("LATENCY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':>12} {'PID':>12} {'Delta':>10}")
    print("-" * 70)
    print(f"{'P50 (µs)':<25} {b_p50:>12.1f} {p_p50:>12.1f} {((p_p50-b_p50)/b_p50*100):>+9.1f}%")
    print(f"{'Mean (µs)':<25} {b_mean:>12.1f} {p_mean:>12.1f} {((p_mean-b_mean)/b_mean*100):>+9.1f}%")
    print(f"{'P95 (µs)':<25} {b_p95:>12.1f} {p_p95:>12.1f} {((p_p95-b_p95)/b_p95*100):>+9.1f}%")

    # PID state
    state = pid_engine.state()
    metrics = pid_engine._metrics

    print("\n" + "=" * 70)
    print("PID STATE")
    print("=" * 70)
    print(f"  Cascade thresholds: L1={state['cascade_thresholds']['L1']} "
          f"L2={state['cascade_thresholds']['L2']} "
          f"L3={state['cascade_thresholds']['L3']}")
    print(f"  Router: {state['route_percent']}% geometric")
    print(f"  Cache: TTL={state['cache_ttl']}s")
    print(f"  Cache hits/misses: {metrics['cache_hits']}/{metrics['cache_misses']}")
    print(f"  Cascade L1/L2/L3/miss: {metrics['cascade_l1']}/{metrics['cascade_l2']}/"
          f"{metrics['cascade_l3']}/{metrics['cascade_miss']}")
    print(f"  Geometric routed: {metrics['geo_routed']}")
    print(f"  Fast routed: {metrics['fast_routed']}")

    # Recall check
    print("\n" + "=" * 70)
    print("RECALL CHECK (100 probes)")
    print("=" * 70)
    probe_idx = np.random.choice(n, 100, replace=False)
    exact = {}
    for idx in probe_idx:
        q = embs[idx]
        sims = embs @ q
        sims[idx] = -np.inf
        exact[idx] = set(np.argsort(-sims)[:K])

    def measure_recall(search_fn):
        hits = 0
        total = 0
        for idx in probe_idx:
            results = search_fn(embs[idx])
            top = {int(pid) for score, (pid, _) in results[:K] if int(pid) != idx}
            if len(top) > 0:
                hits += len(exact[idx] & top)
                total += K
        return hits / total if total > 0 else 0.0

    baseline_recall = measure_recall(
        lambda q: wrapper.search_hybrid_by_embedding(q, k=K, scale=REAL_SCALE))
    pid_recall = measure_recall(
        lambda q: pid_engine.search(q, k=K, scale=REAL_SCALE))

    print(f"  Baseline R@10: {baseline_recall:.3f}")
    print(f"  PID R@10:      {pid_recall:.3f}")
    print(f"  Delta:         {((pid_recall-baseline_recall)/baseline_recall*100) if baseline_recall else 0:+.1f}%")

    os.makedirs("logs", exist_ok=True)
    with open("logs/bench_pid_1m_real_cascade_geo.json", "w") as f:
        json.dump({
            "baseline": {"count": len(baseline_times), "p50_us": b_p50,
                         "mean_us": b_mean, "p95_us": b_p95, "r_at_10": baseline_recall},
            "pid": {"count": len(pid_times), "p50_us": p_p50,
                    "mean_us": p_mean, "p95_us": p_p95, "r_at_10": pid_recall},
            "pid_state": state,
            "pid_metrics": metrics,
        }, f, indent=2)
    print("\nSaved logs/bench_pid_1m_real_cascade_geo.json")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if pid_recall > baseline_recall:
        print(f"✅ PID improves recall: {baseline_recall:.3f} -> {pid_recall:.3f}")
    elif pid_recall < baseline_recall:
        print(f"⚠️  PID recall dropped: {baseline_recall:.3f} -> {pid_recall:.3f}")
    else:
        print("= Recall unchanged")

    if p_p50 <= b_p50 * 1.02:
        print("✅ PID latency neutral")
    elif p_p50 <= b_p50 * 1.10:
        print(f"⚠️  PID latency +{((p_p50-b_p50)/b_p50*100):.1f}% (acceptable)")
    else:
        print(f"❌ PID latency too high: +{((p_p50-b_p50)/b_p50*100):.1f}%")


if __name__ == "__main__":
    main()
