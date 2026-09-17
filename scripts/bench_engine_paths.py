#!/usr/bin/env python3
"""Compare YPEngine search paths: standard vs two-tier."""
import os, sys, time, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from yp_engine import YPEngine

QUERIES = [
    "quantum entanglement",
    "machine learning",
    "neural networks",
    "black holes",
    "climate change",
    "crispr gene editing",
    "dark matter",
    "reinforcement learning",
    "graphene properties",
    "covid vaccine",
]

def bench_path(engine, queries, path_name):
    times = []
    for q in queries:
        t0 = time.perf_counter()
        if path_name == "two_tier":
            res = engine.two_tier_search(q, k=10)
        else:
            res = engine.search(q, top_k=10)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return times

def main():
    print("Loading YPEngine...")
    engine = YPEngine()
    print(f"Engine loaded. HNSW preloaded: {engine._hnsw is not None}")

    # Standard path
    print("\n--- Standard search path ---")
    std_times = bench_path(engine, QUERIES, "standard")
    print(f"P50: {np.percentile(std_times, 50):.0f}µs  P95: {np.percentile(std_times, 95):.0f}µs")

    # Two-tier path (if available)
    if engine._hnsw:
        print("\n--- Two-tier search path ---")
        tt_times = bench_path(engine, QUERIES, "two_tier")
        print(f"P50: {np.percentile(tt_times, 50):.0f}µs  P95: {np.percentile(tt_times, 95):.0f}µs")
    else:
        print("\n--- Two-tier unavailable (no HNSW preloaded) ---")
        tt_times = []

    out = {
        "standard_p50_us": round(float(np.percentile(std_times, 50)), 2),
        "standard_p95_us": round(float(np.percentile(std_times, 95)), 2),
        "two_tier_p50_us": round(float(np.percentile(tt_times, 50)), 2) if tt_times else None,
        "two_tier_p95_us": round(float(np.percentile(tt_times, 95)), 2) if tt_times else None,
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/bench_engine_paths.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to logs/bench_engine_paths.json")

if __name__ == "__main__":
    main()
