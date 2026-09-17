#!/usr/bin/env python3
"""Benchmark Binary HNSW (Rust) vs brute-force Hamming on real ITQ hashes."""

import os, sys, time, json, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from yp_bridge import BinaryHNSW

HASH_PATHS = [
    "paper_hashes_100k.npy",
    "paper_hashes_10k.npy",
    "data/paper_hashes_100k.npy",
    "data/paper_hashes_10k.npy",
]

def load_hashes():
    for p in HASH_PATHS:
        if os.path.exists(p):
            print(f"Loading hashes from {p}")
            arr = np.load(p)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if arr.ndim == 1 and arr.shape[0] % 64 == 0:
                arr = arr.reshape(-1, 64)
            assert arr.shape[1] == 64, f"Expected 64 bytes per hash, got {arr.shape}"
            return arr
    print("No real hash file found — generating 10K synthetic 512-bit hashes")
    return np.random.randint(0, 256, size=(10_000, 64), dtype=np.uint8)

def percentile(times, p):
    return float(np.percentile(times, p))

def bench():
    hashes = load_hashes()
    n = hashes.shape[0]
    print(f"Dataset: {n} vectors, 512 bits each")

    # --- Build HNSW ---
    hnsw = BinaryHNSW()
    t0 = time.perf_counter()
    for i in range(n):
        hnsw.insert(i, bytes(hashes[i]))
    build_s = time.perf_counter() - t0
    print(f"\nHNSW build: {build_s:.3f}s  ({n/build_s:,.0f} vec/s)")
    print(f"HNSW len: {len(hnsw)}")

    # --- Search benchmark ---
    nq = min(1000, n)
    queries = hashes[np.random.choice(n, nq, replace=False)]

    hnsw_times = []
    hnsw_results = []
    for q in queries:
        t0 = time.perf_counter()
        res = hnsw.search(bytes(q), k=100)
        t1 = time.perf_counter()
        hnsw_times.append((t1 - t0) * 1e6)  # microseconds
        hnsw_results.append(res)

    print(f"\nHNSW search (k=100, {nq} queries):")
    print(f"  P50: {percentile(hnsw_times, 50):.1f} µs")
    print(f"  P95: {percentile(hnsw_times, 95):.1f} µs")
    print(f"  P99: {percentile(hnsw_times, 99):.1f} µs")
    print(f"  Mean: {np.mean(hnsw_times):.1f} µs")

    # --- Brute-force baseline (subset for speed) ---
    bf_n = min(10_000, n)
    bf_hashes = hashes[:bf_n]
    bf_queries = queries[:min(100, nq)]
    bf_times = []
    for q in bf_queries:
        t0 = time.perf_counter()
        # XOR + popcount in numpy
        diffs = np.bitwise_xor(bf_hashes, q).reshape(bf_n, 8, 8)
        # popcount via lookup (slower but pure numpy)
        # Use Python int popcount for accuracy on small subset
        dists = []
        for row in diffs:
            d = 0
            for b in row.flat:
                d += b.bit_count()
            dists.append(d)
        # get top 100
        idx = np.argpartition(dists, 100)[:100]
        t1 = time.perf_counter()
        bf_times.append((t1 - t0) * 1e6)

    print(f"\nBrute-force Hamming (n={bf_n}, k=100, {len(bf_queries)} queries):")
    print(f"  P50: {percentile(bf_times, 50):.1f} µs")
    print(f"  P95: {percentile(bf_times, 95):.1f} µs")
    print(f"  Mean: {np.mean(bf_times):.1f} µs")

    speedup = percentile(bf_times, 50) / percentile(hnsw_times, 50)
    print(f"\nSpeedup (P50): {speedup:.1f}×")

    # --- Recall check (self-query on first 100) ---
    hits = 0
    for i in range(min(100, n)):
        res = hnsw.search(bytes(hashes[i]), k=10)
        if any(r[0] == i for r in res):
            hits += 1
    print(f"\nSelf-query recall@10 (first 100): {hits}/100 = {hits}%")

    # Save results
    out = {
        "n_vectors": int(n),
        "build_sec": round(build_s, 3),
        "hnsw_p50_us": round(percentile(hnsw_times, 50), 2),
        "hnsw_p95_us": round(percentile(hnsw_times, 95), 2),
        "hnsw_p99_us": round(percentile(hnsw_times, 99), 2),
        "bf_p50_us": round(percentile(bf_times, 50), 2),
        "speedup_p50x": round(speedup, 1),
        "self_recall_at_10": hits,
    }
    out_path = "logs/bench_binary_hnsw.json"
    os.makedirs("logs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return out

if __name__ == "__main__":
    bench()
