#!/usr/bin/env python3
"""
Trinity Validation Script
Validates that the fast sharded-hash path remains within baseline
latency after Trinity integration. Runs 2,000 stored-hash lookups.
"""
import ctypes
import json
import time
import statistics
import sys
import os
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yp_engine import YPEngine

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data/phoenix_arxiv_1m.db')

def load_hashes(n=2000):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT yp_hash512 FROM papers "
        "WHERE yp_hash512 IS NOT NULL AND yp_hash512 != '' LIMIT ?",
        (n,)
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows

def main():
    print("=" * 60)
    print("TRINITY VALIDATION RUN")
    print("=" * 60)
    
    engine = YPEngine()
    print(f"Engine initialized. Papers loaded: {len(engine.cache)}")
    
    hashes = load_hashes(2000)
    if len(hashes) < 2000:
        print(f"ERROR: only {len(hashes)} stored hashes available")
        return 1
    
    # Warm-up
    print("\n[WARM-UP] 100 stored-hash lookups...")
    for h in hashes[:100]:
        engine.search_sharded_hash(h, top_k=5)
    print("Warm-up complete.")
    
    # Measurement: 2,000 stored-hash lookups
    print("\n[MEASUREMENT] 2,000 stored-hash lookups...")
    latencies = []
    
    for i, h in enumerate(hashes):
        start = time.perf_counter_ns()
        result = engine.search_sharded_hash(h, top_k=5)
        end = time.perf_counter_ns()
        latency_us = (end - start) / 1000.0
        latencies.append(latency_us)
        
        if (i + 1) % 500 == 0:
            print(f"  {i+1} queries done...")
    
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    p999 = latencies[int(len(latencies) * 0.999)]
    mean = statistics.mean(latencies)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total queries:     {len(latencies)}")
    print(f"Mean latency:      {mean:.2f} μs")
    print(f"P50 latency:       {p50:.2f} μs")
    print(f"P99 latency:       {p99:.2f} μs")
    print(f"P99.9 latency:     {p999:.2f} μs")
    print(f"Min latency:       {min(latencies):.2f} μs")
    print(f"Max latency:       {max(latencies):.2f} μs")
    
    print("\n" + "=" * 60)
    print("REGRESSION CHECK")
    print("=" * 60)
    
    baseline_p50 = 2.9
    baseline_p99 = 37.0
    
    ok = True
    if p50 > baseline_p50 * 1.1:
        print(f"FAIL: P50 {p50:.2f} > baseline {baseline_p50} + 10%")
        ok = False
    else:
        print(f"PASS: P50 {p50:.2f} <= baseline {baseline_p50} + 10%")
    
    if p99 > baseline_p99 * 1.1:
        print(f"FAIL: P99 {p99:.2f} > baseline {baseline_p99} + 10%")
        ok = False
    else:
        print(f"PASS: P99 {p99:.2f} <= baseline {baseline_p99} + 10%")
    
    print("\n" + "=" * 60)
    if ok:
        print("VALIDATION COMPLETE - PASS")
    else:
        print("VALIDATION COMPLETE - FAIL")
    print("=" * 60)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
