#!/usr/bin/env python3
"""
Phase B: M3.8 Validation — FIXED
Uses yp_bridge.YPEngine with real titles from engine.cache.
"""
import time, statistics, sys, os, json, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yp_bridge import YPEngine

SEED = 42
random.seed(SEED)

print("=" * 65)
print("PHASE B: M3.8 Validation 10K — FIXED")
print("=" * 65)

print("\n[1/3] Loading engine...")
engine = YPEngine()
total = len(engine.cache)
print(f"      Papers in cache: {total}")

if total < 2000:
    print("FATAL: Need at least 2000 papers.")
    sys.exit(1)

print("\n[2/3] Building held-out set...")
items = list(engine.cache.items())
random.shuffle(items)

split = total // 2
query_items = items[split:]

NUM_QUERIES = min(10000, len(query_items))
query_items = query_items[:NUM_QUERIES]

print(f"      Queries: {len(query_items)}")

print("\n[3/3] Running queries...")
latencies = []
r1_hits = 0
r5_hits = 0
r10_hits = 0

for i, (true_pid, title) in enumerate(query_items):
    t0 = time.perf_counter()
    try:
        results = engine.search(title, top_k=10)
    except Exception as e:
        results = []
    t1 = time.perf_counter()
    
    lat = (t1 - t0) * 1_000_000
    latencies.append(lat)
    
    result_pids = [r[1][0] for r in results] if results else []
    
    if true_pid in result_pids[:1]:
        r1_hits += 1
    if true_pid in result_pids[:5]:
        r5_hits += 1
    if true_pid in result_pids[:10]:
        r10_hits += 1
    
    if (i + 1) % 1000 == 0:
        r1_so_far = r1_hits / (i+1) * 100
        p50_so_far = statistics.median(latencies[-1000:])
        print(f"      {i+1}/{NUM_QUERIES} | R@1: {r1_so_far:.1f}% | P50: {p50_so_far:.1f}µs")

n = NUM_QUERIES
latencies.sort()
p50 = latencies[n // 2]
p99 = latencies[int(n * 0.99)]
mean_lat = statistics.mean(latencies)

r1 = r1_hits / n * 100
r5 = r5_hits / n * 100
r10 = r10_hits / n * 100

print("\n" + "=" * 65)
print("M3.8 VALIDATION RESULTS")
print("=" * 65)
print(f"\n  Queries:      {n}")
print(f"  P50 latency:  {p50:.2f} µs")
print(f"  P99 latency:  {p99:.2f} µs")
print(f"  Mean latency: {mean_lat:.2f} µs")
print(f"\n  R@1:          {r1:.2f}%  {'PASS' if r1 >= 97 else 'FAIL'}  (target: 97%)")
print(f"  R@5:          {r5:.2f}%")
print(f"  R@10:         {r10:.2f}%")

print("\n" + "=" * 65)
if r1 >= 97:
    print("M3.8 TARGET MET — R@1 >= 97%")
else:
    gap = 97.0 - r1
    print(f"M3.8 TARGET NOT MET — gap: {gap:.2f}pp")
print("=" * 65)

out = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'total_queries': n,
    'p50_us': round(p50, 2),
    'p99_us': round(p99, 2),
    'mean_us': round(mean_lat, 2),
    'r1_pct': round(r1, 2),
    'r5_pct': round(r5, 2),
    'r10_pct': round(r10, 2),
    'target_met': r1 >= 97,
}
os.makedirs('logs', exist_ok=True)
with open('logs/m38_validation_10k_fixed.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: logs/m38_validation_10k_fixed.json")
