#!/usr/bin/env python3
"""
Verify that medium queries are now faster after TF-IDF gate.
"""
import sys, os, time, random, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

random.seed(42)
engine = YPEngine()

# Build medium queries
medium_queries = []
for pid, title in list(engine.cache.items()):
    words = title.split()
    if 6 <= len(words) <= 10:
        kept = max(3, int(len(words) * 0.7))
        corrupted = ' '.join(random.sample(words, kept))
        medium_queries.append((pid, corrupted))
    if len(medium_queries) >= 50:
        break

print("=" * 65)
print("CASCADE FIX VERIFICATION")
print("=" * 65)

# Warmup
for _, q in medium_queries[:5]:
    engine.search(q, top_k=5)

# Benchmark
latencies = []
r1_hits = 0

for true_pid, query in medium_queries:
    t0 = time.perf_counter()
    results = engine.search(query, top_k=5)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1_000_000)
    
    result_pids = [r[1][0] for r in results if len(r) > 1 and isinstance(r[1], tuple)]
    if true_pid in result_pids[:1]:
        r1_hits += 1

latencies.sort()
p50 = latencies[len(latencies) // 2]
p99 = latencies[int(len(latencies) * 0.99)]

print(f"\n  Medium queries (4-7 words, n={len(medium_queries)}):")
print(f"    P50 latency: {p50:.0f} µs")
print(f"    P99 latency: {p99:.0f} µs")
print(f"    R@1:         {r1_hits/len(medium_queries)*100:.1f}%")

# Compare to exact title
exact_times = []
for _, title in random.sample(list(engine.cache.items()), 50):
    t0 = time.perf_counter()
    engine.search(title, top_k=1)
    t1 = time.perf_counter()
    exact_times.append((t1 - t0) * 1_000_000)
exact_times.sort()
print(f"\n  Exact titles (n=50):")
print(f"    P50 latency: {exact_times[25]:.0f} µs")

print("\n" + "=" * 65)
if p50 < 15000:
    print("✅ FIX WORKED — medium queries now < 15 ms")
elif p50 < 30000:
    print("⚠️  PARTIAL — medium queries < 30 ms, still room")
else:
    print("❌ STILL SLOW — MiniLM gate not triggering correctly")
print("=" * 65)
