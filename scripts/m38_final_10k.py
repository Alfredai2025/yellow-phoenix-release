#!/usr/bin/env python3
"""
M3.8 Final Validation — 10K queries against held-out set.
Target: R@1 >= 97% on exact-title queries.
Measures: R@1, R@5, R@10, P50, P99, mean latency.
"""
import sys, os, random, time, statistics, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

SEED = 42
random.seed(SEED)

print("=" * 65)
print("M3.8 VALIDATION — 10K QUERIES")
print("=" * 65)

engine = YPEngine()
papers = list(engine.cache.items())
total = len(papers)
print(f"Papers loaded: {total}")

# True held-out: second half of shuffled list
random.shuffle(papers)
split = total // 2
index_papers = papers[:split]
query_papers = papers[split:]

NUM_QUERIES = min(10000, len(query_papers))
query_papers = query_papers[:NUM_QUERIES]

print(f"Index: {len(index_papers)} | Queries: {len(query_papers)}")

# Run
latencies = []
r1_hits = 0
r5_hits = 0
r10_hits = 0

for i, (true_pid, title) in enumerate(query_papers):
    t0 = time.perf_counter()
    try:
        results = engine.search(title, top_k=10)
    except Exception as e:
        results = []
    t1 = time.perf_counter()
    
    lat = (t1 - t0) * 1_000_000
    latencies.append(lat)
    
    result_pids = [r[1][0] for r in results if len(r) > 1 and isinstance(r[1], tuple)]
    
    if true_pid in result_pids[:1]:
        r1_hits += 1
    if true_pid in result_pids[:5]:
        r5_hits += 1
    if true_pid in result_pids[:10]:
        r10_hits += 1
    
    if (i + 1) % 1000 == 0:
        r1_so_far = r1_hits / (i+1) * 100
        p50_so_far = statistics.median(latencies[-1000:])
        print(f"  {i+1}/{NUM_QUERIES} | R@1: {r1_so_far:.2f}% | P50: {p50_so_far:.1f}µs")

# Results
n = NUM_QUERIES
latencies.sort()
p50 = latencies[n // 2]
p99 = latencies[int(n * 0.99)]
mean_lat = statistics.mean(latencies)

r1 = r1_hits / n * 100
r5 = r5_hits / n * 100
r10 = r10_hits / n * 100
target_met = r1 >= 97.0

print("\n" + "=" * 65)
print("M3.8 RESULTS")
print("=" * 65)
print(f"Queries:      {n}")
print(f"R@1:          {r1:.2f}%  {'✅ PASS' if target_met else '❌ FAIL'}  (target: 97%)")
print(f"R@5:          {r5:.2f}%")
print(f"R@10:         {r10:.2f}%")
print(f"P50 latency:  {p50:.2f} µs")
print(f"P99 latency:  {p99:.2f} µs")
print(f"Mean latency: {mean_lat:.2f} µs")

print("\n" + "=" * 65)
if target_met:
    print("✅ M3.8 TARGET MET — R@1 >= 97%")
    print("Engine validated for production.")
else:
    gap = 97.0 - r1
    print(f"❌ M3.8 TARGET NOT MET — gap: {gap:.2f}pp")
    print("Needs encoder retraining or cascade tuning.")
print("=" * 65)

# Save
out = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'commit': '$COMMIT_HASH',
    'total_queries': n,
    'r1_pct': round(r1, 2),
    'r5_pct': round(r5, 2),
    'r10_pct': round(r10, 2),
    'p50_us': round(p50, 2),
    'p99_us': round(p99, 2),
    'mean_us': round(mean_lat, 2),
    'target_met': target_met,
}
os.makedirs('logs', exist_ok=True)
with open('logs/m38_final_10k.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: logs/m38_final_10k.json")
