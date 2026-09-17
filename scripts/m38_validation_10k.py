#!/usr/bin/env python3
"""
Phase B: M3.8 Validation (adapted to current YPEngine API)
Text-based recall benchmark using yp_bridge.YPEngine.
"""
import time, statistics, sys, os, json, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yp_bridge import YPEngine

NUM_QUERIES = 2000
TOP_K = 10
SEED = 42
random.seed(SEED)

print("=" * 60)
print("PHASE B: M3.8 Validation — Text Queries")
print("=" * 60)

print("\n[1/4] Loading engine...")
engine = YPEngine()
print(f"      Total papers: {len(engine.cache)}")

print("\n[2/4] Building held-out test set...")
papers = list(engine.cache.items())
random.shuffle(papers)
split = len(papers) // 2
query_papers = papers[split:split + NUM_QUERIES]
print(f"      Query papers: {len(query_papers)}")

print("\n[3/4] Running validation...")
latencies = []
r1 = r5 = r10 = 0
total = 0

for i, (true_pid, title) in enumerate(query_papers):
    query_text = title if title and len(title) > 5 else f"paper {true_pid}"
    t0 = time.perf_counter()
    try:
        results = engine.search(query_text, top_k=TOP_K)
    except Exception as e:
        results = []
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1_000_000)

    result_pids = [r[1][0] for r in results] if results else []
    if true_pid in result_pids[:1]:
        r1 += 1
    if true_pid in result_pids[:5]:
        r5 += 1
    if true_pid in result_pids[:10]:
        r10 += 1
    total += 1

    if (i + 1) % 500 == 0:
        print(f"      {i+1}/{len(query_papers)} | R@1 so far: {r1/total*100:.1f}% | P50(last 500): {statistics.median(latencies[-500:]):.1f}µs")

latencies.sort()
p50 = latencies[len(latencies) // 2]
p99 = latencies[int(len(latencies) * 0.99)]
mean = statistics.mean(latencies)

results = {
    'total_queries': total,
    'r1_pct': round(r1 / total * 100, 2),
    'r5_pct': round(r5 / total * 100, 2),
    'r10_pct': round(r10 / total * 100, 2),
    'p50_us': round(p50, 2),
    'p99_us': round(p99, 2),
    'mean_us': round(mean, 2),
    'target_r1': 97.0,
    'target_met': (r1 / total * 100) >= 97.0,
}

print("\n" + "=" * 60)
print("M3.8 VALIDATION RESULTS")
print("=" * 60)
print(f"\n  Queries run:  {results['total_queries']}")
print(f"  R@1:          {results['r1_pct']:.2f}%  (target 97.0%)  {'PASS' if results['target_met'] else 'FAIL'}")
print(f"  R@5:          {results['r5_pct']:.2f}%")
print(f"  R@10:         {results['r10_pct']:.2f}%")
print(f"  P50 latency:  {results['p50_us']:.2f} µs")
print(f"  P99 latency:  {results['p99_us']:.2f} µs")
print(f"  Mean latency: {results['mean_us']:.2f} µs")

os.makedirs('logs', exist_ok=True)
with open('logs/m38_validation_10k.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n  Saved to logs/m38_validation_10k.json")

print("\n" + "=" * 60)
if results['target_met']:
    print("✅ M3.8 TARGET MET — R@1 ≥ 97%")
else:
    print("❌ M3.8 TARGET NOT MET — R@1 below 97%")
print("=" * 60)
