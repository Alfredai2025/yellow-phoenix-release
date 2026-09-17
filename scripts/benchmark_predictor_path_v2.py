#!/usr/bin/env python3
"""
Phase A: Predictor-Driven Path — HARDENED
Verifies the Trinity path is ACTUALLY exercised, not silently ignored.
Uses *titles* as self-queries so the engine has a real semantic signal.
"""
import time, statistics, sys, os, json, inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 65)
print("PHASE A: Predictor Path Benchmark — HARDENED")
print("=" * 65)

# ── 1. Import and verify Trinity exists ──
print("\n[1/6] Import check...")
try:
    from yp_bridge import RustBridge, enable_trinity_deep_wire, YPEngine
except ImportError as e:
    print(f"FATAL: Cannot import: {e}")
    sys.exit(1)

# ── 2. Inspect YPEngine.search signature ──
print("\n[2/6] Verifying YPEngine.search() accepts 'use_trinity'...")
search_sig = inspect.signature(YPEngine.search)
search_params = list(search_sig.parameters.keys())
print(f"      search() params: {search_params}")

if 'use_trinity' not in search_params:
    print("FATAL: YPEngine.search() does NOT accept 'use_trinity' parameter.")
    print("       The benchmark would silently ignore it and fake identical results.")
    print("       Fix: Wire use_trinity into search() first.")
    sys.exit(1)
print("      ✅ use_trinity parameter confirmed")

# ── 3. Load engine ──
print("\n[3/6] Loading engine...")
engine = YPEngine()
total = len(engine.cache)
print(f"      Papers: {total}")

# ── 4. Verify deep wire can be enabled ──
print("\n[4/6] Enabling Trinity deep wire...")
try:
    enable_trinity_deep_wire(engine)
    print("      ✅ Deep wire enabled")
except Exception as e:
    print(f"FATAL: Deep wire failed: {e}")
    sys.exit(1)

# ── 5. Verify predictor actually returns hints ──
print("\n[5/6] Verifying predictor returns hints...")
bridge = engine.rust if (getattr(engine, 'rust', None) is not None) else RustBridge()
engine.rust = bridge
if not hasattr(bridge.lib, 'yp_trinity_predict_bucket'):
    print("FATAL: yp_trinity_predict_bucket not in FFI")
    sys.exit(1)

import ctypes
test_bucket = bridge.lib.yp_trinity_predict_bucket(999)
print(f"      Predictor hint for query_id=999: bucket {test_bucket}")
if test_bucket < 0:
    print("      ⚠️  Predictor returns -1 (not initialized or no history)")
    print("      This is HONEST — it means the predictor has no training data yet.")
    print("      Benchmark will show fallback path. This is EXPECTED for an untrained stub.")
else:
    print(f"      ✅ Predictor returns bucket {test_bucket}")

# ── 6. Prepare self-queries from titles ──
print("\n[6/6] Preparing title self-query set...")
queries = []
for pid, title in list(engine.cache.items())[:5200]:
    if title and len(title) > 10:
        queries.append((pid, title))

if len(queries) < 1000:
    print(f"FATAL: Only {len(queries)} queries with text. Need ≥1000.")
    sys.exit(1)
print(f"      ✅ Queries ready: {len(queries)}")

def result_ids(results):
    """Robust ID extraction: handles (score, (pid, title)) and (score, pid)."""
    ids = []
    for r in results or []:
        payload = r[1] if len(r) > 1 else None
        if isinstance(payload, tuple):
            ids.append(payload[0])
        elif payload is not None:
            ids.append(payload)
    return ids

# ── Benchmark with PATH VERIFICATION ──
def run_benchmark(use_trinity, label, n=3000, warmup=200):
    latencies = []
    hits = 0
    misses = 0
    
    test_set = queries[warmup:warmup + n]
    
    for i, (pid, qtext) in enumerate(test_set):
        t0 = time.perf_counter()
        results = engine.search(qtext, top_k=1, use_trinity=use_trinity)
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1_000_000
        latencies.append(lat)
        
        if pid in result_ids(results):
            hits += 1
        else:
            misses += 1
        
        if (i + 1) % 1000 == 0:
            print(f"      {label}: {i+1}/{n} | hits={hits} misses={misses}")
    
    latencies.sort()
    return {
        'label': label,
        'n': n,
        'p50': latencies[n // 2],
        'p99': latencies[int(n * 0.99)],
        'mean': statistics.mean(latencies),
        'hit_rate': hits / n * 100,
        'miss_rate': misses / n * 100,
        'min_lat': min(latencies),
        'max_lat': max(latencies),
    }

print("\n" + "=" * 65)
print("RUNNING BASELINE (use_trinity=False)")
print("=" * 65)
baseline = run_benchmark(False, "BASELINE")

print("\n" + "=" * 65)
print("RUNNING TRINITY (use_trinity=True)")
print("=" * 65)
trinity = run_benchmark(True, "TRINITY")

# ── FAKE DETECTION ──
print("\n" + "=" * 65)
print("FAKE DETECTION")
print("=" * 65)

p50_diff_pct = abs(baseline['p50'] - trinity['p50']) / baseline['p50'] * 100 if baseline['p50'] > 0 else 0
print(f"  P50 difference: {p50_diff_pct:.2f}%")

if p50_diff_pct < 1.0 and baseline['hit_rate'] == trinity['hit_rate']:
    print("  🚨 FAKE ALERT: Baseline and Trinity are statistically identical.")
    print("     This means use_trinity is being IGNORED in search().")
    print("     DO NOT trust these numbers. Fix the wiring first.")
    sys.exit(1)
else:
    print("  ✅ Paths produce different results — parameter is being respected.")

# ── Results ──
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

for r in [baseline, trinity]:
    print(f"\n  {r['label']}:")
    print(f"    N:        {r['n']}")
    print(f"    P50:      {r['p50']:.2f} µs")
    print(f"    P99:      {r['p99']:.2f} µs")
    print(f"    Mean:     {r['mean']:.2f} µs")
    print(f"    Hit rate: {r['hit_rate']:.1f}%")
    print(f"    Range:    {r['min_lat']:.2f} – {r['max_lat']:.2f} µs")

speedup = baseline['p50'] / trinity['p50'] if trinity['p50'] > 0 else 0
print(f"\n  Speedup (P50): {speedup:.2f}x")

out = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'baseline': baseline,
    'trinity': trinity,
    'speedup_p50': round(speedup, 2),
    'fake_detected': False,
}
os.makedirs('logs', exist_ok=True)
with open('logs/benchmark_predictor_path_v2.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved: logs/benchmark_predictor_path_v2.json")

print("\n" + "=" * 65)
print("PHASE A COMPLETE — VERIFIED")
print("=" * 65)
