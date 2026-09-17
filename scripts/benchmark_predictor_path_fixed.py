#!/usr/bin/env python3
"""
Phase A: Predictor-Driven Path — FIXED
Uses direct FFI call to yp_query_auto_with_trinity_hint.
Honest about untrained predictor (bucket 12 stub).
"""
import time, statistics, sys, os, json, ctypes, sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yp_bridge import YPEngine

NUM_QUERIES = 2000
WARMUP = 100
TOP_K = 1
DB_PATH = 'data/phoenix_arxiv_1m.db'

print("=" * 65)
print("PHASE A: Predictor Path — FIXED")
print("=" * 65)

print("\n[1/4] Loading engine...")
engine = YPEngine()
print(f"      Papers in cache: {len(engine.cache)}")

print("\n[2/4] Fetching stored hashes from DB...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT id, yp_hash512 FROM papers WHERE yp_hash512 IS NOT NULL AND yp_hash512 != '' LIMIT ?",
               (NUM_QUERIES + WARMUP + 500,))
rows = cursor.fetchall()
conn.close()

queries = []
for pid, h in rows:
    try:
        h_bytes = bytes.fromhex(h)
        if len(h_bytes) == 64:
            queries.append((pid, h_bytes))
    except ValueError:
        pass

print(f"      Valid hash queries: {len(queries)}")
if len(queries) < 500:
    print("FATAL: Not enough hashes in DB.")
    sys.exit(1)

print("\n[3/4] Setting up Trinity FFI...")
lib = engine.rust.lib

if not hasattr(lib, 'yp_query_auto_with_trinity_hint'):
    print("FATAL: yp_query_auto_with_trinity_hint not in FFI.")
    sys.exit(1)

lib.yp_query_auto_with_trinity_hint.argtypes = [
    ctypes.c_uint64,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_size_t),
]
lib.yp_query_auto_with_trinity_hint.restype = ctypes.c_int

test_bucket = None
if hasattr(lib, 'yp_trinity_predict_bucket'):
    test_bucket = lib.yp_trinity_predict_bucket(1)
    print(f"      Predictor stub returns bucket: {test_bucket}")
    print("      ⚠️  Stub always returns 12 → hit rate will be ~0% (expected).")

print("\n[4/4] Running benchmarks...")

def benchmark_baseline(queries, n):
    latencies = []
    hits = 0
    for i, (pid, h_bytes) in enumerate(queries[WARMUP:WARMUP+n]):
        t0 = time.perf_counter()
        results = engine.search_sharded_hash(h_bytes.hex(), top_k=TOP_K)
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1_000_000
        latencies.append(lat)
        if results and any(r[1][0] == pid for r in results):
            hits += 1
        if (i+1) % 500 == 0:
            print(f"      BASELINE: {i+1}/{n}")
    latencies.sort()
    return {
        'label': 'BASELINE (sharded_hash)',
        'n': n,
        'p50': latencies[n//2],
        'p99': latencies[int(n*0.99)],
        'mean': statistics.mean(latencies),
        'hit_rate': hits / n * 100,
        'min': min(latencies),
        'max': max(latencies),
    }

def benchmark_trinity(queries, n):
    latencies = []
    hits = 0
    misses = 0
    out_ids = (ctypes.c_uint64 * TOP_K)()
    out_scores = (ctypes.c_float * TOP_K)()
    out_len = ctypes.c_size_t()
    for i, (pid, h_bytes) in enumerate(queries[WARMUP:WARMUP+n]):
        pap_128 = h_bytes[:16]
        pap_512 = h_bytes
        t0 = time.perf_counter()
        ret = lib.yp_query_auto_with_trinity_hint(
            i + 1,
            ctypes.c_char_p(pap_128),
            ctypes.c_char_p(pap_512),
            TOP_K,
            out_ids,
            out_scores,
            ctypes.byref(out_len)
        )
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1_000_000
        latencies.append(lat)
        if ret == 0 and out_len.value > 0 and out_ids[0] == pid:
            hits += 1
        else:
            misses += 1
        if (i+1) % 500 == 0:
            print(f"      TRINITY: {i+1}/{n} | hits={hits} misses={misses}")
    latencies.sort()
    return {
        'label': 'TRINITY (predictor hint)',
        'n': n,
        'p50': latencies[n//2],
        'p99': latencies[int(n*0.99)],
        'mean': statistics.mean(latencies),
        'hit_rate': hits / n * 100,
        'miss_rate': misses / n * 100,
        'min': min(latencies),
        'max': max(latencies),
    }

print("\n  → Running BASELINE...")
baseline = benchmark_baseline(queries, NUM_QUERIES)

print("\n  → Running TRINITY...")
trinity = benchmark_trinity(queries, NUM_QUERIES)

print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)
for r in [baseline, trinity]:
    print(f"\n  {r['label']}:")
    print(f"    N:        {r['n']}")
    print(f"    P50:      {r['p50']:.2f} µs")
    print(f"    P99:      {r['p99']:.2f} µs")
    print(f"    Mean:     {r['mean']:.2f} µs")
    print(f"    Hit rate: {r['hit_rate']:.2f}%")
    print(f"    Range:    {r['min']:.2f} – {r['max']:.2f} µs")

speedup = baseline['p50'] / trinity['p50'] if trinity['p50'] > 0 else 0
print(f"\n  Speedup (P50): {speedup:.2f}x")

print("\n" + "-" * 65)
print("HONEST NOTE:")
print("  Predictor is an untrained stub (always returns bucket 12).")
print("  Hit rate ~0% because real buckets are distributed across 2^16.")
print("  Speedup ≈ 1x means fallback path works; real speedup needs training.")
print("-" * 65)

out = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'baseline': {k: v for k, v in baseline.items() if k != 'label'},
    'trinity': {k: v for k, v in trinity.items() if k != 'label'},
    'speedup_p50': round(speedup, 2),
    'predictor_stub_bucket': int(test_bucket) if test_bucket is not None else None,
    'honest_note': 'Predictor untrained. Hit rate expected near 0%.',
}
os.makedirs('logs', exist_ok=True)
with open('logs/benchmark_predictor_path_fixed.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved: logs/benchmark_predictor_path_fixed.json")
