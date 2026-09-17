#!/usr/bin/env python3
"""
Phase A: Benchmark Trinity Predictor-Driven Path
Uses yp_bridge.YPEngine and direct FFI call to yp_query_auto_with_trinity_hint.
Note: the current predictor stub always returns bucket 12, so hit rate reflects
how many stored hashes actually hash to bucket 12.
"""
import time, statistics, sys, os, json, ctypes, sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yp_bridge import YPEngine

NUM_QUERIES = 5000
WARMUP = 200
TOP_K = 1
DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/phoenix_arxiv_1m.db')

print("=" * 60)
print("PHASE A: Predictor-Driven Path Benchmark")
print("=" * 60)

print("\n[1/4] Loading YP engine...")
engine = YPEngine()
print(f"      Papers loaded: {len(engine.cache)}")

print("\n[2/4] Checking Trinity wiring...")
lib = engine.rust.lib
has_trinity = hasattr(lib, 'yp_trinity_init')
print(f"      Trinity FFI available: {has_trinity}")
if not has_trinity:
    print("ERROR: build with --features trinity")
    sys.exit(1)

# Ensure Trinity is initialized
lib.yp_trinity_init()

# Set up direct FFI call
lib.yp_query_auto_with_trinity_hint.argtypes = [
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_size_t),
]
lib.yp_query_auto_with_trinity_hint.restype = ctypes.c_int

print("\n[3/4] Loading stored hashes...")
conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("SELECT id, yp_hash512 FROM papers WHERE yp_hash512 IS NOT NULL AND yp_hash512 != '' LIMIT ?", (NUM_QUERIES + WARMUP,))
rows = cur.fetchall()
conn.close()

queries = []
for pid, hhex in rows:
    try:
        hb = bytes.fromhex(hhex)
        if len(hb) == 64:
            queries.append((pid, hb))
    except Exception:
        pass
print(f"      Valid stored-hash queries: {len(queries)}")

if len(queries) < 100:
    sys.exit(1)

def run_baseline():
    latencies = []
    hits = 0
    for i, (pid, hb) in enumerate(queries[WARMUP:WARMUP + NUM_QUERIES]):
        t0 = time.perf_counter()
        result = engine.search_sharded_hash(hb.hex(), top_k=TOP_K)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1_000_000)
        if result and result[0][1][0] == pid:
            hits += 1
        if (i + 1) % 1000 == 0:
            print(f"      BASELINE: {i+1}/{NUM_QUERIES}")
    return latencies, hits

def run_trinity():
    latencies = []
    hits = 0
    scores_buf = (ctypes.c_float * 10)()
    ids_buf = (ctypes.c_uint64 * 10)()
    len_out = ctypes.c_size_t()
    for i, (pid, hb) in enumerate(queries[WARMUP:WARMUP + NUM_QUERIES]):
        pap_128 = hb[:16]
        pap_512 = hb
        t0 = time.perf_counter()
        rc = lib.yp_query_auto_with_trinity_hint(
            i + 1,
            ctypes.c_void_p(ctypes.addressof((ctypes.c_uint8 * 16)(*pap_128))),
            ctypes.c_void_p(ctypes.addressof((ctypes.c_uint8 * 64)(*pap_512))),
            TOP_K,
            scores_buf,
            ids_buf,
            len_out,
        )
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1_000_000)
        if rc == 0 and len_out.value > 0 and ids_buf[0] == pid:
            hits += 1
        if (i + 1) % 1000 == 0:
            print(f"      TRINITY: {i+1}/{NUM_QUERIES}")
    return latencies, hits

def stats(latencies, hits, label):
    latencies.sort()
    n = len(latencies)
    return {
        'label': label,
        'count': n,
        'p50_us': round(latencies[n // 2], 2),
        'p99_us': round(latencies[int(n * 0.99)], 2),
        'mean_us': round(statistics.mean(latencies), 2),
        'hit_rate_pct': round(hits / n * 100, 2),
        'min_us': round(min(latencies), 2),
        'max_us': round(max(latencies), 2),
    }

print("\n[4/4] Running benchmarks...")
print("\n  → Baseline (search_sharded_hash)...")
base_lat, base_hits = run_baseline()
baseline = stats(base_lat, base_hits, "BASELINE")

print("\n  → Trinity predictor-driven path...")
tri_lat, tri_hits = run_trinity()
trinity = stats(tri_lat, tri_hits, "TRINITY")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
for r in (baseline, trinity):
    print(f"\n  {r['label']}:")
    print(f"    Queries:   {r['count']}")
    print(f"    P50:       {r['p50_us']:.2f} µs")
    print(f"    P99:       {r['p99_us']:.2f} µs")
    print(f"    Mean:      {r['mean_us']:.2f} µs")
    print(f"    Hit rate:  {r['hit_rate_pct']:.2f}%")
    print(f"    Range:     {r['min_us']:.2f} – {r['max_us']:.2f} µs")

speedup_p50 = baseline['p50_us'] / trinity['p50_us'] if trinity['p50_us'] > 0 else 0
speedup_p99 = baseline['p99_us'] / trinity['p99_us'] if trinity['p99_us'] > 0 else 0
print(f"\n  SPEEDUP: P50 {speedup_p50:.2f}x | P99 {speedup_p99:.2f}x")

out = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'baseline': baseline,
    'trinity': trinity,
    'speedup_p50': round(speedup_p50, 2),
    'speedup_p99': round(speedup_p99, 2),
}
os.makedirs('logs', exist_ok=True)
with open('logs/benchmark_predictor_path.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\n  Saved to logs/benchmark_predictor_path.json")
print("\n" + "=" * 60)
print("PHASE A COMPLETE")
print("=" * 60)
