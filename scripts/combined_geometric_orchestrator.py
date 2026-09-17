#!/usr/bin/env python3
"""
Combined Geometric Orchestrator: tensor_spectral + crystal + hologram + vote.
Tests if word-deletion R@1 improves when hash misses trigger geometric fallback.
"""
import sys, os, ctypes, time, random, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

random.seed(42)
engine = YPEngine()
lib = engine.rust.lib

print("=" * 65)
print("COMBINED GEOMETRIC ORCHESTRATOR")
print("=" * 65)

# ── Discover available geometric FFI functions ──
def try_setup(name, argtypes, restype):
    if not hasattr(lib, name):
        return None
    fn = getattr(lib, name)
    try:
        fn.argtypes = argtypes
        fn.restype = restype
    except Exception:
        pass
    return fn

tensor_query = try_setup('yp_tensor_spectral_query',
    [ctypes.c_char_p, ctypes.c_size_t,
     ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_size_t)],
    ctypes.c_int)

crystal_query = try_setup('yp_multi_base_crystal_query',
    [ctypes.c_char_p, ctypes.c_size_t,
     ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_size_t)],
    ctypes.c_int)

geo_query = try_setup('yp_unified_mesh_query_geometric',
    [ctypes.c_char_p, ctypes.c_size_t,
     ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_size_t)],
    ctypes.c_int)

unified_query = try_setup('yp_unified_query_buf',
    [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t,
     ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_size_t)],
    ctypes.c_int)

print("\nGeometric paths available:")
for name, fn in [('tensor_spectral', tensor_query), ('multi_base_crystal', crystal_query),
                 ('unified_mesh_geometric', geo_query), ('unified_query_buf', unified_query)]:
    print(f"  {name:<30} {'✅' if fn else '❌'}")

# ── Geometric fallback: probe all available paths and vote ──
def geometric_vote(query_text, top_k=5):
    """Call all geometric paths, merge by PID, return top-k by max score."""
    h_bytes = engine.rust.encode_text(query_text) if hasattr(engine, 'rust') else b'\x00'*64
    if not h_bytes or len(h_bytes) < 64:
        h_bytes = b'\x00' * 64
    
    candidates = {}  # pid -> (best_score, sources_list)
    
    for path_name, fn in [('tensor', tensor_query), ('crystal', crystal_query),
                          ('geo', geo_query), ('unified', unified_query)]:
        if not fn:
            continue
        out_ids = (ctypes.c_uint64 * top_k)()
        out_scores = (ctypes.c_float * top_k)()
        out_len = ctypes.c_size_t()
        try:
            ret = fn(ctypes.c_char_p(h_bytes), top_k,
                     ctypes.cast(out_ids, ctypes.POINTER(ctypes.c_uint64)),
                     ctypes.cast(out_scores, ctypes.POINTER(ctypes.c_float)),
                     ctypes.byref(out_len))
            if ret == 0 and out_len.value > 0:
                for i in range(out_len.value):
                    rid = out_ids[i]
                    pid = engine.rust_id_map.get(rid, f"unknown_{rid}")
                    score = float(out_scores[i])
                    if pid not in candidates or score > candidates[pid][0]:
                        candidates[pid] = (score, [path_name])
                    elif score == candidates[pid][0]:
                        candidates[pid][1].append(path_name)
        except Exception as e:
            print(f"    {path_name} failed: {e}")
    
    # Sort by score desc, break ties by number of sources (consensus)
    ranked = sorted(candidates.items(),
                    key=lambda x: (x[1][0], len(x[1][1])),
                    reverse=True)
    results = []
    for pid, (score, sources) in ranked[:top_k]:
        title = engine.cache.get(pid, '')
        results.append((score, (pid, title)))
    return results

# ── Benchmark ──
def delete_words(text, ratio=0.3):
    words = text.split()
    keep = max(1, int(len(words) * (1 - ratio)))
    return ' '.join(random.sample(words, keep))

def run_benchmark(name, use_geometric, n=500):
    sample = random.sample(list(engine.cache.items()), n)
    r1_hits = 0
    r5_hits = 0
    latencies = []
    fallback_count = 0
    
    for true_pid, title in sample:
        corrupted = delete_words(title, 0.3)
        
        t0 = time.perf_counter()
        
        # Step 1: fast path (title hash map)
        res = engine.search(corrupted, top_k=5)
        fast_path_worked = bool(res)
        
        # Step 2: if fast path miss AND geometric enabled, try vote
        if use_geometric and not fast_path_worked:
            geo_res = geometric_vote(corrupted, top_k=5)
            if geo_res:
                fallback_count += 1
                # Merge: prefer fast path if it had results, else geometric
                if not res:
                    res = geo_res
        
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1_000_000)
        
        result_pids = []
        for r in res or []:
            payload = r[1] if len(r) > 1 else None
            if isinstance(payload, tuple):
                result_pids.append(payload[0])
            elif payload is not None:
                result_pids.append(payload)
        
        if true_pid in result_pids[:1]:
            r1_hits += 1
        if true_pid in result_pids[:5]:
            r5_hits += 1
    
    latencies.sort()
    return {
        'name': name,
        'n': n,
        'r1': r1_hits / n * 100,
        'r5': r5_hits / n * 100,
        'p50': latencies[n // 2],
        'p99': latencies[int(n * 0.99)],
        'fallbacks': fallback_count,
    }

print("\n" + "=" * 65)
print("BENCHMARK: Word deletion 30%")
print("=" * 65)

baseline = run_benchmark("Baseline (hash only)", use_geometric=False)
combined = run_benchmark("Combined (hash + geometric vote)", use_geometric=True)

for r in [baseline, combined]:
    print(f"\n  {r['name']}:")
    print(f"    R@1:  {r['r1']:.1f}%")
    print(f"    R@5:  {r['r5']:.1f}%")
    print(f"    P50:  {r['p50']:.1f} µs")
    print(f"    P99:  {r['p99']:.1f} µs")
    if r['fallbacks']:
        print(f"    Geometric fallbacks used: {r['fallbacks']}/{r['n']}")

improvement = combined['r1'] - baseline['r1']
print(f"\n  R@1 improvement: {improvement:+.1f} percentage points")
print(f"  {'✅ GEOMETRIC HELPS' if improvement > 0 else '❌ NO IMPROVEMENT'}")

# Save
import json
out = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'baseline': baseline,
    'combined': combined,
    'improvement_r1': round(improvement, 2),
}
os.makedirs('logs', exist_ok=True)
with open('logs/combined_geometric_orchestrator.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: logs/combined_geometric_orchestrator.json")
