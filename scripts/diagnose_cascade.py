#!/usr/bin/env python3
"""
Diagnose cascade: trace where medium queries spend their 66 ms.
Monkey-patches key methods to time each layer without file changes.
"""
import sys, os, time, random, re, inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

random.seed(42)
engine = YPEngine()

# ── 1. Discover internal methods ──
print("=" * 65)
print("CASCADE DIAGNOSTIC")
print("=" * 65)

search_src = inspect.getsource(engine.search)
print("\n[1/4] search() source structure:")
for i, line in enumerate(search_src.split('\n')[:60], 1):
    print(f"  {i:3}: {line}")

# Find internal fallback methods
internal_methods = {}
for name in ['_geometric_fallback', '_unison_search', '_search_normal', 
             '_trinity_search', 'search_sharded_hash']:
    if hasattr(engine, name):
        internal_methods[name] = getattr(engine, name)
        print(f"  Found: {name}")

# ── 2. Monkey-patch timing wrappers ──
timings = {}

def timed_wrapper(fn, label):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t1 = time.perf_counter()
        timings[label] = (t1 - t0) * 1_000_000
        return result
    return wrapper

# Wrap each internal method
for name, fn in internal_methods.items():
    setattr(engine, name, timed_wrapper(fn, name))

# Also wrap encode_text
original_encode = engine.rust.encode_text
engine.rust.encode_text = timed_wrapper(original_encode, 'encode_text')

# ── 3. Run medium queries and trace ──
print("\n[2/4] Running 10 medium queries (4-7 words)...")

# Build medium queries: 70% word retention from 6-10 word titles
medium_queries = []
for pid, title in list(engine.cache.items()):
    words = title.split()
    if 6 <= len(words) <= 10:
        kept = max(3, int(len(words) * 0.7))
        corrupted = ' '.join(random.sample(words, kept))
        medium_queries.append((pid, title, corrupted))
    if len(medium_queries) >= 10:
        break

layer_totals = {name: [] for name in list(internal_methods.keys()) + ['encode_text', 'OTHER']}

for true_pid, orig_title, query in medium_queries:
    timings.clear()
    
    t_total_0 = time.perf_counter()
    results = engine.search(query, top_k=5)
    t_total_1 = time.perf_counter()
    total_us = (t_total_1 - t_total_0) * 1_000_000
    
    # Calculate "OTHER" time (search overhead not captured by wrappers)
    captured = sum(timings.values())
    other = max(0, total_us - captured)
    
    for name in internal_methods:
        layer_totals[name].append(timings.get(name, 0))
    layer_totals['encode_text'].append(timings.get('encode_text', 0))
    layer_totals['OTHER'].append(other)
    
    hit = any(r[1][0] == true_pid for r in results if len(r) > 1 and isinstance(r[1], tuple))
    print(f"\n  Query: '{query[:50]}...'")
    print(f"    Total: {total_us:>7.0f} µs | Hit: {'✅' if hit else '❌'}")
    for name in ['encode_text', 'search_sharded_hash', '_trinity_search', 
                 '_search_normal', '_geometric_fallback', '_unison_search', 'OTHER']:
        if name in timings or name == 'OTHER':
            val = timings.get(name, 0) if name != 'OTHER' else other
            bar = '█' * int(val / total_us * 20) if total_us > 0 else ''
            print(f"    {name:25} {val:>7.0f} µs {bar}")

# ── 4. Summary ──
print("\n[3/4] Summary (average across 10 medium queries):")
print(f"  {'Layer':<25} {'Avg µs':>10} {'% of total':>12}")
print("  " + "-" * 50)
for name, vals in layer_totals.items():
    if vals:
        avg = sum(vals) / len(vals)
        print(f"  {name:<25} {avg:>10.0f}")

# ── 5. Check if fast path is being skipped ──
print("\n[4/4] Fast path check:")
test_title = list(engine.cache.values())[0]
t0 = time.perf_counter()
r1 = engine.search(test_title, top_k=1)
t1 = time.perf_counter()
print(f"  Exact title: {(t1-t0)*1e6:.1f} µs (should be ~4 µs)")

# Check prefix map
if hasattr(engine, '_title_prefix_map'):
    print(f"  Prefix map size: {len(engine._title_prefix_map)}")
    # Try a known prefix
    sample_prefix = list(engine._title_prefix_map.keys())[0]
    t0 = time.perf_counter()
    hit = sample_prefix in engine._title_prefix_map
    t1 = time.perf_counter()
    print(f"  Prefix lookup: {(t1-t0)*1e6:.1f} µs")

# Check if drift recording is slow
if hasattr(engine, '_record_drift'):
    t0 = time.perf_counter()
    engine._record_drift("test query")
    t1 = time.perf_counter()
    print(f"  Drift record: {(t1-t0)*1e6:.1f} µs")

print("\n" + "=" * 65)
print("DIAGNOSTIC COMPLETE")
print("=" * 65)
