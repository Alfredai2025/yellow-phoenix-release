#!/usr/bin/env python3
"""
Profile engine.search() to find which internal path produces 38 µs.
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

print("=" * 65)
print("PROFILING FAST PATH")
print("=" * 65)

engine = YPEngine()

# Test queries that trigger different paths
tests = [
    ("Exact title", None),  # Will pick first title
    ("Synonym swap", None),
    ("Short 2-word", None),
    ("Long phrase", None),
]

# Get sample titles
titles = list(engine.cache.values())
tests[0] = ("Exact title", titles[0])
tests[1] = ("Synonym swap", titles[0].replace('learning', 'training').replace('model', 'framework'))
tests[2] = ("Short 2-word", ' '.join(titles[0].split()[:2]))
tests[3] = ("Long phrase", titles[0])

print("\n[1/1] Profiling paths...")
for label, query in tests:
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        res = engine.search(query, top_k=5)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1_000_000)
    latencies.sort()
    p50 = latencies[50]
    p10 = latencies[10]
    print(f"  {label:<20} | P50: {p50:>8.1f} µs | P10: {p10:>8.1f} µs | query: {query[:50]}...")

# Now test the raw sharded hash path directly
print("\n  Raw sharded_hash path:")
import ctypes
lib = engine.rust.lib
out_ids = (ctypes.c_uint64 * 5)()
out_scores = (ctypes.c_float * 5)()
out_len = ctypes.c_size_t()

# Get a hash
import sqlite3
conn = sqlite3.connect('data/phoenix_arxiv_1m.db')
cursor = conn.cursor()
cursor.execute("SELECT yp_hash512 FROM papers WHERE yp_hash512 IS NOT NULL LIMIT 1")
h = cursor.fetchone()[0]
conn.close()

h_bytes = bytes.fromhex(h)
latencies = []
for _ in range(100):
    t0 = time.perf_counter()
    lib.yp_query_sharded(
        ctypes.c_char_p(h_bytes[:16]),
        5,
        ctypes.cast(out_ids, ctypes.POINTER(ctypes.c_uint64)),
        ctypes.cast(out_scores, ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(out_len)
    )
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1_000_000)
latencies.sort()
print(f"  {'yp_query_sharded':<20} | P50: {latencies[50]:>8.1f} µs | P10: {latencies[10]:>8.1f} µs")

# Test keyword-only path
print("\n  Testing keyword rescue path...")
# Monkey-patch to disable MiniLM temporarily
original_encode = getattr(engine, '_encode_primary', None)
if original_encode:
    engine._encode_primary = lambda text: b'\x00' * 64
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        res = engine.search(titles[0], top_k=5)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1_000_000)
    engine._encode_primary = original_encode
    latencies.sort()
    print(f"  {'Keyword-only':<20} | P50: {latencies[50]:>8.1f} µs | P10: {latencies[10]:>8.1f} µs")
