#!/usr/bin/env python3
"""Smoke test: verify static mesh + sharded path works end-to-end."""
import sys, os
sys.path.insert(0, '/Users/mac/yellow_phoenix')
from yp_bridge import YPEngine

engine = YPEngine()
print(f"Cache: {len(engine.cache)} | Rust id map: {len(getattr(engine, 'rust_id_map', {}))}")

mesh_count = 0
if engine.rust.lib and "yp_mesh_count" in engine.rust.available:
    mesh_count = engine.rust.available["yp_mesh_count"]()
print(f"HYBRID_MESH entries: {mesh_count}")

# Build sharded index
cold_path = os.path.join('/Users/mac/yellow_phoenix', '.tmp', 'sharded_cold.bin')
os.makedirs(os.path.dirname(cold_path), exist_ok=True)
rc = engine.rust.enable_sharding(cold_path=cold_path, num_shards=10)
print(f"enable_sharding rc: {rc}")

if engine.cache:
    first_pid = list(engine.cache.keys())[0]
    first_title = engine.cache[first_pid]
    print(f"\nQuery: '{first_title[:60]}...'")
    results = engine.search_sharded(first_title, top_k=3)
    print(f"Results: {results}")
    if results and results[0][1][0] == first_pid:
        print("✅ PASS: Sharded path returns correct paper")
    else:
        print("❌ FAIL: Sharded path did not return correct paper")
else:
    print("No papers in cache")
