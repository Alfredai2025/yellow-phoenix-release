#!/usr/bin/env python3
"""SAH Live Test — see what Yellow has, get shards, harvest, query."""

import os
import sys
import time

sys.path.insert(0, "/Users/mac/yellow_phoenix")

from yp_bridge import RustBridge, YPEngine
from scripts.sah_registry import SAHRegistry
from scripts.sah_source_finder import SAHSourceFinder
from scripts.sah_harvester import SAHHarvester
from scripts.sah_itq_loader import has_itq_model


def main():
    print("=" * 60)
    print("YELLOW SAH LIVE TEST")
    print("=" * 60)

    bridge = RustBridge()
    engine = YPEngine()
    registry = SAHRegistry()

    # 1. Current state
    summary = registry.coverage_summary()
    print(f"\n[1] CURRENT STATE")
    print(f"    Total beacons:    {summary['total_beacons']}")
    print(f"    Total shards:     {summary['total_shards']}")
    print(f"    Harvested shards: {summary['harvested_shards']}")
    print(f"    By tag:           {summary['by_tag']}")

    # 2. Beacon index count (in-memory)
    count = bridge.beacon_index_count()
    print(f"\n[2] IN-MEMORY BEACON INDEX: {count} entries")

    # 3. ITQ status
    print(f"\n[3] ITQ ROTATION: {'REAL (production)' if has_itq_model() else 'FALLBACK (identity)'}")

    # 4. Check cached models
    finder = SAHSourceFinder()
    cached = list(finder.cache_dir.iterdir()) if finder.cache_dir.exists() else []
    print(f"\n[4] CACHED MODELS: {len(cached)}")
    for c in cached:
        shards = list(finder.find_local_shards(c))
        print(f"    {c.name}: {len(shards)} shard(s)")

    # 5. Download + harvest if empty. If meta sidecar is missing but beacons exist,
    # create a placeholder — do NOT destroy the beacon index for a missing JSON file.
    meta_path = "/Users/mac/yellow_phoenix/data/sah_beacon_meta.json"
    need_harvest = summary['total_beacons'] == 0
    meta_missing = not os.path.exists(meta_path)

    if need_harvest:
        print(f"\n[5] NO BEACONS — downloading Qwen 0.5B (~1GB, may take 1-2 min)...")
        tiny = "qwen/Qwen2.5-0.5B-Instruct"
        try:
            t0 = time.time()
            model_path = finder.download_model(tiny)
            elapsed = time.time() - t0
            shards = finder.find_local_shards(model_path)
            print(f"    Downloaded in {elapsed:.1f}s, {len(shards)} shard(s)")

            # Register
            for shard in shards:
                registry.register_shard(shard, tiny)

            # Harvest
            print(f"    Harvesting beacons...")
            rc = bridge.beacon_index_new(50000)
            assert rc == 0
            harvester = SAHHarvester(bridge, max_dim=512)
            total = 0
            for shard in shards:
                n = harvester.harvest_shard(str(shard), beacons_per_tensor=3)
                total += n
                registry.mark_harvested(shard, n, {})
                print(f"      {shard.name}: {n} beacons")

            # Save index + metadata sidecar
            save_path = "/Users/mac/yellow_phoenix/data/sah_beacon_index.bin"
            bridge.beacon_index_save(save_path)
            print(f"    Saved beacon index: {save_path}")
            harvester.save_meta(meta_path)

            count = bridge.beacon_index_count()
            print(f"    TOTAL BEACONS: {count}")

        except Exception as e:
            print(f"    FAILED: {e}")
            print("    (Network issue? Offline? Try again later.)")
            return
    elif meta_missing:
        print(f"\n[5] METADATA SIDECAR MISSING")
        print(f"    Beacons exist: {summary['total_beacons']}. Index NOT touched.")
        print(f"    To regenerate full metadata with eigenvalues, run:")
        print(f"        python scripts/sah_rebuild_beacon_index.py")
        print(f"    search_hybrid will use synthetic tags until then.")
    else:
        print(f"\n[5] BEACONS EXIST — skipping download")
        print(f"    Metadata sidecar present: {meta_path}")

    # 6. Query test
    print(f"\n[6] QUERY TESTS")
    queries = [
        "machine learning",
        "attention mechanism",
        "neural network",
        "transformer architecture",
        "hello world",
    ]
    for q in queries:
        t0 = time.time()
        result = engine.search_with_sah(q, k=5, use_cascade=True)
        latency = (time.time() - t0) * 1000
        src = result['source']
        n = len(result['results'])
        print(f"    '{q}' → source={src}, latency={latency:.2f}ms, results={n}")

    # 7. Final state
    summary = registry.coverage_summary()
    print(f"\n[7] FINAL STATE")
    print(f"    Beacons: {summary['total_beacons']}")
    print(f"    By tag:  {summary['by_tag']}")

    print(f"\n{'=' * 60}")
    print("SAH LIVE TEST COMPLETE")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
