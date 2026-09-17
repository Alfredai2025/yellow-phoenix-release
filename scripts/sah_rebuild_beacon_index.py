#!/usr/bin/env python3
"""SAH Beacon Index Rebuild — intentionally rebuild from cached local shards.

This script DESTROYS the in-memory beacon index and re-harvests from cached
models. Use it when you want to upgrade metadata (e.g. add eigenvalues) or
recover from a corrupted index. It does NOT download anything new.
"""

import sys

sys.path.insert(0, "/Users/mac/yellow_phoenix")

from yp_bridge import RustBridge
from scripts.sah_harvester import SAHHarvester
from scripts.sah_registry import SAHRegistry
from scripts.sah_source_finder import SAHSourceFinder


def main():
    bridge = RustBridge()
    finder = SAHSourceFinder()
    registry = SAHRegistry()

    shards = []
    for model_dir in finder.cache_dir.iterdir():
        if model_dir.is_dir():
            shards.extend(finder.find_local_shards(model_dir))

    if not shards:
        print("No cached model shards found. Run sah_live_test.py first.")
        return 1

    print(f"Rebuilding beacon index from {len(shards)} cached shard(s)...")
    rc = bridge.beacon_index_new(50000)
    assert rc == 0, "Failed to create fresh beacon index"

    harvester = SAHHarvester(bridge, max_dim=512)
    total = 0
    for shard in shards:
        model_id = shard.parent.parent.name if "qwen" in shard.parent.parent.name.lower() else "unknown"
        n = harvester.harvest_shard(str(shard), beacons_per_tensor=3)
        total += n
        registry.register_shard(shard, model_id)
        registry.mark_harvested(shard, n, {})
        print(f"  {shard.name}: {n} beacons")

    save_path = "/Users/mac/yellow_phoenix/data/sah_beacon_index.bin"
    bridge.beacon_index_save(save_path)
    print(f"Saved beacon index: {save_path}")

    meta_path = "/Users/mac/yellow_phoenix/data/sah_beacon_meta.json"
    harvester.save_meta(meta_path)

    print(f"TOTAL BEACONS: {bridge.beacon_index_count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
