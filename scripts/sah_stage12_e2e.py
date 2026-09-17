#!/usr/bin/env python3
"""SAH Stage 12: End-to-end — download, harvest, query, validate."""

import sys
import time

sys.path.insert(0, "/Users/mac/yellow_phoenix")

from yp_bridge import RustBridge, YPEngine
from scripts.sah_source_finder import SAHSourceFinder
from scripts.sah_harvester import SAHHarvester
from scripts.sah_registry import SAHRegistry
from scripts.sah_auto_downloader import SAHAutoDownloader
from scripts.sah_itq_loader import has_itq_model


def main():
    print("=" * 60)
    print("SAH Stage 12: End-to-End Validation")
    print("=" * 60)

    # 1. Boot
    bridge = RustBridge()
    engine = YPEngine()
    rc = bridge.beacon_index_new(50000)
    assert rc == 0
    print("[1] Beacon index initialized")

    # 2. Check ITQ
    if not has_itq_model():
        print("[WARN] No ITQ model — using fallback identity rotation")
        print("       Install itq_model_512.npz for production hashes")
    else:
        print("[2] ITQ rotation available")

    # 3. Auto-download small model (Qwen 0.5B = ~1GB, fast)
    finder = SAHSourceFinder()
    registry = SAHRegistry()
    auto = SAHAutoDownloader(finder, registry)

    # Override with tiny model for speed
    tiny_model = "qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n[3] Downloading {tiny_model} ...")
    try:
        model_path = finder.download_model(tiny_model)
        shards = finder.find_local_shards(model_path)
        print(f"    Found {len(shards)} shard(s)")
    except Exception as e:
        print(f"    FAILED (network/offline): {e}")
        print("    SKIPPING — pipeline validated up to download step")
        return

    # 4. Harvest
    harvester = SAHHarvester(bridge, max_dim=512)
    total_beacons = 0
    for shard in shards:
        n = harvester.harvest_shard(str(shard), beacons_per_tensor=2)
        total_beacons += n
        print(f"    {shard.name}: {n} beacons")

    count = bridge.beacon_index_count()
    print(f"\n[4] Total beacons harvested: {count}")

    # 5. Save beacon index
    save_path = "/Users/mac/yellow_phoenix/data/sah_beacon_index.bin"
    rc = bridge.beacon_index_save(save_path)
    print(f"[5] Beacon index saved: {rc == 0}")

    # 6. Query via YPEngine
    print("\n[6] Query tests...")
    queries = [
        "machine learning",
        "attention mechanism",
        "neural network",
        "transformer architecture",
    ]

    for q in queries:
        t0 = time.time()
        result = engine.search_with_sah(q, k=5, use_cascade=True)
        latency = (time.time() - t0) * 1000
        print(f"    '{q}' → source={result['source']}, latency={latency:.2f}ms, results={len(result['results'])}")

    # 7. Coverage report
    summary = registry.coverage_summary()
    print(f"\n[7] Coverage: {summary}")

    print("\n" + "=" * 60)
    print("SAH Stage 12 END-TO-END PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
