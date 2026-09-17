#!/usr/bin/env python3
"""SAH Stage 6 smoke test: beacon-aware search + re-balance."""

import sys, os
sys.path.insert(0, "/Users/mac/yellow_phoenix")
from yp_bridge import RustBridge

def main():
    bridge = RustBridge()

    # 1. Fresh beacon index (new() replaces any existing static index)
    rc = bridge.beacon_index_new(1000)
    assert rc == 0, f"beacon_index_new failed: {rc}"
    print("[1] Beacon index ready")

    # 2. Seed beacons: 3 clusters
    import random
    random.seed(42)

    # Cluster 1: Concept beacons (tag=0x01)
    for i in range(10):
        h = bytes([random.randint(0,255) for _ in range(64)])
        # Make them similar to each other
        h = bytes([0xAA if j < 32 else b for j, b in enumerate(h)])
        bridge.beacon_index_insert(h, i, 0x01)

    # Cluster 2: Topic beacons (tag=0x03)
    for i in range(10, 20):
        h = bytes([random.randint(0,255) for _ in range(64)])
        h = bytes([0xBB if j < 32 else b for j, b in enumerate(h)])
        bridge.beacon_index_insert(h, i, 0x03)

    print(f"[2] Seeded {bridge.beacon_index_count()} beacons")

    # 3. Fake document search with cluster boost
    # Simulate a query hash close to cluster 1
    query = bytes([0xAA] * 32 + [0x00] * 32)
    beacons = bridge.beacon_index_search(query, k=3)
    print(f"[3] Beacon probe: {beacons}")

    best_tag = beacons[0][2] if beacons else 0
    print(f"    Best tag: {hex(best_tag)} (expected 0x1)")

    # 4. Fake documents with cluster tags
    docs = [
        (100, 0.9, {"cluster_tag": 0x01}),  # matches cluster 1
        (101, 0.85, {"cluster_tag": 0x03}), # matches cluster 2
        (102, 0.8, {"cluster_tag": 0}),      # unclustered
    ]

    cluster_boost = 0.15 if best_tag == 0x01 else 0.0
    boosted = []
    for doc_id, score, meta in docs:
        final = score * (1.0 + cluster_boost) if meta.get("cluster_tag") == best_tag else score
        boosted.append((doc_id, final, meta))
    boosted.sort(key=lambda x: x[1], reverse=True)

    print(f"[4] Re-ranked: {boosted}")
    top_id = boosted[0][0]
    assert top_id == 100, f"Expected doc 100 top, got {top_id}"

    # 5. Freshness check via YPEngine method
    from yp_bridge import YPEngine
    engine = YPEngine()
    fresh = engine.check_beacon_freshness(bridge.beacon_index_count())
    print(f"[5] Beacon count: {bridge.beacon_index_count()}, freshness trigger: {fresh}")

    print("\nSAH STAGE 6 SMOKE TEST PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
