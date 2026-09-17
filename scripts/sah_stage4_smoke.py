#!/usr/bin/env python3
"""SAH Stage 4 smoke test: beacon hit, beacon miss, fallback, empty."""

import sys
import numpy as np

sys.path.insert(0, "/Users/mac/yellow_phoenix")
from yp_bridge import RustBridge
from scripts.sah_harvester import SAHHarvester
from scripts.sah_cascade import SAHCascade


def fake_production_search(query_hash: bytes, k: int):
    """Stub production search returning fake results."""
    return [(9999, 42, 0x00)]  # (node_id, distance, tag=0)


def main():
    bridge = RustBridge()

    # 1. Fresh beacon index
    rc = bridge.beacon_index_new(1000)
    assert rc == 0
    print("[1] Beacon index created")

    # 2. Seed beacon index with known hashes (simulate harvester output)
    dim = 512
    np.random.seed(123)
    rotation = np.eye(dim, dtype=np.float32).tolist()

    # Insert 3 beacons with different tags
    vec_a = np.random.randn(dim).astype(np.float32)
    vec_a = vec_a / np.linalg.norm(vec_a)
    hash_a = bridge.sah_eigenvector_to_hash(vec_a.tolist(), rotation)

    vec_b = np.random.randn(dim).astype(np.float32)
    vec_b = vec_b / np.linalg.norm(vec_b)
    hash_b = bridge.sah_eigenvector_to_hash(vec_b.tolist(), rotation)

    vec_c = np.random.randn(dim).astype(np.float32)
    vec_c = vec_c / np.linalg.norm(vec_c)
    hash_c = bridge.sah_eigenvector_to_hash(vec_c.tolist(), rotation)

    bridge.beacon_index_insert(hash_a, node_id=1001, tag=0x01)
    bridge.beacon_index_insert(hash_b, node_id=1002, tag=0x02)
    bridge.beacon_index_insert(hash_c, node_id=1003, tag=0x03)
    print("[2] 3 beacons inserted (tags 1,2,3)")

    # 3. Cascade with production fallback
    cascade = SAHCascade(bridge, production_search_fn=fake_production_search)

    # 4. Test: exact match → beacon hit
    result = cascade.query(hash_a, k=3)
    assert result["source"] == "beacon", f"Expected beacon, got {result['source']}"
    assert result["latency_hint"] == "fast"
    assert any(r[0] == 1001 for r in result["results"])
    print(f"[3] Exact match → beacon hit: {result['results']}")

    # 5. Test: near match with small perturbation → beacon hit
    perturbed = bytearray(hash_a)
    perturbed[0] ^= 0x01  # flip 1 bit
    result = cascade.query(bytes(perturbed), k=3)
    # May or may not hit depending on Hamming distance; we just verify no crash
    print(f"[4] 1-bit perturb → source={result['source']}, results={len(result['results'])}")

    # 6. Test: random hash → beacon miss, fallback triggered
    random_hash = bytes(np.random.randint(0, 256, size=64, dtype=np.uint8))
    result = cascade.query(random_hash, k=3)
    assert result["source"] == "fallback", f"Expected fallback, got {result['source']}"
    assert result["latency_hint"] == "slow"
    assert result["results"][0][0] == 9999
    print(f"[5] Random hash → fallback: {result['results']}")

    # 7. Test: no fallback, empty
    cascade_no_fb = SAHCascade(bridge, production_search_fn=None)
    result = cascade_no_fb.query(random_hash, k=3)
    assert result["source"] == "empty"
    assert result["results"] == []
    print("[6] No fallback → empty")

    # 8. Batch query
    batch = [hash_a, hash_b, bytes(np.random.randint(0, 256, size=64, dtype=np.uint8))]
    results = cascade.query_batch(batch, k=2)
    sources = [r["source"] for r in results]
    assert sources.count("beacon") >= 2, f"Expected >=2 beacon hits, got {sources}"
    print(f"[7] Batch query sources: {sources}")

    print("\nSAH Stage 4 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
