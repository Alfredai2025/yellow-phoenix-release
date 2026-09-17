#!/usr/bin/env python3
"""SAH Stage 2 smoke test: Eigenvector -> Hash -> Beacon Index."""

import numpy as np
import sys
sys.path.insert(0, "/Users/mac/yellow_phoenix")
from yp_bridge import RustBridge


def main():
    bridge = RustBridge()

    # 1. Create beacon index
    rc = bridge.beacon_index_new(1000)
    assert rc == 0, f"beacon_index_new failed: {rc}"
    print("[1] Beacon index created")

    # 2. Build a fake ITQ rotation matrix (QR-orthogonalized noise)
    dim = 512
    np.random.seed(42)
    noise = np.random.randn(dim, dim).astype(np.float32) * 0.01
    rotation = np.eye(dim, dtype=np.float32) + noise
    rotation, _ = np.linalg.qr(rotation)
    rotation = rotation.astype(np.float32)

    # 3. Generate a random normalized eigenvector
    eigenvector = np.random.randn(dim).astype(np.float32)
    eigenvector = eigenvector / np.linalg.norm(eigenvector)

    # 4. Convert to hash
    hash_bytes = bridge.sah_eigenvector_to_hash(eigenvector.tolist(), rotation.tolist())
    assert len(hash_bytes) == 64, f"Hash length wrong: {len(hash_bytes)}"
    print(f"[2] Eigenvector -> Hash: {hash_bytes.hex()[:32]}...")

    # 5. Insert into beacon index with tag=0x01 (concept beacon)
    rc = bridge.beacon_index_insert(hash_bytes, node_id=1001, tag=0x01)
    assert rc == 0, f"beacon_index_insert failed: {rc}"
    print("[3] Beacon inserted with tag=0x01")

    # 6. Search and verify retrieval
    results = bridge.beacon_index_search(hash_bytes, k=3)
    assert len(results) > 0, "Search returned empty"
    top_id, top_dist, top_tag = results[0]
    assert top_id == 1001, f"Wrong node_id: {top_id}"
    assert top_tag == 0x01, f"Wrong tag: {top_tag}"
    print(f"[4] Search retrieved: id={top_id}, dist={top_dist}, tag={top_tag}")

    # 7. Batch convert test
    batch_vecs = [
        (np.random.randn(dim).astype(np.float32) / np.linalg.norm(np.random.randn(dim))).tolist()
        for _ in range(5)
    ]
    batch_hashes = bridge.sah_batch_eigenvector_to_hash(batch_vecs, rotation.tolist())
    assert len(batch_hashes) == 5
    assert all(len(h) == 64 for h in batch_hashes)
    print("[5] Batch convert: 5/5 hashes valid")

    # 8. Insert batch into beacon index
    for i, h in enumerate(batch_hashes):
        bridge.beacon_index_insert(h, node_id=2000 + i, tag=0x02)
    count = bridge.beacon_index_count()
    assert count == 6, f"Expected 6 beacons, got {count}"
    print(f"[6] Beacon index count: {count}")

    print("\nSAH Stage 2 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
