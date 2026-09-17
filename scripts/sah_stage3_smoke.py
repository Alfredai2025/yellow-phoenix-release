#!/usr/bin/env python3
"""SAH Stage 3 smoke test: synthetic shard -> harvester -> beacon index."""

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, "/Users/mac/yellow_phoenix")
from yp_bridge import RustBridge
from scripts.sah_harvester import SAHHarvester


def main():
    bridge = RustBridge()

    # 1. Create beacon index
    rc = bridge.beacon_index_new(10000)
    assert rc == 0, f"beacon_index_new failed: {rc}"
    print("[1] Beacon index created")

    # 2. Fake ITQ rotation (same method as Stage 2)
    dim = 512
    np.random.seed(42)
    noise = np.random.randn(dim, dim).astype(np.float32) * 0.01
    rotation = np.eye(dim, dtype=np.float32) + noise
    rotation, _ = np.linalg.qr(rotation)
    rotation = rotation.astype(np.float32).tolist()

    # 3. Build synthetic safetensors shard (small matrices for speed)
    from safetensors.numpy import save_file
    tmpdir = tempfile.mkdtemp()
    shard_path = os.path.join(tmpdir, "test_model.safetensors")

    tensors = {
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(512, 512).astype(np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.random.randn(512, 512).astype(np.float32),
        "model.embed_tokens.weight": np.random.randn(512, 512).astype(np.float32),
    }
    save_file(tensors, shard_path)
    print(f"[2] Synthetic shard written: {shard_path}")

    # 4. Harvest
    harvester = SAHHarvester(bridge, rotation, max_dim=512)
    n = harvester.harvest_shard(shard_path, beacons_per_tensor=2)
    assert n > 0, "No beacons inserted"
    print(f"[3] Harvested {n} beacons")

    # 5. Verify count
    count = bridge.beacon_index_count()
    assert count == n, f"Count mismatch: index={count}, harvested={n}"
    print(f"[4] Beacon index count: {count}")

    # 6. Verify tag distribution
    # (We can't read tags back individually without search, but we verify no crash)
    print(f"[5] Stats: {harvester.stats}")

    # 7. Search smoke: convert one tensor's eigenvector to hash and search
    q_vec = harvester.compute_eigenvectors(tensors["model.layers.0.self_attn.q_proj.weight"], k=1)[0]
    q_hash = bridge.sah_eigenvector_to_hash(q_vec.tolist(), rotation)
    results = bridge.beacon_index_search(q_hash, k=3)
    assert len(results) > 0, "Search returned empty"
    print(f"[6] Search returned {len(results)} results")

    # Cleanup
    os.remove(shard_path)
    os.rmdir(tmpdir)

    print("\nSAH Stage 3 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
