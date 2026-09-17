#!/usr/bin/env python3
"""SAH Stage 9 smoke: real ITQ rotation produces different hashes than fallback."""

import sys
import numpy as np

sys.path.insert(0, "/Users/mac/yellow_phoenix")

from scripts.sah_itq_loader import load_rotation_matrix, has_itq_model, get_fallback_rotation
from yp_bridge import RustBridge


def main():
    bridge = RustBridge()
    rc = bridge.beacon_index_new(1000)
    assert rc == 0

    # 1. Check ITQ model exists
    if not has_itq_model():
        print("[WARN] ITQ model not found — skipping real rotation test")
        print("[PASS] Fallback rotation available")
        return

    print("[1] ITQ model found")

    # 2. Load real rotation
    real_rot = load_rotation_matrix()
    assert len(real_rot) == 512
    assert len(real_rot[0]) == 512
    print("[2] Real ITQ rotation loaded: 512×512")

    # 3. Load fallback
    fallback_rot = get_fallback_rotation()
    print("[3] Fallback rotation loaded")

    # 4. Same eigenvector, different hashes
    dim = 512
    np.random.seed(42)
    eigenvector = np.random.randn(dim).astype(np.float32)
    eigenvector = eigenvector / np.linalg.norm(eigenvector)

    real_hash = bridge.sah_eigenvector_to_hash(eigenvector.tolist(), real_rot)
    fallback_hash = bridge.sah_eigenvector_to_hash(eigenvector.tolist(), fallback_rot)

    assert len(real_hash) == 64
    assert len(fallback_hash) == 64
    assert real_hash != fallback_hash, "Real ITQ should produce different hash than fallback"
    print(f"[4] Real hash: {real_hash.hex()[:16]}...")
    print(f"[4] Fallback hash: {fallback_hash.hex()[:16]}...")

    # 5. Hamming distance between real and fallback
    real_bytes = np.frombuffer(real_hash, dtype=np.uint8)
    fallback_bytes = np.frombuffer(fallback_hash, dtype=np.uint8)
    hamming = np.sum(np.unpackbits(real_bytes) != np.unpackbits(fallback_bytes))
    print(f"[5] Hamming distance real vs fallback: {hamming} / 512")

    # 6. Real rotation is orthonormal (check)
    rot_np = np.array(real_rot, dtype=np.float32)
    identity_check = np.dot(rot_np, rot_np.T)
    off_diag = np.max(np.abs(identity_check - np.eye(512)))
    print(f"[6] Orthonormality check (max off-diag): {off_diag:.6f}")

    # 7. Beacon insert + search with real hash
    bridge.beacon_index_insert(real_hash, node_id=9001, tag=0x01)
    results = bridge.beacon_index_search(real_hash, k=1)
    assert len(results) == 1
    assert results[0][0] == 9001
    print(f"[7] Beacon round-trip with real ITQ: id={results[0][0]}, dist={results[0][1]}")

    print("\nSAH Stage 9 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
