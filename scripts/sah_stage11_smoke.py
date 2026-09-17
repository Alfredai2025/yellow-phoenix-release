#!/usr/bin/env python3
"""SAH Stage 11 smoke: beacon index save/load round-trip."""

import sys
import os
import tempfile

sys.path.insert(0, "/Users/mac/yellow_phoenix")
from yp_bridge import RustBridge


def main():
    bridge = RustBridge()

    # 1. Create and populate
    rc = bridge.beacon_index_new(1000)
    assert rc == 0
    fake_hash = bytes(range(64))  # deterministic
    bridge.beacon_index_insert(fake_hash, node_id=1234, tag=0x02)
    bridge.beacon_index_insert(fake_hash, node_id=5678, tag=0x03)
    assert bridge.beacon_index_count() == 2
    print("[1] Inserted 2 beacons")

    # 2. Save
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "beacon_index.bin")
    rc = bridge.beacon_index_save(path)
    assert rc == 0, f"Save failed: {rc}"
    assert os.path.exists(path)
    print(f"[2] Saved to {path} ({os.path.getsize(path)} bytes)")

    # 3. Load into fresh index
    rc = bridge.beacon_index_load(path)
    assert rc == 0, f"Load failed: {rc}"
    count = bridge.beacon_index_count()
    assert count == 2, f"Expected 2, got {count}"
    print(f"[3] Loaded, count={count}")

    # 4. Search still works
    results = bridge.beacon_index_search(fake_hash, k=3)
    assert len(results) == 2
    ids = [r[0] for r in results]
    assert 1234 in ids and 5678 in ids
    print(f"[4] Search after load: {results}")

    # 5. Tags preserved
    tags = [r[2] for r in results]
    assert 0x02 in tags and 0x03 in tags
    print("[5] Tags preserved through save/load")

    os.remove(path)
    os.rmdir(tmpdir)

    print("\nSAH Stage 11 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
