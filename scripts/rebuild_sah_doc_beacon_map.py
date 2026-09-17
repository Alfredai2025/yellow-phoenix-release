#!/usr/bin/env python3
"""
Rebuild the document-beacon -> paper metadata sidecar.

The binary beacon index (`data/sah_beacon_index.bin`) stores 5,000 document
beacons with sequential node IDs starting at 2,000,000.  Their hashes were
produced by the current ITQ encoder, while the DB `yp_hash512` column may
still contain legacy hashes from an older pipeline.

This script reconstructs the mapping by:
  1. Reading every candidate paper title from the DB.
  2. Encoding each title with the current ITQ encoder.
  3. Matching those hashes against the hashes stored in the beacon index.
  4. Writing `data/sah_beacon_doc_map.json`.

The beacon fast path in `yp_bridge.py` then uses this map to return real
(pid, title) results without trusting the legacy `yp_hash512` column.
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yp_bridge import RustBridge, ITQHasher, YP_ROOT


def main(db_path: str | None = None, beacon_path: str | None = None, out_path: str | None = None):
    db_path = db_path or os.path.join(YP_ROOT, "data", "phoenix_arxiv_1m.db")
    beacon_path = beacon_path or os.path.join(YP_ROOT, "data", "sah_beacon_index.bin")
    out_path = out_path or os.path.join(YP_ROOT, "data", "sah_beacon_doc_map.json")

    print("Loading beacon index...")
    rb = RustBridge()
    rc = rb.beacon_index_load(beacon_path)
    count = rb.beacon_index_count()
    print(f"  load rc={rc}, count={count}")
    if rc != 0 or count == 0:
        raise RuntimeError(f"Beacon index not available: rc={rc}, count={count}")

    print("Loading ITQ hasher...")
    hasher = ITQHasher()

    print(f"Fetching candidate titles from {db_path}...")
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT id, title FROM papers WHERE title IS NOT NULL AND length(title) <= 80"
    )
    candidates = list(cur)
    conn.close()
    print(f"  candidates: {len(candidates)}")

    # Build a hash -> (pid, title) lookup.  Duplicates are unlikely but keep first.
    hash_map: dict[bytes, tuple[str, str]] = {}
    batch_size = 256
    t0 = time.time()
    for i in range(0, len(candidates), batch_size):
        chunk = candidates[i : i + batch_size]
        texts = [title for _, title in chunk]
        bits = hasher.encode_batch(texts)
        packed = np.packbits(bits, axis=1)
        for (pid, title), h in zip(chunk, packed):
            hb = h.tobytes()
            if hb not in hash_map:
                hash_map[hb] = (pid, title)
    print(f"  encoded in {time.time() - t0:.1f}s, unique hashes: {len(hash_map)}")

    # Match document beacon node IDs against the title hashes.
    # Node IDs begin at 2,000,000 and are contiguous for 5,000 entries.
    doc_map: dict[str, dict[str, str]] = {}
    missing: list[int] = []
    for nid in range(2_000_000, 2_005_000):
        hb = rb.beacon_index_get_hash(nid)
        if hb is None:
            missing.append(nid)
            continue
        match = hash_map.get(hb)
        if match is None:
            missing.append(nid)
            continue
        pid, title = match
        doc_map[str(nid)] = {"pid": pid, "title": title}

    print(f"  matched: {len(doc_map)}, missing: {len(missing)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc_map, f)
    print(f"Saved document-beacon map: {out_path} ({len(doc_map)} entries)")


if __name__ == "__main__":
    main(*sys.argv[1:])
