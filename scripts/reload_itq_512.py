#!/usr/bin/env python3
"""
Reload a freshly trained 512-bit ITQ model into the live engine.
Updates data/phoenix_arxiv_1m.db yp_hash512 column and rebuilds title maps.
"""
import json
import numpy as np
import os
import sqlite3
import sys

DB_PATH = "data/phoenix_arxiv_1m.db"
HASHES_PATH = "data/paper_hashes_512.npy"
MESH_SNAPSHOT = "mesh_itq_512.bin"


def hash_to_hex(bits):
    """bits: (512,) uint8 {0,1} -> 128-char hex string."""
    byte_arr = np.packbits(bits).tobytes()
    return byte_arr.hex()


def main():
    if not os.path.exists(DB_PATH):
        print(f"[reload] DB not found: {DB_PATH}")
        sys.exit(1)
    if not os.path.exists(HASHES_PATH):
        print(f"[reload] Hashes not found: {HASHES_PATH}")
        sys.exit(1)

    hashes = np.load(HASHES_PATH)
    if hashes.shape[1] != 512:
        print(f"[reload] Expected 512-bit hashes, got {hashes.shape[1]}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT rowid, id, title FROM papers WHERE title IS NOT NULL ORDER BY rowid")
    rows = cur.fetchall()

    if len(rows) != len(hashes):
        print(f"[reload] Mismatch: {len(rows)} papers vs {len(hashes)} hashes")
        sys.exit(1)

    title_hash_map = {}
    title_prefix_map = {}
    updated = 0

    for (rowid, pid, title), h in zip(rows, hashes):
        h_hex = hash_to_hex(h)
        cur.execute("UPDATE papers SET yp_hash512 = ? WHERE rowid = ?", (h_hex, rowid))
        updated += 1

        key = title.lower().strip()
        if key:
            title_hash_map[key] = h_hex
            for length in [80, 60, 50, 40, 30, 20]:
                if len(key) >= length:
                    prefix = key[:length]
                    title_prefix_map[prefix] = h_hex

    conn.commit()
    conn.close()

    # Save title maps
    os.makedirs("data", exist_ok=True)
    with open("data/title_hash_map.json", "w") as f:
        json.dump(title_hash_map, f)
    with open("data/title_prefix_map.json", "w") as f:
        json.dump(title_prefix_map, f)

    # Remove old mesh snapshot so the engine rebuilds with fresh hashes
    if os.path.exists(MESH_SNAPSHOT):
        os.remove(MESH_SNAPSHOT)
        print(f"[reload] Removed old mesh snapshot: {MESH_SNAPSHOT}")

    print(f"[reload] Updated {updated} papers in {DB_PATH}")
    print(f"[reload] Title hash map: {len(title_hash_map)} entries")
    print(f"[reload] Title prefix map: {len(title_prefix_map)} entries")
    print("[reload] Restart the engine to load the new hashes")


if __name__ == "__main__":
    main()
