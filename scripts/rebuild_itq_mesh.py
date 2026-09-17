#!/usr/bin/env python3
"""Rebuild mesh with real ITQ semantic hashes."""
import numpy as np
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from yp_bridge import YPEngine


def main():
    print("=" * 60)
    print("REBUILD: Mesh with Real ITQ 512-bit Hashes")
    print("=" * 60)

    # 1. Load DB papers
    print("[1/6] Loading papers from DB...")
    conn = sqlite3.connect("data/phoenix_arxiv_1m.db")
    c = conn.cursor()
    c.execute("SELECT id, title FROM papers WHERE title IS NOT NULL ORDER BY id")
    papers = c.fetchall()
    conn.close()
    print(f"      DB papers: {len(papers)}")

    # 2. Load ITQ hashes
    print("[2/6] Loading ITQ hashes...")
    hashes = np.load("itq_hashes_512.npy")
    print(f"      ITQ hashes: {len(hashes)}")

    # 3. Pair by index (safe: use min count)
    n = min(len(papers), len(hashes))
    print(f"[3/6] Pairing {n} papers with hashes...")
    if len(papers) != len(hashes):
        print(f"      WARNING: Mismatch! DB={len(papers)}, hashes={len(hashes)}")
        print(f"      Using first {n} entries. Extra hashes/papers ignored.")

    # 4. Build fresh mesh: temporarily hide existing mesh so engine starts fresh
    mesh_path = os.path.expanduser("~/yellow_phoenix/mesh_itq_512.bin")
    backup_path = mesh_path + ".old"
    if os.path.exists(mesh_path):
        if os.path.exists(backup_path):
            os.remove(backup_path)
        shutil.move(mesh_path, backup_path)
        print(f"      Moved existing mesh to {backup_path}")

    print("[4/6] Building fresh mesh...")
    engine = YPEngine()

    skipped = 0
    for i in range(n):
        pid, title = papers[i]
        hash_bytes = hashes[i].tobytes()
        if np.all(hashes[i] == 0):
            skipped += 1
            continue
        engine.insert(pid, title or "", hash_bytes)
        if (i + 1) % 1000 == 0:
            print(f"      ...{i+1}/{n} inserted")

    print(f"[5/6] Inserted {n - skipped} papers (skipped {skipped} zero hashes). Saving mesh...")
    engine.save_mesh()

    print("      Enabling sharded mesh for validation...")
    engine.enable_sharding(num_shards=10, pre_fault=True)

    # 6. Self-query validation
    print("[6/6] Running self-query validation...")
    correct = 0
    test_n = min(100, n)
    for i in range(test_n):
        results = engine.search_by_hash(hashes[i].tobytes(), top_k=5)
        found = any(rpid == papers[i][0] for score, (rpid, rtitle) in results)
        if found:
            correct += 1

    r1 = correct / test_n * 100
    print(f"      Self-search R@1: {r1:.1f}% ({correct}/{test_n})")

    if r1 > 90:
        print("      ✅ MESH IS GOOD. Real ITQ hashes working.")
    elif r1 > 50:
        print("      ⚠️  PARTIAL. Check hash-to-ID mapping.")
    else:
        print("      ❌ BAD. Hash-to-ID mapping is wrong.")

    print("=" * 60)
    print("REBUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
