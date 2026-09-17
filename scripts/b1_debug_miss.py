#!/usr/bin/env python3
import os, sys, sqlite3, random, json, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from yp_bridge import RustBridge

DB_PATH = os.path.join(ROOT, "data/phoenix_arxiv_1m.db")
HASH_PATH = os.path.join(ROOT, "hashes_12702_mlp.npy")
COLD_PATH = os.path.join(ROOT, ".tmp", "b1_debug_cold.bin")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT id, title FROM papers WHERE title IS NOT NULL AND title != '' ORDER BY rowid")
rows = [(pid, title.strip()) for pid, title in cur.fetchall() if title and len(title.strip()) > 3]
conn.close()
N = len(rows)
hash_bytes = np.load(HASH_PATH)[:N]

bridge = RustBridge()
for i in range(N):
    h = hash_bytes[i]
    bridge.lib.yp_insert_to_mesh_bytes(i + 1, h[:16].tobytes(), 16, h[:64].tobytes(), 64)
os.makedirs(os.path.dirname(COLD_PATH), exist_ok=True)
rc = bridge.enable_sharding(cold_path=COLD_PATH, num_shards=10, pre_fault=False)
print(f"enable_sharding rc={rc}")

random.seed(42)
queries = list(range(N)); random.shuffle(queries)
mismatch = 0
for qi in queries:
    hhex = hash_bytes[qi].tobytes().hex()
    raw = bridge.query_sharded(hhex)
    results = raw.get("results", [])
    if not results or int(results[0]["id"]) != qi + 1:
        pid, title = rows[qi]
        print(f"miss: qi={qi} id={qi+1} pid={pid} returned={results}")
        print(f"  title: {title[:120]}")
        print(f"  hash prefix: {hhex[:32]}")
        # check duplicates of prefix
        prefix = hash_bytes[qi, :16].tobytes()
        dup = [j for j in range(N) if hash_bytes[j, :16].tobytes() == prefix]
        print(f"  duplicate prefix indices: {dup[:10]} (total {len(dup)})")
        mismatch += 1
        if mismatch >= 10:
            break
print(f"first {mismatch} mismatches shown")
