#!/usr/bin/env python3
"""B1: Rebuild a unified mesh from MLP hashes and test self-query recall."""
import ctypes
import json
import os
import random
import sqlite3
import sys
import time

import numpy as np

# Allow imports from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from yp_bridge import RustBridge, MINILM_PATH

DB_PATH = os.path.join(ROOT, "data/phoenix_arxiv_1m.db")

def load_titles():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title FROM papers WHERE title IS NOT NULL AND title != ''")
    rows = [(pid, title.strip()) for pid, title in cur.fetchall() if title and len(title.strip()) > 3]
    conn.close()
    return rows

def main():
    print("Loading papers from DB...")
    rows = load_titles()
    ids = [r[0] for r in rows]
    titles = [r[1] for r in rows]
    N = len(titles)
    print(f"Papers: {N}")

    print("Loading MiniLM...")
    from sentence_transformers import SentenceTransformer
    minilm = SentenceTransformer(MINILM_PATH, device="cpu", local_files_only=True)

    print("Batch encoding titles...")
    t0 = time.time()
    embs = minilm.encode(titles, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    print(f"  Embeddings: {embs.shape} in {time.time() - t0:.1f}s")

    print("Loading MLP hash model...")
    mlp = np.load("itq_model_512_v2.npz")
    W1, b1 = mlp["W1"].astype(np.float32), mlp["b1"].astype(np.float32)
    W2, b2 = mlp["W2"].astype(np.float32), mlp["b2"].astype(np.float32)
    W3, b3 = mlp["W3"].astype(np.float32), mlp["b3"].astype(np.float32)

    print("Hashing with MLP...")
    t0 = time.time()
    z1 = np.maximum(0, embs @ W1.T + b1)
    z2 = np.maximum(0, z1 @ W2.T + b2)
    z3 = z2 @ W3.T + b3
    bits = z3 >= 0
    hash_bytes = np.packbits(bits.astype(np.uint8), axis=1)
    print(f"  Hashes: {hash_bytes.shape} in {time.time() - t0:.1f}s")

    out_hash = os.path.join(ROOT, "hashes_12702_mlp.npy")
    np.save(out_hash, hash_bytes)
    print(f"Saved: {out_hash}")

    print("Creating unified mesh handle...")
    bridge = RustBridge()
    handle = bridge.lib.yp_unified_mesh_new_512(20000, 20000)
    print(f"  Handle: {handle}")

    print("Inserting into handle...")
    t0 = time.time()
    for i in range(N):
        fine_hex = hash_bytes[i].tobytes().hex().encode("utf-8")
        bridge.lib.yp_unified_mesh_insert_512(handle, b"", fine_hex)
        if i % 2000 == 0 and i > 0:
            print(f"  {i // 1000}K inserted ({time.time() - t0:.1f}s)")
    print(f"  Insert done in {time.time() - t0:.1f}s")

    print("Building edges...")
    bridge.lib.yp_unified_mesh_build_edges(handle, 8)

    print("Testing recall on 100 queries...")
    random.seed(42)
    q_idx = random.sample(range(N), 100)
    q_set = set(q_idx)
    idx_idx = [i for i in range(N) if i not in q_set]
    idx_hashes = hash_bytes[idx_idx]
    idx_id_for_position = {pos: idx_idx[pos] for pos in range(len(idx_idx))}

    r1 = 0
    returned = 0
    t0 = time.time()
    for i, qi in enumerate(q_idx):
        # Exact Hamming ground truth over the remaining 12,602 hashes
        diffs = np.bitwise_xor(idx_hashes, hash_bytes[qi])
        hamming = np.unpackbits(diffs, axis=1).sum(axis=1)
        gt_pos = int(np.argmin(hamming))
        gt_idx = idx_idx[gt_pos]

        fine_hex = hash_bytes[qi].tobytes().hex().encode("utf-8")
        result_ptr = bridge.lib.yp_unified_mesh_query_512(handle, fine_hex, len(fine_hex), 1)
        yp_id = None
        if result_ptr:
            result_json = ctypes.string_at(result_ptr).decode("utf-8")
            try:
                data = json.loads(result_json)
                if data.get("results"):
                    yp_id = int(data["results"][0]["id"])
            except Exception:
                yp_id = None
            if hasattr(bridge.lib, "yp_free_string"):
                bridge.lib.yp_free_string(result_ptr)
        if yp_id is not None:
            returned += 1
        # IDs were assigned starting at 1, so map back to 0-based index
        if yp_id is not None and yp_id - 1 == gt_idx:
            r1 += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/100 done ({time.time() - t0:.1f}s) — R@1 so far {r1}/{i + 1}")

    print(f"\nRecall@1: {r1}/100 = {r1}%")
    print(f"YP returned results for {returned}/100 queries")

    mesh_path = os.path.join(ROOT, "mesh_mlp_512.bin")
    print("Saving mesh...")
    bridge.lib.yp_unified_mesh_save(handle, mesh_path.encode("utf-8"))
    bridge.lib.yp_unified_mesh_free(handle)
    print(f"Done. Saved: {mesh_path}")

if __name__ == "__main__":
    main()
