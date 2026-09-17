#!/usr/bin/env python3
"""Deploy 512-bit ITQ and rebuild the mesh with real semantic hashes."""
import numpy as np
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from yp_bridge import YPEngine


def load_itq_model(path="itq_model_512.npz"):
    """Load ITQ rotation matrix and mean."""
    data = np.load(path)
    return data["R"], data["m"]


def encode_512bit(vec, R, m):
    """Encode a vector to a 512-bit binary hash (64 bytes)."""
    centered = vec - m
    rotated = centered @ R
    bits = (rotated > 0).astype(np.uint8)
    byte_arr = np.packbits(bits)
    if len(byte_arr) < 64:
        byte_arr = np.pad(byte_arr, (0, 64 - len(byte_arr)), mode="constant")
    return byte_arr.tobytes()


def main():
    print("[512-DEPLOY] Loading ITQ model...")
    R, m = load_itq_model("itq_model_512.npz")
    print(f"[512-DEPLOY] R shape: {R.shape}, m shape: {m.shape}")

    print("[512-DEPLOY] Loading papers...")
    with open("data/papers.jsonl") as f:
        papers = [json.loads(line) for line in f if line.strip()]
    print(f"[512-DEPLOY] Papers: {len(papers)}")

    print("[512-DEPLOY] Loading embeddings...")
    embeddings = np.load("paper_embeddings.npy")
    print(f"[512-DEPLOY] Embeddings: {embeddings.shape}")
    assert len(papers) == len(embeddings), "Paper/embedding count mismatch!"

    # Backup existing mesh
    mesh_backup = "data/mesh_backup_pre512"
    if os.path.exists("data/mesh"):
        if os.path.exists(mesh_backup):
            shutil.rmtree(mesh_backup)
        shutil.copytree("data/mesh", mesh_backup)
        print(f"[512-DEPLOY] Backed up mesh to {mesh_backup}")

    engine = YPEngine()
    print("[512-DEPLOY] Re-encoding all papers with 512-bit ITQ...")
    zero_count = 0
    for i, (paper, emb) in enumerate(zip(papers, embeddings)):
        hash_512 = encode_512bit(emb, R, m)
        if all(b == 0 for b in hash_512):
            zero_count += 1
        engine.insert(paper["id"], paper.get("title", ""), hash_512)
        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(papers)}")

    print(f"[512-DEPLOY] Inserted {len(papers)} papers, zero hashes: {zero_count}")
    engine.save_mesh()

    print("[512-DEPLOY] Quick benchmark...")
    latencies = []
    for i in range(100):
        vec = embeddings[i % len(embeddings)]
        t0 = time.perf_counter()
        _ = engine.search("benchmark", vec, top_k=5)
        latencies.append((time.perf_counter() - t0) * 1_000_000)
    print(
        f"[512-DEPLOY] Latency: avg={np.mean(latencies):.1f}μs, "
        f"P50={np.percentile(latencies, 50):.1f}μs, "
        f"P99={np.percentile(latencies, 99):.1f}μs"
    )
    print("[512-DEPLOY] DONE.")


if __name__ == "__main__":
    main()
