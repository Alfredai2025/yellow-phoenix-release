#!/usr/bin/env python3
"""Head-to-head: FAISS HNSW (384-d float) vs YP Binary HNSW (512-bit hash)."""

import os, sys, time, json, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from yp_bridge import BinaryHNSW

EMB_PATHS = ["data/paper_embeddings_100k.npy", "paper_embeddings_100k.npy"]
HASH_PATHS = ["data/paper_hashes_100k.npy", "paper_hashes_100k.npy"]

def load_data():
    ep = next((p for p in EMB_PATHS if os.path.exists(p)), None)
    hp = next((p for p in HASH_PATHS if os.path.exists(p)), None)
    if not ep or not hp:
        raise FileNotFoundError("Need paper_embeddings_100k.npy and paper_hashes_100k.npy")
    embs = np.load(ep).astype(np.float32)
    hashes = np.load(hp)
    if hashes.dtype != np.uint8:
        hashes = hashes.astype(np.uint8)
    if hashes.ndim == 1:
        hashes = hashes.reshape(-1, 64)
    return embs, hashes

def bench_faiss(xb, xq, k=10):
    import faiss
    dim = xb.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128

    t0 = time.perf_counter()
    index.add(xb)
    build_s = time.perf_counter() - t0

    # Warm-up
    index.search(xq[:10], k)

    times = []
    for q in xq:
        t0 = time.perf_counter()
        _, _ = index.search(q.reshape(1, -1), k)
        times.append((time.perf_counter() - t0) * 1e6)

    # Memory estimate: write index to temp file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        faiss.write_index(index, f.name)
        mem_bytes = os.path.getsize(f.name)
        os.unlink(f.name)

    return build_s, times, mem_bytes, index

def bench_binary_hnsw(hashes, xq_hashes, k=10):
    hnsw = BinaryHNSW()
    t0 = time.perf_counter()
    for i in range(len(hashes)):
        hnsw.insert(i, bytes(hashes[i]))
    build_s = time.perf_counter() - t0

    times = []
    for q in xq_hashes:
        t0 = time.perf_counter()
        hnsw.search(bytes(q), k)
        times.append((time.perf_counter() - t0) * 1e6)

    # Memory: hashes (n*64) + rough graph estimate (n*M*4 bytes per edge)
    n = len(hashes)
    mem_bytes = n * 64 + n * 32 * 4  # conservative
    return build_s, times, mem_bytes, hnsw

def recall_faiss(index, xq, gt, k=10):
    """gt is list of lists of brute-force top-k indices."""
    hits = []
    for i, q in enumerate(xq):
        _, I = index.search(q.reshape(1, -1), k)
        pred = set(I[0])
        gt_set = set(gt[i])
        hits.append(len(pred & gt_set) / k)
    return float(np.mean(hits)) * 100

def recall_binary(hnsw, xq_hashes, gt, k=10):
    hits = []
    for i, q in enumerate(xq_hashes):
        res = hnsw.search(bytes(q), k)
        pred = {int(r[0]) for r in res}
        gt_set = set(gt[i])
        hits.append(len(pred & gt_set) / k)
    return float(np.mean(hits)) * 100

def brute_force_gt(xb, xq, k=10):
    gt = []
    for q in xq:
        sims = (xb @ q) / (np.linalg.norm(xb, axis=1) * np.linalg.norm(q) + 1e-10)
        topk = np.argpartition(-sims, k)[:k]
        topk = topk[np.argsort(-sims[topk])]
        gt.append(list(topk))
    return gt

def percentile(arr, p):
    return float(np.percentile(arr, p))

def main():
    print("Loading data...")
    embs, hashes = load_data()
    n = len(embs)
    nq = 500
    xb = embs
    xq = embs[:nq]
    xq_hashes = hashes[:nq]
    print(f"Vectors: {n}, Queries: {nq}, Dim: {embs.shape[1]}")

    print("\n--- Brute-force ground truth (cosine top-10) ---")
    t0 = time.perf_counter()
    gt = brute_force_gt(xb, xq, k=10)
    print(f"GT computed in {(time.perf_counter()-t0):.2f}s")

    print("\n--- FAISS HNSW (384-d float) ---")
    f_build, f_times, f_mem, faiss_idx = bench_faiss(xb, xq, k=10)
    print(f"Build: {f_build:.2f}s")
    print(f"Search P50: {percentile(f_times, 50):.1f}µs  P95: {percentile(f_times, 95):.1f}µs  P99: {percentile(f_times, 99):.1f}µs")
    print(f"Index memory: {f_mem/1024/1024:.1f} MB")

    print("\n--- YP Binary HNSW (512-bit hash) ---")
    b_build, b_times, b_mem, hnsw = bench_binary_hnsw(hashes, xq_hashes, k=10)
    print(f"Build: {b_build:.2f}s")
    print(f"Search P50: {percentile(b_times, 50):.1f}µs  P95: {percentile(b_times, 95):.1f}µs  P99: {percentile(b_times, 99):.1f}µs")
    print(f"Index memory: {b_mem/1024/1024:.1f} MB")

    f_recall = recall_faiss(faiss_idx, xq, gt, k=10)
    b_recall = recall_binary(hnsw, xq_hashes, gt, k=10)

    print(f"\nRecall@10 vs brute-force cosine:")
    print(f"  FAISS HNSW:  {f_recall:.1f}%")
    print(f"  YP Binary:   {b_recall:.1f}%")

    results = {
        "n_vectors": int(n),
        "n_queries": int(nq),
        "faiss": {
            "build_sec": round(f_build, 2),
            "p50_us": round(percentile(f_times, 50), 2),
            "p95_us": round(percentile(f_times, 95), 2),
            "p99_us": round(percentile(f_times, 99), 2),
            "memory_mb": round(f_mem / 1024 / 1024, 2),
            "recall_at_10": round(f_recall, 2),
        },
        "yp_binary": {
            "build_sec": round(b_build, 2),
            "p50_us": round(percentile(b_times, 50), 2),
            "p95_us": round(percentile(b_times, 95), 2),
            "p99_us": round(percentile(b_times, 99), 2),
            "memory_mb": round(b_mem / 1024 / 1024, 2),
            "recall_at_10": round(b_recall, 2),
        },
    }

    os.makedirs("logs", exist_ok=True)
    out = "logs/bench_faiss_vs_binary_hnsw.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

if __name__ == "__main__":
    main()
