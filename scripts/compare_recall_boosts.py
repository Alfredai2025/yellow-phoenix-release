#!/usr/bin/env python3
"""Compare 3 strategies to boost recall@10: Multi-Probe, Multi-Index, Graph Diffusion."""

import os, sys, time, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from yp_bridge import BinaryHNSW

HASH_PATHS = ["data/paper_hashes_100k.npy", "paper_hashes_100k.npy"]
EMB_PATHS = ["data/paper_embeddings_100k.npy", "paper_embeddings_100k.npy"]

def load_data():
    hp = next((p for p in HASH_PATHS if os.path.exists(p)), None)
    ep = next((p for p in EMB_PATHS if os.path.exists(p)), None)
    if not hp or not ep:
        raise FileNotFoundError("Need paper_hashes_100k.npy and paper_embeddings_100k.npy")
    hashes = np.load(hp)
    if hashes.dtype != np.uint8:
        hashes = hashes.astype(np.uint8)
    if hashes.ndim == 1:
        hashes = hashes.reshape(-1, 64)
    embs = np.load(ep).astype(np.float32)
    return hashes, embs

def cosine_sim(query, candidates):
    qn = query / (np.linalg.norm(query) + 1e-10)
    cn = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-10)
    return qn @ cn.T

def brute_force_topk(query_emb, embs, k=10):
    sims = cosine_sim(query_emb, embs)
    topk = np.argpartition(-sims, k)[:k]
    return topk[np.argsort(-sims[topk])]

def build_main_index(hashes):
    idx = BinaryHNSW()
    for i in range(len(hashes)):
        idx.insert(i, bytes(hashes[i]))
    return idx

def build_range_index(hashes, start_byte, end_byte):
    """Build HNSW on a sub-range of bytes."""
    idx = BinaryHNSW()
    sub = hashes[:, start_byte:end_byte]
    # Pad to 64 bytes with zeros so BinaryHNSW accepts it
    padded = np.zeros((len(sub), 64), dtype=np.uint8)
    padded[:, :sub.shape[1]] = sub
    for i in range(len(padded)):
        idx.insert(i, bytes(padded[i]))
    return idx

def search_range_index(idx, query_hash, start_byte, end_byte, k=100):
    """Search a range index with padded query."""
    sub = query_hash[start_byte:end_byte]
    padded = np.zeros(64, dtype=np.uint8)
    padded[:len(sub)] = sub
    return idx.search(bytes(padded), k)

def perturb_hash(hash_bytes, pattern):
    """Flip bits according to pattern."""
    h = bytearray(hash_bytes)
    for pos in pattern:
        byte_idx = pos // 8
        bit_idx = pos % 8
        h[byte_idx] ^= (1 << bit_idx)
    return bytes(h)

def recall_at_k(pred, gt, k):
    return len(set(pred[:k]) & set(gt[:k])) / k

def main():
    print("Loading data...")
    hashes, embs = load_data()
    n = len(hashes)
    nq = 200  # 200 queries for speed
    print(f"Vectors: {n}, Queries: {nq}")

    # Build main index
    print("\nBuilding main HNSW index...")
    t0 = time.perf_counter()
    main_idx = build_main_index(hashes)
    print(f"Main index built in {time.perf_counter()-t0:.1f}s")

    # Build range indexes for Option B
    print("Building 3 range indexes (Option B)...")
    range_idxs = [
        (build_range_index(hashes, 0, 21), 0, 21),    # bytes 0-20
        (build_range_index(hashes, 21, 43), 21, 43),   # bytes 21-42
        (build_range_index(hashes, 43, 64), 43, 64),   # bytes 43-63
    ]
    print("Range indexes built")

    # Ground truth
    print("\nComputing brute-force ground truth...")
    gt_list = []
    for i in range(nq):
        gt_list.append(list(brute_force_topk(embs[i], embs, k=10)))
    print("GT done")

    # Perturbation patterns for Option A
    probe_patterns = [
        [],                                      # original
        [0, 64, 128, 192, 256, 320, 384, 448],   # 1 bit per 64-bit chunk
        [1, 65, 129, 193, 257, 321, 385, 449],   # shifted by 1
    ]

    methods = {
        "baseline": {"candidates": [], "times": []},
        "multi_probe": {"candidates": [], "times": []},
        "multi_index": {"candidates": [], "times": []},
        "graph_diffusion": {"candidates": [], "times": []},
    }

    print("\nRunning 200 queries across 4 methods...")
    for qi in range(nq):
        q_hash, q_emb = hashes[qi], embs[qi]

        # --- BASELINE: single search k=100 ---
        t0 = time.perf_counter()
        res = main_idx.search(bytes(q_hash), k=100)
        base_cands = [int(r[0]) for r in res]
        base_time = (time.perf_counter() - t0) * 1e6
        methods["baseline"]["candidates"].append(base_cands)
        methods["baseline"]["times"].append(base_time)

        # --- OPTION A: Multi-Probe (3 perturbed hashes) ---
        t0 = time.perf_counter()
        all_cands = set()
        for pattern in probe_patterns:
            ph = perturb_hash(bytes(q_hash), pattern)
            res = main_idx.search(ph, k=100)
            for r in res:
                all_cands.add(int(r[0]))
        probe_cands = list(all_cands)
        probe_time = (time.perf_counter() - t0) * 1e6
        methods["multi_probe"]["candidates"].append(probe_cands)
        methods["multi_probe"]["times"].append(probe_time)

        # --- OPTION B: Multi-Index Ensemble (3 range indexes) ---
        t0 = time.perf_counter()
        all_cands = set()
        for idx, sb, eb in range_idxs:
            res = search_range_index(idx, q_hash, sb, eb, k=100)
            for r in res:
                all_cands.add(int(r[0]))
        range_cands = list(all_cands)
        range_time = (time.perf_counter() - t0) * 1e6
        methods["multi_index"]["candidates"].append(range_cands)
        methods["multi_index"]["times"].append(range_time)

        # --- OPTION C: Graph Diffusion (candidates + their neighbors) ---
        t0 = time.perf_counter()
        all_cands = set(base_cands)
        for cand in base_cands[:20]:  # top 20 candidates' neighbors
            res = main_idx.search(bytes(hashes[cand]), k=10)
            for r in res:
                all_cands.add(int(r[0]))
        diff_cands = list(all_cands)
        diff_time = (time.perf_counter() - t0) * 1e6
        methods["graph_diffusion"]["candidates"].append(diff_cands)
        methods["graph_diffusion"]["times"].append(diff_time)

        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{nq} done")

    # Re-rank and compute recall for each method
    print("\nRe-ranking and computing recall...")
    results = {}
    for name, data in methods.items():
        r1, r5, r10 = [], [], []
        for qi in range(nq):
            cands = data["candidates"][qi]
            if not cands:
                continue
            # Re-rank by cosine
            cand_embs = embs[cands]
            sims = cosine_sim(embs[qi], cand_embs)
            ranked = [cands[i] for i in np.argsort(-sims)]
            gt = gt_list[qi]
            r1.append(recall_at_k(ranked, gt, 1))
            r5.append(recall_at_k(ranked, gt, 5))
            r10.append(recall_at_k(ranked, gt, 10))

        times = data["times"]
        results[name] = {
            "avg_candidates": int(np.mean([len(c) for c in data["candidates"]])),
            "p50_us": round(float(np.percentile(times, 50)), 1),
            "p95_us": round(float(np.percentile(times, 95)), 1),
            "r1_pct": round(float(np.mean(r1)) * 100, 2),
            "r5_pct": round(float(np.mean(r5)) * 100, 2),
            "r10_pct": round(float(np.mean(r10)) * 100, 2),
        }

    # Print table
    print("\n" + "="*80)
    print(f"{'Method':<20} {'Candidates':>10} {'P50(µs)':>10} {'R@1%':>8} {'R@5%':>8} {'R@10%':>8}")
    print("="*80)
    for name in ["baseline", "multi_probe", "multi_index", "graph_diffusion"]:
        r = results[name]
        label = {"baseline": "Baseline (k=100)",
                 "multi_probe": "A: Multi-Probe",
                 "multi_index": "B: Multi-Index",
                 "graph_diffusion": "C: Graph Diffusion"}[name]
        print(f"{label:<20} {r['avg_candidates']:>10} {r['p50_us']:>10} {r['r1_pct']:>8.1f} {r['r5_pct']:>8.1f} {r['r10_pct']:>8.1f}")
    print("="*80)

    # Save
    os.makedirs("logs", exist_ok=True)
    with open("logs/compare_recall_boosts.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to logs/compare_recall_boosts.json")

if __name__ == "__main__":
    main()
