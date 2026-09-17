#!/usr/bin/env python3
"""Border-crossing benchmark: geometric structure vs cosine re-rank."""

import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from yp_bridge import HybridSpectralHNSW
import ctypes

EMBS = "data/paper_embeddings_100k.npy"
BASIS = "data/spectral_basis_100k_64.npz"
NQ = 50  # queries to test
K = 10   # top-k
CANDIDATES = 200


def load_ffi():
    import glob
    dylib = glob.glob("target/release/libpams*.dylib")[0]
    lib = ctypes.CDLL(dylib)
    lib.yp_geometric_score.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.yp_geometric_score.restype = ctypes.c_float
    return lib


def brute_force_topk(embs, q_idx, k):
    q = embs[q_idx]
    sims = embs @ q
    sims[q_idx] = -np.inf
    top = np.argpartition(-sims, k)[:k]
    return set(top[np.argsort(-sims[top])])


def main():
    print("Loading data...")
    embs = np.load(EMBS).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    basis = np.load(BASIS)
    V = basis["V"]  # (384, 64)
    C = embs @ V    # (100000, 64)

    print("Building HNSW...")
    engine = HybridSpectralHNSW(EMBS, BASIS, hnsw_ef=25)

    print("Loading FFI...")
    ffi = load_ffi()

    print(f"Running {NQ} queries...")
    results = {"cosine": {"times": [], "recalls": []}, "geometric": {"times": [], "recalls": []}}

    for i in range(NQ):
        q_idx = i  # use first NQ papers as queries
        q_384 = embs[q_idx]
        q_64 = C[q_idx].astype(np.float32)
        gt = brute_force_topk(embs, q_idx, K)

        # HNSW candidates
        cands = engine.search_hnsw(q_384, k=CANDIDATES)
        cand_ids = np.asarray(cands, dtype=np.int64)

        # --- BASELINE: exact cosine on 384-d ---
        t0 = time.perf_counter()
        sims = embs[cand_ids] @ q_384
        top_local = np.argpartition(-sims, K - 1)[:K]
        top_local = top_local[np.argsort(-sims[top_local])]
        pred_cos = set(cand_ids[top_local])
        t1 = time.perf_counter()
        results["cosine"]["times"].append((t1 - t0) * 1e6)
        results["cosine"]["recalls"].append(len(pred_cos & gt) / K)

        # --- CHALLENGER: geometric score on 64-d ---
        t0 = time.perf_counter()
        scores = []
        for cid in cand_ids:
            c_64 = C[cid].astype(np.float32)
            score = ffi.yp_geometric_score(
                q_64.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c_64.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(q_64),
            )
            scores.append(score)
        scores = np.array(scores, dtype=np.float32)
        top_local = np.argpartition(-scores, K - 1)[:K]
        top_local = top_local[np.argsort(-scores[top_local])]
        pred_geo = set(cand_ids[top_local])
        t1 = time.perf_counter()
        results["geometric"]["times"].append((t1 - t0) * 1e6)
        results["geometric"]["recalls"].append(len(pred_geo & gt) / K)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{NQ} done")

    def summarize(name):
        r = results[name]
        return {
            "mean_recall": float(np.mean(r["recalls"])),
            "p50_latency_us": float(np.percentile(r["times"], 50)),
            "p95_latency_us": float(np.percentile(r["times"], 95)),
        }

    out = {
        "cosine": summarize("cosine"),
        "geometric": summarize("geometric"),
        "winner": "geometric" if summarize("geometric")["mean_recall"] > summarize("cosine")["mean_recall"] + 0.001 else "cosine",
    }

    print("\n" + "=" * 60)
    print(f"{'Method':<15} {'R@10':>8} {'P50 µs':>10} {'P95 µs':>10}")
    print("=" * 60)
    for m in ["cosine", "geometric"]:
        s = out[m]
        print(f"{m:<15} {s['mean_recall']*100:>7.1f}% {s['p50_latency_us']:>9.1f} {s['p95_latency_us']:>9.1f}")
    print("=" * 60)
    print(f"\nWINNER: {out['winner'].upper()}")

    Path("logs").mkdir(exist_ok=True)
    with open("logs/bench_geometric_border.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
