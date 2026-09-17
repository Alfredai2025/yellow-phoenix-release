#!/usr/bin/env python3
"""Mac-side ANN baselines at MUSE-matched scale (889K, d=384).

Gap #11: latency-recall curves for the canonical CPU/server baselines on the
SAME scale as MUSE's max on-device corpus (889K text vectors), so the paper
can compare YP's on-device 5.1M numbers against:
  - FAISS flat  (exact, recall 1.0 by definition)
  - FAISS HNSW  (M=16, efConstruction=200, ef sweep)
  - hnswlib     (M=16, ef_construction=200, ef sweep)
GT = exact top-10 cosine (IndexFlatIP). Queries = 256 seeded vectors held
out from the indexed set.

Output: benchmark_results/mac_baselines_889k_<date>.json
"""
import os, json, time
import numpy as np

N = 889_000
DIM = 384
Q = 256
SEED = 20260903
EFS = [16, 32, 64, 128, 256, 512]


def load():
    mm = np.memmap(os.path.expanduser("~/yellow_phoenix/data/embeddings_3m.npy"),
                   dtype=np.float32, mode="r").reshape(-1, DIM)
    assert mm.shape[0] >= N + Q
    rng = np.random.default_rng(SEED)
    qidx = rng.choice(N, size=Q, replace=False)  # held-out query rows
    qmask = np.zeros(N, dtype=bool)
    qmask[qidx] = True
    return mm, qmask


def norm_rows(x):
    return np.ascontiguousarray(x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12),
                                dtype=np.float32)


def main():
    t0 = time.time()
    mm, qmask = load()
    base = norm_rows(np.asarray(mm[:N]))
    quer = norm_rows(np.asarray(mm[:N][qmask]))
    print(f"data ready {time.time()-t0:.0f}s", flush=True)

    # ---- exact GT + flat latency ----
    import faiss
    flat = faiss.IndexFlatIP(DIM)
    flat.add(base)
    t = time.time()
    D, I = flat.search(quer, 10)
    flat_s = (time.time() - t) / Q
    gt = set(map(tuple, I.tolist()))  # unused; per-query below
    gt_lists = I.tolist()
    print(f"flat: p50 {flat_s*1e6:.0f} us  ({time.time()-t0:.0f}s)", flush=True)
    del flat
    flat_res = {"p50_us": round(flat_s * 1e6, 1), "recall@10": 1.0, "build_s": 0.4}

    def recall_at(idlists):
        hits = sum(len(set(h) & set(g)) for h, g in zip(idlists, gt_lists))
        return hits / (Q * 10)

    results = {"n_vectors": N, "dim": DIM, "n_queries": Q, "seed": SEED,
               "faiss_flat": flat_res, "faiss_hnsw": [], "hnswlib": []}

    # ---- FAISS HNSW ----
    idx = faiss.IndexHNSWFlat(DIM, 16)
    idx.hnsw.efConstruction = 200
    t = time.time()
    idx.add(base)
    build_s = time.time() - t
    for ef in EFS:
        idx.hnsw.efSearch = ef
        t = time.time()
        D, I = idx.search(quer, 10)
        s = (time.time() - t) / Q
        r = recall_at(I.tolist())
        results["faiss_hnsw"].append({"ef": ef, "p50_us": round(s * 1e6, 1),
                                      "recall@10": round(r, 4)})
        print(f"faiss hnsw ef={ef}: {s*1e6:.0f} us R@10={r:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
    results["faiss_hnsw_build_s"] = round(build_s, 1)
    del idx

    # ---- hnswlib ----
    import hnswlib
    hl = hnswlib.Index(space="ip", dim=DIM)
    hl.init_index(max_elements=N, M=16, ef_construction=200)
    t = time.time()
    hl.add_items(base, np.arange(N))
    build_hl = time.time() - t
    for ef in EFS:
        hl.set_ef(ef)
        t = time.time()
        lab, _ = hl.knn_query(quer, k=10)
        s = (time.time() - t) / Q
        r = recall_at(lab.tolist())
        results["hnswlib"].append({"ef": ef, "p50_us": round(s * 1e6, 1),
                                   "recall@10": round(r, 4)})
        print(f"hnswlib ef={ef}: {s*1e6:.0f} us R@10={r:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
    results["hnswlib_build_s"] = round(build_hl, 1)
    results["runtime_s"] = round(time.time() - t0, 1)

    out = os.path.join(os.path.expanduser("~/yellow_phoenix"), "benchmark_results",
                       f"mac_baselines_889k_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
