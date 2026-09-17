#!/usr/bin/env python3
"""FAISS driver for the 100K cascade-vs-FAISS benchmark.

Usage: yp100k_bench_faiss.py hnsw|ivf
- hnsw: IndexHNSWFlat(d=384, M=12), efConstruction=400, efSearch=400.
- ivf:  IndexIVFFlat(nlist=316, METRIC_INNER_PRODUCT on L2-normalized vectors),
        nprobe swept, chosen to match HNSW warm p50 (reads /tmp/yp100k/results_faiss_hnsw.json).
Strict R@1 vs brute-force float GT (exact cosine argmin, exclude-self).
Cold = median of first 50 queries in a fresh process; warm pass discards one
full 1000-query warm-up pass before the measured pass.
Single-threaded. RSS via psutil.
"""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import psutil

OUT = "/tmp/yp100k"
D = 384

def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))

def measure(fn):
    """Returns (cold_us[50], warm_us[1000], results[1000]). fn(i) -> returned_id."""
    cold, warm, res = [], [], []
    for i in range(50):
        t0 = time.perf_counter_ns(); r = fn(i); cold.append(time.perf_counter_ns() - t0); res.append(r)
    for i in range(1000):  # discarded warm-up pass
        fn(i)
    for i in range(1000):
        t0 = time.perf_counter_ns(); r = fn(i); warm.append(time.perf_counter_ns() - t0); res.append(r)
    return cold, warm, res

def main():
    system = sys.argv[1]
    import faiss
    faiss.omp_set_num_threads(1)

    x = np.load("/Users/mac/yellow_phoenix/data/paper_embeddings_100k.npy")
    assert x.shape == (100000, 384) and x.dtype == np.float32
    x = np.ascontiguousarray(x / np.linalg.norm(x, axis=1, keepdims=True), dtype=np.float32)
    qids = np.load(f"{OUT}/queries.npy")
    gt = np.load(f"{OUT}/gt.npy")
    xq = np.ascontiguousarray(x[qids])
    proc = psutil.Process()
    rss_base = proc.memory_info().rss

    if system == "hnsw":
        t0 = time.time()
        index = faiss.IndexHNSWFlat(D, 12)
        index.hnsw.efConstruction = 400
        index.add(x)
        build_s = time.time() - t0
        index.hnsw.efSearch = 400
        rss_idx = proc.memory_info().rss

        def fn(i):
            D_, I = index.search(xq[i:i+1], 5)
            for j in range(5):
                if int(I[0, j]) != int(qids[i]):
                    return int(I[0, j])
            return -1
        nprobe = None
        sweep = None
    elif system == "ivf":
        hnsw_res = json.load(open(f"{OUT}/results_faiss_hnsw.json"))
        target_p50 = hnsw_res["warm_p50_ms"]
        nlist = 316
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
        t0 = time.time()
        index.train(x)
        index.add(x)
        build_s = time.time() - t0
        rss_idx = proc.memory_info().rss

        sweep = {}
        def mk(np_):
            index.nprobe = np_
            def f(i):
                D_, I = index.search(xq[i:i+1], 5)
                for j in range(5):
                    if int(I[0, j]) != int(qids[i]):
                        return int(I[0, j])
                return -1
            return f
        for np_ in [1, 2, 4, 8, 16, 32, 64, 128, 256, 316]:
            cold, warm, res = measure(mk(np_))
            p50 = pct(warm, 50) / 1e6; p95 = pct(warm, 95) / 1e6
            r1 = float(np.mean([r == g for r, g in zip(res, gt)]))
            sweep[np_] = dict(p50_ms=p50, p95_ms=p95, r1=r1, cold_p50_ms=pct(cold, 50) / 1e6)
            print(f"nprobe={np_}: p50={p50:.3f}ms p95={p95:.3f}ms cold_p50={sweep[np_]['cold_p50_ms']:.3f}ms R@1={r1:.4f}", flush=True)
        # choose nprobe with p50 closest to target from above (match latency, not beat it)
        cands = sorted((abs(sweep[np_]["p50_ms"] - target_p50), np_) for np_ in sweep)
        cands_above = [c for c in cands if sweep[c[1]]["p50_ms"] >= target_p50 * 0.9]
        nprobe = (cands_above[0][1] if cands_above else cands[0][1])
        fn = mk(nprobe)
        # final measured numbers for chosen nprobe (fresh passes)
        cold, warm, res = measure(fn)
        build_s = build_s  # noqa
        r = dict(system=f"faiss_ivfflat_nlist316_nprobe{nprobe}", nlist=nlist, nprobe=nprobe,
                 build_s=build_s, rss_base_mb=rss_base / 2**20, rss_index_mb=rss_idx / 2**20,
                 rss_query_mb=proc.memory_info().rss / 2**20,
                 strict_r1=float(np.mean([rr == g for rr, g in zip(res, gt)])),
                 warm_p50_ms=pct(warm, 50) / 1e6, warm_p95_ms=pct(warm, 95) / 1e6,
                 cold_p50_ms=pct(cold, 50) / 1e6, n=1000,
                 sweep=sweep, target_hnsw_p50_ms=target_p50)
        json.dump(r, open(f"{OUT}/results_faiss_ivf.json", "w"), indent=2)
        print(json.dumps({k: v for k, v in r.items() if k != "sweep"}, indent=2))
        return

    cold, warm, res = measure(fn)
    r = dict(system="faiss_hnsw_m12_ef400", m=12, ef_construction=400, ef_search=400,
             build_s=build_s, rss_base_mb=rss_base / 2**20, rss_index_mb=rss_idx / 2**20,
             rss_query_mb=proc.memory_info().rss / 2**20,
             strict_r1=float(np.mean([rr == g for rr, g in zip(res, gt)])),
             warm_p50_ms=pct(warm, 50) / 1e6, warm_p95_ms=pct(warm, 95) / 1e6,
             cold_p50_ms=pct(cold, 50) / 1e6, n=1000)
    json.dump(r, open(f"{OUT}/results_faiss_hnsw.json", "w"), indent=2)
    print(json.dumps(r, indent=2))

if __name__ == "__main__":
    main()
