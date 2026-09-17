#!/usr/bin/env python3
"""Yellow Phoenix driver for the 100K cascade-vs-FAISS benchmark.

Usage: yp100k_bench_yp.py binary|cascade
- binary:  BinaryHNSW (M=12, efConstruction=400, efSearch=400) over 512-bit ITQ
           hashes, via yp_bridge FFI (libpams.dylib). NO geometric re-rank.
           Strict R@1 = top-1 hamming neighbor (excluding self) == float GT argmin.
- cascade: BinaryHNSW ef=400 top-100 candidates -> tensor_spectral_512
           arbitration: candidates re-scored by spectral distance of query hash
           (full spectral scan restricted to candidate ids), argmin wins.
Both compare against the same brute-force float GT (exclude-self).
"""
import os, sys, json, time, ctypes

import numpy as np
import psutil

YP = "/Users/mac/yellow_phoenix"
OUT = "/tmp/yp100k"
os.chdir(YP)
sys.path.insert(0, YP)

def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))

def main():
    system = sys.argv[1]
    from yp_bridge import RustBridge, BinaryHNSW

    hashes = np.load(f"{OUT}/hashes.npy")
    qids = np.load(f"{OUT}/queries.npy")
    gt = np.load(f"{OUT}/gt.npy")
    proc = psutil.Process()
    rss_base = proc.memory_info().rss

    bridge = RustBridge()
    hnsw = BinaryHNSW(bridge)
    t0 = time.time()
    n = hnsw.load(f"{OUT}/binary_hnsw_100k_m12.bin")
    load_s = time.time() - t0
    assert n == 100000, n
    # explicit ef=400 (file already baked with ef_search=400)
    if hasattr(bridge.lib, "yp_binary_hnsw_set_ef"):
        bridge.lib.yp_binary_hnsw_set_ef.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        bridge.lib.yp_binary_hnsw_set_ef.restype = ctypes.c_int
        bridge.lib.yp_binary_hnsw_set_ef(hnsw._handle, 400)
    rss_idx = proc.memory_info().rss

    qb = [hashes[q].tobytes() for q in qids]

    if system == "binary":
        def fn(i):
            for nid, _d in hnsw.search(qb[i], 2):
                if nid != int(qids[i]):
                    return nid
            return -1
    else:
        # build tensor_spectral_512 index over all 100K hashes
        lib = bridge.lib
        lib.yp_tensor_spectral_512_build.restype = ctypes.c_int
        lib.yp_tensor_spectral_512_build.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ]
        ids = np.arange(100000, dtype=np.uint64)
        hh = np.ascontiguousarray(hashes)
        t0 = time.time()
        rc = lib.yp_tensor_spectral_512_build(
            ctypes.c_void_p(1),
            hh.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_size_t(100000),
        )
        spectral_build_s = time.time() - t0
        assert rc == 0, rc
        rss_idx = proc.memory_info().rss

        lib.yp_tensor_spectral_512_query.restype = ctypes.c_size_t
        lib.yp_tensor_spectral_512_query.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ]
        K_ALL = 100000
        out_ids = np.zeros(K_ALL, dtype=np.uint64)
        out_scores = np.zeros(K_ALL, dtype=np.float32)

        def spectral_all(qhash):
            arr = np.frombuffer(qhash, dtype=np.uint8)
            k = lib.yp_tensor_spectral_512_query(
                ctypes.c_void_p(1),
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.c_size_t(K_ALL),
                out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_size_t(K_ALL),
            )
            return out_ids[:k], out_scores[:k]

        def fn(i):
            cands = [nid for nid, _d in hnsw.search(qb[i], 100) if nid != int(qids[i])]
            if not cands:
                return -1
            sids, sscores = spectral_all(qb[i])
            score = dict(zip((int(s) for s in sids), (float(v) for v in sscores)))
            # spectral score = -euclidean_sq; higher = nearer
            return max(cands, key=lambda c: score.get(c, -np.inf))

    # cold pass (first 50), discarded warm-up, measured pass
    cold, warm, res = [], [], []
    for i in range(50):
        t0 = time.perf_counter_ns(); r = fn(i); cold.append(time.perf_counter_ns() - t0); res.append(r)
    for i in range(1000):
        fn(i)
    res = []
    for i in range(1000):
        t0 = time.perf_counter_ns(); r = fn(i); warm.append(time.perf_counter_ns() - t0); res.append(r)

    r = dict(system=("yp_binary_hnsw_m12_ef400" if system == "binary"
                     else "yp_geometric_cascade_hnsw_spectral"),
             m=12, ef_construction=400, ef_search=400, hash_bits=512,
             load_s=load_s,
             spectral_build_s=(spectral_build_s if system == "cascade" else None),
             rss_base_mb=rss_base / 2**20, rss_index_mb=rss_idx / 2**20,
             rss_query_mb=proc.memory_info().rss / 2**20,
             strict_r1=float(np.mean([rr == g for rr, g in zip(res, gt)])),
             warm_p50_ms=pct(warm, 50) / 1e6, warm_p95_ms=pct(warm, 95) / 1e6,
             cold_p50_ms=pct(cold, 50) / 1e6, n=1000)
    json.dump(r, open(f"{OUT}/results_yp_{system}.json", "w"), indent=2)
    print(json.dumps(r, indent=2))

if __name__ == "__main__":
    main()
