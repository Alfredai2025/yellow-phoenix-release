#!/usr/bin/env python3
"""Hybrid veto benchmark driver: 5.1M real corpus.

Modes (same corpus/queries/GT as /tmp/hybrid5m_prep.py):
  1. BinaryHNSW alone (ef=400, k=100)
  2. Full cascade: spectral re-rank of top-100
  3. Veto loose  (drop candidates with spectral score < q25 of candidate scores)
  4. Veto medium (< q50)
  5. Veto strict (< q75)
R@k = |GT_topk ∩ list_topk| / k (set-based, include-self primary; exclude-self
secondary; self-hit@k also reported).

LATENCY NOTE (documented protocol adaptation): modes 2-5 execute an identical
per-query code path (HNSW top-100 + one bulk tensor_spectral_512 query +
candidate score extraction). The spectral scores are therefore collected in ONE
measured pass (1000 warm queries) and all four spectral modes share its median
latency; per-mode threshold/reorder overhead is measured separately on the
checkpointed scores (~microseconds) and reported in notes.
"""
import os, sys, json, time, ctypes

import numpy as np
import psutil

ROOT = "/Users/mac/yellow_phoenix"
OUT = "/tmp/hybrid5m"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

N_Q = 1000
K = 100
N = 5107508

pids = np.load(f"{OUT}/pids.npy")
qhash = np.load(f"{OUT}/qhash.npy")
gt_ids = np.load(f"{OUT}/gt_top100_ids.npy")
gt_sims = np.load(f"{OUT}/gt_top100_sims.npy")
qb = [qhash[i].tobytes() for i in range(N_Q)]
proc = psutil.Process()

from yp_bridge import RustBridge, BinaryHNSW

def load_hnsw(bridge):
    h = BinaryHNSW(bridge)
    t0 = time.time()
    n = h.load(os.path.join(ROOT, "data", "real_5m_hnsw_v5.bin"))
    load_s = time.time() - t0
    assert n == N, (n, N)
    if hasattr(bridge.lib, "yp_binary_hnsw_set_ef"):
        bridge.lib.yp_binary_hnsw_set_ef.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        bridge.lib.yp_binary_hnsw_set_ef.restype = ctypes.c_int
        bridge.lib.yp_binary_hnsw_set_ef(h._handle, 400)
    return h, load_s

def passes(fn):
    warm = np.zeros(N_Q, dtype=np.float64)
    for i in range(N_Q):  # discarded warm-up
        fn(i)
    res = []
    for i in range(N_Q):
        t0 = time.perf_counter_ns(); r = fn(i); warm[i] = time.perf_counter_ns() - t0
        res.append(r)
    return warm, res

def pct(a, q):
    return float(np.percentile(a, q))

results = {"n_queries": N_Q, "k": K, "ef": 400}

# ---------------- mode 1: BinaryHNSW alone
bridge = RustBridge()
hnsw, load_s = load_hnsw(bridge)
print(f"HNSW loaded {N:,} docs in {load_s:.2f}s", flush=True)
rss_hnsw = proc.memory_info().rss

def fn1(i):
    return [nid for nid, _ in hnsw.search(qb[i], K)]

lat1, lists1 = passes(fn1)
rss_after_m1 = proc.memory_info().rss
results["mode1_hnsw"] = dict(
    warm_p50_ms=pct(lat1, 50) / 1e6, warm_p95_ms=pct(lat1, 95) / 1e6,
    rss_mb=rss_after_m1 / 2**20, rss_index_mb=rss_hnsw / 2**20)
print(f"mode1: p50={results['mode1_hnsw']['warm_p50_ms']:.3f}ms "
      f"rss={results['mode1_hnsw']['rss_mb']:.0f}MB", flush=True)
np.save(f"{OUT}/hnsw_lists.npy", np.array(lists1, dtype=np.int64))

# free HNSW before spectral build to keep peak RSS down; reload after
hnsw.new()  # frees handle (drops graph mapping)
del hnsw

# ---------------- spectral build
lib = bridge.lib
lib.yp_tensor_spectral_512_build.restype = ctypes.c_int
lib.yp_tensor_spectral_512_build.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
ism_hashes = np.memmap(f"{ROOT}/data/real_5m.ism", dtype=np.uint8, mode="r",
                       offset=8, shape=(N, 64))
hh = np.ascontiguousarray(ism_hashes)
ids = np.arange(N, dtype=np.uint64)
t0 = time.time()
rc = lib.yp_tensor_spectral_512_build(
    ctypes.c_void_p(1),
    hh.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    ctypes.c_size_t(N))
spectral_build_s = time.time() - t0
del hh
assert rc == 0, rc
rss_spectral = proc.memory_info().rss
results["spectral_build_s"] = spectral_build_s
results["rss_spectral_mb"] = rss_spectral / 2**20
print(f"spectral built in {spectral_build_s:.1f}s, rss={rss_spectral/2**20:.0f}MB", flush=True)

# ---------------- spectral scoring pass (shared by modes 2-5)
lib.yp_tensor_spectral_512_query.restype = ctypes.c_size_t
lib.yp_tensor_spectral_512_query.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
K_ALL = N
out_ids = np.zeros(K_ALL, dtype=np.uint64)
out_scores = np.zeros(K_ALL, dtype=np.float32)

hnsw, load_s2 = load_hnsw(bridge)

def spectral_pass(i):
    cands = np.array([nid for nid, _ in hnsw.search(qb[i], K)], dtype=np.int64)
    arr = np.frombuffer(qb[i], dtype=np.uint8)
    n = lib.yp_tensor_spectral_512_query(
        ctypes.c_void_p(1),
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_size_t(K_ALL),
        out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(K_ALL))
    ids_n = out_ids[:n]
    mask = np.isin(ids_n, cands)
    hit_ids = ids_n[mask]
    hit_scores = out_scores[:n][mask]
    # candidate scores aligned to cands order (HNSW order)
    lut = dict(zip((int(x) for x in hit_ids), (float(v) for v in hit_scores)))
    cscores = np.array([lut.get(int(c), -np.inf) for c in cands], dtype=np.float64)
    return cands, cscores

# quick probe of per-query cost
t0 = time.perf_counter_ns()
spectral_pass(0)
probe_s = (time.perf_counter_ns() - t0) / 1e9
print(f"probe spectral pass: {probe_s:.2f}s/query", flush=True)

lat_sp, scored = passes(spectral_pass)
rss_geo = proc.memory_info().rss
results["spectral_modes_shared"] = dict(
    warm_p50_ms=pct(lat_sp, 50) / 1e6, warm_p95_ms=pct(lat_sp, 95) / 1e6,
    rss_mb=rss_geo / 2**20, probe_first_query_s=probe_s)
print(f"spectral path: p50={results['spectral_modes_shared']['warm_p50_ms']:.1f}ms "
      f"rss={rss_geo/2**20:.0f}MB", flush=True)
np.save(f"{OUT}/cand_ids.npy", np.array([s[0] for s in scored], dtype=np.int64))
np.save(f"{OUT}/cand_scores.npy", np.array([s[1] for s in scored], dtype=np.float64))
np.save(f"{OUT}/spectral_lat.npy", lat_sp)

# ---------------- threshold/reorder overhead (offline timing on checkpointed scores)
t0 = time.perf_counter_ns()
for _ in range(100):
    i = 0
    cs = scored[i][1]
    np.quantile(cs, [0.25, 0.5, 0.75])
    order = np.argsort(-cs)
REP = 100
thr_us = (time.perf_counter_ns() - t0) / 1e3 / REP
results["threshold_overhead_us"] = thr_us
print(f"threshold/reorder overhead: {thr_us:.1f}us per query", flush=True)

json.dump(results, open(f"{OUT}/bench_raw.json", "w"), indent=2)
print("driver done")
