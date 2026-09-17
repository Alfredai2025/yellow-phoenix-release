#!/usr/bin/env python3
"""Experiment A — learned hash-space re-ranker (kill the float store).

Stages (each checkpointed to /tmp/learned_rerank/):
  1. pairs:     20K random corpus docs as pseudo-queries; top-100 HNSW
                neighbors each (self included) -> (qid, cid) pairs
  2. features:  per-pair byte-wise XOR popcount (64) + 8x8-byte block
                popcounts (8) + total hamming (1) + hamming/512 (1) = 74 feats
  3. labels:    exact float cosine from floats_5m.f16 (corpus is L2-normalized
                -> cosine = dot), computed in chunks
  4. train:     sklearn HistGradientBoostingRegressor on 2M pairs
  5. eval:      1000 text-space queries (/tmp/hybrid5m), include-self:
                (1) HNSW order top-50, (2) exact float re-rank top-50
                (production path), (3) learned re-rank top-50.
                R@1, R@10, p50 overhead, % of float-rerank gain recovered.
  6. report:    benchmark_results/learned_rerank_5m_<date>.{json,md} + Entry 61.

Env overrides for smoke: A_N_TRAIN_DOCS (default 20000), A_N_EVAL (default 1000),
A_SKIP_PAIRS=1 to reuse checkpointed pairs.
"""
import os, sys, json, time, ctypes, datetime

import numpy as np

ROOT = "/Users/mac/yellow_phoenix"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
OUT = "/tmp/learned_rerank"
os.makedirs(OUT, exist_ok=True)

N_ALL = 5107508
DIM = 384
SEED = 20260916
N_TRAIN_DOCS = int(os.environ.get("A_N_TRAIN_DOCS", "20000"))
N_EVAL = int(os.environ.get("A_N_EVAL", "1000"))
TOPK = 50
DATE = datetime.datetime.now().strftime("%Y%m%d")

# popcount LUT
_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def pair_features(qh, ch):
    """qh (B,64) uint8, ch (B,64) uint8 -> (B,74) float32."""
    x = _LUT[np.bitwise_xor(qh, ch)]          # (B,64) per-byte popcount
    blocks = x.reshape(len(x), 8, 8).sum(axis=2)  # (B,8)
    total = x.sum(axis=1, keepdims=True).astype(np.float32)
    feats = np.concatenate([x, blocks, total, total / 512.0], axis=1)
    return feats.astype(np.float32)

# ---------------- stage 1: pairs
pairs_path = f"{OUT}/train_pairs.npy"
if os.environ.get("A_SKIP_PAIRS") == "1" and os.path.exists(pairs_path):
    pairs = np.load(pairs_path)
    print(f"reused pairs {pairs.shape}", flush=True)
else:
    from yp_bridge import RustBridge, BinaryHNSW
    bridge = RustBridge()
    h = BinaryHNSW(bridge)
    h.load(os.path.join(ROOT, "data/real_5m_hnsw_v5.bin"))
    bridge.lib.yp_binary_hnsw_set_ef.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    bridge.lib.yp_binary_hnsw_set_ef(h._handle, 400)
    rng = np.random.default_rng(SEED)
    tdocs = rng.choice(N_ALL, size=N_TRAIN_DOCS, replace=False)
    ism = np.memmap("data/real_5m.ism", dtype=np.uint8, mode="r", offset=8, shape=(N_ALL, 64))
    rows = []
    t0 = time.time()
    for i, d in enumerate(tdocs):
        qb = np.asarray(ism[d]).tobytes()
        for nid, _ in h.search(qb, 100):
            rows.append((int(d), int(nid)))
        if (i + 1) % 2000 == 0:
            print(f"  pairs {i+1}/{N_TRAIN_DOCS} ({time.time()-t0:.0f}s)", flush=True)
    pairs = np.array(rows, dtype=np.int64)
    np.save(pairs_path, pairs)
    print(f"pairs: {pairs.shape}", flush=True)

# ---------------- stage 2+3: features + labels (chunked over pairs)
X_path, y_path = f"{OUT}/X.npy", f"{OUT}/y.npy"
if os.path.exists(X_path) and os.path.exists(y_path) and \
   len(np.load(X_path, mmap_mode="r")) == len(pairs):
    X = np.load(X_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    print(f"reused X {X.shape} y {y.shape}", flush=True)
else:
    ism = np.memmap("data/real_5m.ism", dtype=np.uint8, mode="r", offset=8, shape=(N_ALL, 64))
    f16 = np.memmap("data/floats_5m.f16", dtype=np.float16, mode="r", offset=32, shape=(N_ALL, DIM))
    n = len(pairs)
    X = np.zeros((n, 74), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    CH = 200_000
    t0 = time.time()
    for s in range(0, n, CH):
        e = min(s + CH, n)
        q = np.asarray(ism[pairs[s:e, 0]])
        c = np.asarray(ism[pairs[s:e, 1]])
        X[s:e] = pair_features(q, c)
        vq = np.asarray(f16[pairs[s:e, 0]]).astype(np.float32)
        vc = np.asarray(f16[pairs[s:e, 1]]).astype(np.float32)
        y[s:e] = np.einsum("ij,ij->i", vq, vc)
        if (s // CH) % 5 == 0:
            print(f"  feats {e}/{n} ({time.time()-t0:.0f}s)", flush=True)
    np.save(X_path, X)
    np.save(y_path, y)
    print(f"X {X.shape}, y mean {y.mean():.4f}", flush=True)

# ---------------- stage 4: train
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
model_path = f"{OUT}/model.joblib"
if os.path.exists(model_path) and os.environ.get("A_SKIP_PAIRS") == "1":
    model = joblib.load(model_path)
    print("reused model", flush=True)
else:
    t0 = time.time()
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                          max_leaf_nodes=63, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=SEED)
    model.fit(X, y)
    joblib.dump(model, model_path)
    print(f"trained in {time.time()-t0:.0f}s; train MSE {np.mean((model.predict(X[:50000]) - y[:50000])**2):.5f}", flush=True)

# ---------------- stage 5: eval
qhash = np.load("/tmp/hybrid5m/qhash.npy")[:N_EVAL]
qemb = np.load("/tmp/hybrid5m/qemb.npy")[:N_EVAL]
gt = np.load("/tmp/hybrid5m/gt_top100_ids.npy")[:N_EVAL]
cand = np.load("/tmp/hybrid5m/cand_ids.npy")[:N_EVAL, :TOPK]  # HNSW order top-50
ism = np.memmap("data/real_5m.ism", dtype=np.uint8, mode="r", offset=8, shape=(N_ALL, 64))
f16 = np.memmap("data/floats_5m.f16", dtype=np.float16, mode="r", offset=32, shape=(N_ALL, DIM))

r1_h, r10_h = [], []
r1_f, r10_f = [], []
r1_l, r10_l = [], []
overhead = []
# warm-up pass: touch candidate-hash pages + model caches so the timed
# overhead reflects production (hashes resident, stored inside the graph)
for i in range(min(N_EVAL, 200)):
    ids = cand[i]
    ch = np.asarray(ism[ids])
    feats = pair_features(np.broadcast_to(qhash[i], (len(ids), 64)), ch)
    model.predict(feats)
t0 = time.time()
for i in range(N_EVAL):
    g10 = set(int(x) for x in gt[i, :10])
    g1 = int(gt[i, 0])
    ids = cand[i]
    # (1) HNSW order
    r1_h.append(int(ids[0] == g1))
    r10_h.append(len(set(int(x) for x in ids[:10]) & g10) / 10)
    # (2) exact float re-rank (production path)
    vq = np.asarray(f16[ids]).astype(np.float32)
    sims = vq @ qemb[i]
    ord_f = ids[np.argsort(-sims, kind="stable")]
    r1_f.append(int(ord_f[0] == g1))
    r10_f.append(len(set(int(x) for x in ord_f[:10]) & g10) / 10)
    # (3) learned re-rank + overhead timing
    tl = time.perf_counter_ns()
    ch = np.asarray(ism[ids])
    feats = pair_features(np.broadcast_to(qhash[i], (len(ids), 64)), ch)
    pred = model.predict(feats)
    overhead.append(time.perf_counter_ns() - tl)
    ord_l = ids[np.argsort(-pred, kind="stable")]
    r1_l.append(int(ord_l[0] == g1))
    r10_l.append(len(set(int(x) for x in ord_l[:10]) & g10) / 10)
    if (i + 1) % 200 == 0:
        print(f"  eval {i+1}/{N_EVAL} ({time.time()-t0:.0f}s)", flush=True)

m = lambda a: float(np.mean(a))
_pct = lambda r: "undefined (float gain ~0)" if r is None else f"{r*100:.1f}%"
overhead = np.array(overhead) / 1e6
_d1 = m(r1_f) - m(r1_h)
_d10 = m(r10_f) - m(r10_h)
rec_r1 = float((m(r1_l) - m(r1_h)) / _d1) if _d1 >= 0.02 else None
rec_r10 = float((m(r10_l) - m(r10_h)) / _d10) if _d10 >= 0.02 else None
res = dict(
    n_eval=N_EVAL, topk=TOPK, n_train_pairs=int(len(X)), seed=SEED,
    hnsw=dict(R1=m(r1_h), R10=m(r10_h)),
    float_rerank=dict(R1=m(r1_f), R10=m(r10_f)),
    learned=dict(R1=m(r1_l), R10=m(r10_l),
                 overhead_p50_ms=float(np.median(overhead)),
                 overhead_p95_ms=float(np.percentile(overhead, 95))),
    gain_recovery=dict(R1=rec_r1, R10=rec_r10),
)
_r1 = rec_r1 if rec_r1 is not None else 0.0
decision = ("VALIDATED" if _r1 >= 0.80 and res["learned"]["overhead_p50_ms"] <= 1.0 else
            "PARTIAL" if _r1 >= 0.50 else "KILLED")
res["decision"] = decision
# marginal model cost (sklearn 1.9 predict has ~17ms per-call constant overhead;
# measure marginal per-row cost to characterize an optimized serving path)
_t0 = time.perf_counter_ns()
model.predict(np.zeros((50000, 74), dtype=np.float32))
_marg_s = (time.perf_counter_ns() - _t0) / 1e9
res["learned"]["sklearn_percall_overhead_ms"] = res["learned"]["overhead_p50_ms"]
res["learned"]["marginal_predict_ms_per_1000rows"] = _marg_s
json.dump(res, open(f"{OUT}/learned_rerank_result.json", "w"), indent=2)
print(json.dumps(res, indent=2))

# ---------------- stage 6: report + proof entry
md = f"""# Experiment A — Learned hash-space re-ranker (5.1M, text-space queries)

- Date: {DATE}; seed {SEED}; corpus data/real_5m_hnsw_v5.bin + floats_5m.f16
- Training: {len(X):,} (query-hash, candidate-hash, exact-cosine) pairs from
  {N_TRAIN_DOCS:,} random docs x top-100 HNSW neighbors (self included);
  74 features (64 per-byte XOR popcounts + 8 block popcounts + hamming +
  hamming/512); sklearn HistGradientBoostingRegressor (no lightgbm in venv).
- Eval: {N_EVAL} text-space queries ({'/tmp/hybrid5m'}, seed 20260916),
  include-self, top-{TOPK} candidates from Entry 59 checkpoints.

| System | R@1 | R@10 | overhead p50 ms |
|---|---|---|---|
| HNSW order (baseline) | {m(r1_h):.4f} | {m(r10_h):.4f} | 0 |
| Exact float re-rank top-{TOPK} (production) | {m(r1_f):.4f} | {m(r10_f):.4f} | (float store required) |
| Learned hash-space re-rank | {m(r1_l):.4f} | {m(r10_l):.4f} | {res['learned']['overhead_p50_ms']:.3f} (p95 {res['learned']['overhead_p95_ms']:.3f}) |

Gain recovery vs float re-rank: R@1 {_pct(rec_r1)}, R@10 {_pct(rec_r10)}.
Zero float storage; features are pure hash XORs.

Pre-registered decision: **{decision}** (>=80% recovery & <=1ms -> VALIDATED;
50-80% -> partial; <50% -> KILLED). R@1 recovery {_pct(rec_r1)},
overhead {res['learned']['overhead_p50_ms']:.3f} ms.

Caveat: training pseudo-queries are corpus docs (hash-space distribution),
eval queries are fresh text encodes; hash-space features transfer by design
but this is a documented distribution gap.
"""
os.makedirs("benchmark_results", exist_ok=True)
open(f"benchmark_results/learned_rerank_5m_{DATE}.md", "w").write(md)
json.dump(res, open(f"benchmark_results/learned_rerank_5m_{DATE}.json", "w"), indent=2)

entry = f"""
Entry 61: Experiment A — Learned Hash-Space Re-Ranker ({decision})
Date: {DATE}
Git Hash: engine worktree 62eb8ae1 (+ uncommitted; scripts + node_label accessor)
Status: MEASURED ON DESKTOP (M3-class, single-threaded eval).
  5.1M corpus; {len(X):,} training pairs ({N_TRAIN_DOCS}K docs x top-100 HNSW,
  labels = exact float cosine from floats_5m.f16); 74 XOR-popcount features;
  sklearn HistGradientBoosting (no lightgbm available). Eval on 1000
  text-space queries (Entry 59 checkpoints, include-self, top-{TOPK}):
  HNSW R@1 {m(r1_h):.4f} R@10 {m(r10_h):.4f}; float re-rank R@1 {m(r1_f):.4f}
  R@10 {m(r10_f):.4f}; learned R@1 {m(r1_l):.4f} R@10 {m(r10_l):.4f} at
  {res['learned']['overhead_p50_ms']:.3f}ms p50 overhead. Gain recovery
  R@1 {_pct(rec_r1)} / R@10 {_pct(rec_r10)}; zero float storage.
  Pre-registered decision: {decision}.
Artifacts: benchmark_results/learned_rerank_5m_{DATE}.{{json,md}},
  scripts/learned_rerank_5m.py, /tmp/learned_rerank/ (pairs, X, y, model)
Next: {'product integration eval + paper section' if decision == 'VALIDATED' else 'per decision rule'}
"""
guard = f"{OUT}/entry61_appended"
if not os.path.exists(guard):
    with open("logs/proof/proof_of_life_chain.txt", "a") as f:
        f.write(entry)
    open(guard, "w").write("done")
    print("report + Entry 61 written; decision:", decision)
else:
    print("report regenerated; Entry 61 already in chain (guard); decision:", decision)
