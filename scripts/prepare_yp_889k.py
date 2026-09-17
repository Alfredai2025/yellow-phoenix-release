#!/usr/bin/env python3
"""Prepare YP@889K bench inputs, EXACTLY matching bench_mac_baselines_889k.py:
same N, same seeded held-out query rows, same corpus rows.

Outputs:
  data/yp_889k.ism          FlatIndex ISM: [u64 count][count x 64B ITQ hashes][count x 8B ids]
  data/yp_889k_queries.ism  same layout, 256 query hashes (ids = row indices)
  benchmark_results/yp_889k_gt.json  exact float-cosine top-10 per query (GT for recall)
"""
import os, json
import numpy as np

N = 889_000
DIM = 384
Q = 256
SEED = 20260903
TOPK = 10

data = np.memmap(os.path.expanduser("~/yellow_phoenix/data/embeddings_3m.npy"),
                 dtype=np.float32, mode="r").reshape(-1, DIM)
assert data.shape[0] >= N + Q

# IDENTICAL sampling to bench_mac_baselines_889k.py
rng = np.random.default_rng(SEED)
qidx = rng.choice(N, size=Q, replace=False)

m = np.load(os.path.expanduser("~/yellow_phoenix/data/itq_model_512_fixed.npz"))
mean = m["mean"].astype(np.float32)
proj = m["proj"].astype(np.float32)

def bits(x):
    return np.packbits(((x - mean) @ proj) > 0, axis=1)  # n x 64

base = np.asarray(data[:N])
qb = np.asarray(base[qidx])

out_ism = os.path.expanduser("~/yellow_phoenix/data/yp_889k.ism")
with open(out_ism, "wb") as f:
    f.write(np.uint64(N).tobytes())
    f.write(bits(base).tobytes())
    f.write(np.arange(N, dtype=np.uint64).tobytes())
print("wrote", out_ism, os.path.getsize(out_ism))

out_q = os.path.expanduser("~/yellow_phoenix/data/yp_889k_queries.ism")
with open(out_q, "wb") as f:
    f.write(np.uint64(Q).tobytes())
    f.write(bits(qb).tobytes())
    f.write(np.arange(Q, dtype=np.uint64).tobytes())
print("wrote", out_q, os.path.getsize(out_q))

# exact float GT (cosine) top-10, chunked
qn = qb / (np.linalg.norm(qb, axis=1, keepdims=True) + 1e-12)
best_s = np.full((Q, TOPK), -np.inf, dtype=np.float32)
best_i = np.full((Q, TOPK), -1, dtype=np.int64)
CH = 50_000
for lo in range(0, N, CH):
    hi = min(lo + CH, N)
    x = base[lo:hi]
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    s = qn @ xn.T                                    # Q x k
    idx = np.argpartition(-s, TOPK, axis=1)[:, :TOPK]
    cs = np.take_along_axis(s, idx, 1)
    ii = idx + lo
    ms = np.concatenate([best_s, cs], axis=1)
    mi = np.concatenate([best_i, ii], axis=1)
    top = np.argpartition(-ms, TOPK, axis=1)[:, :TOPK]
    best_s = np.take_along_axis(ms, top, 1)
    best_i = np.take_along_axis(mi, top, 1)
order = np.argsort(-best_s, axis=1)
gt = np.take_along_axis(best_i, order, 1)
out_gt = os.path.expanduser("~/yellow_phoenix/benchmark_results/yp_889k_gt.json")
with open(out_gt, "w") as f:
    json.dump({"n": N, "q": Q, "seed": SEED, "gt_top10": gt.tolist()}, f)
print("wrote", out_gt)
print("GT sample row0:", gt[0][:5].tolist())
