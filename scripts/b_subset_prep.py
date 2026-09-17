#!/usr/bin/env python3
"""Experiment B subset prep: sample a subset of the 5.1M corpus, write a
sequential-id ISM (position space), the position->original-id map, the query
ISM (500 text-space queries from /tmp/hybrid5m), and GT top-10 within the
subset (in position space).

Usage: b_subset_prep.py <N_SUBSET> <OUTDIR>
"""
import os, sys, json, time
import numpy as np

ROOT = "/Users/mac/yellow_phoenix"
os.chdir(ROOT)

N_ALL = 5107508
DIM = 384
SEED = 20260916

n_sub = int(sys.argv[1])
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)

rng = np.random.default_rng(SEED)
# position-space: subset positions ARE 0..n_sub-1; subset_pos = chosen row indices
pos = np.sort(rng.choice(N_ALL, size=n_sub, replace=False))  # positions in 5.1M files
np.save(f"{outdir}/subset_rows.npy", pos)  # position -> original 5.1M row/id

ism = np.memmap("data/real_5m.ism", dtype=np.uint8, mode="r", offset=8, shape=(N_ALL, 64))
hashes = np.ascontiguousarray(ism[pos])
with open(f"{outdir}/subset.ism", "wb") as f:
    f.write(np.array([n_sub], dtype="<u8").tobytes())
    f.write(hashes.tobytes())
    f.write(np.arange(n_sub, dtype="<u8").tobytes())  # sequential ids = position
print(f"wrote {outdir}/subset.ism ({n_sub} docs)")

# queries: first 500 text-space queries
qhash = np.load("/tmp/hybrid5m/qhash.npy")[:500]
pids = np.load("/tmp/hybrid5m/pids.npy")[:500]
with open(f"{outdir}/queries.ism", "wb") as f:
    f.write(np.array([len(qhash)], dtype="<u8").tobytes())
    f.write(np.ascontiguousarray(qhash).tobytes())
    f.write(np.arange(len(qhash), dtype="<u8").tobytes())
np.save(f"{outdir}/query_pids.npy", pids)

# GT top-10 within subset (position space), by float cosine of query emb
qemb = np.load("/tmp/hybrid5m/qemb.npy")[:500]
f16 = np.memmap("data/floats_5m.f16", dtype=np.float16, mode="r", offset=32, shape=(N_ALL, DIM))
CH = 200_000
gt = np.zeros((len(qemb), 10), dtype=np.uint64)
best = np.full((len(qemb), 10), -np.inf, dtype=np.float32)
t0 = time.time()
for s in range(0, n_sub, CH):
    e = min(s + CH, n_sub)
    rows = pos[s:e]
    c = np.asarray(f16[rows]).astype(np.float32)
    sims = qemb @ c.T
    idx = np.argpartition(sims, -10, axis=1)[:, -10:]
    r = np.take_along_axis(sims, idx, axis=1).argsort(axis=1)[:, ::-1]
    idx = np.take_along_axis(idx, r, axis=1)
    sim = np.take_along_axis(sims, idx, axis=1)
    ji = np.concatenate([gt, (idx + s).astype(np.uint64)], axis=1)
    js = np.concatenate([best, sim], axis=1)
    k = np.argpartition(js, -10, axis=1)[:, -10:]
    kr = np.take_along_axis(js, k, axis=1).argsort(axis=1)[:, ::-1]
    k = np.take_along_axis(k, kr, axis=1)
    gt = np.take_along_axis(ji, k, axis=1)
    best = np.take_along_axis(js, k, axis=1)
gt.tofile(f"{outdir}/gt.bin")
in_subset = np.array([int(p) in set(pos.tolist()) for p in pids])
json.dump({"n_subset": n_sub, "seed": SEED, "n_queries": len(qemb),
           "gt_wall_s": time.time() - t0,
           "query_source_self_in_subset_frac": float(in_subset.mean())},
          open(f"{outdir}/subset_meta.json", "w"), indent=2)
print(f"GT done ({time.time()-t0:.0f}s), query self-in-subset frac={in_subset.mean():.3f}")
