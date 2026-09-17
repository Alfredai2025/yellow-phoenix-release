#!/usr/bin/env python3
"""Entry 63 — Matched-bytes final: int8 low-rank factors vs OPQ-style PQ.

Closes the kill-the-floats line (Contribution A2). Question: at equal bytes/doc,
does truncating to k int8 PCA dims beat product-quantizing all 384 dims, when
re-ranking the SAME HNSW top-50 candidates on the FROZEN exam?

Arms (per byte budget B in {64, 128, 192}):
  pqB    : PCA-rotate -> kmeans per subspace (256 centroids), B codes/doc
  facB   : PCA-rotate -> keep top-B dims, per-dim int8 (shared scale), B bytes/doc
  float16: production re-rank upper bound (768 bytes/doc) — reference only

Quantizers train on corpus float rows ONLY (random 600k sample). The frozen
exam (/tmp/hybrid5m: 1000 text queries, GT top-100 exact, HNSW cand top-200)
is used strictly for evaluation. Same candidate slice as Entry 61 (top-50).

PRE-REGISTERED DECISION RULE:
  A2 SURVIVES if factors beat PQ by >=3pp R@10 at ANY matched budget
  A2 CLOSED   if PQ >= factors at every budget (line dies, paper paragraph)
  (being within noise: |diff| < 1pp counts as tie -> PQ wins on simplicity)

Read-only w.r.t. project data; writes only benchmark_results/matched_bytes_*.json
"""
import json, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
F16 = ROOT / "data/floats_5m.f16"
EXAM = Path("/tmp/hybrid5m")
OUT = ROOT / "benchmark_results"
N_ALL, DIM, OFF = 5107508, 384, 32
SEED, TOPK = 20260916, 50
N_TRAIN = 600_000
BUDGETS = [256]  # Entry 63 extension row: 64/128/192 already closed (PQ swept)
rng = np.random.default_rng(SEED)

X = np.memmap(F16, dtype=np.float16, mode="r", offset=OFF, shape=(N_ALL, DIM))

# ---------------- frozen exam ----------------
qemb = np.load(EXAM / "qemb.npy")                    # (1000, 384) float32 text queries
gt = np.load(EXAM / "gt_top100_ids.npy")[:, :10]     # (1000, 10)
cand = np.load(EXAM / "cand_ids.npy")[:, :TOPK]      # (1000, 50) HNSW order
gt_sets = [set(r.tolist()) for r in gt]

def r_at_k(ranked_ids, k):
    return float(np.mean([len(set(r[:k].tolist()) & g) / k for r, g in zip(ranked_ids, gt_sets)]))

# ---------------- load train sample + PCA rotation ----------------
t0 = time.time()
tr_idx = rng.choice(N_ALL, size=N_TRAIN, replace=False)
tr = np.empty((N_TRAIN, DIM), dtype=np.float32)
CH = 100_000
for i in range(0, N_TRAIN, CH):
    j = min(i + CH, N_TRAIN)
    tr[i:j] = X[tr_idx[i:j]].astype(np.float32)
print(f"train sample loaded ({time.time()-t0:.0f}s)", flush=True)

mu = tr.mean(axis=0)
cov = ((tr - mu).T @ (tr - mu)) / N_TRAIN
evals, V = np.linalg.eigh(cov)
V = V[:, ::-1]                                        # PCs descending
R = V.T                                               # rotation x_rot = (x-mu) @ V
tr_rot = (tr - mu) @ V
print(f"PCA done ({time.time()-t0:.0f}s)", flush=True)

# ---------------- candidate + query corpus ----------------
cand_flat = np.unique(cand)                           # union of all candidates
C = X[cand_flat].astype(np.float32)                   # encode target docs
q_rot = (qemb - mu) @ V
C_rot = (C - mu) @ V
print(f"corpus encoded: {len(cand_flat)} unique candidates ({time.time()-t0:.0f}s)", flush=True)
def rank_by_scores(scores):
    # scores: (nq, ncand) over C rows -> top-10 doc ids
    order = scores.argsort(axis=1)[:, ::-1][:, :10]
    return cand_flat[order]

# ---------------- float16 reference ----------------
Cf = X[cand_flat].astype(np.float32)
qf = qemb / np.maximum(np.linalg.norm(qemb, axis=1, keepdims=True), 1e-12)
Cf_n = Cf / np.maximum(np.linalg.norm(Cf, axis=1, keepdims=True), 1e-12)
ref_rank = rank_by_scores(qf @ Cf_n.T)
r1_ref, r10_ref = r_at_k(ref_rank, 1), r_at_k(ref_rank, 10)
print(f"float16 ref: R@1={r1_ref:.4f} R@10={r10_ref:.4f}  ({time.time()-t0:.0f}s)", flush=True)

# ---------------- arms ----------------
results = {}
for B in BUDGETS:
    # --- facB: top-B dims int8, shared per-dim scale from train sample ---
    s = np.abs(tr_rot[:, :B]).max(axis=0) / 127.0
    s = np.maximum(s, 1e-12)
    q_fac = np.round(q_rot[:, :B] / s).astype(np.float32) * s
    C_fac = np.round(C_rot[:, :B] / s).astype(np.float32) * s
    # cosine on reconstructed-then-renormalized vectors
    qn = q_fac / np.maximum(np.linalg.norm(q_fac, axis=1, keepdims=True), 1e-12)
    Cn = C_fac / np.maximum(np.linalg.norm(C_fac, axis=1, keepdims=True), 1e-12)
    r_fac = rank_by_scores(qn @ Cn.T)
    fac = {"r1": r_at_k(r_fac, 1), "r10": r_at_k(r_fac, 10)}

    # --- pqB: PCA-rotated product quantizer ---
    # subspace widths: spread DIM across B subs (first DIM%B subs get one extra dim)
    base, rem = divmod(DIM, B)
    widths = [base + (1 if s < rem else 0) for s in range(B)]
    cols_list = []
    st = 0
    for w in widths:
        cols_list.append(slice(st, st + w))
        st += w
    centroids = []
    codes_c = np.empty((len(C_rot), B), dtype=np.uint8)
    for sub, cols in enumerate(cols_list):
        sd = widths[sub]
        pts = tr_rot[rng.choice(N_TRAIN, size=200_000, replace=False), cols]
        cent = pts[rng.choice(len(pts), 256, replace=False)].copy()
        for _ in range(12):
            d2 = ((pts[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
            a = d2.argmin(axis=1)
            cnt = np.bincount(a, minlength=256).astype(np.float32)
            newc = cent.copy()
            for dcol in range(sd):
                newc[:, dcol] = np.bincount(a, weights=pts[:, dcol], minlength=256) / np.maximum(cnt, 1)
            cent = np.where((cnt > 0)[:, None], newc, cent)
        centroids.append(cent)
        d2 = ((C_rot[:, None, cols] - cent[None, :, :]) ** 2).sum(axis=2)
        codes_c[:, sub] = d2.argmin(axis=1)
    # ADC scoring: per-query per-sub distance table, then gather over codes
    scores = np.zeros((len(q_rot), len(C_rot)), dtype=np.float32)
    for sub, cols in enumerate(cols_list):
        qtab = ((q_rot[:, None, cols] - centroids[sub][None, :, :]) ** 2).sum(axis=2)  # (nq, 256)
        scores += qtab[:, codes_c[:, sub]]          # (nq, 256) x codes (ncand,) -> (nq, ncand)
    r_pq = rank_by_scores(-scores)                   # PQ scores are distances
    pq = {"r1": r_at_k(r_pq, 1), "r10": r_at_k(r_pq, 10)}
    print(f"  B={B}: fac R@10={fac['r10']:.4f}  pq R@10={pq['r10']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    results[f"fac{B}"] = fac
    results[f"pq{B}"] = pq

# ---------------- decision ----------------
summary = {"float16_ref": {"r1": r1_ref, "r10": r10_ref, "bytes_per_doc": 768}, "arms": {},
           "date": "20260916", "seed": SEED, "topk": TOPK, "budgets": BUDGETS}
verdicts = []
for B in BUDGETS:
    f, p = results[f"fac{B}"], results[f"pq{B}"]
    diff = f["r10"] - p["r10"]
    if diff >= 0.03:
        v = "FACTORS_WIN"
    elif abs(diff) < 0.01:
        v = "TIE_PQ_WINS"
    else:
        v = "PQ_WINS"
    verdicts.append(v)
    summary["arms"][f"B={B}"] = {"fac": f, "pq": p, "diff_r10": round(diff, 4), "verdict": v}
summary["decision"] = "A2_SURVIVES" if any(v == "FACTORS_WIN" for v in verdicts) else "A2_CLOSED"

print(json.dumps(summary, indent=2))
OUT.mkdir(exist_ok=True)
(OUT / "matched_bytes_a2_20260916.json").write_text(json.dumps(summary, indent=2))
print("WROTE", OUT / "matched_bytes_a2_20260916.json")
print("DECISION:", summary["decision"])
