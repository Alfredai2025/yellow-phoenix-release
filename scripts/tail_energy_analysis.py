#!/usr/bin/env python3
"""Tail-energy analysis for kill-the-floats feasibility (Contribution A2).

READ-ONLY: loads data/floats_5m.f16 via memmap, never writes project files.
Outputs go to benchmark_results/tail_energy_*.json and stdout.

Quarantine: corpus float rows are the TRAINING signal. The frozen text-space
exam (/tmp/hybrid5m, 1000 queries) is NEVER touched here. Recall-sim queries
are corpus docs held out from the corpus half (hash-space distribution) --
documented distribution gap, same caveat as Entry 61.

Decision rule (pre-registered, from the A2 brief):
  k <= 128 and tail <= 5%  -> kill-the-floats feasible (3.8GB -> ~350-500MB)
  k 128-200, tail <= 5%    -> feasible, ship with distilled scorer
  k > 200 or tail > 10%    -> embeddings fatter than assumed; hybrid early

Usage:
  python3 tail_energy_analysis.py              # Part 1: spectrum + tail table
  python3 tail_energy_analysis.py --recall-sim # Part 2: + truncation R@10 sim
"""
import json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
F16 = ROOT / "data/floats_5m.f16"
OUT = ROOT / "benchmark_results"
N_ALL, DIM, OFF = 5107508, 384, 32
SEED = 20260916
KS = [32, 48, 64, 96, 128, 160, 192, 256, 320]

rng = np.random.default_rng(SEED)
X = np.memmap(F16, dtype=np.float16, mode="r", offset=OFF, shape=(N_ALL, DIM))

# ---------- Part 1: covariance spectrum (exact; 384x384 eigh) ----------
t0 = time.time()
# sample rows for the spectrum estimate (1M samples is plenty for 384 dims)
n_spec = 1_000_000
idx = rng.choice(N_ALL, size=n_spec, replace=False)
Xs = np.empty((n_spec, DIM), dtype=np.float32)
CH = 100_000
for i in range(0, n_spec, CH):
    j = min(i + CH, n_spec)
    Xs[i:j] = X[idx[i:j]].astype(np.float32)
    if (i // CH) % 5 == 0:
        print(f"spec load {j}/{n_spec} ({time.time()-t0:.0f}s)", flush=True)

mu = Xs.mean(axis=0)
Xc = Xs - mu
cov = (Xc.T @ Xc) / n_spec
evals, evecs = np.linalg.eigh(cov)  # ascending
evals = evals[::-1]
evecs = evecs[:, ::-1]
sv = np.sqrt(np.maximum(evals, 0))
total = float((sv**2).sum())
tail = {k: float(np.sqrt((sv[k:] ** 2).sum() / total)) for k in KS}
# effective rank (participation ratio)
eff_rank = float((evals.sum()) ** 2 / (evals**2).sum())
k_at = {}
for thr in (0.01, 0.02, 0.05, 0.10):
    cum = np.cumsum(sv**2) / total
    k_at[f"tail<={int(thr*100)}%"] = int(np.searchsorted(cum, 1 - thr**2)) + 1

print("\n=== TAIL ENERGY (raw PCA branch) ===")
print(f"effective rank: {eff_rank:.1f} of {DIM}")
for k in KS:
    print(f"  k={k:4d}  tail={tail[k]*100:6.2f}%")
print(f"k recommendation: {json.dumps(k_at)}")

res = {
    "date": "20260916", "seed": SEED, "n_spec": n_spec,
    "effective_rank": eff_rank, "tail_energy_pct": {str(k): round(v*100, 3) for k, v in tail.items()},
    "k_at_thresholds": k_at,
    "top20_sv": sv[:20].tolist(),
}
OUT.mkdir(exist_ok=True)
(OUT / "tail_energy_20260916.json").write_text(json.dumps(res, indent=2))

# ---------- Part 2 (optional): truncation-only recall sim, train-internal ----------
if "--recall-sim" not in sys.argv:
    print("\ndone (add --recall-sim for truncation R@10). elapsed", round(time.time()-t0), "s")
    sys.exit(0)

t1 = time.time()
N_CORPUS, N_Q = 100_000, 1_000
# disjoint query/corpus split from the SAME sampled pool used above (held-out queries)
sim_idx = idx[: N_CORPUS + N_Q]           # reuse first 101k of the 1M sample
q_rows, c_rows = sim_idx[:N_Q], sim_idx[N_Q:]
C = Xs[N_Q : N_Q + N_CORPUS]              # matches sim_idx[N_Q:] order
Q = Xs[:N_Q]

def l2n(M):
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)

Qn, Cn = l2n(Q), l2n(C)
gt = (Qn @ Cn.T).argpartition(-10, axis=1)[:, -10:]  # top-10 exact (set; ties fine at this scale)
gt_sets = [set(r.tolist()) for r in gt]

print(f"\n=== RECALL SIM ({time.time()-t1:.0f}s in) ===  corpus={N_CORPUS} queries={N_Q}")
sim = {}
V = evecs  # PCs from Part 1
for k in KS:
    Vk = V[:, :k]
    Qr = l2n((Q - mu) @ Vk)          # truncated+renormalized queries
    Cr = l2n((C - mu) @ Vk)
    approx = (Qr @ Cr.T).argpartition(-10, axis=1)[:, -10:]
    r10 = np.mean([len(set(r.tolist()) & g) / 10 for r, g in zip(approx, gt_sets)])
    sim[k] = round(float(r10), 4)
    print(f"  k={k:4d}  trunc-only R@10={r10:.4f}   (elapsed {time.time()-t1:.0f}s)", flush=True)

# residual distribution: ||x - reconstruct_k(x)|| for k=128 (tail-fallback fraction)
k0 = 128
Vk = V[:, :k0]
rec = (Xc[:200_000] @ Vk) @ Vk.T
resid = np.linalg.norm(Xc[:200_000] - rec, axis=1)
resid_pct = {str(p): round(float(np.percentile(resid, p)), 4) for p in (50, 90, 95, 99, 99.5, 99.9)}

print("\nresidual norm pctiles (k=128):", json.dumps(resid_pct))
print("exact-baseline note: GT here is exact cosine on the 100k corpus half;")
print("R@10 of trunc rows is the LOWER bound (int8 factor quantization not simulated).")

res["recall_sim"] = {"corpus": N_CORPUS, "queries": N_Q, "r10_trunc_only": sim,
                     "resid_pctiles_k128": resid_pct}
(OUT / "tail_energy_20260916.json").write_text(json.dumps(res, indent=2))
print("\nWROTE", OUT / "tail_energy_20260916.json", " total elapsed", round(time.time()-t0), "s")
