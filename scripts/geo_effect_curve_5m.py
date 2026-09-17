#!/usr/bin/env python3
"""Phase 0 pre-registered screen: region-wise effect curve for the geometric
layer (tensor_spectral re-rank) on the 5.1M corpus, using ONLY checkpointed
artifacts from /tmp/hybrid5m/ (Entry 59 run).

Per-query features:
  tie_ratio        fraction of top-100 HNSW candidates at the minimum
                   Hamming distance within the top-100
  spectral_gap     eigenvalue-decay slope of the query's local neighborhood:
                   hamming k-means (k=64) over a 200K-doc subsample of the
                   5.1M ITQ hashes; query assigned to nearest centroid; the
                   cluster's slope = OLS slope of log(eigenvalue) over the top
                   32 eigenvalues of the centered bit covariance (512x512,
                   numpy eigh); query inherits its cluster's slope.
  hash_entropy     |ones(qhash) - 256|  (documented: deviation from balanced)
  candidate_spread variance of the 100 Hamming distances
Per-query outcome:
  delta = |GT[:10] ∩ top10(cascade order)| - |GT[:10] ∩ top10(HNSW order)|
  where cascade order = same 100 candidates sorted by spectral distance asc;
  delta in [-10, +10].
Decision rule (pre-registered): any bin/cell with >=100 queries and mean delta
> +1.0 -> PRECEDENT (router justified); all bins <= +0.5 -> REFUTED (museum).
"""
import os, json, time
import numpy as np

ROOT = "/Users/mac/yellow_phoenix"
OUT = "/tmp/hybrid5m"
os.chdir(ROOT)

N_Q = 1000
K = 100
N = 5107508
SEED = 20260916

pids = np.load(f"{OUT}/pids.npy")
qhash = np.load(f"{OUT}/qhash.npy")
gt = np.load(f"{OUT}/gt_top100_ids.npy")
cand_ids = np.load(f"{OUT}/cand_ids.npy")      # (1000,100) HNSW order, includes self
cand_dist = np.load(f"{OUT}/cand_scores.npy")  # (1000,100) spectral squared distance, smaller=nearer
ism = np.memmap(f"{ROOT}/data/real_5m.ism", dtype=np.uint8, mode="r", offset=8, shape=(N, 64))

# ---------------- candidate hamming distances
t0 = time.time()
hd = np.zeros((N_Q, K), dtype=np.int16)  # hamming distance query->candidate
for i in range(N_Q):
    ch = np.asarray(ism[cand_ids[i]])              # (100,64)
    hd[i] = np.unpackbits(ch ^ qhash[i], axis=1).sum(axis=1)
print(f"hamming dists: {time.time()-t0:.1f}s")

# ---------------- features
tie_ratio = (hd == hd.min(axis=1, keepdims=True)).mean(axis=1)
tie_count = (hd == hd.min(axis=1, keepdims=True)).sum(axis=1)  # 1..100, granularity fix
cand_spread = hd.var(axis=1)
hash_ones = np.unpackbits(qhash, axis=1).sum(axis=1).astype(np.int64)
hash_entropy = np.abs(hash_ones - 256)

# ---------------- spectral_gap: hamming k-means (k=64) on 200K subsample
rng = np.random.default_rng(SEED)
sub_idx = rng.choice(N, size=200_000, replace=False)
sub = np.unpackbits(np.asarray(ism[sub_idx]), axis=1)  # (200K, 512) uint8 0/1
KC = 64
cent = sub[rng.choice(200_000, size=KC, replace=False)].astype(np.int16)  # int for majority
assign = np.zeros(200_000, dtype=np.int64)
for it in range(8):
    t_it = time.time()
    for s in range(0, 200_000, 10_000):
        e = min(s + 10_000, 200_000)
        # hamming via XOR + popcount lookup
        x = sub[s:e, None, :] != cent[None, :, :]     # (chunk,64,512) bool
        d = x.sum(axis=2)                              # popcount
        assign[s:e] = d.argmin(axis=1)
    # recompute centroids = per-bit majority
    newc = np.zeros((KC, 512), dtype=np.int16)
    cnts = np.bincount(assign, minlength=KC)
    for c in range(KC):
        m = assign == c
        if m.any():
            newc[c] = (sub[m].sum(axis=0) >= (m.sum() / 2)).astype(np.int16)
        else:
            newc[c] = cent[c]
    shift = int((newc != cent).sum())
    cent = newc
    print(f"kmeans it{it}: shifted {shift} centroid-bits, sizes[{cnts.min()}-{cnts.max()}], {time.time()-t_it:.0f}s")
    if shift == 0:
        break

# per-cluster eigen-decay slope on centered bit covariance (all 200K subsample docs)
slopes = np.full(KC, np.nan)
for c in range(KC):
    m = assign == c
    X = sub[m].astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / len(X)
    ev = np.linalg.eigvalsh(cov)[::-1][:32]
    ev = np.maximum(ev, 1e-12)
    slope = np.polyfit(np.arange(32), np.log(ev), 1)[0]
    slopes[c] = slope
print("cluster slopes:", np.round(slopes, 4))

# assign each query to nearest centroid; inherit slope
qs = np.unpackbits(qhash, axis=1).astype(np.int16)
qassign = np.zeros(N_Q, dtype=np.int64)
for s in range(0, N_Q, 200):
    e = min(s + 200, N_Q)
    x = qs[s:e, None, :] != cent[None, :, :]
    qassign[s:e] = x.sum(axis=2).argmin(axis=1)
spectral_gap = slopes[qassign]

# ---------------- delta
delta = np.zeros(N_Q)
for i in range(N_Q):
    g10 = set(int(x) for x in gt[i, :10])
    h10 = set(int(x) for x in cand_ids[i, :10])
    c_ord = cand_ids[i][np.argsort(cand_dist[i], kind="stable")]
    c10 = set(int(x) for x in c_ord[:10])
    delta[i] = len(g10 & c10) - len(g10 & h10)
print(f"delta: mean={delta.mean():.4f} (overall cascade effect, matches Entry 59)")

# ---------------- binning
def bin_table(feat, name, nbins):
    edges = np.quantile(feat, np.linspace(0, 1, nbins + 1))
    edges[0] -= 1e-9
    rows = []
    for b in range(nbins):
        m = (feat > edges[b]) & (feat <= edges[b + 1]) if b > 0 else (feat <= edges[b + 1])
        n = int(m.sum())
        if n == 0:
            rows.append(dict(bin=b, n=0)); continue
        d = delta[m]
        rows.append(dict(bin=b, lo=float(edges[b]), hi=float(edges[b + 1]), n=n,
                         mean_delta=float(d.mean()),
                         ci95=float(1.96 * d.std(ddof=1) / np.sqrt(n))))
    return rows

tables = {}
for name, feat in [("tie_ratio", tie_ratio), ("spectral_gap", spectral_gap),
                   ("hash_entropy", hash_entropy), ("candidate_spread", cand_spread)]:
    tables[name + "_quartiles"] = bin_table(feat, name, 4)
    tables[name + "_deciles"] = bin_table(feat, name, 10)
# tie_ratio is quantized at 1/100 -> quartile bins degenerate; use tie_count cut
# (1 = unique argmin, 2, 3, >=4 tied candidates at min distance)
tc_edges = [0, 1, 2, 3, 100]
tie_rows = []
for b in range(4):
    m = (tie_count > tc_edges[b]) & (tie_count <= tc_edges[b + 1])
    n = int(m.sum())
    dd = delta[m]
    tie_rows.append(dict(bin=b, lo=tc_edges[b], hi=tc_edges[b + 1], n=n,
                         mean_delta=float(dd.mean()),
                         ci95=float(1.96 * dd.std(ddof=1) / np.sqrt(n))))
tables["tie_count_1_2_3_4plus"] = tie_rows

# 4x4 grid. NOTE: tie_ratio has ZERO variance on this corpus (all 1000 queries
# have tie_count==1: unique hamming argmin among top-100), so the hypothesized
# tie_ratio x spectral_gap grid is degenerate by construction. Substitute the
# only feature with both variance and a monotone effect: candidate_spread.
ge = np.quantile(cand_spread, [0, .25, .5, .75, 1]); se = np.quantile(spectral_gap, [0, .25, .5, .75, 1])
grid = []
for a in range(4):
    for b in range(4):
        m = (cand_spread > ge[a]) & (cand_spread <= ge[a + 1] + (1e-9 if a == 3 else 0)) & \
            (spectral_gap > se[b]) & (spectral_gap <= se[b + 1] + (1e-9 if b == 3 else 0))
        n = int(m.sum())
        row = dict(spread_bin=a, gap_bin=b, n=n)
        if n:
            d = delta[m]
            row.update(mean_delta=float(d.mean()), ci95=float(1.96 * d.std(ddof=1) / np.sqrt(n)))
        grid.append(row)

# ---------------- topic-cluster test (cs.LG / cs.CV)
import sqlite3
con = sqlite3.connect(f"{ROOT}/data/papers_5m.db")
cats = dict(con.execute(
    f"SELECT id, categories FROM meta WHERE id IN ({','.join('?' * N_Q)})",
    [int(p) for p in pids]).fetchall())
is_cslg = np.array([("cs.LG" in (cats.get(int(p)) or "")) for p in pids])
is_cscv = np.array([("cs.CV" in (cats.get(int(p)) or "")) for p in pids])
is_cs = np.array([("cs." in (cats.get(int(p)) or "")) for p in pids])
topic = dict(
    cs_LG_n=int(is_cslg.sum()), cs_LG_mean_delta=float(delta[is_cslg].mean()) if is_cslg.any() else None,
    cs_CV_n=int(is_cscv.sum()), cs_CV_mean_delta=float(delta[is_cscv].mean()) if is_cscv.any() else None,
    cs_any_n=int(is_cs.sum()), cs_any_mean_delta=float(delta[is_cs].mean()) if is_cs.any() else None,
    rest_mean_delta=float(delta[~is_cs].mean()),
    rest_ci95=float(1.96 * delta[~is_cs].std(ddof=1) / np.sqrt((~is_cs).sum())))

# ---------------- decision
max_bin = max((r.get("mean_delta", -99), r["n"], k) for k, t in tables.items() for r in t if r.get("n", 0) >= 100)
max_cell = max(((c.get("mean_delta", -99), c["n"]) for c in grid if c.get("n", 0) >= 100), default=(-99, 0))
decision = "PRECEDENT" if (max_bin[0] > 1.0 or max_cell[0] > 1.0) else \
           ("REFUTED" if max_bin[0] <= 0.5 and max_cell[0] <= 0.5 else "MARGINAL")

# ---------------- save
import csv
with open(f"{OUT}/router_features.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pid", "tie_ratio", "tie_count", "spectral_gap", "hash_entropy", "candidate_spread",
                "delta", "kmeans_cluster", "cs_LG", "cs_CV"])
    for i in range(N_Q):
        w.writerow([int(pids[i]), float(tie_ratio[i]), int(tie_count[i]), float(spectral_gap[i]),
                    float(hash_entropy[i]), float(cand_spread[i]), float(delta[i]),
                    int(qassign[i]), int(is_cslg[i]), int(is_cscv[i])])

out = dict(seed=SEED, n_queries=N_Q,
           overall_mean_delta=float(delta.mean()),
           decision=decision,
           max_bin_ge100=dict(mean_delta=float(max_bin[0]), n=int(max_bin[1]), table=max_bin[2]),
           max_grid_cell_ge100=dict(mean_delta=float(max_cell[0]), n=int(max_cell[1])),
           tables=tables, grid=grid, topic=topic,
           features_doc=dict(
               tie_ratio="fraction of top-100 candidates at min hamming distance within top-100",
               spectral_gap="cluster-inherited OLS slope of log(top-32 eigenvalues) of centered bit covariance; hamming k-means k=64 on seeded 200K subsample of 5.1M hashes, 8 iterations, majority-vote centroids",
               hash_entropy="|ones(qhash)-256|",
               candidate_spread="variance of the 100 query-candidate hamming distances",
               delta="|GT[:10] ∩ top10(spectral-asc order)| - |GT[:10] ∩ top10(HNSW order)|, same 100 candidates"))
json.dump(out, open(f"{OUT}/geo_effect_curve.json", "w"), indent=2)
print(json.dumps(dict(decision=decision, max_bin=max_bin[:3], max_cell=max_cell,
                      overall=float(delta.mean()), topic=topic), indent=2))
print("saved /tmp/hybrid5m/geo_effect_curve.json + router_features.csv")
