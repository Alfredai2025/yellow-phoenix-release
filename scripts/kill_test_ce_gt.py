#!/usr/bin/env python3
"""CROSS-ENCODER KILL TEST (audit item #1).

Worry: our ground truth (E-space MiniLM-cosine nearest DIFFERENT doc) is produced by
the same embedding family the hashes quantize — self-graded exam. This test hires an
independent judge: cross-encoder/ms-marco-MiniLM-L-6-v2, which reads
(query_text, candidate_text) jointly and is never used anywhere in the pipeline.

Protocol (695 flagship eval queries, payload_sem_39m):
  candidates per query = MiniLM-E top-32 (excl. self) UNION our gt top-10
  CE top-1 = the judge's chosen nearest doc.
  (a_set)  CE top-1 in our gt top-10 SET        -> GT validity at R@10 granularity
  (a_str)  CE top-1 == the exact payload target -> strictest agreement
  (b)      CE top-1 in hash-space top-10        -> device recall UNDER CE ground truth
  (c)      payload target in hash top-10        -> must reproduce 86.5% (self-check)

KILL CRITERION: (b) < 76.5%  (>10pt drop vs the 86.5% headline) -> rebuild GT.
"""
import json, os, sys, time
import numpy as np

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

ROOT = os.path.expanduser("~/yellow_phoenix")
os.chdir(ROOT)
SRC = "data/vines_10m"
N = 5_660_333
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

log("loading E (MiniLM embeddings, memmap)")
E = np.memmap(f"{SRC}/embeddings.bin", dtype=np.float32, mode="r", shape=(N, 384))

log("loading payload queries")
qs = json.load(open("data/watch_ceiling/payload_sem_39m.json"))["queries"]
qpos = np.array([q["qpos"] for q in qs])
tgt_ids = np.array([q["id"] for q in qs], dtype=np.uint64)
nq = len(qs)

log("loading hashes (index) for hash-space top-10")
raw = np.fromfile("data/index_10m_hashes.bin", dtype=np.uint8, offset=14)
H = raw.reshape(N, 72)[:, 8:]
rowids = raw.reshape(N, 72)[:, :8].copy().view("<Q").ravel()
LIMIT = 3_900_000  # flat-scan prefix

log("loading exam gt top-10")
eq = np.load("data/vines_10m_max/eval_qidx.npy")
gt = np.load("data/vines_10m_max/eval_qidx_gt.npy")
row_of = {int(q): i for i, q in enumerate(eq)}
gtset = np.array([rowids[gt[row_of[int(p)]]] for p in qpos])  # (nq,10) rowids

# ---- MiniLM-E top-32 per query (excl. self) ----
log("computing MiniLM-E top-32 shortlists")
norms = np.sqrt((E[:LIMIT] * E[:LIMIT]).sum(1))
qn = np.sqrt((E[qpos] * E[qpos]).sum(1))
TOPM = 32
e_short = np.zeros((nq, TOPM), dtype=np.int64)
for s in range(0, LIMIT, 500_000):
    e = min(s + 500_000, LIMIT)
    sim = (np.asarray(E[s:e]) @ np.asarray(E[qpos]).T).T
    sim /= (norms[s:e][None, :] * qn[:, None])
    self_mask = (qpos >= s) & (qpos < e)
    sim[np.arange(nq)[self_mask], qpos[self_mask] - s] = -np.inf
    part = np.argpartition(-sim, TOPM, axis=1)[:, :TOPM]
    pv = np.take_along_axis(sim, part, 1)
    if s == 0:
        topv, topi = pv, part + s
    else:
        mv = np.concatenate([topv, pv], 1); mi = np.concatenate([topi, part + s], 1)
        sel = np.argpartition(-mv, TOPM, axis=1)[:, :TOPM]
        topv = np.take_along_axis(mv, sel, 1); topi = np.take_along_axis(mi, sel, 1)
e_short = topi
log("E shortlists done")

# ---- candidate union + text fetch ----
need = set()
for i in range(nq):
    need.update(int(x) for x in e_short[i])
    need.update(int(x) for x in gtset[i])          # these are ROWIDS; map to positions
# gtset holds rowids -> positions via searchsorted over the FULL index
# (gt top-10 members may lie outside the 3.9M prefix; CE judges them anyway,
# while hash-space top-10 stays prefix-bound — a CE pick outside the prefix
# correctly counts as a device miss)
full_pre = rowids
want = np.unique(gtset.ravel())
ins = np.searchsorted(full_pre, want)
ins = np.clip(ins, 0, N - 1)
ok = full_pre[ins] == want
assert ok.all(), f"{(~ok).sum()} gt rowids not in index"
rid2pos = dict(zip(want.tolist(), ins.tolist()))
gtpos = np.vectorize(rid2pos.get)(gtset)
for i in range(nq):
    need.update(int(x) for x in gtpos[i])
need.update(int(p) for p in qpos)
log(f"collecting texts for {len(need)} docs")

texts = {}
with open(f"{SRC}/corpus_texts.jsonl") as f:
    for line in f:
        r = json.loads(line)
        i = r["i"]
        if i in need:
            texts[i] = r["t"]
        if len(texts) == len(need):
            break
log(f"texts fetched: {len(texts)}/{len(need)}")

# ---- cross-encoder scoring ----
log("loading cross-encoder/ms-marco-MiniLM-L-6-v2")
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

def trunc(t, n=512):
    return t if len(t) <= n else t[:n]

ce_top1 = np.zeros(nq, dtype=np.int64)   # POSITION of CE's chosen doc
ce_top1_rid = np.zeros(nq, dtype=np.uint64)
for i in range(nq):
    cand = list(dict.fromkeys(list(e_short[i]) + list(gtpos[i])))
    pairs = [(trunc(texts[int(qpos[i])]), trunc(texts[int(c)])) for c in cand]
    scores = ce.predict(pairs)
    best = cand[int(np.argmax(scores))]
    ce_top1[i] = best
    ce_top1_rid[i] = rowids[best]
    if (i + 1) % 100 == 0:
        log(f"CE {i+1}/{nq}")
log("CE scoring done")

# ---- hash-space top-10 (device answer set) ----
log("computing hash-space top-10 per query (3.9M prefix)")
hash_top10 = np.zeros((nq, 10), dtype=np.uint64)  # rowids
Hp = H[:LIMIT]
for i in range(nq):
    qh = Hp[qpos[i]]
    bd, bi = [], []
    for s in range(0, LIMIT, 500_000):
        e = min(s + 500_000, LIMIT)
        d = POP[np.bitwise_xor(Hp[s:e], qh)].sum(axis=1)
        idx = np.argpartition(d, min(10, e - s - 1))[:10]
        bd.append(d[idx]); bi.append(idx + s)
    dd = np.concatenate(bd); ii = np.concatenate(bi)
    sel = np.argpartition(dd, 10)[:10]
    sel = sel[np.argsort(dd[sel], kind="stable")]
    hash_top10[i] = rowids[ii[sel]]
    if (i + 1) % 100 == 0:
        log(f"hash top10 {i+1}/{nq}")

# ---- verdicts ----
gtset_set = [set(x.tolist()) for x in gtset]
a_set = np.mean([ce_top1_rid[i] in gtset_set[i] for i in range(nq)])
a_str = np.mean([ce_top1_rid[i] == tgt_ids[i] for i in range(nq)])
b = np.mean([ce_top1_rid[i] in set(hash_top10[i].tolist()) for i in range(nq)])
c = np.mean([tgt_ids[i] in set(hash_top10[i].tolist()) for i in range(nq)])

log("=" * 64)
log(f"KILL TEST RESULTS (n={nq})")
log(f"  (a_set) CE top-1 in our gt top-10 set : {100*a_set:.1f}%")
log(f"  (a_str) CE top-1 == exact payload target: {100*a_str:.1f}%")
log(f"  (b)     CE-GT in hash top-10 (device recall under judge): {100*b:.1f}%")
log(f"  (c)     our-GT in hash top-10 (self-check, expect 86.5%) : {100*c:.1f}%")
verdict = "PASS" if b >= 76.5 else "KILL"
log(f"  KILL criterion: b >= 76.5% required -> VERDICT: {verdict}")
json.dump({"n": nq, "a_set": float(a_set), "a_str": float(a_str),
           "b_ce_device_recall": float(b), "c_selfcheck": float(c),
           "verdict": verdict},
          open("benchmark_results/kill_test_ce_gt_20261003.json", "w"), indent=1)
np.savez("benchmark_results/kill_test_ce_gt_20261003_detail.npz",
         ce_top1_rid=ce_top1_rid, tgt_ids=tgt_ids, qpos=qpos,
         hash_top10=hash_top10, gtset=gtset)
log("saved detail npz for cluster diagnostics")
log("saved benchmark_results/kill_test_ce_gt_20261003.json")
