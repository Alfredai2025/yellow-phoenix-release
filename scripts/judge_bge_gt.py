#!/usr/bin/env python3
"""JUDGE #2 (local): BGE-small-en-v1.5 bi-encoder, independent model family.
Same protocol as kill_test_ce_gt.py: candidates = MiniLM-E top-32 UNION our gt top-10;
judge picks its own nearest by embedding cosine; same metrics + cluster breakdown."""
import json, os, time
import numpy as np

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
os.chdir(os.path.expanduser("~/yellow_phoenix"))
SRC = "data/vines_10m"; N = 5_660_333; LIMIT = 3_900_000

E = np.memmap(f"{SRC}/embeddings.bin", dtype=np.float32, mode="r", shape=(N, 384))
qs = json.load(open("data/watch_ceiling/payload_sem_39m.json"))["queries"]
qpos = np.array([q["qpos"] for q in qs]); tgt_ids = np.array([q["id"] for q in qs], dtype=np.uint64)
nq = len(qs)
raw = np.fromfile("data/index_10m_hashes.bin", dtype=np.uint8, offset=14)
H = raw.reshape(N, 72)[:, 8:]
rowids = raw.reshape(N, 72)[:, :8].copy().view("<Q").ravel()
eq = np.load("data/vines_10m_max/eval_qidx.npy"); gt = np.load("data/vines_10m_max/eval_qidx_gt.npy")
row_of = {int(q): i for i, q in enumerate(eq)}
gtset = np.array([rowids[gt[row_of[int(p)]]] for p in qpos])

log("MiniLM-E top-32")
norms = np.sqrt((E[:LIMIT] * E[:LIMIT]).sum(1)); qn = np.sqrt((E[qpos] * E[qpos]).sum(1))
TOPM = 32; e_short = np.zeros((nq, TOPM), dtype=np.int64)
for s in range(0, LIMIT, 500_000):
    e = min(s + 500_000, LIMIT)
    sim = (np.asarray(E[s:e]) @ np.asarray(E[qpos]).T).T / (norms[s:e][None, :] * qn[:, None])
    m = (qpos >= s) & (qpos < e); sim[np.arange(nq)[m], qpos[m] - s] = -np.inf
    part = np.argpartition(-sim, TOPM, axis=1)[:, :TOPM]; pv = np.take_along_axis(sim, part, 1)
    if s == 0: topv, topi = pv, part + s
    else:
        mv = np.concatenate([topv, pv], 1); mi = np.concatenate([topi, part + s], 1)
        sel = np.argpartition(-mv, TOPM, axis=1)[:, :TOPM]
        topv = np.take_along_axis(mv, sel, 1); topi = np.take_along_axis(mi, sel, 1)
e_short = topi

full_pre = rowids
want = np.unique(gtset.ravel()); ins = np.searchsorted(full_pre, want); ins = np.clip(ins, 0, N - 1)
assert (full_pre[ins] == want).all()
rid2pos = dict(zip(want.tolist(), ins.tolist())); gtpos = np.vectorize(rid2pos.get)(gtset)

need = set(int(p) for p in qpos)
for i in range(nq):
    need.update(int(x) for x in e_short[i]); need.update(int(x) for x in gtpos[i])
log(f"texts for {len(need)} docs")
texts = {}
with open(f"{SRC}/corpus_texts.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["i"] in need: texts[r["i"]] = r["t"]
        if len(texts) == len(need): break
log(f"texts {len(texts)}/{len(need)}")

log("loading BGE-small-en-v1.5 (offline)")
from sentence_transformers import SentenceTransformer
os.environ["HF_HUB_OFFLINE"] = "1"
bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
def trunc(t, n=512): return t if len(t) <= n else t[:n]

judge_top1 = np.zeros(nq, dtype=np.uint64)
for i in range(nq):
    cand = list(dict.fromkeys(list(e_short[i]) + list(gtpos[i])))
    emb = bge.encode([trunc(texts[int(qpos[i])])] + [trunc(texts[int(c)]) for c in cand],
                     normalize_embeddings=True)
    sims = emb[1:] @ emb[0]
    judge_top1[i] = rowids[cand[int(np.argmax(sims))]]
    if (i + 1) % 200 == 0: log(f"judge {i+1}/{nq}")
log("judge done")

log("hash top-10 (3.9M)")
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
Hp = H[:LIMIT]; hash_top10 = np.zeros((nq, 10), dtype=np.uint64)
for i in range(nq):
    qh = Hp[qpos[i]]; bd, bi = [], []
    for s in range(0, LIMIT, 500_000):
        e = min(s + 500_000, LIMIT)
        d = POP[np.bitwise_xor(Hp[s:e], qh)].sum(axis=1)
        idx = np.argpartition(d, min(10, e - s - 1))[:10]
        bd.append(d[idx]); bi.append(idx + s)
    dd = np.concatenate(bd); ii = np.concatenate(bi)
    sel = np.argpartition(dd, 10)[:10]; sel = sel[np.argsort(dd[sel], kind="stable")]
    hash_top10[i] = rowids[ii[sel]]
    if (i + 1) % 200 == 0: log(f"hash {i+1}/{nq}")

gtset_set = [set(x.tolist()) for x in gtset]
a_set = np.mean([judge_top1[i] in gtset_set[i] for i in range(nq)])
a_str = np.mean([judge_top1[i] == tgt_ids[i] for i in range(nq)])
b = np.mean([judge_top1[i] in set(hash_top10[i].tolist()) for i in range(nq)])
c = np.mean([tgt_ids[i] in set(hash_top10[i].tolist()) for i in range(nq)])
clu = np.fromfile(f"{SRC}/cluster_ids_bdc.u32", dtype=np.uint32)
ui = np.searchsorted(full_pre, np.unique(np.r_[judge_top1, tgt_ids])); mp = dict(zip(np.unique(np.r_[judge_top1, tgt_ids]).tolist(), ui.tolist()))
jpos = np.array([mp[int(r)] for r in judge_top1]); tpos = np.array([mp[int(r)] for r in tgt_ids])
same_clu = clu[jpos] == clu[tpos]

log("=" * 64)
log(f"JUDGE BGE-small (n={nq})")
log(f"  (a_set) judge top-1 in our gt top-10 : {100*a_set:.1f}%")
log(f"  (a_str) judge top-1 == payload target: {100*a_str:.1f}%")
log(f"  (b)     judge-GT in hash top-10      : {100*b:.1f}%   [kill line 76.5%]")
log(f"  (c)     our-GT in hash top-10        : {100*c:.1f}%")
log(f"  same-cluster as our target           : {100*same_clu.mean():.1f}%")
log(f"  VERDICT: {'PASS' if b >= 76.5 else 'KILL'}")
np.savez("benchmark_results/judge_bge_detail.npz", judge_top1=judge_top1, tgt_ids=tgt_ids, qpos=qpos, hash_top10=hash_top10)
