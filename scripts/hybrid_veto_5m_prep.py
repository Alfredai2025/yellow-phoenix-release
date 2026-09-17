#!/usr/bin/env python3
"""Hybrid veto benchmark prep: 5.1M real corpus.

1. Sample 1000 docs (seed 20260916), text = "{title}. {abstract}"[:512].
2. ENCODER CALIBRATION: which (text encoder, ITQ model) matches the stored
   corpus hashes in data/real_5m.ism? Test 4 combos, pick lowest mean hamming.
3. Embed queries with the calibrated encoder, checkpoint embeddings+hashes.
4. GT: exact cosine top-100 of each query embedding against all 5.1M f16->f32
   corpus vectors (chunked matmul). Wall time recorded.
Artifacts under /tmp/hybrid5m/.
"""
import os, sys, json, time, struct, sqlite3

import numpy as np

ROOT = "/Users/mac/yellow_phoenix"
DATA = os.path.join(ROOT, "data")
OUT = "/tmp/hybrid5m"
os.makedirs(OUT, exist_ok=True)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

SEED = 20260916
N_Q = 1000
DIM = 384
ISM = os.path.join(DATA, "real_5m.ism")
F16 = os.path.join(DATA, "floats_5m.f16")
DB = os.path.join(DATA, "papers_5m.db")
ABS = os.path.join(DATA, "abstracts_5m.bin")

# ---------------- corpus index files
with open(ISM, "rb") as f:
    N = int(np.frombuffer(f.read(8), dtype="<u8")[0])
print(f"corpus docs: {N:,}")
ism_hashes = np.memmap(ISM, dtype=np.uint8, mode="r", offset=8, shape=(N, 64))
f16 = np.memmap(F16, dtype=np.float16, mode="r", offset=32, shape=(N, DIM))

# ---------------- abstracts store (from scripts/eval_semantic_text_recall.py)
def read_varint(buf, pos):
    shift = 0; result = 0
    while True:
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7

class AbstractStore:
    def __init__(self, path):
        import zstandard as zstd
        f = open(path, "rb"); self.f = f
        assert f.read(4) == b"YPA1"; f.read(1)
        hdr = f.read(24)
        self.n_blocks = int(np.frombuffer(hdr[0:8], "<u8")[0])
        dlen = int(np.frombuffer(hdr[16:20], "<u4")[0])
        dctx = zstd.ZstdDecompressor(dict_data=zstd.ZstdCompressionDict(f.read(dlen)))
        self.decomp = lambda b: dctx.decompress(b, max_output_size=1 << 22)
        f.seek(0, 2); end = f.tell()
        f.seek(end - self.n_blocks * 24)
        self.index = [tuple(struct.unpack("<QQII", f.read(24))) for _ in range(self.n_blocks)]

    def get(self, pid):
        lo, hi = 0, self.n_blocks - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.index[mid][0] <= pid: lo = mid
            else: hi = mid - 1
        first_id, offset, nrec, clen = self.index[lo]
        if not (first_id <= pid < first_id + nrec): return None
        self.f.seek(offset)
        raw = self.decomp(self.f.read(clen))
        cur, pos = read_varint(raw, 0)
        for k in range(nrec):
            ln, p2 = read_varint(raw, pos)
            if cur == pid:
                return raw[p2:p2 + ln].decode("utf-8", "replace")
            pos = p2 + ln
            if k < nrec - 1:
                delta, pos = read_varint(raw, pos); cur += delta
        return None

# ---------------- queries
db = sqlite3.connect(DB)
total = db.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
assert total == N, (total, N)
rng = np.random.default_rng(SEED)
pids = np.sort(rng.choice(total, size=N_Q, replace=False))
store = AbstractStore(ABS)
titles = dict(db.execute(
    f"SELECT id, title FROM meta WHERE id IN ({','.join('?' * len(pids))})",
    [int(p) for p in pids]).fetchall())
texts = []
for pid in pids:
    t = titles.get(int(pid)) or ""
    a = store.get(int(pid)) or ""
    texts.append((f"{t}. {a}")[:512])
np.save(f"{OUT}/pids.npy", pids)
json.dump({"seed": SEED, "n": N_Q, "corpus": "data/real_5m.ism/floats_5m.f16/papers_5m.db (5,107,508 docs)",
           "text": "{title}. {abstract}\"[:512]\""}, open(f"{OUT}/meta.json", "w"), indent=2)
print("queries built")

# ---------------- encoders
def itq_pack(emb, mean, proj):
    v = (emb.astype(np.float64) - mean.astype(np.float64)) @ proj.astype(np.float64)
    return np.packbits((v >= 0).astype(np.uint8))

m1 = np.load(f"{DATA}/itq_model_1m_512.npz")
fixed = np.load(f"{DATA}/itq_model_512_fixed.npz")
MODELS = {
    "itq_1m": (m1["mean"].astype(np.float32), (m1["W"] @ m1["R"]).astype(np.float32)),
    "itq_fixed": (fixed["mean"].astype(np.float32), fixed["proj"].astype(np.float32)),
}

from sentence_transformers import SentenceTransformer
import torch
dev = "mps" if torch.backends.mps.is_available() else "cpu"
st_model = SentenceTransformer(os.path.join(ROOT, "models/all-MiniLM-L6-v2"), device=dev)

def st_embed(texts):
    return st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False).astype(np.float32)

class OnnxEnc:
    def __init__(self):
        from tokenizers import Tokenizer
        import onnxruntime as ort
        d = os.path.join(ROOT, "models/minilm_onnx_int8_ptq")
        self.tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        self.tok.enable_truncation(max_length=128)
        self.sess = ort.InferenceSession(os.path.join(d, "model.onnx"),
                                         providers=["CPUExecutionProvider"])
    def embed(self, text):
        e = self.tok.encode(text)
        ids = np.array([e.ids], dtype=np.int64)
        mask = np.array([e.attention_mask], dtype=np.int64)
        tt = np.zeros_like(ids)
        out = self.sess.run(None, {"input_ids": ids, "attention_mask": mask,
                                   "token_type_ids": tt})[0][0]
        m = mask[0].astype(np.float32)
        emb = (out * m[:, None]).sum(0) / max(m.sum(), 1.0)
        n = np.linalg.norm(emb)
        return (emb / n).astype(np.float32) if n > 0 else emb.astype(np.float32)

onx = OnnxEnc()

# ---------------- calibration on 32 docs
cal_ids = pids[:32]
cal_emb_st = st_embed([texts[i] for i in range(32)])
cal_emb_onx = np.stack([onx.embed(texts[i]) for i in range(32)])
report = {}
for mname, (mean, proj) in MODELS.items():
    for enc_name, embs in [("st_fp32", cal_emb_st), ("onnx_int8ptq", cal_emb_onx)]:
        ds = []
        for i in range(32):
            h = itq_pack(embs[i], mean, proj)
            ds.append(int(np.unpackbits(h ^ np.asarray(ism_hashes[int(cal_ids[i])])).sum()))
        report[f"{enc_name}+{mname}"] = float(np.mean(ds))
        print(f"calib {enc_name}+{mname}: mean hamming to stored hash = {np.mean(ds):.1f}")
best = min(report, key=report.get)
print("BEST:", best, report[best])
json.dump(report, open(f"{OUT}/calibration.json", "w"), indent=2)

# brief-preferred combo is st_fp32+itq_1m; use it if calibration says it matches
# (mean hamming <= 60), else the best-calibrated combo (deviation recorded).
if report["st_fp32+itq_1m"] <= 60:
    chosen_enc, chosen_model = "st_fp32", "itq_1m"
else:
    chosen_enc = best.split("+")[0]
    chosen_model = best.split("+")[1]
print("CHOSEN:", chosen_enc, chosen_model)

# ---------------- embed all queries
t0 = time.time()
if chosen_enc == "st_fp32":
    qemb = st_embed(texts)
else:
    qemb = np.stack([onx.embed(t) for t in texts]).astype(np.float32)
    qemb /= np.linalg.norm(qemb, axis=1, keepdims=True)
embed_s = time.time() - t0
mean, proj = MODELS[chosen_model]
qhash = np.stack([itq_pack(qemb[i], mean, proj) for i in range(N_Q)])
np.save(f"{OUT}/qemb.npy", qemb)
np.save(f"{OUT}/qhash.npy", qhash)
print(f"embedded {N_Q} queries in {embed_s:.1f}s")

# self-hamming sanity (encoder match at full scale)
dself = np.unpackbits(qhash ^ np.asarray(ism_hashes[pids]), axis=1).sum(1)
print(f"self hamming: mean={dself.mean():.1f} p50={np.percentile(dself,50):.0f} p95={np.percentile(dself,95):.0f}")

# ---------------- GT: exact cosine top-100 vs all 5.1M f16->f32
t0 = time.time()
CH = 150_000
top_ids = np.zeros((N_Q, 100), dtype=np.int64)
top_sims = np.zeros((N_Q, 100), dtype=np.float32)
for s in range(0, N, CH):
    e = min(s + CH, N)
    c = np.asarray(f16[s:e]).astype(np.float32)
    sims = qemb @ c.T
    idx = np.argpartition(sims, -100, axis=1)[:, -100:]
    ridx = np.take_along_axis(sims, idx, axis=1).argsort(axis=1)[:, ::-1]
    idx = np.take_along_axis(idx, ridx, axis=1)
    sim = np.take_along_axis(sims, idx, axis=1)
    joined_ids = np.concatenate([top_ids, idx + s], axis=1)
    joined_sims = np.concatenate([top_sims, sim], axis=1)
    k = np.argpartition(joined_sims, -100, axis=1)[:, -100:]
    kr = np.take_along_axis(joined_sims, k, axis=1).argsort(axis=1)[:, ::-1]
    k = np.take_along_axis(k, kr, axis=1)
    top_ids = np.take_along_axis(joined_ids, k, axis=1)
    top_sims = np.take_along_axis(joined_sims, k, axis=1)
    if (s // CH) % 10 == 0:
        print(f"GT chunk {s:,}/{N:,} ({time.time()-t0:.0f}s)", flush=True)
gt_s = time.time() - t0
np.save(f"{OUT}/gt_top100_ids.npy", top_ids)
np.save(f"{OUT}/gt_top100_sims.npy", top_sims)
print(f"GT done in {gt_s:.1f}s")
json.dump({"chosen_encoder": chosen_enc, "chosen_model": chosen_model,
           "calibration": report, "embed_s": embed_s, "gt_wall_s": gt_s,
           "self_hamming_mean": float(dself.mean())},
          open(f"{OUT}/prep_summary.json", "w"), indent=2)
print("done")
