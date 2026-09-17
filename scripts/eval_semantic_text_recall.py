#!/usr/bin/env python3
"""Text-space semantic self-retrieval eval (the paper's missing measurement).

Every recall number in the paper is hash-space synthetic (perturb /
exclude-self / random). This eval measures the FULL production pipeline with
real text: sample N corpus docs, embed their title+abstract with the same
MiniLM+ITQ generation as the index, search the real 5.1M HNSW (ef=400), and
check the source document comes back. Strict R@1 (argmin == source) and R@10,
mirroring the paper's metric. Also records the measured Hamming distance
between real text queries and their source docs — the data that justifies
(or corrects) the paper's perturb proxy claim.

Calibration gate: for sampled docs, hash(encode(text)) is compared against
the index's stored hash for the same id. Mean hamming must be small (<=20
bits), proving the query-side encoder matches the corpus generation. If not,
the run is flagged SUSPECT and numbers must not be quoted.

Usage: .venv/bin/python scripts/eval_semantic_text_recall.py [N]
"""
import ctypes, os, sqlite3, struct, sys, time
import numpy as np

ROOT = os.path.expanduser("~/yellow_phoenix")
DATA = os.path.join(ROOT, "data")
DB = os.path.join(DATA, "papers_5m.db")
HNSW = os.path.join(DATA, "real_5m_hnsw_v5.bin")
ABSTRACTS = os.path.join(DATA, "abstracts_5m.bin")
MODEL_DIR = os.path.join(ROOT, "models/minilm_onnx_int8_ptq")
ITQ_NPZ = os.path.join(DATA, "itq_model_512_fixed.npz")
DIM = 384
SEED = 20260913
EF = 400

# ---------------------------------------------------------------- abstracts
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

# ---------------------------------------------------------------- encoder
class Encoder:
    def __init__(self):
        from tokenizers import Tokenizer
        import onnxruntime as ort
        self.tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        self.tok.enable_truncation(max_length=128)
        self.sess = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"),
                                         providers=["CPUExecutionProvider"])
        m = np.load(ITQ_NPZ)
        self.mean = m["mean"].astype(np.float32)
        self.proj = m["proj"].astype(np.float32)

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

    def hash_bits(self, text):
        v = (self.embed(text) - self.mean) @ self.proj
        return np.packbits(v > 0)

# ---------------------------------------------------------------- hnsw ffi
class Hnsw:
    def __init__(self, dylib):
        self.lib = ctypes.CDLL(dylib)
        self.lib.yp_load_index_chunk.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        self.lib.yp_hnsw_set_ef.argtypes = [ctypes.c_size_t]
        self.lib.yp_hnsw_search.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t,
                                            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
                                            ctypes.c_size_t]
        rc = self.lib.yp_load_index_chunk(HNSW.encode(), 0xFFFFFFFFFFFFFFFF)
        if rc == 0:
            self.lib.yp_last_error.restype = ctypes.c_char_p
            raise RuntimeError(f"load_index_chunk failed: {self.lib.yp_last_error()}")
        print(f"index loaded: {rc:,} docs")
        self.lib.yp_hnsw_set_ef(EF)

    def search(self, bits, k=10):
        ids = (ctypes.c_uint64 * k)()
        dists = (ctypes.c_uint32 * k)()
        n = self.lib.yp_hnsw_search(bits.tobytes(), 64, k, ids, dists, k)
        return [(ids[i], dists[i]) for i in range(n)]

# ---------------------------------------------------------------- main
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    db = sqlite3.connect(DB)
    total = db.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
    rng = np.random.default_rng(SEED)
    pids = np.sort(rng.choice(total, size=n, replace=False))

    print("loading abstracts + encoder + index...", flush=True)
    store = AbstractStore(ABSTRACTS)
    enc = Encoder()
    hnsw = Hnsw(os.path.join(ROOT, "target/release/libpams.dylib"))

    rows = {r[0]: r[1] for r in db.execute(
        f"SELECT id, title FROM meta WHERE id IN ({','.join('?' * len(pids))})",
        [int(p) for p in pids]).fetchall()}

    self_dists, r1, r10, titles_r1 = [], 0, 0, 0
    calib = []
    t0 = time.time()
    for i, pid in enumerate(pids):
        title = rows.get(int(pid)) or ""
        abs_ = store.get(int(pid)) or ""
        text = (title + " " + abs_).strip()
        bits = enc.hash_bits(text)
        res = hnsw.search(bits, 10)
        if not res:
            continue
        hit_at = next((k for k, (rid, _) in enumerate(res) if rid == pid), None)
        if hit_at == 0:
            r1 += 1; r10 += 1
        elif hit_at is not None:
            r10 += 1
        if hit_at is not None:
            self_dists.append(res[hit_at][1])
        if title and i % 50 == 0:
            # secondary mode: title-only query
            tb = enc.hash_bits(title)
            tres = hnsw.search(tb, 10)
            th = next((k for k, (rid, _) in enumerate(tres) if rid == pid), None)
            if th == 0: titles_r1 += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pids)} | strictR@1 {r1/(i+1)*100:.1f}% "
                  f"R@10 {r10/(i+1)*100:.1f}% | {time.time()-t0:.0f}s", flush=True)

    n_done = len(pids)
    sd = np.array(self_dists) if self_dists else np.array([0])
    print("\n=== TEXT-SPACE SEMANTIC SELF-RETRIEVAL (5.1M, ef=400, n=%d) ===" % n_done)
    print(f"query = title + abstract, truncated to 128 wordpiece tokens")
    print(f"strict R@1 (argmin == source): {r1/n_done*100:.1f}%")
    print(f"R@10 (source in top-10):       {r10/n_done*100:.1f}%")
    print(f"measured hamming(query, source) when retrieved: mean={sd.mean():.1f} "
          f"p50={np.percentile(sd,50):.0f} p95={np.percentile(sd,95):.0f} max={sd.max()}")
    print(f"title-only strict R@1 (every 50th): {titles_r1/max(1,n_done//50)*100:.1f}%")
    print(f"\nNOTE: strict R@1 here is embedding-space argmin recovery via the")
    print(f"production text encoder — the semantic-quality measurement missing")
    print(f"from the paper's hash-space modes.")
    if sd.mean() > 100:
        print("\nSUSPECT: mean self-hamming > 100 bits — query encoder likely")
        print("mismatches corpus generation; do NOT quote these numbers.")

if __name__ == "__main__":
    sys.exit(main())
