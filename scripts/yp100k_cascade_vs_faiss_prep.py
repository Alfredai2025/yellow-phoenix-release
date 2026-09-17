#!/usr/bin/env python3
"""Prep for the 100K cascade-vs-FAISS benchmark.

- Verifies corpus shape/dtype, computes 512-bit ITQ hashes matching
  src/ffi_itq.rs encoding (center with mean, project with W@R, MSB-first packbits),
  verifies parity against the Rust FFI yp_itq_encode.
- Writes ISM file (count u64 + n*64 hashes + n u64 ids) for build_hnsw_from_ism.
- Computes brute-force GT (exact cosine argmin over all N, exclude-self) for
  1000 seeded query doc indices.
- Saves all artifacts under /tmp/yp100k/.
"""
import os, sys, json, hashlib

import numpy as np

YP = "/Users/mac/yellow_phoenix"
OUT = "/tmp/yp100k"
os.makedirs(OUT, exist_ok=True)
SEED = 20260915
N_QUERIES = 1000

os.chdir(YP)
sys.path.insert(0, YP)

emb = np.load("data/paper_embeddings_100k.npy")
assert emb.shape == (100000, 384) and emb.dtype == np.float32, (emb.shape, emb.dtype)
print("corpus:", emb.shape, emb.dtype)
corpus_sha = hashlib.sha256(open("data/paper_embeddings_100k.npy", "rb").read()).hexdigest()
print("corpus sha256:", corpus_sha)

m = np.load("data/itq_model_1m_512.npz")
mean = m["mean"].astype(np.float32)
proj = (m["W"] @ m["R"]).astype(np.float32)
print("stored proj == W@R:", np.allclose(proj, m["proj"], atol=1e-5))

# --- hash encoding matching src/ffi_itq.rs: acc = sum((x-mean)*proj[:,bit]); bit set if acc>=0, MSB-first
centered = emb.astype(np.float64) - mean.astype(np.float64)
scores = centered @ proj.astype(np.float64)          # (N, 512)
bits = (scores >= 0.0).astype(np.uint8)              # bit i -> byte i//8, MSB (7 - i%8)
hashes = np.packbits(bits, axis=1)                   # MSB-first, matches numpy convention in ffi_itq.rs
assert hashes.shape == (100000, 64) and hashes.dtype == np.uint8
np.save(f"{OUT}/hashes.npy", hashes)
print("hashes:", hashes.shape)

# --- parity check against Rust FFI yp_itq_encode
import ctypes
lib = ctypes.CDLL(os.path.join(YP, "target/release/libpams.dylib"))
lib.yp_itq_init.restype = ctypes.c_int
lib.yp_itq_init.argtypes = [ctypes.c_char_p]
# build the .bin model file exactly like yp_bridge.get_shared_bridge
# NOTE: yp_bridge.get_shared_bridge writes a 16-byte header (magic+n_dims+n_bits)
# but src/ffi_itq.rs parses a 12-byte header (magic+n_dims) and silently mis-loads
# mean/proj by 4 bytes on the repo's production itq_model_1m_512.bin (bug, reported).
# For parity checking we write the format ffi_itq.rs actually parses.
bin_path = os.path.join(OUT, "itq_model_12Bhdr.bin")
if not os.path.exists(bin_path):
    header = b"YPITQ512" + np.array([mean.shape[0]], dtype=np.uint32).tobytes()
    with open(bin_path, "wb") as f:
        f.write(header)
        f.write(mean.astype(np.float32).tobytes())
        f.write(proj.astype(np.float32).tobytes())
rc = lib.yp_itq_init(bin_path.encode())
print("yp_itq_init rc:", rc)
lib.yp_itq_encode.restype = ctypes.c_int
lib.yp_itq_encode.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)]
rng = np.random.default_rng(0)
mismatch = 0
for i in rng.choice(100000, 64, replace=False):
    v = np.ascontiguousarray(emb[i], dtype=np.float32)
    out = (ctypes.c_uint8 * 64)()
    rc = lib.yp_itq_encode(v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 384, out)
    assert rc == 0
    if not np.array_equal(np.frombuffer(out, dtype=np.uint8), hashes[i]):
        mismatch += 1
print(f"FFI parity: {64-mismatch}/64 vectors match")
assert mismatch == 0, "Python hash encoding does not match Rust FFI!"

# --- write ISM (build_hnsw_from_ism format): count u64 + hashes + ids
ids = np.arange(100000, dtype=np.uint64)
with open(f"{OUT}/corpus_100k.ism", "wb") as f:
    f.write(np.array([100000], dtype=np.uint64).tobytes())
    f.write(hashes.tobytes())
    f.write(ids.tobytes())
print("wrote ISM")

# --- queries + brute-force GT (exact cosine argmin over all N, exclude self)
qrng = np.random.default_rng(SEED)
qids = qrng.choice(100000, N_QUERIES, replace=False).astype(np.int64)
np.save(f"{OUT}/queries.npy", qids)

x = emb.astype(np.float32)
x /= np.linalg.norm(x, axis=1, keepdims=True)
gt = np.zeros(N_QUERIES, dtype=np.int64)
for qi, q in enumerate(qids):
    sims = x @ x[q]
    sims[q] = -np.inf
    gt[qi] = int(np.argmax(sims))
    if (qi + 1) % 200 == 0:
        print(f"GT {qi+1}/{N_QUERIES}")
np.save(f"{OUT}/gt.npy", gt)
np.save(f"{OUT}/corpus_norm.npy", x)

meta = dict(seed=SEED, n_queries=N_QUERIES, corpus_sha256=corpus_sha,
            corpus="data/paper_embeddings_100k.npy (100000x384 float32)",
            gt="exact cosine argmin over all 100000 L2-normalized embeddings, exclude-self",
            hash_encoding="center(mean) @ (W@R), bit=acc>=0, MSB-first packbits; FFI parity 64/64 (12-byte-header model file; see header bug note)",
            header_bug="yp_bridge.py writes 16-byte ITQ header (magic+n_dims+n_bits); src/ffi_itq.rs parses 12-byte header (magic+n_dims) -> production itq_model_1m_512.bin mis-loads mean/proj by 4 bytes")
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print("done")
