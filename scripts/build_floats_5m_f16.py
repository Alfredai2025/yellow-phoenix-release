#!/usr/bin/env python3
"""Build floats_5m.f16: id-ordered float16 MiniLM-384 embeddings for the 5.1M corpus.

Row order == papers_5m.db id == ISM id == HNSW label id, so the on-device
re-ranker can fetch a paper's exact embedding by id.

File layout (little-endian):
  [0]  magic "YPF1" (4 bytes)
  [4]  u32 version = 1
  [8]  u32 dim = 384
  [12] u64 count
  [20] reserved (12 bytes, zero)
  [32] count * dim * 2 bytes of float16

Verification: for sampled ids, re-encode the stored f16 (upcast to f32)
with the ITQ model and compare against the hash in real_5m_hnsw.bin.
"""
import os
import struct
import sys
import time

import numpy as np

ROOT = os.path.expanduser("~/yellow_phoenix")
DATA = os.path.join(ROOT, "data")
WORK = os.path.join(DATA, "_build_papers5m")
OUT = os.path.join(DATA, "floats_5m.f16")
SOURCES = [
    ("data/embeddings_3m.npy", "raw"),
    ("data/embeddings_oai439k.npy", "npy"),
    ("data/embeddings_pubmed_new.npy", "npy"),
    ("data/embeddings_pubmed_new2.npy", "npy"),
    ("data/droplet_embeddings_aligned.npy", "npy"),
]
CHUNK = 100_000
DIM = 384


def open_source(rel, kind):
    path = os.path.join(ROOT, rel)
    if kind == "raw":
        mm = np.memmap(path, dtype=np.float32, mode="r").reshape(-1, DIM)
        return mm
    arr = np.load(path)
    assert arr.dtype == np.float32 and arr.shape[1] == DIM, f"{rel}: {arr.shape}"
    return arr


def main():
    t0 = time.time()
    kept = [np.load(os.path.join(WORK, f"kept_src{i}.npy")) for i in range(5)]
    bounds = np.cumsum([0] + [len(a) for a in kept])
    total = int(bounds[-1])
    print(f"corpus: {total:,} papers -> {total*DIM*2/1e9:.2f} GB f16", flush=True)

    with open(OUT, "wb") as f:
        f.write(struct.pack("<4sIIQ12x", b"YPF1", 1, DIM, total))
        for si, (rel, kind) in enumerate(SOURCES):
            arr = open_source(rel, kind)
            rows = kept[si]
            n = len(rows)
            for lo in range(0, n, CHUNK):
                hi = min(lo + CHUNK, n)
                x = np.ascontiguousarray(arr[rows[lo:hi]], dtype=np.float32)
                f.write(x.astype(np.float16).tobytes())
            print(f"  src{si}: {n:,} rows | {time.time()-t0:.0f}s", flush=True)
            del arr

    size = os.path.getsize(OUT)
    expect = 32 + total * DIM * 2
    assert size == expect, f"size {size} != expected {expect}"
    print(f"wrote {OUT} ({size/1e9:.2f} GB) in {time.time()-t0:.0f}s")

    # --- verify: stored f16 re-encodes to the HNSW hash (small hamming ok) ---
    print("verifying against HNSW hashes...", flush=True)
    m = np.load(os.path.join(DATA, "itq_model_512_fixed.npz"))
    mean = m["mean"].astype(np.float32)
    proj = m["proj"].astype(np.float32)

    mm = np.memmap(OUT, dtype=np.float16, mode="r", offset=32)
    mm = mm.reshape(total, DIM)

    rng = np.random.default_rng(20260906)
    sample = np.sort(rng.choice(total, size=64, replace=False))
    dists = []
    for pid in sample:
        emb = np.asarray(mm[pid], dtype=np.float32)
        bits = np.packbits(((emb - mean) @ proj) > 0)
        # HNSW stored hash via the ISM (id -> hash row order == id order)
        row = ism_hash_row(pid)
        d = int(np.count_nonzero(bits != row))
        dists.append(d)
    dists = np.array(dists)
    print(f"hamming(f16-derived hash, stored hash) over 64 samples: "
          f"mean={dists.mean():.2f} max={dists.max()} "
          f"({'OK' if dists.max() <= 8 else 'SUSPECT — investigate'})")


def ism_hash_row(pid):
    """Read the 64-byte hash of ISM row `pid` from real_5m.ism
    (layout: [u64 count][count x 64B hashes][count x 8B ids])."""
    path = os.path.join(DATA, "real_5m.ism")
    with open(path, "rb") as f:
        f.seek(8 + pid * 64)
        return np.frombuffer(f.read(64), dtype=np.uint8)


if __name__ == "__main__":
    sys.exit(main())
