#!/usr/bin/env python3
"""Build the true-5M real-papers shootout pair.

Streams float32 embedding sources in chunks, ITQ-encodes with the fixed 512-bit
model, and writes the FlatIndex ISM file directly (no merged-float intermediate):

    [u64 count LE][count x 64B hashes][count x 8B ids]     ids = global 0-based row

Sources (must each be L2-normalized MiniLM-384 in the same space, row order =
global row order):
    1. data/embeddings_3m.npy           raw float32, 3,025,752 x 384 (no npy hdr)
    2. data/embeddings_oai439k.npy      npy float32, 439,400 x 384
    3. data/embeddings_pubmed_new.npy   npy float32, P x 384  (new PubMed rows)
    4. data/droplet_embeddings_aligned.npy  npy float32, 106,298 x 384

Usage: build_5m_real.py [out_ism]
"""
import os
import sys

import numpy as np

DATA = os.path.expanduser("~/yellow_phoenix/data")

SOURCES = [
    ("data/embeddings_3m.npy", "raw", 3_025_752),
    ("data/embeddings_oai439k.npy", "npy", None),
    ("data/embeddings_pubmed_new.npy", "npy", None),
    ("data/droplet_embeddings_aligned.npy", "npy", 106_298),
]

CHUNK = 50_000
HASH_BYTES = 64


def open_source(rel, kind, expect_n):
    path = os.path.join(os.path.expanduser("~/yellow_phoenix"), rel)
    if kind == "raw":
        assert os.path.getsize(path) == expect_n * 384 * 4, f"{rel}: bad size"
        mm = np.memmap(path, dtype=np.float32, mode="r").reshape(-1, 384)
        return mm
    arr = np.load(path)
    assert arr.dtype == np.float32 and arr.shape[1] == 384, f"{rel}: bad arr {arr.shape} {arr.dtype}"
    if expect_n:
        assert arr.shape[0] == expect_n, f"{rel}: {arr.shape[0]} != {expect_n}"
    return arr


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "real_5m.ism")

    print("[+] loading ITQ model")
    m = np.load(os.path.join(DATA, "itq_model_512_fixed.npz"))
    mean = m["mean"].astype(np.float32)
    proj = m["proj"].astype(np.float32)  # 384 x 512
    assert proj.shape == (384, 512) and mean.shape == (384,)

    sources = []
    total = 0
    for rel, kind, expect_n in SOURCES:
        arr = open_source(rel, kind, expect_n)
        sources.append((rel, arr))
        total += arr.shape[0]
        print(f"    {rel}: {arr.shape[0]:,} rows")
    print(f"[+] total rows: {total:,}  -> hashes {total*HASH_BYTES/1e9:.2f} GB")

    with open(out_path, "wb") as out:
        out.write(np.uint64(total).tobytes())
        pos = 0
        for rel, arr in sources:
            n = arr.shape[0]
            for lo in range(0, n, CHUNK):
                hi = min(lo + CHUNK, n)
                x = np.asarray(arr[lo:hi], dtype=np.float32)
                bits = ((x - mean) @ proj) > 0
                codes = np.packbits(bits, axis=1)
                assert codes.shape == (hi - lo, HASH_BYTES)
                out.write(codes.tobytes())
                pos += hi - lo
                if pos % 500_000 < CHUNK:
                    print(f"    encoded {pos:,}/{total:,}")
            del arr
        # ids: global 0-based index
        id_buf = np.arange(total, dtype=np.uint64)
        for lo in range(0, total, CHUNK * 16):
            hi = min(lo + CHUNK * 16, total)
            out.write(id_buf[lo:hi].tobytes())
    size = os.path.getsize(out_path)
    assert size == 8 + total * (HASH_BYTES + 8), f"size {size} != expected"
    print(f"[+] wrote {out_path} ({size/1e9:.3f} GB, {total:,} rows)")

    # sanity: reload a few rows, verify bit stats and id roundtrip
    with open(out_path, "rb") as f:
        cnt = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        assert cnt == total
        f.seek(8 + 12345 * HASH_BYTES)
        h = np.frombuffer(f.read(HASH_BYTES), dtype=np.uint8)
        f.seek(8 + total * HASH_BYTES + 12345 * 8)
        rid = np.frombuffer(f.read(8), dtype=np.uint64)[0]
    print(f"[+] sanity: row 12345 mean-bits={np.unpackbits(h).mean():.3f} id={rid}")


if __name__ == "__main__":
    main()
