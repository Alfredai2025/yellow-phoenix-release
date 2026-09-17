#!/usr/bin/env python3
"""Build the deduped true-5M real-papers shootout pair.

Streams float32 embedding sources in chunks, ITQ-encodes with the fixed
512-bit model, and writes the FlatIndex ISM file directly. Every encoded hash
is checked against a seen-set: the first row carrying a given hash is kept,
all later rows (same paper ingested under another source/id) are dropped.
This makes the final paper count equal to the count of unique contents.

Output (FlatIndex): [u64 count LE][count x 64B hashes][count x 8B ids]
ids = final 0-based row index.

Usage: build_5m_real_dedup.py [out_ism]
"""
import os
import sys

import numpy as np

DATA = os.path.expanduser("~/yellow_phoenix/data")

SOURCES = [
    ("data/embeddings_3m.npy", "raw"),
    ("data/embeddings_oai439k.npy", "npy"),
    ("data/embeddings_pubmed_new.npy", "npy"),
    ("data/embeddings_pubmed_new2.npy", "npy"),
    ("data/droplet_embeddings_aligned.npy", "npy"),
]

CHUNK = 50_000
HASH_BYTES = 64


def open_source(rel, kind):
    path = os.path.join(os.path.expanduser("~/yellow_phoenix"), rel)
    if kind == "raw":
        mm = np.memmap(path, dtype=np.float32, mode="r").reshape(-1, 384)
        assert mm.shape[0] == 3_025_752, f"{rel}: {mm.shape}"
        return mm
    arr = np.load(path)
    assert arr.dtype == np.float32 and arr.shape[1] == 384, f"{rel}: {arr.shape} {arr.dtype}"
    return arr


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "real_5m_dedup.ism")

    print("[+] loading ITQ model")
    m = np.load(os.path.join(DATA, "itq_model_512_fixed.npz"))
    mean = m["mean"].astype(np.float32)
    proj = m["proj"].astype(np.float32)
    assert proj.shape == (384, 512) and mean.shape == (384,)

    seen = set()
    kept = 0
    scanned = 0
    id_buf = []

    with open(out_path, "wb") as out:
        out.write(b"\x00" * 8)  # count placeholder
        for rel, kind in SOURCES:
            arr = open_source(rel, kind)
            n = arr.shape[0]
            print(f"[+] {rel}: {n:,} rows")
            for lo in range(0, n, CHUNK):
                hi = min(lo + CHUNK, n)
                x = np.asarray(arr[lo:hi], dtype=np.float32)
                bits = ((x - mean) @ proj) > 0
                codes = np.packbits(bits, axis=1)
                assert codes.shape == (hi - lo, HASH_BYTES)
                cv = np.ascontiguousarray(codes).view(np.dtype((np.void, HASH_BYTES))).ravel()
                for j in range(len(cv)):
                    h = cv[j].tobytes()
                    if h not in seen:
                        seen.add(h)
                        out.write(h)
                        id_buf.append(kept)
                        kept += 1
                scanned += hi - lo
                if scanned % 500_000 < CHUNK:
                    print(f"    scanned {scanned:,} | kept {kept:,} | dup {scanned-kept:,}")
            del arr
        # ids
        ids = np.array(id_buf, dtype=np.uint64)
        for lo in range(0, kept, CHUNK * 16):
            out.write(ids[lo:lo + CHUNK * 16].tobytes())
        # patch count
        out.seek(0)
        out.write(np.uint64(kept).tobytes())

    size = os.path.getsize(out_path)
    assert size == 8 + kept * (HASH_BYTES + 8), f"size {size}"
    print(f"[+] wrote {out_path}: {kept:,} unique papers "
          f"({size/1e9:.3f} GB), dropped {scanned-kept:,} dup rows")

    with open(out_path, "rb") as f:
        cnt = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        assert cnt == kept
        f.seek(8 + 999999 * HASH_BYTES)
        h = np.frombuffer(f.read(HASH_BYTES), dtype=np.uint8)
        f.seek(8 + kept * HASH_BYTES + 999999 * 8)
        rid = np.frombuffer(f.read(8), dtype=np.uint64)[0]
    print(f"[+] sanity: row 999999 mean-bits={np.unpackbits(h).mean():.3f} id={rid}")


if __name__ == "__main__":
    main()
