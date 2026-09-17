#!/usr/bin/env python3
"""Generate a CONSISTENT synthetic 1M dataset for the iOS re-rank benchmark.

The current data/synth_1m.ism holds uniformly random hashes with no related
embeddings, so a cosine re-rank stage cannot be demonstrated against it.
This script regenerates the 1M set the way production data looks:

  embeddings (384-d MiniLM-like) --ITQ--> V (512-d float, pre-binarization)
                                      |--> bits = V >= 0 --> 512-bit hashes

Outputs (in data/):
  synth_1m_embeddings.npy  (1_000_000 x 512 float32, L2-normalized, ~2.0 GiB)
  synth_1m.ism             (u64 count + count*64 hash bytes + count*u64 ids)

The HNSW graph is built afterwards with:
  cargo run --release --bin build_hnsw_from_ism -- data/synth_1m.ism data/synth_1m_hnsw.bin
"""
import shutil
import struct
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

EMB_PATH = DATA / "paper_embeddings_1m_synthetic.npy"
ITQ_PATH = DATA / "itq_model_512_fixed.npz"
OUT_NPY = DATA / "synth_1m_embeddings.npy"
OUT_ISM = DATA / "synth_1m.ism"
BACKUP_DIR = DATA / "_backup_random_1m"

CHUNK = 100_000


def main() -> None:
    t0 = time.time()

    # --- backup old random 1M artifacts (first run only) ---
    BACKUP_DIR.mkdir(exist_ok=True)
    for name in ("synth_1m.ism", "synth_1m_hnsw.bin"):
        src = DATA / name
        dst = BACKUP_DIR / name
        if src.exists() and not dst.exists():
            print(f"Backing up {name} -> {dst.relative_to(ROOT)}")
            shutil.copy2(src, dst)

    # --- load inputs ---
    print(f"Loading embeddings {EMB_PATH.name} (mmap)...")
    X = np.load(EMB_PATH, mmap_mode="r")
    n, d_in = X.shape
    print(f"  shape {X.shape} {X.dtype}")

    z = np.load(ITQ_PATH)
    mean = z["mean"].astype(np.float64)
    proj = z["proj"].astype(np.float64)
    n_bits = int(z["n_bits"])
    assert n_bits == 512 and proj.shape == (d_in, 512), (n_bits, proj.shape)
    print(f"ITQ model: mean{mean.shape} proj{proj.shape} bits={n_bits}")

    out = np.lib.format.open_memmap(OUT_NPY, mode="w+", dtype="<f4", shape=(n, 512))
    ism_hashes = np.empty((n, 64), dtype=np.uint8)

    print("Rotating, normalizing, binarizing in chunks...")
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        block = np.asarray(X[start:end], dtype=np.float64)
        V = (block - mean) @ proj                       # (chunk, 512) pre-binarization
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        Vn = (V / norms).astype(np.float32)             # L2-normalized for cosine dot
        out[start:end] = Vn
        bits = (V >= 0).astype(np.uint8)                # binarize the UN-normalized V
        ism_hashes[start:end] = np.packbits(bits, axis=1)
        if (start // CHUNK) % 5 == 0:
            print(f"  {end:,}/{n:,}  ({time.time()-t0:.1f}s)")

    out.flush()
    del out

    print(f"Writing {OUT_ISM.name}...")
    with open(OUT_ISM, "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(ism_hashes.tobytes())
        f.write(np.arange(n, dtype=np.uint64).tobytes())

    print("\nDone.")
    print(f"  {OUT_NPY.name}: {OUT_NPY.stat().st_size:,} bytes")
    print(f"  {OUT_ISM.name}: {OUT_ISM.stat().st_size:,} bytes")
    print(f"  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
