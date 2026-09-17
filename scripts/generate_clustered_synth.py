#!/usr/bin/env python3
"""Generate clustered-Gaussian synthetic embeddings and ITQ-encode them.

Unlike generate_all_synthetic.py (i.i.d. uniform random bytes — an unlearnable
NN task), this produces data with REAL neighborhood structure:
  - cluster centers uniform on the 384-d unit sphere
  - points = normalize(center + sigma * gaussian_noise)
  - encoded with the REAL itq512_W_mean.bin model (mean-subtract, W dim-major,
    MSB-first packbits) so hashes live in the same convention/space as the
    real 5.1M corpus.

With this data, perturb/exclude_self recall against brute-force GT is a
meaningful probe of index quality at scale.

Output: raw ISM format (same as generate_all_synthetic.py):
  count(u64 LE) + count*64 hash bytes + count*u64 ids   (ids = 0..n-1)

Usage:
  python3 generate_clustered_synth.py --n 1000000 --clusters 1000 \
      --out data/synth_clustered_1m.ism
"""
import argparse
import numpy as np
import os
import sys
import time

MODEL = "/Users/mac/yellow_phoenix_mobile/YPPhone/Resources/itq512_W_mean.bin"
DIM = 384
N_BITS = 512
CHUNK = 100_000


def load_itq(path):
    b = np.fromfile(path, dtype=np.uint8)
    assert bytes(b[:8]) == b"YPITQ512", "bad ITQ magic"
    n_dims = int.from_bytes(b[8:12], "little")
    assert n_dims == DIM
    # 12-byte header: magic(8) + n_dims(4); then mean; then W (dim-major d*512+b)
    mean = np.fromfile(path, dtype=np.float32, count=DIM, offset=12)
    W = np.fromfile(path, dtype=np.float32, count=DIM * N_BITS, offset=12 + DIM * 4)
    W = W.reshape(DIM, N_BITS)
    return mean, W


def itq_encode(x, mean, W):
    """x: (rows, DIM) f32 -> (rows, 64) uint8, MSB-first (packbits)."""
    acc = (x - mean) @ W          # (rows, 512) f32
    return np.packbits(acc >= 0.0, axis=1)   # bit j of row i -> byte j//8, MSB-first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--clusters", type=int, required=True)
    ap.add_argument("--sigma", type=float, default=0.55,
                    help="within-cluster noise (relative to unit-norm centers)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()

    assert args.n % args.clusters == 0 or args.n >= args.clusters
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    mean, W = load_itq(args.model)
    print(f"ITQ model loaded: dim {DIM}, bits {N_BITS}", flush=True)

    # Cluster centers: uniform on the unit sphere.
    C = rng.normal(size=(args.clusters, DIM)).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)

    per = args.n // args.clusters
    print(f"n={args.n} clusters={args.clusters} (~{per} pts/cluster) sigma={args.sigma}",
          flush=True)

    tmp = args.out + ".part"
    with open(tmp, "wb") as f:
        f.write(args.n.to_bytes(8, "little"))
        done = 0
        labels = np.repeat(np.arange(args.clusters), per)
        while done < args.n:
            rows = min(CHUNK, args.n - done)
            lbl = labels[done:done + rows]
            noise = rng.normal(size=(rows, DIM)).astype(np.float32)
            x = C[lbl] + args.sigma * noise
            x /= np.linalg.norm(x, axis=1, keepdims=True)   # unit-norm, like MiniLM
            f.write(itq_encode(x, mean, W).tobytes())
            done += rows
            if done % 1_000_000 < CHUNK:
                print(f"  encoded {done}/{args.n} ({time.time()-t0:.0f}s)", flush=True)
        ids = np.arange(args.n, dtype="<u8")
        f.write(ids.tobytes())
    os.replace(tmp, args.out)

    sz = os.path.getsize(args.out)
    print(f"DONE: {args.out} ({sz/1e6:.0f} MB) in {time.time()-t0:.0f}s", flush=True)

    # Sanity: hamming distance between two points of the same cluster should be
    # far below 256 (random). Quick check on the first cluster's first two points.
    with open(args.out, "rb") as f:
        f.seek(8)
        h = np.fromfile(f, dtype=np.uint8, count=2 * 64).reshape(2, 64)
        hd_same = int(np.unpackbits(h[0] ^ h[1]).sum())
    print(f"sanity: hamming(point0, point1) same-cluster = {hd_same} (random ~256)",
          flush=True)


if __name__ == "__main__":
    main()
