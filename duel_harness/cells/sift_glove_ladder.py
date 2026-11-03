#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MAP CELL — SIFT-1M (L2) + GloVe-1M (cosine) per-dim Lloyd-Max ladder,
bits/dim = 1..8, BOTH rulers per depth. This is the our-codec side of the
published SIFT/GloVe duel cells (see expected_outputs/published_cells.csv).

Datasets (ann-benchmarks hdf5, as distributed):
  sift-128-euclidean.hdf5   1,000,000 x 128 train, 10,000 test, L2
  glove-100-angular.hdf5    1,183,514 x 100 train, 10,000 test, cosine
Ground truth = the bundle's first-10 neighbors; 1000 queries per measurement.

Campaign training recipe (for exact level reproduction): train on the first
400k rows, per-dim MiniBatchKMeans(K=2^b, batch_size=8192, n_init=3,
random_state=0). Books/codes are cached under --work-dir so a re-run skips
re-training; recall cells are exactly re-runnable from the cache.

Usage:
  python cells/sift_glove_ladder.py --data-dir DATA --work-dir WORK \
      --out-dir OUT [--datasets sift glove] [--bits 1 2 3 4 5 6 7 8]
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codec import PerDimCodec            # noqa: E402
from rulers import dual_ruler, unit_normalize  # noqa: E402

NQ = 1000
TR = 400_000

DATASETS = {
    "sift": ("sift-128-euclidean.hdf5", "l2"),
    "glove": ("glove-100-angular.hdf5", "cosine"),
}


def run(name, args, t0):
    fn, metric = DATASETS[name]
    import h5py
    f = h5py.File(os.path.join(args.data_dir, fn), "r")
    X = np.asarray(f["train"], np.float32)
    Q = np.asarray(f["test"][:NQ], np.float32)
    GT = np.asarray(f["neighbors"][:NQ, :10])
    N, d = X.shape
    if metric == "cosine":
        X = unit_normalize(X)
        Q = unit_normalize(Q)
    out = os.path.join(args.out_dir, f"ladder_{name}.csv")
    done = set()
    if os.path.exists(out):
        with open(out) as f2:
            for row in csv.reader(f2):
                if row and row[0] == name:
                    done.add((int(row[1]), row[3]))
    for b in args.bits:
        K = 1 << b
        B = d * b // 8
        codec = PerDimCodec(d, b)
        bp = os.path.join(args.work_dir, f"ladder_{name}_{b}b_books.npy")
        cp = os.path.join(args.work_dir, f"ladder_{name}_{b}b_codes.npy")
        if os.path.exists(bp) and os.path.exists(cp):
            codec.mu = np.load(os.path.join(args.work_dir,
                                            f"ladder_{name}_{b}b_mu.npy"))
            codec.books = np.load(bp)
            codes = np.load(cp)
        else:
            codec.fit(X[:TR], seeds=0)
            np.save(bp, codec.books)
            np.save(os.path.join(args.work_dir, f"ladder_{name}_{b}b_mu.npy"),
                    codec.mu)
            codes = codec.encode(X)
            np.save(cp, codes)
            print(f"{name} b={b} encoded ({time.time() - t0:.0f}s)", flush=True)
        rf, rw = dual_ruler(lambda lo, hi: codec.decode(codes[lo:hi]),
                            X, Q, GT, metric, chunk=200_000)
        with open(out, "a", newline="") as f2:
            for tag, v in (("flat", rf), ("w10k", rw)):
                if (b, tag) not in done:
                    csv.writer(f2).writerow(
                        [name, b, B, tag, f"{v:.4f}", int(time.time() - t0)])
        print(f"LADDER {name} b={b} ({B}B): flat={rf:.4f} w10k={rw:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del codes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="dir with the hdf5 bundles")
    p.add_argument("--work-dir", required=True, help="books/codes cache dir")
    p.add_argument("--out-dir", required=True, help="CSV output dir")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS),
                   choices=list(DATASETS))
    p.add_argument("--bits", nargs="+", type=int, default=list(range(1, 9)))
    args = p.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    for name in args.datasets:
        run(name, args, t0)
    print("SIFT/GLOVE LADDER DONE", flush=True)


if __name__ == "__main__":
    main()
