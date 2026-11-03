#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MAP CELL — Yandex T2I-10M per-dim Lloyd-Max ladder, bits/dim = 1..8
(25..200 bytes/vector), BOTH rulers per depth. This is the our-codec side of
the published 10M duel cells (see expected_outputs/published_cells.csv).

Dataset (public, CC-BY, as distributed — see README for fetch):
  base_10m_norm.f32   unit-normalized 10,000,000 x 200 f32 payload (the
                      script builds this cache from the raw downloaded base)
  queries_100k.fbin   100,000 x 200 f32 (uint32 nq, uint32 dim, then rows)
  gt.npy              (nq, 100) int ground-truth indices, top-100 by inner
                      product as distributed
Protocol: 1000 queries, cosine, gt first-10. Engine ruler re-rank fetches
shortlist rows SEQUENTIALLY from the memmap (random order = IO death).

Campaign training recipe: train on the first 400k normalized rows, per-dim
MiniBatchKMeans(K=2^b, batch_size=8192, n_init=3, max_iter=20,
random_state=dim index). Books/codes cached under --work-dir per depth.

Usage:
  python cells/yandex_10m_ladder.py --data-dir DATA --work-dir WORK \
      --out-dir OUT [--bits 1 2 3 4 5 6 7 8]
"""

import argparse
import csv
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codec import PerDimCodec                       # noqa: E402
from rulers import dual_ruler, unit_normalize       # noqa: E402

NQ = 1000
TR = 400_000
N, D = 10_000_000, 200


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bits", nargs="+", type=int, default=list(range(1, 9)))
    args = p.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    nf = os.path.join(args.data_dir, "base_10m_norm.f32")
    if os.path.exists(nf):
        base = np.memmap(nf, dtype=np.float32, mode="r", shape=(N, D))
    else:
        raw = np.memmap(os.path.join(args.data_dir, "base_10m.f32"),
                        dtype=np.float32, mode="r", shape=(N, D))
        print("normalizing 10M x 200 payload (one-time, 8GB cache)...", flush=True)
        base = np.memmap(nf, dtype=np.float32, mode="w+", shape=(N, D))
        for lo in range(0, N, 1_000_000):
            hi = min(lo + 1_000_000, N)
            base[lo:hi] = unit_normalize(np.asarray(raw[lo:hi]))
        base.flush()
        del raw

    gt = np.load(os.path.join(args.data_dir, "gt.npy"))[:NQ]
    assert gt.max() < N
    qp = os.path.join(args.data_dir, "queries_100k.fbin")
    with open(qp, "rb") as f:
        nq, dq = struct.unpack("<II", f.read(8))
    Q = np.memmap(qp, dtype=np.float32, mode="r", offset=8,
                  shape=(nq, dq))[:NQ].copy()
    Q = unit_normalize(Q)

    out = os.path.join(args.out_dir, "ladder_yandex10m.csv")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for row in csv.reader(f):
                if row:
                    done.add((int(row[0]), row[3]))

    for b in args.bits:
        codec = PerDimCodec(D, b)
        cp = os.path.join(args.work_dir, f"yandex_ladder_{b}b_codes.npy")
        if os.path.exists(cp):
            codec.mu = np.load(os.path.join(args.work_dir,
                                            f"yandex_ladder_{b}b_mu.npy"))
            codec.books = np.load(os.path.join(args.work_dir,
                                               f"yandex_ladder_{b}b_books.npy"))
            codes = np.load(cp)
        else:
            print(f"bits={b}: training {D} 1-d kmeans (K={codec.k})...", flush=True)
            codec.fit(np.asarray(base[:TR]), seeds=list(range(D)), max_iter=20)
            np.save(os.path.join(args.work_dir, f"yandex_ladder_{b}b_books.npy"),
                    codec.books)
            np.save(os.path.join(args.work_dir, f"yandex_ladder_{b}b_mu.npy"),
                    codec.mu)
            print(f"  encoding 10M x {D} via searchsorted "
                  f"({time.time() - t0:.0f}s)...", flush=True)
            codes = np.empty((N, D), dtype=np.uint8)
            for j in range(D):
                codes[:, j] = codec.encode(np.asarray(base[:, j])[:, None])[:, 0]
            np.save(cp, codes)
            print(f"  encoded ({time.time() - t0:.0f}s)", flush=True)
        rf, rw = dual_ruler(lambda lo, hi: codec.decode(codes[lo:hi]),
                            base, Q, gt, "cosine", chunk=250_000,
                            rescore_renormalize=True)
        B = D * b // 8
        with open(out, "a", newline="") as f:
            for tag, rec in (("flat", rf), ("w10k", rw)):
                if (b, tag) not in done:
                    csv.writer(f).writerow(
                        [b, codec.k, B, tag, f"{rec:.4f}", int(time.time() - t0)])
        print(f"YandexT2I {b}b/dim ({B}B): flat={rf:.4f} w10k={rw:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del codes
    print("YANDEX 10M LADDER DONE", flush=True)


if __name__ == "__main__":
    main()
