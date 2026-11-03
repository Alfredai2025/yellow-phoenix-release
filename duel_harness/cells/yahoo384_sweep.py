#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MAP CELL — Yahoo-384 ledge sweep: PCA-rotate a 384-d embedding corpus,
then per-dim 4-bit quantizers on the top-M PCA components (M = 2*bytes),
bytes in {96, 128, 160, 176, 192, 208}. Flat-ruler (code-score) recall@10 —
the ledge test of the map paper's supplement: does recall jump when the
quantizer covers all 384 dims (192B)?

DATA NOTE: the published numbers used the campaign's Yahoo-384 payload
(Yahoo Answers answers embedded with a MiniLM-class encoder, 384-d, cosine;
a VIBE-campaign artifact that is NOT redistributed — see README). This
script takes any 384-d f32 payload + gt.npy + query embeddings; the protocol
is fully specified here.

Campaign recipe: unit-normalize corpus + queries at load; mean-center on the
first 400k rows; PCA via eigh of the train covariance (rows = top
components); per-dim MiniBatchKMeans(16, batch_size=8192, n_init=3,
max_iter=20, random_state=dim index) on the 400k rotated train rows; encode
each vector by nearest level on covered dims; scores = squared L2 of the
covered-dim reconstruction in PCA space; recall vs gt first-10.

The campaign run scored with torch (MPS); this port is numpy-only. Tie
order inside the top-10 boundary can differ in the last ulp, so treat Yahoo
cells as matched to ~3e-4 (the paper's documented boundary-instability
band); the SIFT/GloVe/Yandex cells are the exact-match anchors.

Usage:
  python cells/yahoo384_sweep.py --base payload.f32 --gt gt.npy --emb emb.npy \
      --query-offset 671164 --work-dir WORK --out-dir OUT
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rulers import merge_into, unit_normalize  # noqa: E402

NQ = 1000
TR = 400_000
D = 384
BYTES = [96, 128, 160, 176, 192, 208]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="raw 384-d f32 payload (n x 384)")
    p.add_argument("--gt", required=True, help="gt.npy (nq, >=10)")
    p.add_argument("--emb", required=True, help="query embeddings f32 (m x 384)")
    p.add_argument("--query-offset", type=int, default=671164,
                   help="first row of --emb to use as the 1000 queries")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bytes", nargs="+", type=int, default=BYTES)
    args = p.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    base_raw = np.memmap(args.base, dtype=np.float32).reshape(-1, D)
    gt = np.load(args.gt)[:NQ]
    Q = np.ascontiguousarray(
        np.load(args.emb, mmap_mode="r")[
            args.query_offset:args.query_offset + NQ]).astype(np.float32)
    base = unit_normalize(base_raw)          # ~1GB in RAM, fine
    del base_raw
    Q = unit_normalize(Q)
    N = len(base)
    print(f"Yahoo-384: {N} x {D}, queries {NQ}", flush=True)

    mu = base[:TR].mean(0, keepdims=True)
    Xc = (base[:TR] - mu).astype(np.float32)
    cov = (Xc.T @ Xc) / TR
    evals, evecs = np.linalg.eigh(cov)
    R = evecs[:, ::-1].T.astype(np.float32)   # PCA rows, top-first
    print(f"PCA done; top-half var share "
          f"{evals[::-1][:192].sum() / evals.sum():.3f} ({time.time() - t0:.0f}s)",
          flush=True)
    Z = Xc @ R.T                             # train in PCA space (400k, 384)

    out = os.path.join(args.out_dir, "sweep_yahoo384.csv")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for row in csv.reader(f):
                if row:
                    done.add(int(row[0]))

    for B in args.bytes:
        M = min(2 * B, D)                    # cap at D (208B would need 416)
        if B in done:
            print(f"B={B} already in CSV, skipping", flush=True)
            continue
        tB = time.time()
        cp = os.path.join(args.work_dir, f"yahoo_sweep_{B}B_codes.npy")
        bp = os.path.join(args.work_dir, f"yahoo_sweep_{B}B_books.npy")
        if os.path.exists(cp) and os.path.exists(bp):
            codes = np.load(cp)
            books = np.load(bp)
        else:
            from sklearn.cluster import MiniBatchKMeans
            books = np.stack([
                MiniBatchKMeans(n_clusters=16, batch_size=8192, n_init=3,
                                max_iter=20, random_state=j)
                .fit(Z[:, j:j + 1]).cluster_centers_[:, 0]
                for j in range(M)])
            codes = np.empty((N, M), dtype=np.uint8)
            for lo in range(0, N, 200_000):
                hi = min(lo + 200_000, N)
                z = ((base[lo:hi] - mu) @ R.T).astype(np.float32)
                codes[lo:hi] = np.abs(z[:, :M, None] - books[None, :, :]).argmin(2)
            np.save(cp, codes)
            np.save(bp, books)
        # score: squared L2 on covered dims in PCA space (larger = farther).
        # Per-dim accumulation (memory-safe numpy port of the campaign's
        # torch loop): dist += (q_j - book_j[code_j])^2 over covered dims.
        Zq = (Q @ R.T).astype(np.float32)[:, :M]
        gbd = np.full((NQ, 10), np.float32(1e30), np.float32)
        gbi = np.zeros((NQ, 10), np.int64)
        for lo in range(0, N, 50_000):
            hi = min(lo + 50_000, N)
            dist = np.zeros((NQ, hi - lo), np.float32)
            for j in range(M):
                dist += (Zq[:, j, None] - books[j, codes[lo:hi, j]][None, :]) ** 2
            part = np.argpartition(dist, 10, axis=1)[:, :10]
            gbd, gbi = merge_into(-gbd, gbi, -np.take_along_axis(dist, part, 1),
                                  part + lo)
        rec = 0.0
        for j in range(NQ):
            rec += len(set(gbi[j].tolist()) & set(gt[j].tolist())) / 10.0
        rec /= NQ
        cov_d = min(M, D)
        with open(out, "a", newline="") as f:
            csv.writer(f).writerow([B, M, cov_d, f"{rec:.4f}",
                                    int(time.time() - tB)])
        print(f"Yahoo384 B={B} M={M} covered={cov_d}/384 recall={rec:.4f} "
              f"({time.time() - tB:.0f}s)", flush=True)
        del codes
    print("YAHOO-384 SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
