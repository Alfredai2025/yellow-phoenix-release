#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Duel runner: Google's PUBLIC RSLM reference codecs (Apache-2.0, vendored
under vendor/rslm/) under OUR rulers — the byte-matched comparison of the
duel/map paper.

Their codec, our measurement protocol (identical to the campaign scripts):
  flat = recall@10, code scores only, full pool
  w10k = recall@10 after exact rescore of the code-score top-10,000

Subcommands (datasets as distributed, 1000 queries, bundle ground truth):
  glove      GloVe-100-angular hdf5 (1,183,514 x 100, cosine; encode the
             unit-normalized corpus, like the campaign run)
  sift       SIFT-128-euclidean hdf5 (1M x 128, L2; RSLM compresses the raw
             corpus — its norm correction is inside the codec)
  yandex10m  Yandex T2I-10M (10M x 200 f32 base + 100k-query fbin + gt.npy,
             cosine on the unit-normalized base; memory-safe streaming —
             the RSLM full decompress would build a 10M-python-list bomb,
             so slices of 250k are decompressed and merged)

Output CSV rows: codec,bytes,tag,recall,sec (append-only, resumable).

These are the vendor-side cells of the published duel table; the our-codec
side is cells/sift_glove_ladder.py + cells/yandex_10m_ladder.py, and the
byte-matched pairings with the published numbers are in
expected_outputs/published_cells.csv.
"""

import argparse
import csv
import importlib.util
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rulers import dual_ruler, unit_normalize  # noqa: E402

NQ = 1000
CODEC_CLASSES = ["Rslm1Codec", "Rslm2Codec", "Rslm3Codec", "Rslm4Codec", "Rslm4LiteCodec"]


def load_rslm():
    spec = importlib.util.spec_from_file_location(
        "rslm_codec", os.path.join(HERE, "vendor", "rslm", "cells_codec.py"))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    return rc


def load_hdf5(data_dir, name):
    import h5py
    f = h5py.File(os.path.join(data_dir, name), "r")
    X = np.asarray(f["train"], dtype=np.float32)
    Q = np.asarray(f["test"][:NQ], dtype=np.float32)
    GT = np.asarray(f["neighbors"][:NQ, :10])
    return X, Q, GT


def append_rows(out, done, rows):
    with open(out, "a", newline="") as f:
        for r in rows:
            if (r[0], r[2]) in done:
                continue
            csv.writer(f).writerow(r)


def load_done(out):
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for row in csv.reader(f):
                if row:
                    done.add((row[0], row[2]))
    return done


def run_glove(args):
    """Vendored RSLM codecs on GloVe-100-angular under the dual ruler."""
    rc = load_rslm()
    t0 = time.time()
    X, Q, GT = load_hdf5(args.data_dir, "glove-100-angular.hdf5")
    Xn = unit_normalize(X)
    Qn = unit_normalize(Q)
    del X, Q
    N, d = Xn.shape
    print(f"GloVe: {N} x {d}, queries {NQ}", flush=True)
    out = os.path.join(args.out_dir, "vendor_rslm_glove.csv")
    done = load_done(out)
    for cname in args.codecs:
        cls = getattr(rc, cname)
        c = cls(d, is_residual=False)
        nbytes = c.compressed_size
        if (cname, "flat") in done and (cname, "w10k") in done:
            continue
        tp = time.time()
        packed = c.compress(Xn)                      # encode unit-normalized corpus
        recon = c.decompress(packed, as_numpy=True)
        del packed
        print(f"{cname}: encoded {nbytes}B/vec ({time.time()-tp:.0f}s)", flush=True)
        rn = unit_normalize(recon)
        del recon
        rf, rw = dual_ruler(lambda lo, hi: rn[lo:hi], Xn, Qn, GT, "cosine",
                            chunk=200_000)
        append_rows(out, done, [
            [cname, nbytes, "flat", f"{rf:.4f}", int(time.time() - t0)],
            [cname, nbytes, "w10k", f"{rw:.4f}", int(time.time() - t0)]])
        print(f"VENDOR {cname} ({nbytes}B): flat={rf:.4f} w10k={rw:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        done |= {(cname, "flat"), (cname, "w10k")}
    print("VENDOR GLOVE DONE", flush=True)


def run_sift(args):
    """Vendored RSLM codecs on SIFT-128-euclidean under the dual ruler."""
    rc = load_rslm()
    t0 = time.time()
    X, Q, GT = load_hdf5(args.data_dir, "sift-128-euclidean.hdf5")
    N, d = X.shape
    out = os.path.join(args.out_dir, "vendor_rslm_sift.csv")
    done = load_done(out)
    for cname in args.codecs:
        cls = getattr(rc, cname)
        c = cls(d, is_residual=False)
        nbytes = c.compressed_size
        if (cname, "flat") in done and (cname, "w10k") in done:
            continue
        packed = c.compress(X)                       # RAW SIFT; norm fix inside
        def recon_at(lo, hi):
            return c.decompress(packed[lo * nbytes:hi * nbytes], as_numpy=True)
        rf, rw = dual_ruler(recon_at, X, Q, GT, "l2", chunk=250_000)
        del packed
        append_rows(out, done, [
            [cname, nbytes, "flat", f"{rf:.4f}", int(time.time() - t0)],
            [cname, nbytes, "w10k", f"{rw:.4f}", int(time.time() - t0)]])
        print(f"VENDOR-SIFT {cname} ({nbytes}B): flat={rf:.4f} w10k={rw:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        done |= {(cname, "flat"), (cname, "w10k")}
    print("VENDOR SIFT DONE", flush=True)


def run_yandex10m(args):
    """Vendored RSLM codecs on Yandex T2I-10M under the dual ruler.

    Memory-safe (campaign machine had 18 GB RAM): packed bytes are cached to
    disk per codec (re-runnable without re-encode); decompress is called per
    250k-vector slice and the ruler merge streams slice by slice.
    """
    rc = load_rslm()
    t0 = time.time()
    n, d = 10_000_000, 200
    base = np.memmap(os.path.join(args.data_dir, "base_10m_norm.f32"),
                     dtype=np.float32, mode="r", shape=(n, d))
    gt = np.load(os.path.join(args.data_dir, "gt.npy"))[:NQ]
    assert gt.max() < n
    with open(os.path.join(args.data_dir, "queries_100k.fbin"), "rb") as f:
        nq, dq = struct.unpack("<II", f.read(8))
    Q = np.memmap(os.path.join(args.data_dir, "queries_100k.fbin"),
                  dtype=np.float32, mode="r", offset=8, shape=(nq, dq))[:NQ].copy()
    Q = unit_normalize(Q)
    out = os.path.join(args.out_dir, "vendor_rslm_yandex10m.csv")
    done = load_done(out)
    for cname in args.codecs:
        cls = getattr(rc, cname)
        c = cls(d, is_residual=False)
        nbytes = c.compressed_size
        if (cname, "flat") in done and (cname, "w10k") in done:
            continue
        pkp = os.path.join(args.work_dir, f"vendor10m_{cname}.bin")
        if os.path.exists(pkp):
            packed = open(pkp, "rb").read()
            print(f"{cname}: loaded cached packed ({len(packed) / 1e6:.0f}MB)", flush=True)
        else:
            te = time.time()
            packed = c.compress(base)                # memmap input, chunked internally
            with open(pkp, "wb") as f:
                f.write(packed)
            print(f"{cname}: encoded {nbytes}B/vec ({time.time()-te:.0f}s)", flush=True)
        def recon_at(lo, hi):
            recon = c.decompress(packed[lo * nbytes:hi * nbytes], as_numpy=True)
            return unit_normalize(recon)
        rf, rw = dual_ruler(recon_at, base, Q, gt, "cosine", chunk=250_000)
        del packed
        append_rows(out, done, [
            [cname, nbytes, "flat", f"{rf:.4f}", int(time.time() - t0)],
            [cname, nbytes, "w10k", f"{rw:.4f}", int(time.time() - t0)]])
        print(f"VENDOR10M {cname} ({nbytes}B): flat={rf:.4f} w10k={rw:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        done |= {(cname, "flat"), (cname, "w10k")}
    print("VENDOR 10M DONE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="dataset", required=True)
    for name, fn, base_hint in (
            ("glove", run_glove, "dir with glove-100-angular.hdf5"),
            ("sift", run_sift, "dir with sift-128-euclidean.hdf5"),
            ("yandex10m", run_yandex10m,
             "dir with base_10m_norm.f32 (unit-normalized 10M x 200), "
             "queries_100k.fbin, gt.npy")):
        sp = sub.add_parser(name)
        sp.add_argument("--data-dir", required=True, help=base_hint)
        sp.add_argument("--out-dir", required=True, help="CSV output directory")
        sp.add_argument("--work-dir", default=None,
                        help="scratch dir (packed-bytes cache; default = out-dir)")
        sp.add_argument("--codecs", nargs="+", default=CODEC_CLASSES,
                        choices=CODEC_CLASSES, help="RSLM codec classes to run")
        sp.set_defaults(fn=fn)
    args = p.parse_args()
    if args.work_dir is None:
        args.work_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
