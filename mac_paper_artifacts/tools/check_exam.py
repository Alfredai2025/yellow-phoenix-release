#!/usr/bin/env python3
"""check_exam.py — clean-room verifier for the Mac paper's candidate-depth claims.

Reproduces Mac paper Table 1 (candidate-depth effect) from the frozen exam
ONLY, plus the corpus vector store. No engine imports, numpy only.

Protocol (identical to the paper's scripts/test_candidate_depth.py):
  for candidate depth k in {50, 100, 200}:
    pool     = released candidate ids for the query, truncated to depth k
    scores   = cosine(query_emb, corpus_f16_vec) for each id in pool
    top10    = the 10 pool ids with highest cosine score
    R@10     = mean over queries of |top10 & GT_top10| / 10
  (queries and corpus vectors are both L2-normalized before the dot product;
   corpus vectors are the released float16 rows, upcast float32)

Candidate-depth 200 caveat (stated in the paper itself): the released exam
ships only 100 candidate ids per query, so depth 200 cannot be evaluated from
the released ids alone. If --hashes is given (corpus hash store, 64 bytes/doc),
an AUXILIARY clean-room protocol is also run: exact brute-force Hamming
top-200 against the full 5.1M hash store, re-ranked the same way. This is NOT
the paper's protocol (the paper used a graph index at ef=400); it is labeled
in-line as an auxiliary/scout number.

Expected values are read from benchmarks/claims_layer.json (shipped in this
bundle), not hardcoded here.

Usage:
  python3 check_exam.py \
      --exam frozen_exam_hybrid5m_20260916 \
      --vectors /path/to/floats_5m.f16 \
      [--hashes /path/to/real_5m.ism] \
      [--claims benchmarks/claims_layer.json] \
      [--depths 50 100 200]

Exit code 0 iff every REQUIRED depth reproduces its published value exactly
(at 4-decimal rounding). Auxiliary numbers never affect the exit code.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

CORPUS_N, DIM, F16_OFF, HASH_OFF = 5107508, 384, 32, 8


def cosine_rerank_r10(qemb, cand, gt10, X):
    """Exact f16 cosine re-rank of candidate rows; returns (R@10, ceiling)."""
    N = len(qemb)
    qn = qemb / np.linalg.norm(qemb, axis=1, keepdims=True)
    C = X[cand.ravel()].astype(np.float32).reshape(N, -1, DIM)
    cn = C / np.linalg.norm(C, axis=2, keepdims=True)
    s = np.einsum("nd,ncd->nc", qn, cn)
    top = np.argsort(-s, axis=1)[:, :10]
    ranked = np.take_along_axis(cand, top, axis=1)
    gt_sets = [set(r.tolist()) for r in gt10]
    r10 = float(np.mean([len(set(ranked[i].tolist()) & gt_sets[i]) / 10
                         for i in range(N)]))
    ceil = float(np.mean([len(set(cand[i].tolist()) & gt_sets[i]) / 10
                          for i in range(N)]))
    return r10, ceil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", required=True)
    ap.add_argument("--vectors", required=True, help="corpus float16 store (row = doc id)")
    ap.add_argument("--hashes", default=None, help="corpus hash store (optional; enables auxiliary Hamming top-200)")
    ap.add_argument("--claims", default=None)
    ap.add_argument("--depths", type=int, nargs="+", default=[50, 100, 200])
    a = ap.parse_args()

    exam = Path(a.exam)
    qemb = np.load(exam / "qemb.npy").astype(np.float32)
    gt10 = np.load(exam / "gt_top100_ids.npy")[:, :10]
    cand_ids = np.load(exam / "cand_ids.npy")
    N = len(qemb)
    print(f"exam: n={N} queries, released candidates per query: {cand_ids.shape[1]}")

    X = np.memmap(a.vectors, dtype=np.float16, mode="r",
                  offset=F16_OFF, shape=(CORPUS_N, DIM))

    expected = {}
    if a.claims and Path(a.claims).exists():
        cl = json.loads(Path(a.claims).read_text())["claims_layer"]["claims"]
        for c in cl:
            if c.get("metric") == "R@10" and "candidate_depth" in c:
                expected[c["candidate_depth"]] = c["value"]

    ok_all = True
    for k in a.depths:
        exp = expected.get(k)
        if k <= cand_ids.shape[1]:
            r10, ceil = cosine_rerank_r10(qemb, cand_ids[:, :k], gt10, X)
            got = round(r10, 4)
            ok = (exp is None) or (got == exp)
            ok_all &= ok
            print(f"k={k:<4} released-ids re-rank: R@10 = {got:.4f}"
                  + (f"  expected {exp:.4f}  {'PASS' if ok else 'FAIL'}" if exp is not None else "")
                  + f"  (candidate ceiling {ceil:.4f})")
        else:
            print(f"k={k:<4} SKIP: released exam ships only {cand_ids.shape[1]} "
                  f"candidate ids per query; depth {k} needs the (unreleased) "
                  f"graph-search candidate set — disclosed in the paper itself.")
            if exp is not None:
                print(f"      published value {exp:.4f} is therefore NOT "
                      f"reproducible from released artifacts alone: FAIL")
                ok_all = False
            if a.hashes:
                H = np.memmap(a.hashes, dtype=np.uint64, mode="r",
                              offset=HASH_OFF, shape=(CORPUS_N, 8))
                qhash = np.load(exam / "qhash.npy")
                pool = np.zeros((N, k), dtype=np.int64)
                for i in range(N):
                    q = np.frombuffer(qhash[i].tobytes(), dtype=np.uint64)
                    d = np.bitwise_count(H ^ q).sum(axis=1)
                    pool[i] = np.argpartition(d, k)[:k]
                r10, ceil = cosine_rerank_r10(qemb, pool, gt10, X)
                print(f"      [AUXILIARY/scout — NOT the paper protocol] exact "
                      f"Hamming top-{k} pool from the hash store: "
                      f"R@10 = {round(r10, 4):.4f} (ceiling {ceil:.4f}) "
                      f"vs published {exp if exp else '?'}")

    print("\nVERDICT:", "ALL REQUIRED DEPTHS REPRODUCED" if ok_all
          else "MISMATCH — at least one published value not reproduced (see above)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
