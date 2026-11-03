#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The dual ruler — the two evaluation protocols of the duel/map paper.

Every codec cell is measured under BOTH rulers:

  * flat_recall  — "the field's standard ruler": recall@10 of the code scores
    alone over the FULL pool. A codec-only lottery: no re-rank, no engine.

  * engine_ruler — "the deployment semantics": recall@10 after an exact
    re-rank of the code-score top-W shortlist (W = 10,000). This is what a
    real engine does: cheap codes propose, exact vectors dispose.

Protocol (matches the map paper, 1000 queries, ground truth = the bundle's
first-10 neighbors, cosine or L2 as distributed):

  * metric="cosine": corpus and queries are unit-normalized; code scores are
    dots of unit-normalized reconstructions; the exact re-rank is a dot on
    the original normalized corpus rows.
  * metric="l2": code scores are negated squared L2 of reconstructions
    (2*r.q - |r|^2 - |q|^2, larger = closer); the exact re-rank is the true
    squared L2 on the original corpus rows.

The streaming top-k merge (chunked argpartition with a -1e30 sentinel, worst
case carried across chunks) is byte-for-byte the convention of the campaign
scripts — it was triple-verified against brute force — and the per-chunk
score expressions preserve the campaign op order so cached codes reproduce
the published cells exactly.
"""

import numpy as np

SENTINEL = np.float32(-1e30)


def merge_into(bv, bi, vals, idx):
    """Merge a chunk's top candidates into the running global top-k.

    bv/bi: running (nq, k) best scores/indices. vals/idx: chunk's (nq, k').
    Returns updated (bv, bi). Sentinel scores rank last; identical behavior
    to the campaign scripts (verified vs brute force).
    """
    ad = np.concatenate([bv, vals], 1)
    ai = np.concatenate([bi, idx], 1)
    sel = np.argpartition(-ad, vals.shape[1], axis=1)[:, :vals.shape[1]]
    return np.take_along_axis(ad, sel, 1), np.take_along_axis(ai, sel, 1)


def _score_chunk(recon, Q, metric, qx):
    """Code scores of one reconstruction chunk. Returns (nq, chunk) C-contig.

    recon: (chunk, d) f32 (RAW reconstruction; cosine normalizes internally).
    qx:    (nq,) f32 |q|^2 (l2 only).
    """
    if metric == "cosine":
        rn = recon / np.maximum(np.linalg.norm(recon, axis=1, keepdims=True), 1e-12)
        return np.ascontiguousarray((rn @ Q.T).T)
    if metric == "l2":
        rx = (recon * recon).sum(1)
        return np.ascontiguousarray((2.0 * (recon @ Q.T) - rx[:, None] - qx[None, :]).T)
    raise ValueError(metric)


def _rescore(X, Q, metric, idx, j, renorm=False):
    """Exact re-rank of shortlist rows idx for query j against originals X.

    renorm: re-normalize the fetched rows before the cosine dot (the Yandex
    campaign cell did this; all other cells score the stored rows directly).
    """
    cand = X[idx]
    if metric == "cosine":
        if renorm:
            cand = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
        top = np.argpartition(-(cand @ Q[j]), 10)[:10]
    else:
        top = np.argpartition(((cand - Q[j]) ** 2).sum(1), 10)[:10]
    return idx[top]


def _stream_scores(recon_at, N, Q, metric, chunk, progress=None):
    """Stream code scores over the pool via recon_at(lo, hi) -> (hi-lo, d).

    Returns (top10_scores, top10_idx, topW_scores, topW_idx) global merges.
    """
    nq = Q.shape[0]
    W = 10_000
    qx = (Q * Q).sum(1) if metric == "l2" else None
    b10v = np.full((nq, 10), SENTINEL, np.float32)
    b10i = np.zeros((nq, 10), np.int64)
    bwv = np.full((nq, W), SENTINEL, np.float32)
    bwi = np.zeros((nq, W), np.int64)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        s2 = _score_chunk(recon_at(lo, hi), Q, metric, qx)
        p10 = np.argpartition(-s2, 10, axis=1)[:, :10]
        b10v, b10i = merge_into(b10v, b10i, np.take_along_axis(s2, p10, 1), p10 + lo)
        kw = min(W, s2.shape[1] - 1)   # guard: kth <= slice-1 on short tail slices
        pw = np.argpartition(-s2, kw, axis=1)[:, :W]
        bwv, bwi = merge_into(bwv, bwi, np.take_along_axis(s2, pw, 1), pw + lo)
        if progress:
            progress(hi, N)
    return b10v, b10i, bwv, bwi


def flat_recall(recon_at, N, Q, GT, metric, chunk=200_000, progress=None):
    """FLAT ruler: recall@10 of code scores alone over the full pool.

    recon_at(lo, hi) -> (hi-lo, d) f32 reconstruction chunk.
    Q: (nq, d) f32, unit-normalized if cosine. GT: (nq, >=10) int.
    """
    _, b10i, _, _ = _stream_scores(recon_at, N, Q, metric, chunk, progress)
    nq = Q.shape[0]
    rf = 0.0
    for j in range(nq):
        rf += len(set(b10i[j].tolist()) & set(GT[j].tolist())) / 10.0
    return rf / nq


def engine_ruler(recon_at, X, Q, GT, metric, W=10_000, chunk=200_000,
                 progress=None, rescore_renormalize=False):
    """ENGINE ruler: recall@10 after exact re-rank of the code-score top-W.

    X: original corpus rows (memmap ok) — normalized if cosine, raw if l2 —
    used ONLY for the exact re-rank of the shortlist, exactly as a deployed
    engine would fetch full vectors for its shortlist.
    """
    _, _, bwv, bwi = _stream_scores(recon_at, X.shape[0], Q, metric, chunk, progress)
    nq = Q.shape[0]
    rw = 0.0
    for j in range(nq):
        idx = np.sort(bwi[j])          # sequential fetch (random = IO death)
        top = _rescore(X, Q, metric, idx, j, rescore_renormalize)
        rw += len(set(top.tolist()) & set(GT[j].tolist())) / 10.0
    return rw / nq


def dual_ruler(recon_at, X, Q, GT, metric, chunk=200_000, progress=None,
               rescore_renormalize=False):
    """Both rulers in one streaming pass. Returns (flat, w10k) recalls."""
    _, b10i, _, bwi = _stream_scores(recon_at, X.shape[0], Q, metric, chunk, progress)
    nq = Q.shape[0]
    rf = rw = 0.0
    for j in range(nq):
        g = set(GT[j].tolist())
        rf += len(set(b10i[j].tolist()) & g) / 10.0
        idx = np.sort(bwi[j])
        top = _rescore(X, Q, metric, idx, j, rescore_renormalize)
        rw += len(set(top.tolist()) & g) / 10.0
    return rf / nq, rw / nq


def unit_normalize(X):
    return (X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)).astype(np.float32)
