#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-dimension Lloyd-Max codec (the "plain recipe" that won the duel).

Per-dimension scalar quantization of a mean-centered corpus: fit an
independent K-level 1-D quantizer per dimension (k-means is the empirical
Lloyd-Max quantizer of that coordinate marginal), encode each coordinate to
its nearest level, decode as books[dim, code] + mu. b bits/dimension costs
d*b/8 bytes per vector, packed one uint8 code per coordinate (b <= 8).

This is the codec family whose byte-matched duel against Google's published
RSLM reference implementation (google-research/rslm, Apache-2.0) is
reproduced by this harness — see README.md and expected_outputs/.
"""

import numpy as np


class PerDimCodec:
    """Per-dimension k-means (Lloyd-Max) scalar quantizer.

    Training uses sklearn MiniBatchKMeans with the exact hyper-parameters of
    the campaign scripts so a full re-run reproduces the published levels
    (modulo sklearn/numpy version drift; recall cells are exactly re-runnable
    from cached codes/books — see README "Self-check").
    """

    def __init__(self, dim, bits):
        assert 1 <= bits <= 8, "bits/dim must be in 1..8 (one uint8 code per coordinate)"
        self.dim = dim
        self.bits = bits
        self.k = 1 << bits
        self.mu = None      # (d,) f32 corpus mean (train subset)
        self.books = None   # (d, K) f32, levels SORTED ascending per dim

    @property
    def bytes_per_vec(self):
        return self.dim * self.bits // 8

    def fit(self, X_train, seeds=0, max_iter=None):
        """Fit mu + per-dim books on a train subset (campaign: first 400k rows).

        seeds: int (same random_state for every dim — SIFT/GloVe campaign
        scripts used 0) or a length-d sequence of ints (Yandex-10M campaign
        script used seeds=dim index, max_iter=20).
        """
        from sklearn.cluster import MiniBatchKMeans
        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        self.mu = X_train.mean(0).astype(np.float32)
        Xc = (X_train - self.mu).astype(np.float32)
        d, K = self.dim, self.k
        if np.isscalar(seeds):
            seeds = [int(seeds)] * d
        books = np.empty((d, K), np.float32)
        kw = dict(batch_size=8192, n_init=3)
        if max_iter is not None:
            kw["max_iter"] = max_iter
        for j in range(d):
            books[j] = MiniBatchKMeans(K, random_state=int(seeds[j]),
                                       **kw).fit(Xc[:, j:j + 1]).cluster_centers_.ravel()
        books.sort(axis=1)
        self.books = books
        return self

    def encode(self, X):
        """X (n, d) f32 -> codes (n, d) uint8, nearest sorted level per dim."""
        assert self.books is not None, "fit() first"
        X = np.asarray(X, dtype=np.float32) - self.mu
        n, d = X.shape
        K = self.k
        codes = np.empty((n, d), np.uint8)
        for j in range(d):
            lv = self.books[j]
            col = X[:, j]
            pos = np.clip(np.searchsorted(lv, col), 1, K - 1)
            l = np.abs(col - lv[pos - 1])
            r = np.abs(col - lv[pos])
            codes[:, j] = np.where(l <= r, pos - 1, pos)
        return codes

    def decode(self, codes):
        """codes (n, d) uint8 -> recon (n, d) f32 = books[code] + mu."""
        codes = np.asarray(codes)
        n = codes.shape[0]
        rowoff = (np.arange(self.dim) * self.k).astype(np.int32)[None, :]
        recon = self.books.ravel()[(codes.astype(np.int32) + rowoff).ravel()]
        return recon.reshape(n, self.dim).astype(np.float32) + self.mu

    # -- persistence: books/mu round-trip through npz (codes are plain npy) --
    def save(self, path):
        np.savez(path, dim=self.dim, bits=self.bits, mu=self.mu, books=self.books)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        c = cls(int(z["dim"]), int(z["bits"]))
        c.mu = z["mu"].astype(np.float32)
        c.books = z["books"].astype(np.float32)
        return c
