#!/usr/bin/env python3
"""
PQReranker: lightweight Product Quantization re-rank for the final stage.
Phase 4 of the 10M master plan. Pure Python, additive only.
"""

import numpy as np
from typing import List, Tuple, Optional


try:
    from sklearn.cluster import MiniBatchKMeans
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


class PQReranker:
    """
    Simple PQ re-ranker: database vectors are encoded as m bytes;
    query stays float and uses asymmetric distance tables.
    """

    def __init__(self, dim: int = 384, m: int = 8, nbits: int = 8):
        """
        m: number of subspaces (8)
        nbits: bits per subspace (8 -> 256 centroids)
        """
        self.dim = dim
        self.m = m
        self.nbits = nbits
        self.dsub = dim // m  # 48 for 384-dim / 8 subspaces

        # codebooks[subspace, centroid, dsub]
        self.codebooks: Optional[np.ndarray] = None
        # codes[doc_id] -> (m,) uint8
        self.codes: dict = {}

    def _train_subspace(self, sub_embs: np.ndarray) -> np.ndarray:
        """Train 256 centroids for one subspace."""
        if _HAS_SKLEARN:
            kmeans = MiniBatchKMeans(
                n_clusters=256,
                batch_size=256,
                max_iter=10,
                random_state=42,
                n_init=3,
            )
            kmeans.fit(sub_embs)
            return kmeans.cluster_centers_.astype(np.float32)

        # Fallback: random initialization + one Lloyd iteration
        N = len(sub_embs)
        rng = np.random.default_rng(42)
        indices = rng.choice(N, 256, replace=False)
        centroids = sub_embs[indices].copy()
        dists = np.linalg.norm(sub_embs[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        for k in range(256):
            mask = labels == k
            if mask.any():
                centroids[k] = sub_embs[mask].mean(axis=0)
        return centroids.astype(np.float32)

    def train(self, embeddings: np.ndarray, ids: Optional[np.ndarray] = None):
        """
        Train codebooks and encode all vectors.

        embeddings: (N, dim) float32
        ids:        (N,) optional int64 identifiers (defaults to 0..N-1)
        """
        N = len(embeddings)
        if ids is None:
            ids = np.arange(N, dtype=np.int64)

        self.codebooks = np.zeros((self.m, 256, self.dsub), np.float32)
        all_codes = np.zeros((N, self.m), dtype=np.uint8)

        for subspace in range(self.m):
            start = subspace * self.dsub
            end = start + self.dsub
            sub_embs = embeddings[:, start:end].astype(np.float32)

            centroids = self._train_subspace(sub_embs)
            self.codebooks[subspace] = centroids

            dists = np.linalg.norm(sub_embs[:, None, :] - centroids[None, :, :], axis=2)
            all_codes[:, subspace] = np.argmin(dists, axis=1).astype(np.uint8)

        for i in range(N):
            self.codes[int(ids[i])] = all_codes[i]

    def asymmetric_distance(self, query: np.ndarray, doc_id: int) -> float:
        """PQ asymmetric distance between float query and coded vector."""
        codes = self.codes.get(doc_id)
        if codes is None:
            return float("inf")
        dist = 0.0
        for subspace in range(self.m):
            start = subspace * self.dsub
            end = start + self.dsub
            qsub = query[start:end]
            dists = np.linalg.norm(self.codebooks[subspace] - qsub, axis=1)
            dist += float(dists[codes[subspace]])
        return dist

    def rerank(self, query: np.ndarray,
               candidate_ids: List[int],
               k: int = 10) -> List[Tuple[int, float]]:
        """Re-rank candidates by PQ distance; return top-k."""
        query = query.astype(np.float32)
        scored = [(doc_id, self.asymmetric_distance(query, doc_id))
                  for doc_id in candidate_ids]
        scored.sort(key=lambda x: x[1])
        return scored[:k]
