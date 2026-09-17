#!/usr/bin/env python3
"""
ShardRouter: 10M-scale routing across independent 1M HybridMesh384 shards.
Phase 2 of the 10M master plan. Additive only — no Rust changes.
"""

import numpy as np
from typing import List, Tuple

from scripts.scale_engine_pid import ScaleEnginePID


class ShardRouter:
    """
    Routes queries to top-K shards by 64-bit hash-prefix similarity.
    Each shard is a standalone ScaleEnginePID with its own HybridMesh384.
    """

    def __init__(self, n_shards: int = 10, shard_capacity: int = 1_100_000):
        self.n_shards = n_shards
        self.shard_capacity = shard_capacity
        self.shards: List[ScaleEnginePID] = []
        self.shard_prefixes: List[int] = []  # 64-bit center prefix per shard

    def _shard_id_from_hash(self, hash_bytes: bytes) -> int:
        """Deterministic shard by first 8 bytes."""
        prefix = int.from_bytes(hash_bytes[:8], "little")
        return prefix % self.n_shards

    def _top_shards_for_query(self, query_hash: bytes, n_top: int = 2) -> List[int]:
        """Return top-N shard IDs by XOR prefix distance."""
        q_prefix = int.from_bytes(query_hash[:8], "little")
        distances = []
        for sid, center in enumerate(self.shard_prefixes):
            d = bin(q_prefix ^ center).count("1")
            distances.append((d, sid))
        distances.sort()
        return [sid for _, sid in distances[:n_top]]

    def build_from_embeddings(self, ids: np.ndarray, embeddings: np.ndarray,
                              hashes: np.ndarray = None, n_probes: int = 1):
        """
        Build n_shards independent HybridMesh384 indexes from contiguous chunks.

        ids:        (N,) int64
        embeddings: (N, 384) float32
        hashes:     (N, 48) uint8 primary ITQ hashes; computed if None
        n_probes:   1 = single hash, 3 = multi-probe ITQ per vector
        """
        n = len(ids)
        if hashes is None:
            hasher = ScaleEnginePID(scale_engine=None)
            hashes = hasher._compute_hashes(embeddings)

        shard_size = n // self.n_shards

        for sid in range(self.n_shards):
            start = sid * shard_size
            end = n if sid == self.n_shards - 1 else (sid + 1) * shard_size

            shard = ScaleEnginePID(scale_engine=None)
            # Let the shard compute hashes so multi-probe can be applied.
            shard._build_hybrid_mesh384(
                embeddings[start:end],
                ids[start:end],
                n_probes=n_probes,
            )
            self.shards.append(shard)

            # Shard center = median 64-bit prefix
            prefixes = [int.from_bytes(h[:8], "little") for h in hashes[start:end]]
            self.shard_prefixes.append(int(np.median(prefixes)))

            print(f"[ShardRouter] Shard {sid}: {end - start:,} docs")

    def search(self, query_embedding: np.ndarray, query_hash: bytes,
               k: int = 10, k_fast: int = 500,
               n_shards_probe: int = 2) -> List[Tuple[int, float, str]]:
        """
        Search top-N shards and merge/deduplicate.
        Returns: list of (doc_id, score, source_shard)
        """
        shard_ids = self._top_shards_for_query(query_hash, n_shards_probe)

        all_results = []
        for sid in shard_ids:
            results = self.shards[sid].search_production(
                query_embedding, k=k, k_fast=k_fast
            )
            for doc_id in results:
                # HybridMesh384 search returns ids only; score is not exposed here,
                # so we tag with shard name for debugging.
                all_results.append((doc_id, 0.0, f"shard_{sid}"))

        # Deduplicate by doc_id, keep first occurrence
        seen = {}
        for doc_id, score, src in all_results:
            if doc_id not in seen:
                seen[doc_id] = (score, src)

        final = [(doc_id, score, src) for doc_id, (score, src) in seen.items()]
        return final[:k]
