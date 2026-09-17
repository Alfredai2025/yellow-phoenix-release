#!/usr/bin/env python3
"""
Phase 6b: Random Sharded HNSW Index
Assigns vectors to shards via random permutation. Each shard is a representative
sample of the full dataset. Query aggregates top-k_per_shard across all shards,
deduplicates, and returns global top-k by L2 distance.
"""

import numpy as np
import hnswlib
import time
from typing import List, Tuple, Optional


class RandomShardedIndex:
    def __init__(
        self,
        dim: int,
        n_shards: int = 4,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        space: str = "l2",
        seed: int = 42,
    ):
        self.dim = dim
        self.n_shards = n_shards
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.space = space
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.shards: List[hnswlib.Index] = []
        self.shard_ids: List[np.ndarray] = []  # global IDs stored in each shard
        self.n_total = 0

    def build(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None, num_threads: int = -1) -> float:
        """
        Build all shards. Randomly permute vectors, split into shards, build hnswlib index per shard.
        num_threads: -1 uses all cores, 1 is single-threaded.
        """
        n_total = len(vectors)
        self.n_total = n_total
        if ids is None:
            ids = np.arange(n_total, dtype=np.int64)

        # Random permutation: each shard gets a representative sample
        perm = self.rng.permutation(n_total)
        shard_size = (n_total + self.n_shards - 1) // self.n_shards

        t0 = time.time()
        self.shards = []
        self.shard_ids = []

        for i in range(self.n_shards):
            start = i * shard_size
            end = min((i + 1) * shard_size, n_total)
            if start >= end:
                break

            idx = perm[start:end]
            shard_vectors = vectors[idx].astype(np.float32)
            shard_ids = ids[idx]

            index = hnswlib.Index(space=self.space, dim=self.dim)
            index.init_index(
                max_elements=len(shard_vectors),
                ef_construction=self.ef_construction,
                M=self.M,
            )
            # Build with thread parallelism (avoids macOS multiprocessing spawn issues)
            index.add_items(shard_vectors, shard_ids, num_threads=num_threads)
            index.set_ef(self.ef_search)

            self.shards.append(index)
            self.shard_ids.append(shard_ids)

        build_time = time.time() - t0
        print(f"[RandomShardedIndex] Built {len(self.shards)} shards ({n_total} vectors) in {build_time:.2f}s")
        return build_time

    def search(
        self,
        query: np.ndarray,
        k_per_shard: int = 200,
        k_final: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query all shards, aggregate candidates, return top-k_final by distance.
        Returns: (global_ids, distances)
        """
        all_ids = []
        all_dists = []

        for index in self.shards:
            labels, distances = index.knn_query(query, k=k_per_shard)
            all_ids.extend(labels[0])
            all_dists.extend(distances[0])

        # Aggregate: sort by distance, deduplicate, take top-k_final
        combined = list(zip(all_dists, all_ids))
        combined.sort(key=lambda x: x[0])

        seen = set()
        final_ids = []
        final_dists = []
        for dist, idx in combined:
            if idx not in seen:
                seen.add(idx)
                final_ids.append(idx)
                final_dists.append(dist)
                if len(final_ids) >= k_final:
                    break

        return np.array(final_ids, dtype=np.int64), np.array(final_dists, dtype=np.float32)

    def batch_search(
        self,
        queries: np.ndarray,
        k_per_shard: int = 200,
        k_final: int = 10,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Search multiple queries."""
        return [self.search(q, k_per_shard, k_final) for q in queries]


if __name__ == "__main__":
    # Smoke test
    print("[*] RandomShardedIndex smoke test...")
    dim = 384
    n = 100_000
    vectors = np.random.randn(n, dim).astype(np.float32)

    idx = RandomShardedIndex(dim=dim, n_shards=4, M=16, ef_search=50)
    t = idx.build(vectors, num_threads=8)
    print(f"    Build: {t:.2f}s")

    q = vectors[0]
    t0 = time.perf_counter()
    ids, dists = idx.search(q, k_per_shard=50, k_final=10)
    lat = (time.perf_counter() - t0) * 1e6
    print(f"    Query: {lat:.1f} µs, ids={ids[:3]}")
    print("[+] Smoke test passed")
