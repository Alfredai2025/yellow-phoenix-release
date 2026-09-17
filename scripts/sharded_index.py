#!/usr/bin/env python3
"""
Phase 6: Sharded HNSW Index
Splits N vectors into S shards, builds in parallel, queries with aggregation.
"""

import numpy as np
import hnswlib
import multiprocessing as mp
import time
from typing import List, Tuple


def _build_shard_worker(args):
    """Worker: build one hnswlib index from a chunk of vectors."""
    shard_idx, vectors, ids, dim, space, M, ef_construction, ef_search = args
    n = len(vectors)
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(vectors, ids)
    index.set_ef(ef_search)
    return (shard_idx, index)


class ShardedIndex:
    def __init__(self, dim: int, n_shards: int = 4, M: int = 16, ef_construction: int = 200, ef_search: int = 50, space: str = 'l2'):
        self.dim = dim
        self.n_shards = n_shards
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.space = space
        self.shards: List[hnswlib.Index] = []
        self.shard_offsets: List[int] = []  # global ID offset per shard
        self.id_to_shard: dict = {}  # global ID -> (shard_idx, local_idx)

    def build(self, vectors: np.ndarray, ids: np.ndarray = None):
        """Build all shards in parallel."""
        n_total = len(vectors)
        if ids is None:
            ids = np.arange(n_total)

        # Split into shards
        shard_size = (n_total + self.n_shards - 1) // self.n_shards
        chunks = []
        self.shard_offsets = []

        for i in range(self.n_shards):
            start = i * shard_size
            end = min((i + 1) * shard_size, n_total)
            if start >= end:
                break
            chunk_vectors = vectors[start:end]
            chunk_ids = ids[start:end]
            self.shard_offsets.append(start)
            for local_idx, global_id in enumerate(chunk_ids):
                self.id_to_shard[int(global_id)] = (i, local_idx)
            chunks.append((i, chunk_vectors, chunk_ids, self.dim, self.space, self.M, self.ef_construction, self.ef_search))

        # Parallel build
        t0 = time.time()
        with mp.Pool(processes=min(self.n_shards, mp.cpu_count())) as pool:
            results = pool.map(_build_shard_worker, chunks)

        # Sort by shard_idx and store
        results.sort(key=lambda x: x[0])
        self.shards = [r[1] for r in results]
        build_time = time.time() - t0
        print(f"[ShardedIndex] Built {len(self.shards)} shards ({n_total} vectors) in {build_time:.2f}s")
        return build_time

    def search(self, query: np.ndarray, k_per_shard: int = 50, k_final: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query all shards, aggregate candidates, return top-k_final by distance.
        Returns: (ids, distances)
        """
        all_ids = []
        all_dists = []

        for shard_idx, index in enumerate(self.shards):
            labels, distances = index.knn_query(query, k=k_per_shard)
            # labels are local IDs within shard; map to global IDs
            offset = self.shard_offsets[shard_idx]
            global_ids = labels[0] + offset
            all_ids.extend(global_ids)
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

        return np.array(final_ids), np.array(final_dists)

    def batch_search(self, queries: np.ndarray, k_per_shard: int = 50, k_final: int = 10):
        """Search multiple queries. Returns list of (ids, distances)."""
        results = []
        for q in queries:
            ids, dists = self.search(q, k_per_shard, k_final)
            results.append((ids, dists))
        return results


if __name__ == "__main__":
    # Quick smoke test
    print("[*] ShardedIndex smoke test...")
    dim = 384
    n = 100_000
    vectors = np.random.randn(n, dim).astype(np.float32)

    idx = ShardedIndex(dim=dim, n_shards=4, M=16, ef_search=50)
    t0 = time.time()
    idx.build(vectors)
    print(f"    Build: {time.time()-t0:.2f}s")

    q = vectors[0]
    t0 = time.perf_counter()
    ids, dists = idx.search(q, k_per_shard=50, k_final=10)
    lat = (time.perf_counter() - t0) * 1e6
    print(f"    Query: {lat:.1f} µs, ids={ids[:3]}")
    print("[+] Smoke test passed")
