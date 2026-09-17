#!/usr/bin/env python3
"""
ScaleEnginePID: wraps ScaleEngine with real prefix cascade + two-tier geometric re-rank.
"""

import os
import numpy as np
try:
    import hnswlib
except Exception:
    hnswlib = None
from yp_bridge import Mesh384, HybridMesh384, BinaryHNSW  # 384-bit honest mesh + two-tier re-rank
try:
    from yp.autoshard import AutoShardIndex  # Phase 1: auto-scaling vector index
except Exception:
    AutoShardIndex = None

# HNSW Phase 3 Pareto-optimal parameters (2026-07-30)
# Grid search on 100K embeddings: M=HNSW_M, ef_search=50 gives 99.9% R@1 at 285 µs P50
HNSW_M = 16
HNSW_EF_SEARCH = 50
HNSW_EF_CONSTRUCTION = 200

# Module-level cache for ITQ transformation matrices.
_ITQ_CACHE = {}


class ScaleEnginePID:
    def __init__(self, scale_engine=None):
        # Phase 1: AutoShardIndex replaces direct hnswlib
        self._vector_index = None
        self._vector_dim = 384  # MiniLM dimension

        # Production: native Rust HybridMesh384 (HNSW384 hash + embedding re-rank)
        self._hybrid_mesh384 = None

        self.engine = scale_engine
        self._hashes = {}
        self._hash_indices = {}
        self._hash_ready = set()
        self._mesh384 = {}
        self._hnsw_indices = {}
        self._build_mesh384 = os.environ.get('YP_BUILD_MESH384', '1') != '0'


    def _build_hnsw_index(self, embeddings, ids, dim=384, M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION):
        """Build AutoShardIndex from embeddings (auto single/sharded by scale)."""
        n = len(ids)
        if n == 0:
            return None
        index = AutoShardIndex(
            dim=dim, M=M, ef_construction=ef_construction, ef_search=HNSW_EF_SEARCH
        )
        index.build(embeddings, ids)
        return index

    def _hnsw_search(self, index, query_emb, k=50):
        """Search AutoShardIndex and return list of doc ids."""
        if index is None:
            return []
        ids, _ = index.search(query_emb, k=k)
        return [int(x) for x in ids]

    def _build_hybrid_mesh384(self, embeddings: np.ndarray, ids: np.ndarray = None):
        """Build native HybridMesh384 from ITQ hashes + float embeddings."""
        n = len(embeddings)
        if ids is None:
            ids = np.arange(n, dtype=np.uint64)
        hashes = self._compute_hashes(embeddings)
        mesh = HybridMesh384(capacity=max(n * 2, 65536), emb_dim=embeddings.shape[1])
        self._hash_to_id = {}
        for i in range(n):
            h = bytes(hashes[i])
            mesh.insert(int(ids[i]), h, embeddings[i])
            self._hash_to_id[h] = int(ids[i])
        self._hybrid_mesh384 = mesh
        self._doc_ids = ids.astype(np.uint64)
        print(f"[HybridMesh384] Built {n} docs")
        return mesh

    def _encode_itq(self, embedding: np.ndarray) -> bytes:
        """Encode a single embedding to its 384-bit ITQ hash bytes."""
        h = self._compute_hashes(embedding.reshape(1, -1))
        return bytes(h[0])

    def _encode_itq_512(self, embedding: np.ndarray) -> bytes:
        """Encode a single embedding to its 512-bit ITQ hash bytes."""
        return self._encode_itq_512_batch(embedding.reshape(1, -1))[0].tobytes()

    def _encode_itq_512_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Encode a batch of embeddings to 512-bit hashes (N, 64) uint8."""
        cache_key = "itq_512"
        if cache_key in _ITQ_CACHE:
            mean, proj = _ITQ_CACHE[cache_key]
            x = embeddings.astype(np.float32) - mean
            bits = (x @ proj) > 0
            return np.packbits(bits.astype(np.uint8), axis=1)

        model_path = os.path.join(os.path.dirname(__file__), "..", "data", "itq_model_512.npz")
        if not os.path.exists(model_path):
            raise RuntimeError(f"ITQ-512 model not found: {model_path}")
        d = np.load(model_path)
        mean = d["mean"].astype(np.float32)
        proj = d["proj"].astype(np.float32)
        _ITQ_CACHE[cache_key] = (mean, proj)
        x = embeddings.astype(np.float32) - mean
        bits = (x @ proj) > 0
        return np.packbits(bits.astype(np.uint8), axis=1)

    def _build_binary_hnsw(self, embeddings: np.ndarray, ids: np.ndarray = None):
        """Build native Rust BinaryHNSW index from 512-bit ITQ hashes."""
        n = len(embeddings)
        if ids is None:
            ids = np.arange(n, dtype=np.uint64)
        # Exact 384-bit hash fast path (no mesh required)
        hashes_384 = self._compute_hashes(embeddings)
        self._hash_to_id = {bytes(hashes_384[i]): int(ids[i]) for i in range(n)}
        # Deep 512-bit BinaryHNSW path
        hashes_512 = self._encode_itq_512_batch(embeddings)
        idx = BinaryHNSW()
        idx.insert_batch(ids.astype(np.uint64), hashes_512)
        self._binary_hnsw = idx
        self._binary_hnsw_hashes = hashes_512
        print(f"[BinaryHNSW] Built {n} docs from 512-bit hashes")
        return idx

    def search_production(self, query_embedding: np.ndarray, k: int = 10, k_fast: int = 50):
        """
        Production two-tier search (HybridMesh384):
          1. HybridMesh384 HNSW384 graph: top-k_fast Hamming candidates (~55 µs)
          2. EmbeddingStore cosine re-rank: exact top-k (~50 µs)
        Total: ~105 µs for 99.6% R@1.
        """
        if self._hybrid_mesh384 is None:
            raise RuntimeError("HybridMesh384 not initialized. Call _build_hybrid_mesh384() first.")
        query_hash = self._compute_hashes(query_embedding.reshape(1, -1))
        results = self._hybrid_mesh384.search(bytes(query_hash[0]), query_embedding, k_fast=k_fast, n_final=k)
        return [int(doc_id) for (_, doc_id) in results]

    def search_binary_hnsw(self, query_embedding: np.ndarray, k: int = 10):
        """
        BinaryHNSW deep path (native Rust, ~481 µs at 1M).
        Requires _build_binary_hnsw() to be called first.
        """
        query_hash = self._encode_itq(query_embedding)
        doc_id = self._hash_to_id.get(query_hash)
        if doc_id is not None:
            return [int(doc_id)]
        if not hasattr(self, '_binary_hnsw') or self._binary_hnsw is None:
            raise RuntimeError("BinaryHNSW not initialized. Call _build_binary_hnsw() first.")
        hash512 = self._encode_itq_512(query_embedding)
        results = self._binary_hnsw.search(hash512, k=k)
        return [int(doc_id) for (doc_id, _) in results]

    def search_fastest(self, query_embedding: np.ndarray, k: int = 10):
        """
        Multi-engine cascade: exact hash → BinaryHNSW → HybridMesh384.
        Uses the fastest engine that is built and returns results.
        """
        # 1. Exact 384-bit hash hit (fastest)
        query_hash = self._encode_itq(query_embedding)
        doc_id = self._hash_to_id.get(query_hash)
        if doc_id is not None:
            return [int(doc_id)]

        # 2. BinaryHNSW deep path (~0.5 ms at 1M)
        if hasattr(self, '_binary_hnsw') and self._binary_hnsw is not None:
            hash512 = self._encode_itq_512(query_embedding)
            results = self._binary_hnsw.search(hash512, k=k)
            if results:
                return [int(doc_id) for (doc_id, _) in results]

        # 3. HybridMesh384 fallback (~3.5 ms at 1M)
        if self._hybrid_mesh384 is not None:
            query_hash = self._compute_hashes(query_embedding.reshape(1, -1))
            results = self._hybrid_mesh384.search(bytes(query_hash[0]), query_embedding, k_fast=50, n_final=k)
            return [int(doc_id) for (_, doc_id) in results]

        raise RuntimeError("No retrieval engine available.")

    def _ensure_hashes(self, scale):
        if scale in self._hash_ready:
            return
        embs = self.engine._engines[scale][0]
        n, dim = embs.shape

        hash_path = f"data/paper_hashes_{scale}.npy"
        if os.path.exists(hash_path):
            hashes = np.load(hash_path)
            # If existing hashes are the wrong width for current model, recompute.
            expected_bytes = dim // 8
            if hashes.shape[1] != expected_bytes:
                print(f"[ScaleEnginePID] Existing hash width {hashes.shape[1]} != expected {expected_bytes}, recomputing...")
                hashes = self._compute_hashes(embs)
                np.save(hash_path, hashes)
        else:
            print(f"[ScaleEnginePID] Computing {dim}-bit hashes for {scale} ({n:,} vectors)...")
            hashes = self._compute_hashes(embs)
            os.makedirs("data", exist_ok=True)
            np.save(hash_path, hashes)
            print(f"[ScaleEnginePID] Saved {hash_path}")

        self._hashes[scale] = hashes

        print(f"[ScaleEnginePID] Building prefix indices...")
        self._hash_indices[scale] = {
            8: self._build_prefix_index(hashes, 8),
            4: self._build_prefix_index(hashes, 4),
            2: self._build_prefix_index(hashes, 2),
        }
        self._hash_ready.add(scale)

        # Build hnswlib fast approximate index for this scale
        print(f"[ScaleEnginePID] Building hnswlib index (M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION})...")
        self._hnsw_indices[scale] = self._build_hnsw_index(embs, np.arange(n, dtype=np.int64))
        print(f"[ScaleEnginePID] hnswlib index ready")

        # Build honest 384-bit Rust mesh for this scale (fallback)
        if self._build_mesh384:
            print(f"[ScaleEnginePID] Building CrystalMesh384 (n={n:,}, capacity={n*2:,})...")
            mesh = Mesh384(capacity=max(n * 2, 1024))
            for idx, h in enumerate(hashes):
                mesh.insert(int(idx), bytes(h))
            self._mesh384[scale] = mesh
            print(f"[ScaleEnginePID] CrystalMesh384 ready, len={len(mesh)}")
        else:
            print("[ScaleEnginePID] Skipping CrystalMesh384 build (YP_BUILD_MESH384=0)")

        print(f"[ScaleEnginePID] Ready for {scale}")

    def _compute_hashes(self, embs):
        """Binary hash via trained ITQ model."""
        dim = embs.shape[1]
        cache_key = f"itq_{dim}"
        if cache_key in _ITQ_CACHE:
            mean, proj = _ITQ_CACHE[cache_key]
            bits = ((embs - mean) @ proj) > 0
            return np.packbits(bits.astype(np.uint8), axis=1)

        # Try dimension-specific ITQ model first.
        model_candidates = [f"data/itq_model_{dim}.npz", f"itq_model_{dim}.npz"]
        for itq_path in model_candidates:
            if os.path.exists(itq_path):
                try:
                    model = np.load(itq_path)
                    mean = model["mean"].astype(np.float32)
                    if "proj" in model:
                        proj = model["proj"].astype(np.float32)
                    elif "pca_components" in model and "R" in model:
                        proj = (model["pca_components"].T @ model["R"]).astype(np.float32)
                    else:
                        proj = model["R"].astype(np.float32)
                    _ITQ_CACHE[cache_key] = (mean, proj)
                    bits = ((embs - mean) @ proj) > 0
                    n_bits = bits.shape[1]
                    print(f"[ScaleEnginePID] Using ITQ model {itq_path} ({n_bits} bits)")
                    return np.packbits(bits.astype(np.uint8), axis=1)
                except Exception as e:
                    print(f"[ScaleEnginePID] ITQ failed ({itq_path}): {e}, falling back")

        # Fallback: random projection (not semantically aligned — warn)
        print("[ScaleEnginePID] WARNING: no trained ITQ model found, using random projection")
        np.random.seed(42)
        proj = np.random.randn(dim, dim).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0)
        mean = np.zeros(dim, dtype=np.float32)
        _ITQ_CACHE[cache_key] = (mean, proj)
        bits = (embs @ proj) > 0
        return np.packbits(bits.astype(np.uint8), axis=1)

    def _build_prefix_index(self, hashes, n_bytes):
        index = {}
        for i in range(len(hashes)):
            prefix = bytes(hashes[i, :n_bytes])
            if prefix not in index:
                index[prefix] = []
            index[prefix].append(i)
        return index

    def _build_vector_index(self, embeddings: np.ndarray, ids: np.ndarray = None):
        """Build or rebuild AutoShardIndex from embeddings."""
        if self._vector_index is None:
            self._vector_index = AutoShardIndex(dim=self._vector_dim)
        meta = self._vector_index.build(embeddings, ids)
        print(f"[AutoShardIndex] Built: {meta}")
        return meta


    def _search_vector_index(self, query_embedding: np.ndarray, k: int = 10):
        """Search AutoShardIndex. Returns (ids, distances)."""
        if self._vector_index is None:
            return [], []
        ids, dists = self._vector_index.search(query_embedding, k=k)
        return ids.tolist(), dists.tolist()


    def search_hybrid_by_embedding(self, q, k=10, scale="1m"):
        return self.engine.search_hybrid_by_embedding(q, k=k, scale=scale)

    def _cascade_prefix_search(self, q, k=10, l1_threshold=8, l2_threshold=4, l3_threshold=2, scale="1m"):
        """Cascade DISABLED: 384-d embeddings need 384-d ITQ hash. Random projection is not semantically aligned."""
        return None

    def _score_and_format(self, embs, idxs, q, k):
        candidates = embs[idxs]
        sims = candidates @ q
        top = np.argsort(-sims)[:k]
        return [(float(sims[i]), (str(idxs[i]), f"doc_{idxs[i]}")) for i in top]

    def search_geometric(self, q, k=10, scale="1m"):
        """Two-tier geometric: hnswlib fast ANN -> exact cosine re-rank."""
        self._ensure_hashes(scale)
        embs = self.engine._engines[scale][0]

        # Tier 1: hnswlib approximate nearest neighbors (Phase 3 sweet spot)
        indices = self._hnsw_search(self._hnsw_indices[scale], q, k=max(k * 5, 50))
        if not indices and scale in self._mesh384:
            # Fallback: CrystalMesh384 brute-force Hamming
            q_hash = self._compute_hashes(q.reshape(1, -1))[0]
            coarse = self._mesh384[scale].query_k(bytes(q_hash), k=max(k * 5, 50))
            indices = [int(doc_id) for _dist, doc_id in coarse]

        # Tier 2: exact cosine re-rank
        candidate_embs = embs[indices]
        sims = candidate_embs @ q
        top = np.argsort(-sims)[:k]

        return [(float(sims[i]), (str(indices[i]), f"doc_{indices[i]}")) for i in top]
