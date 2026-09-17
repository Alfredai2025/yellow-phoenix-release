# YP Functor Audit — Phase A/B
# Date: 2026-07-30
# Author: Marc John Sawyer
# Status: Float-embedding spectral index measured on real 100K data

## 1. Category Definitions

### C_hash — 512-bit ITQ Hash Space
- **Objects:** Binary vectors h ∈ {0,1}^512
- **Morphisms:** Hamming distance d_H(h₁, h₂) = popcount(h₁ XOR h₂)
- **Identity:** d_H(h, h) = 0
- **Composition:** Triangle inequality holds: d_H(a,c) ≤ d_H(a,b) + d_H(b,c)

### C_spectral — Tensor Spectral Eigenspace
- **Objects:** Real vectors v ∈ ℝ^k (eigenvector coefficients, k ≈ 16–64)
- **Morphisms:** Spectral angle d_θ(v₁, v₂) = arccos( (v₁·v₂) / (||v₁|| ||v₂||) )
- **Identity:** d_θ(v, v) = 0
- **Composition:** Triangle inequality holds (spherical metric)

### C_hnsw — Binary HNSW Graph
- **Objects:** Graph nodes n indexed by document ID
- **Morphisms:** Shortest path length d_G(n₁, n₂) in edge hops
- **Identity:** d_G(n, n) = 0
- **Composition:** Triangle inequality holds (graph metric)

### C_geometric — 512-D 0/1 Multivector Space
- **Objects:** 512-bit hashes interpreted as 0/1 vectors
- **Morphisms:** Geometric product score s(a, b) = a·b + 0.1 · |a ∧ b|
- **Identity:** s(a, a) = ||a||² + 0.1·0 = scalar norm
- **Composition:** Bilinear, associative

## 2. Functor Definitions

### F₁: C_hash → C_spectral
- **Object map:** h ↦ c(h) = projection of h onto top-k spectral eigenvectors
- **Morphism map:** d_H(h₁, h₂) ↦ d_θ(c₁, c₂)
- **Functoriality target:** d_θ(F₁(h₁), F₁(h₂)) ≤ k₁ · d_H(h₁, h₂)
- **Empirical k₁:** measured via linear regression on sampled pairs

### F₂: C_spectral → C_hnsw
- **Object map:** v ↦ n(v) where layer L = floor(log2(||v||))
- **Morphism map:** d_θ(v₁, v₂) ↦ d_G(n₁, n₂)
- **Functoriality target:** d_G(n₁, n₂) ≤ f(d_θ, M, ef)
- **Empirical bound:** measured as HNSW rank of the spectral nearest neighbor

### F₃: C_hnsw → C_geometric
- **Object map:** {n₁...n_k} ↦ M = Σᵢ wᵢ · eᵢ
- **Morphism map:** graph proximity ↦ geometric score
- **Functoriality target:** order-preservation
- **Empirical check:** Kendall tau between HNSW ranking and geometric-score ranking

### F₄ = F₃ ∘ F₂ ∘ F₁: C_hash → C_geometric
- **Master functor:** End-to-end recall bound
- **Theorem target:** R@1_geometric ≥ R@1_hash − δ
- **Target δ:** < 0.03 (3 percentage points)

## 3. Measurement Plan
- F₁: Sample random pairs, compute (d_H, d_θ), fit linear regression
- F₂: For each query, compare spectral NN (excluding self) to HNSW result rank
- F₃: Verify geometric-score ordering matches HNSW distance ordering
- F₄: Compare `HybridMesh::query_auto` vs HNSW+geometric rerank against brute-force Hamming ground truth

## 4. Acceptance Criteria
- [ ] k₁ measured with R² > 0.8
- [ ] F₂ bound f() established with 95% confidence
- [ ] F₃ order-preservation verified on 1K pairs
- [ ] F₄ end-to-end bound δ < 0.03

## 5. Phase B — Real API Wiring

### Rust type map
| Category | Rust type | Path |
|----------|-----------|------|
| C_hash object source | `HybridMesh` / `CrystalMesh512` | `crate::hybrid_mesh::HybridMesh` |
| C_spectral object (512-bit) | `TensorSpectralIndex512` | `crate::math::tensor_spectral_512::TensorSpectralIndex512` |
| C_spectral object (float) | `TensorSpectralIndexFloat` | `crate::math::tensor_spectral_float::TensorSpectralIndexFloat` |
| C_hnsw object | `BinaryHNSW` | `crate::binary_hnsw::BinaryHNSW` |
| C_geometric score | `geometric_score()` helper | `src/math/functor_bounds.rs` |

### API calls used
| Functor | Call | Returns |
|---------|------|---------|
| F₁ | `mesh.fine.slots` → `Slot512 { id, pap }` | 512-bit hash |
| F₁ (float) | `embeddings[id * dim .. (id+1) * dim]` | float embedding slice |
| F₁/F₂ (float) | `spectral.query(embedding, top_k)` | `Vec<(score, id)>` |
| F₂ | `hnsw_graph.search(pap_512, 500)` | `Vec<(hamming, node_idx)>` |
| F₂ | `hnsw_graph.node(idx).unwrap().id` | `u64` doc id |
| F₃ | `hnsw_graph.search(query_hash, 20)` | HNSW candidates |
| F₃ | `geometric_score(query_vec, candidate_vec)` | `f32` score |
| F₄ | `mesh.query_auto(pap_128, pap_512, 10)` | `Vec<(score, id)>` |
| F₄ | brute-force exact Hamming over `mesh.fine.slots` (excluding self) | ground-truth id |

### Files changed
- `src/math/functor_bounds.rs` — real measurement algorithms.
- `src/math/functor_bounds_float.rs` — float-embedding F1/F2.
- `src/math/tensor_spectral_512.rs` — 512-bit spectral index.
- `src/math/tensor_spectral_float.rs` — float-embedding spectral index.
- `src/math.rs` — re-exports new modules.
- `src/bin/functor_audit.rs` — end-to-end runner on `.bin` records.
- `src/bin/functor_audit_float.rs` — float-embedding runner.
- `scripts/npy_hashes_to_bin.py` — converts `N x 64 uint8` `.npy` hashes to `.bin`.
- `scripts/npy_to_f32bin.py` — converts float `.npy` embeddings to `.f32bin`.
- `scripts/itq_grid_search.py` — ITQ iteration grid search.
- `src/tensor_spectral.rs` — added `vector_by_id(id)` helper.
- `scripts/verify_functor.py` — runs tests, reports status.
- `docs/functor_audit.md` — this file.

### Verification
```bash
cargo check
cargo test --lib functor_bounds::tests
python3 scripts/verify_functor.py
cargo run --release --bin functor_audit_float -- data/paper_hashes_100k.bin data/paper_embeddings_100k.f32bin
```

## 6. Real-Data Results

### 6.1 512-bit spectral index on `paper_hashes_100k.npy`

```json
{
  "records": 100000,
  "F1": { "k1": -1.1e-7, "r_squared": 9.5e-6, "sample_pairs": 5000 },
  "F2": { "mean_rank": 50.0, "p95_rank": 50.0, "queries": 500 },
  "F3": { "kendall_tau": 0.2612, "queries": 500 },
  "F4": { "delta": -0.0580, "test_ids": 500 }
}
```

**Diagnosis:** PCA on 512-bit 0/1 vectors does not capture ITQ neighborhood structure. Contracts fail.

### 6.2 Float-embedding spectral index on `paper_embeddings_100k.npy`

```json
{
  "F1": { "k1": 0.004261, "r_squared": 0.739291 },
  "F2": { "mean_rank": 49.96, "p95_rank": 50.0 }
}
```

| Metric | Value | Target | Verdict |
|--------|-------|--------|---------|
| F1 R² | **0.739** | > 0.80 | ⚠️ Close — much better than hash-only, but still slightly below target |
| F2 mean rank | **49.96** | < 5.0 | ❌ Spectral embedding NN is almost never in HNSW top-500 |

### 6.3 ITQ Hash Quality vs. Rerank Depth

A sweep over Hamming candidate set size shows how many binary candidates are needed so that re-ranking by L2 recovers the true embedding nearest neighbor:

```json
[
  { "k": 10,   "r1": 0.9400 },
  { "k": 20,   "r1": 0.9700 },
  { "k": 50,   "r1": 0.9960 },
  { "k": 100,  "r1": 1.0000 },
  { "k": 200,  "r1": 1.0000 }
]
```

| Candidate set size | Reranked R@1 | Notes |
|--------------------|--------------|-------|
| 10 | 94.0% | Current production default in some paths |
| 20 | 97.0% | — |
| **50** | **99.6%** | **Sweet spot** |
| 100 | 100.0% | Diminishing returns |
| 200 | 100.0% | Overkill |

### 6.4 Interpretation
- **F1 is nearly valid.** Hamming distance on the ITQ hashes correlates reasonably with spectral angle on the original 384-d MiniLM embeddings (R² ≈ 0.74).
- **F2 is a design-level failure.** The spectral nearest neighbor in embedding space is not preserved by Hamming HNSW.
- **Hash quantization is lossy at R@1.** Direct Hamming nearest neighbor has **0%** exact R@1 vs. L2 ground truth.
- **Reranking saves it.** Fetching the top-50 Hamming candidates and re-ranking by L2 achieves **99.6% R@1**.

### 6.5 Engineering Recommendation

The encoder/ITQ is not the bottleneck for 99% recall — the **retrieval depth** is.

If using a hash-first pipeline:
- Change candidate fetch from `top_k=10` to `top_k=50`.
- Re-rank the 50 candidates with the original float embedding (cosine or L2).
- This jumps recall from ~94% to **99.6%** at the cost of ~5–10× more distance computations in the rerank stage (still negligible vs. full brute force).

If using the existing `ScaleEngine` (float HNSW via `hnswlib`):
- The same principle applies: fetch more HNSW candidates before cosine rerank.
- Current code fetches `k+10` for 100K. Raising this to `k*5` (i.e., 50 for k=10) is the analogous fix.
