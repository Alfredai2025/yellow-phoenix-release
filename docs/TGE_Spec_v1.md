# Trained Geometric Embedding (TGE) Spec v1.0

## Problem
The geometric brain (Cl(5)) receives binary hash bytes — garbage in, garbage out.
Bivector weight trains to 0.010. R@1 = 0%.

## Solution
Train a neural network f: ℝ³⁸⁴ → Cl(5) grade-1 (unit vector in ℝ⁵)
such that the geometric product f(q) * f(p) carries retrievable signal.

## Architecture

### Encoder (CPU-Numpy Prototype)
```
W1: (384, 128)  → ReLU
W2: (128, 32)   → ReLU  
W3: (32, 5)     → L2 normalize
```

### Relationship Scorer
Given query vector q̂ = f(q) and paper vector p̂ = f(p):
```
R = q̂ · p̂          (scalar, grade 0)
  + q̂ ∧ p̂          (bivector, grade 2)

score(q,p) = w₀ * |R_scalar| + w₂ * ‖R_bivector‖₂
```

### Grade Weights (Learned)
w₀, w₂ ≥ 0.01, initialized to w₀=1.0, w₂=0.5.
The bivector weight w₂ is the critical parameter.
If w₂ → 0 during training, the geometric brain collapses to cosine.
If w₂ > 0.3, the bivector carries independent signal.

## Loss Function

### Triplet Loss
```
L = Σ max(0, margin - score(q, pos) + score(q, neg))
```

### Hard Negative Mining
For each query, select negatives from:
- Random: easy, low loss
- Top-k cosine: hard, forces geometric to find signal cosine misses
- Same cluster, different topic: structural hard negative

### Regularization
- ‖W‖²_F < 10.0 (weight decay)
- ‖f(q)‖ = 1.0 enforced by L2 norm (not learned)
- w₂ ≥ 0.01 (bivector preservation constraint)

## Training Data

### Self-Supervised (Tonight, no labels)
Query = paper embedding
Positive = k-NN neighbor (k=5) in 384-d cosine space
Negative = random far point OR high-cosine wrong-topic point

### Supervised (Future, with query logs)
Query = actual user query embedding
Positive = clicked paper
Negative = shown but not clicked

Minimum: 10,000 triplets for convergence.
Optimal: 100,000+ triplets with hard negatives.

## Training Schedule

| Phase | Epochs | Learning Rate | Batch Size | Negatives |
|---|---|---|---|---|
| Warmup | 50 | 0.01 | 32 | Random |
| Hard negative mining | 200 | 0.001 | 64 | Top-10 cosine |
| Fine-tune weights | 50 | 0.0001 | 128 | Structural |

## Evaluation Metrics

1. **Geometric R@1** vs Cosine R@1 on held-out queries
2. **Bivector weight evolution** w₂(t) — must stay > 0.3
3. **Grade distribution** of top-100 relationships
4. **Norm drift** — ‖f(q)‖ should be 1.0 ± 0.001
5. **Visualization** — 5-D PCA of f(q) vs raw MiniLM PCA

## Implementation Roadmap

### Phase 0: CPU Prototype (Tonight)
- Numpy MLP, 1000 triplets, 300 epochs
- Goal: w₂ > 0.3, geometric R@1 > 20% (vs cosine baseline)

### Phase 1: PyTorch CPU
- 10,000 triplets, proper autograd
- Goal: w₂ > 0.5, geometric R@1 > 50%

### Phase 2: PyTorch GPU
- 100,000 triplets, hard negative mining
- Goal: w₂ > 0.8, geometric R@1 > 80%, beats cosine on tie-breaks

### Phase 3: Rust Inference
- Export W1, W2, W3 as const matrices
- SIMD matrix multiply 384→5
- Geometric product in Rust Cl(5) engine
- Latency target: < 50 µs per query-paper pair

## Files
- `scripts/tge_prototype.py` — Phase 0 numpy trainer
- `scripts/tge_eval.py` — evaluation harness
- `src/tge_encoder.rs` — Phase 3 Rust inference
- `data/tge_W1.npy`, `W2.npy`, `W3.npy`, `weights.npy` — trained parameters

## Success Criteria
TGE is viable ONLY if:
1. w₂ > 0.3 after 300 epochs (bivector lives)
2. Geometric R@1 > 50% on held-out set (better than random)
3. Geometric R@5 > 80% (usable for re-ranking top-100)

If any criterion fails, TGE goes to MUSEUM.
