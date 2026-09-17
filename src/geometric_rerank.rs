// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Tier 3: 5-D Geometric Re-rank
//! Projects 384-dim embeddings into a 5-D subspace (first 5 dims normalized)
//! and scores candidates by the geometric product: inner + wedge + dual.

use std::cmp::Ordering;

/// Normalize a 5-D vector to unit length.
fn normalize_5d(v: &[f32]) -> [f32; 5] {
    let mut out = [0.0f32; 5];
    let mut norm_sq = 0.0f32;
    for i in 0..5.min(v.len()) {
        out[i] = v[i];
        norm_sq += v[i] * v[i];
    }
    let norm = norm_sq.sqrt();
    if norm > 1e-8 {
        for i in 0..5 {
            out[i] /= norm;
        }
    }
    out
}

/// Inner product (scalar part) — how aligned two unit vectors are.
fn inner_5d(a: [f32; 5], b: [f32; 5]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Wedge magnitude (bivector part) — how much they span a 2-D plane.
/// For unit vectors: wedge = sqrt(1 - inner^2) = sin(theta)
fn wedge_magnitude_5d(a: [f32; 5], b: [f32; 5]) -> f32 {
    let mut sum_sq = 0.0f32;
    for i in 0..5 {
        for j in (i + 1)..5 {
            let w = a[i] * b[j] - a[j] * b[i];
            sum_sq += w * w;
        }
    }
    sum_sq.sqrt()
}

/// Dual complement magnitude (4-vector part) — how they fill the 5-D volume.
fn dual_magnitude_5d(a: [f32; 5], b: [f32; 5]) -> f32 {
    // Simplified: volume spanned by a, b, and three orthonormal basis vectors
    // For two vectors in 5-D, the dual is proportional to the complement of their span.
    // We approximate as the product of the unused dimensions.
    let mut unused = [1.0f32; 5];
    for i in 0..5 {
        unused[i] = (1.0 - a[i].abs()) * (1.0 - b[i].abs());
    }
    unused.iter().product::<f32>().sqrt()
}

/// Geometric score: weighted combination of inner, wedge, dual.
/// High inner = same direction (good for NN).
/// Low wedge = same line (collinear, good).
/// High dual = fills space (bad for NN, we want concentrated).
fn geometric_score(a_emb: &[f32], b_emb: &[f32]) -> f32 {
    let a = normalize_5d(a_emb);
    let b = normalize_5d(b_emb);

    let inner = inner_5d(a, b);
    let wedge = wedge_magnitude_5d(a, b);
    let dual = dual_magnitude_5d(a, b);

    // Weighted: alignment is primary, planar structure secondary, volume penalty
    let score = inner
        + 0.1 * (1.0 - wedge)  // reward low wedge (same plane)
        - 0.01 * dual;         // penalize high dual (too spread out)

    score
}

/// Re-rank top-k candidates by geometric score.
/// Returns (score, original_index) sorted by score descending.
pub fn rerank_geometric(query: &[f32], candidates: &[&[f32]], k: usize) -> Vec<(f32, usize)> {
    let mut scored: Vec<(f32, usize)> = candidates
        .iter()
        .enumerate()
        .map(|(idx, emb)| (geometric_score(query, emb), idx))
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let v = [1.0f32, 0.0, 0.0, 0.0, 0.0];
        let s = geometric_score(&v, &v);
        assert!(s > 0.9, "identical vectors should score > 0.9, got {}", s);
    }

    #[test]
    fn test_orthogonal() {
        let a = [1.0f32, 0.0, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0, 0.0];
        let s = geometric_score(&a, &b);
        assert!(s < 0.2, "orthogonal should score low, got {}", s);
    }

    #[test]
    fn test_rerank() {
        let q = [1.0f32, 0.0, 0.0, 0.0, 0.0];
        let c0 = [1.0f32, 0.0, 0.0, 0.0, 0.0]; // identical
        let c1 = [0.0f32, 1.0, 0.0, 0.0, 0.0]; // orthogonal
        let c2 = [0.9f32, 0.1, 0.0, 0.0, 0.0]; // close

        let candidates: Vec<&[f32]> = vec![&c1, &c0, &c2];
        let ranked = rerank_geometric(&q, &candidates, 2);

        assert_eq!(ranked[0].1, 1); // c0 (identical) should be first
        assert_eq!(ranked[1].1, 2); // c2 (close) should be second
    }
}
