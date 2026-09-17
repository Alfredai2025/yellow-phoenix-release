// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Spectral Hologram — low-rank projection index.
//!
//! Built from a precomputed spectral basis V (d x k) and projections C (n x k).
//! Query projects into k-dim spectral space and scores against all stored projections.
//! The field H = C^T C captures the manifold covariance (optional diffusion scoring).

use alloc::vec::Vec;

pub struct SpectralHologram {
    d: usize,
    k: usize,
    n: usize,
    basis: Vec<f32>,      // V, column-major or row-major? Use row-major: basis[i*k + j] = V[i][j]
    projections: Vec<f32>, // C, row-major: projections[i*k + j] = C[i][j]
    field: Vec<f32>,      // H = C^T C / n, row-major k x k
}

impl SpectralHologram {
    /// basis: d x k row-major matrix (each row is one original dimension's coefficients)
    /// projections: n x k row-major matrix (each row is one paper's spectral coords)
    pub fn new(basis: Vec<f32>, d: usize, k: usize, projections: Vec<f32>, n: usize) -> Self {
        assert_eq!(basis.len(), d * k);
        assert_eq!(projections.len(), n * k);

        // Build normalized covariance field H = (C^T C) / n
        let mut field = vec![0.0f32; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut sum = 0.0f32;
                for row in 0..n {
                    sum += projections[row * k + i] * projections[row * k + j];
                }
                field[i * k + j] = sum / (n as f32);
            }
        }

        Self { d, k, n, basis, projections, field }
    }

    /// Project raw embedding into spectral space: c = q^T V
    fn project(&self, query: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; self.k];
        for j in 0..self.k {
            let mut sum = 0.0f32;
            for i in 0..self.d {
                sum += query[i] * self.basis[i * self.k + j];
            }
            c[j] = sum;
        }
        c
    }

    /// Direct spectral scoring: score_i = c_q · c_i
    pub fn query_direct(&self, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        let c_q = self.project(query);
        let mut scores: Vec<(u64, f32)> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mut score = 0.0f32;
            for j in 0..self.k {
                score += c_q[j] * self.projections[i * self.k + j];
            }
            scores.push((i as u64, score));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// MUSEUM — Diffusion scoring: score_i = (c_q^T H) · c_i
    ///
    /// `H = C^T C` is dominated by the first eigenvalue and drowns the query
    /// signal (R@10 ≈ 0.2%). Kept under the `holographic-diffusion` feature for
    /// reference. Production path: `query_direct` (64-d spectral dot) or
    /// `YPEngine::search_hybrid` (HNSW + exact cosine). See commit bdea142b.
    #[cfg(feature = "holographic-diffusion")]
    pub fn query_diffusion(&self, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        let c_q = self.project(query);

        // v = c_q^T H  (1 x k)
        let mut v = vec![0.0f32; self.k];
        for j in 0..self.k {
            let mut sum = 0.0f32;
            for l in 0..self.k {
                sum += c_q[l] * self.field[l * self.k + j];
            }
            v[j] = sum;
        }

        let mut scores: Vec<(u64, f32)> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mut score = 0.0f32;
            for j in 0..self.k {
                score += v[j] * self.projections[i * self.k + j];
            }
            scores.push((i as u64, score));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    pub fn len(&self) -> usize { self.n }
    pub fn dims(&self) -> (usize, usize) { (self.d, self.k) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_identity_basis(d: usize, k: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; d * k];
        for i in 0..k.min(d) {
            v[i * k + i] = 1.0;
        }
        v
    }

    #[test]
    fn test_direct_query_finds_self() {
        let d = 4;
        let k = 4;
        let n = 5;
        let basis = approx_identity_basis(d, k);
        let mut projections = vec![0.0f32; n * k];
        for i in 0..n {
            projections[i * k + (i % k)] = 1.0;
        }
        let holo = SpectralHologram::new(basis, d, k, projections, n);
        let query = vec![0.0f32, 0.0, 1.0, 0.0]; // matches paper 2
        let r = holo.query_direct(&query, 3);
        assert_eq!(r[0].0, 2);
    }

    #[test]
    #[cfg(feature = "holographic-diffusion")]
    fn test_diffusion_query_runs() {
        let d = 4;
        let k = 4;
        let n = 5;
        let basis = approx_identity_basis(d, k);
        let mut projections = vec![0.0f32; n * k];
        for i in 0..n {
            projections[i * k + (i % k)] = 1.0;
        }
        let holo = SpectralHologram::new(basis, d, k, projections, n);
        let query = vec![0.0f32, 0.0, 1.0, 0.0];
        let r = holo.query_diffusion(&query, 3);
        assert!(!r.is_empty());
    }
}
