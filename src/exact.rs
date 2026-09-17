// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

// GHOST MODULE — NOT WIRED INTO src/lib.rs
// This file exists on disk but is NOT declared in src/lib.rs.
// It does NOT compile into the library and is NOT reachable from Python.
//
// AUDIT DATE: 2026-07-27
// ACTION: Preserved per "Wire First, Delete Never" policy.
//         Do not modify unless wiring into the build.
//
use alloc::vec::Vec;
use core::cmp::Ordering;

pub struct ExactIndex {
    vectors: Vec<f32>,
    dim: usize,
    num_vectors: usize,
}

impl ExactIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dim,
            num_vectors: 0,
        }
    }

    pub fn add(&mut self, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim);
        self.vectors.extend_from_slice(vector);
        self.num_vectors += 1;
    }

    pub fn query(&self, query: &[f32], candidates: &[usize], top_k: usize) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.dim);

        let mut results = Vec::with_capacity(top_k.min(candidates.len()));
        let mut worst_best = f32::MAX;

        for &idx in candidates {
            let start = idx * self.dim;
            let mut dist = 0.0f32;
            for d in 0..self.dim {
                let diff = query[d] - self.vectors[start + d];
                dist += diff * diff;
            }

            if dist < worst_best || results.len() < top_k {
                results.push((idx, dist));
                results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                if results.len() > top_k {
                    results.pop();
                }
                worst_best = results.last().map(|(_, d)| *d).unwrap_or(f32::MAX);
            }
        }

        results
    }

    pub fn len(&self) -> usize {
        self.num_vectors
    }
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ranks_correctly() {
        let mut idx = ExactIndex::new(4);
        idx.add(&[1.0, 0.0, 0.0, 0.0]);
        idx.add(&[0.0, 1.0, 0.0, 0.0]);
        idx.add(&[0.0, 0.0, 1.0, 0.0]);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = idx.query(&query, &[0, 1, 2], 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-6);
    }
}
