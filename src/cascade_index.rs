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

use crate::exact::ExactIndex;
use crate::pq::PQIndex;
use crate::query::LSHIndex;
use crate::types::multivector::BinaryMultivector;

pub struct CascadeIndex {
    pub lsh: LSHIndex,
    pub pq: PQIndex,
    pub exact: ExactIndex,
    _dim: usize,
}

impl CascadeIndex {
    pub fn new(lsh_buckets: usize, lsh_hashes: usize, pq_m: usize, pq_subspace_dim: usize, exact_dim: usize) -> Self {
        Self {
            lsh: LSHIndex::new(lsh_buckets, lsh_hashes),
            pq: PQIndex::new(pq_m, pq_subspace_dim),
            exact: ExactIndex::new(exact_dim),
            _dim: exact_dim,
        }
    }

    pub fn train_pq(&mut self, vectors: &[f32], num_vectors: usize) {
        self.pq.train(vectors, num_vectors);
    }

    pub fn add(&mut self, binary: BinaryMultivector, pq_vector: &[f32], exact_vector: &[f32]) {
        self.lsh.add(binary);
        self.pq.add(pq_vector);
        self.exact.add(exact_vector);
    }

    pub fn query(
        &self,
        binary_query: &BinaryMultivector,
        float_query: &[f32],
        top_k: usize,
        stage1_radius: i16,
        stage1_max: usize,
        stage2_max: usize,
    ) -> Vec<(usize, f32)> {
        let mut candidates = self.lsh.collect_candidates(binary_query, stage1_radius);

        if stage1_max > 0 && candidates.len() > stage1_max {
            let mut scored: Vec<(usize, f32)> = candidates
                .iter()
                .map(|&idx| {
                    let d = crate::distance::pap_distance(binary_query, &self.lsh.get_prototype(idx));
                    (idx, d)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            candidates = scored.into_iter().take(stage1_max).map(|(idx, _)| idx).collect();
        }

        let pq_results = self.pq.query(float_query, &candidates, stage2_max);
        let pq_ids: Vec<usize> = pq_results.into_iter().map(|(id, _)| id).collect();

        self.exact.query(float_query, &pq_ids, top_k)
    }

    pub fn query_fast(&self, binary_query: &BinaryMultivector, top_k: usize, threshold: f32) -> Vec<(usize, f32)> {
        self.lsh.query(binary_query, top_k, threshold)
    }

    pub fn query_medium(
        &self,
        binary_query: &BinaryMultivector,
        float_query: &[f32],
        top_k: usize,
        stage1_radius: i16,
        stage1_max: usize,
    ) -> Vec<(usize, f32)> {
        let mut candidates = self.lsh.collect_candidates(binary_query, stage1_radius);
        if stage1_max > 0 && candidates.len() > stage1_max {
            let mut scored: Vec<(usize, f32)> = candidates
                .iter()
                .map(|&idx| {
                    let d = crate::distance::pap_distance(binary_query, &self.lsh.get_prototype(idx));
                    (idx, d)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            candidates = scored.into_iter().take(stage1_max).map(|(idx, _)| idx).collect();
        }
        self.pq.query(float_query, &candidates, top_k)
    }

    pub fn len(&self) -> usize {
        self.lsh.len()
    }
    pub fn is_empty(&self) -> bool {
        self.lsh.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::multivector::BinaryMultivector;

    const CHUNKS: usize = 2;

    fn make_binary(indices: &[usize]) -> BinaryMultivector {
        let mut chunks = [0u64; CHUNKS];
        for &i in indices {
            chunks[i / 64] |= 1u64 << (i % 64);
        }
        BinaryMultivector(chunks)
    }

    fn train_data(n: usize, dim: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                v.push(((i * 13 + d * 7) % 100) as f32 / 100.0);
            }
        }
        v
    }

    #[test]
    fn cascade_full_pipeline() {
        let mut idx = CascadeIndex::new(64, 1, 2, 4, 8);
        let train = train_data(1000, 8);
        idx.train_pq(&train, 1000);

        for i in 0..10 {
            let binary = make_binary(&[i, i + 1, i + 2]);
            let mut vec = vec![0.0f32; 8];
            vec[i % 8] = 1.0;
            idx.add(binary, &vec, &vec);
        }

        let q_binary = make_binary(&[0, 1, 2]);
        let mut q_float = vec![0.0f32; 8];
        q_float[0] = 1.0;

        let results = idx.query(&q_binary, &q_float, 3, 1, 20, 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn cascade_medium_skips_exact() {
        let mut idx = CascadeIndex::new(64, 1, 2, 4, 8);
        let train = train_data(1000, 8);
        idx.train_pq(&train, 1000);

        let mut vec = vec![0.0f32; 8];
        vec[0] = 1.0;
        idx.add(make_binary(&[0, 1]), &vec, &vec);

        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = idx.query_medium(&make_binary(&[0, 1]), &q, 1, 1, 10);
        assert_eq!(results.len(), 1);
    }
}
