// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Flat embedding store for two-tier HNSW re-ranking.
//! Maps doc_id -> float embedding vector. Dense or sparse.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct EmbeddingStore {
    dim: usize,
    data: HashMap<u64, Vec<f32>>,
}

impl EmbeddingStore {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: u64, embedding: Vec<f32>) {
        if embedding.len() == self.dim {
            self.data.insert(id, embedding);
        }
    }

    pub fn get(&self, id: u64) -> Option<&[f32]> {
        self.data.get(&id).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Cosine similarity between two vectors. Returns -1..1.
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    /// Re-rank candidate IDs by cosine similarity to query embedding.
    /// Returns top_n (score, id) sorted by score descending.
    pub fn rerank_by_cosine(&self, query: &[f32], candidates: &[u64], top_n: usize) -> Vec<(f32, u64)> {
        let mut scored: Vec<(f32, u64)> = candidates
            .iter()
            .filter_map(|&id| {
                self.get(id).map(|emb| (Self::cosine(query, emb), id))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(top_n);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert!((EmbeddingStore::cosine(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(EmbeddingStore::cosine(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_rerank() {
        let mut store = EmbeddingStore::new(2);
        store.insert(1, vec![1.0, 0.0]);
        store.insert(2, vec![0.0, 1.0]);
        store.insert(3, vec![0.9, 0.1]);
        
        let query = vec![1.0, 0.0];
        let results = store.rerank_by_cosine(&query, &[1, 2, 3], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1, 1); // exact match
        assert_eq!(results[1].1, 3); // close match
    }
}
