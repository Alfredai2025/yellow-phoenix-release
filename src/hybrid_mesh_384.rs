// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Honest 384-bit HybridMesh — single layer, no padding, no ghost slots.
//! Uses CrystalMesh384 for direct Hamming search + HNSW graph overlay.

use crate::binary_hnsw_384::BinaryHNSW384;
use crate::crystal_mesh_384::{CrystalMesh384, PAP_384_BYTES};
use crate::embedding_store::EmbeddingStore;

#[derive(Clone, Debug)]
pub struct HybridMesh384 {
    pub mesh: CrystalMesh384,
    pub hnsw: BinaryHNSW384,
    pub embeddings: EmbeddingStore,
    pub graph_hits: u64,
    pub direct_fallbacks: u64,
    pub failed_inserts: u64,
}

impl HybridMesh384 {
    pub fn new(capacity: usize, emb_dim: usize) -> Self {
        Self {
            mesh: CrystalMesh384::new(capacity),
            hnsw: BinaryHNSW384::new(),
            embeddings: EmbeddingStore::new(emb_dim),
            graph_hits: 0,
            direct_fallbacks: 0,
            failed_inserts: 0,
        }
    }

    /// Insert document into both mesh and HNSW graph.
    pub fn insert(&mut self, id: u64, pap: [u8; PAP_384_BYTES], embedding: Vec<f32>) -> bool {
        if !self.mesh.insert(id, pap) {
            self.failed_inserts += 1;
            return false;
        }
        self.hnsw.insert(id, pap);
        self.embeddings.insert(id, embedding);
        true
    }

    /// Direct Hamming nearest neighbor in the mesh.
    pub fn query_direct(&self, pap: &[u8; PAP_384_BYTES]) -> Option<u64> {
        self.mesh.query_direct(pap)
    }

    /// Top-k by Hamming (brute force over slots).
    pub fn query_k(&self, pap: &[u8; PAP_384_BYTES], k: usize) -> Vec<(u32, u64)> {
        self.mesh.query_k(pap, k)
    }

    /// HNSW graph search for approximate neighbors.
    pub fn query_graph(&self, pap: &[u8; PAP_384_BYTES], k: usize) -> Vec<(u32, u64)> {
        self.hnsw
            .search(pap, k)
            .into_iter()
            .filter_map(|(dist, idx)| self.hnsw.node(idx).map(|n| (dist, n.id)))
            .collect()
    }

    /// Auto query: try HNSW first, fall back to direct Hamming if empty.
    pub fn query_auto(&mut self, pap: &[u8; PAP_384_BYTES], k: usize) -> Vec<(u32, u64)> {
        let graph_results = self.query_graph(pap, k);
        if !graph_results.is_empty() {
            self.graph_hits += 1;
            graph_results
        } else {
            self.direct_fallbacks += 1;
            self.query_k(pap, k)
        }
    }

    /// Number of documents.
    pub fn len(&self) -> usize {
        self.mesh.len
    }

    /// Check if mesh is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Batch insert for efficiency.
    pub fn batch_insert(&mut self, items: &[(u64, [u8; PAP_384_BYTES], Vec<f32>)]) -> usize {
        let mut inserted = 0;
        for (id, pap, emb) in items {
            if self.insert(*id, *pap, emb.clone()) {
                inserted += 1;
            }
        }
        inserted
    }

    /// Check if mesh, HNSW, and embeddings are in sync.
    pub fn check_integrity(&self) -> Result<(), String> {
        let mesh_count = self.mesh.len;
        let hnsw_count = self.hnsw.len();
        let emb_count = self.embeddings.len();
        if mesh_count != hnsw_count || mesh_count != emb_count {
            return Err(format!(
                "integrity mismatch: mesh={}, hnsw={}, embeddings={}",
                mesh_count, hnsw_count, emb_count
            ));
        }
        Ok(())
    }

    /// Stats summary.
    pub fn stats(&self) -> String {
        let mut s = String::new();
        s.push_str(r#"{"entries":"#);
        s.push_str(&self.len().to_string());
        s.push_str(r#","graph_hits":"#);
        s.push_str(&self.graph_hits.to_string());
        s.push_str(r#","direct_fallbacks":"#);
        s.push_str(&self.direct_fallbacks.to_string());
        s.push_str(r#","failed_inserts":"#);
        s.push_str(&self.failed_inserts.to_string());
        s.push('}');
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_384_insert_query() {
        let mut hybrid = HybridMesh384::new(1024, 3);
        let mut pap = [0u8; PAP_384_BYTES];
        pap[0] = 0xAB;
        assert!(hybrid.insert(42, pap, vec![1.0, 0.0, 0.0]));
        let result = hybrid.query_direct(&pap);
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_hybrid_384_auto_fallback() {
        let mut hybrid = HybridMesh384::new(1024, 3);
        let pap = [0xAB; PAP_384_BYTES];
        // Empty HNSW, should fallback to direct Hamming
        let results = hybrid.query_auto(&pap, 5);
        assert!(results.is_empty()); // no docs inserted
    }
}

// ==================== Two-Tier Search: HNSW + Embedding Re-rank ====================

impl HybridMesh384 {
    /// Two-tier search: HNSW fast retrieval + embedding re-rank.
    /// 
    /// Tier 1: HNSW graph search on binary hash → top_k_fast candidates.
    /// Tier 2: Fetch embeddings, cosine re-rank → top_n_final.
    pub fn search_with_rerank(
        &self,
        pap: &[u8; PAP_384_BYTES],
        query_embedding: &[f32],
        embeddings: &EmbeddingStore,
        k_fast: usize,
        n_final: usize,
    ) -> Vec<(f32, u64)> {
        // Tier 1: HNSW fast approximate search
        let hnsw_results = self.query_graph(pap, k_fast);
        let candidate_ids: Vec<u64> = hnsw_results.iter().map(|(_, id)| *id).collect();
        
        if candidate_ids.is_empty() {
            // Fallback: brute-force Hamming direct search
            let direct = self.query_k(pap, k_fast);
            let direct_ids: Vec<u64> = direct.iter().map(|(_, id)| *id).collect();
            return embeddings.rerank_by_cosine(query_embedding, &direct_ids, n_final);
        }
        
        // Tier 2: Re-rank by embedding cosine similarity
        embeddings.rerank_by_cosine(query_embedding, &candidate_ids, n_final)
    }
}

#[cfg(test)]
mod rerank_tests {
    use super::*;
    use crate::crystal_mesh_384::PAP_384_BYTES;
    use crate::embedding_store::EmbeddingStore;

    #[test]
    fn test_two_tier_search() {
        let mut hybrid = HybridMesh384::new(1024, 3);
        
        let pap = [0xAB; PAP_384_BYTES];
        hybrid.insert(1, pap, vec![1.0, 0.0, 0.0]);
        
        let query_emb = vec![1.0, 0.0, 0.0];
        let results = hybrid.search_with_rerank(&pap, &query_emb, &hybrid.embeddings, 5, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 1);
    }
}
