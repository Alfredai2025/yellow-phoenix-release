// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! hash_stage.rs — Stage 0: hash lookup and candidate filter.

use crate::hybrid_mesh::{HybridMesh, PAP_128_BYTES, PAP_512_BYTES};

#[derive(Clone, Debug)]
pub struct FeatureResult {
    pub score: f32,
    pub confidence: f32,
    pub candidates: Vec<(u64, f32)>,
}

const FAST_PATH_CONFIDENCE_THRESHOLD: f32 = 0.95;

pub struct HashStage {
    mesh: HybridMesh,
}

impl HashStage {
    pub fn new(mesh: HybridMesh) -> Self {
        Self { mesh }
    }

    /// Borrow the underlying mesh (used by batch fusion to share the index).
    pub fn mesh(&self) -> &HybridMesh {
        &self.mesh
    }

    /// Try the fast bucket-only path when confidence is high.
    pub fn query_with_confidence(
        &self,
        pap_128: &[u8; PAP_128_BYTES],
        pap_512: &[u8; PAP_512_BYTES],
        top_k: usize,
        confidence: f32,
    ) -> FeatureResult {
        if confidence >= FAST_PATH_CONFIDENCE_THRESHOLD {
            if let Some(results) = self.mesh.query_fast_path(pap_128, top_k.max(5)) {
                return Self::feature_result_from(results);
            }
        }
        self.query(pap_128, pap_512, top_k)
    }

    fn feature_result_from(candidates: Vec<(f32, u64)>) -> FeatureResult {
        let candidates: Vec<(u64, f32)> = candidates.into_iter().map(|(score, id)| (id, score)).collect();
        let confidence = if candidates.len() == 1 {
            0.98
        } else if candidates.len() >= 2 && candidates[0].1 > 0.0 {
            let ratio = candidates[0].1 / (candidates[1].1 + 1e-6);
            ratio.min(1.0)
        } else {
            0.0
        };
        FeatureResult {
            score: candidates.first().map(|(_, s)| *s).unwrap_or(0.0),
            confidence,
            candidates,
        }
    }

    pub fn query(&self, pap_128: &[u8; PAP_128_BYTES], pap_512: &[u8; PAP_512_BYTES], top_k: usize) -> FeatureResult {
        let results = self.mesh.query_auto(pap_128, pap_512, top_k.max(10));
        let candidates: Vec<(u64, f32)> = results.into_iter().map(|(score, id)| (id, score)).collect();

        // Confidence based on how much the top score dominates.
        let confidence = if candidates.len() >= 2 && candidates[0].1 > 0.0 {
            let ratio = candidates[0].1 / (candidates[1].1 + 1e-6);
            ratio.min(1.0)
        } else if candidates.len() == 1 {
            0.95
        } else {
            0.0
        };

        FeatureResult {
            score: candidates.first().map(|(_, s)| *s).unwrap_or(0.0),
            confidence,
            candidates,
        }
    }
}
