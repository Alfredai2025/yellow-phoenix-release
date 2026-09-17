// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! wedge_stage.rs — Stage 2: wedge top-K contrastive re-rank.

use crate::hash_stage::FeatureResult;
use crate::hybrid_mesh::PAP_512_BYTES;

pub struct WedgeStage;

impl WedgeStage {
    pub fn new() -> Self {
        Self
    }

    pub fn query(
        &self,
        _pap_512: &[u8; PAP_512_BYTES],
        spectral_result: &FeatureResult,
    ) -> FeatureResult {
        // Wedge re-rank: penalize candidates whose score is too close to the next best.
        let mut candidates = spectral_result.candidates.clone();
        if candidates.len() >= 2 {
            let mut reranked: Vec<(u64, f32)> = candidates
                .windows(2)
                .enumerate()
                .map(|(i, w)| {
                    let (id, score) = w[0];
                    let gap = (score - w[1].1).max(0.0);
                    let bonus = if i == 0 { gap } else { gap * 0.5 };
                    (id, score + bonus)
                })
                .collect();
            if let Some(last) = candidates.last() {
                reranked.push(*last);
            }
            candidates = reranked;
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(10);
        }

        let confidence = if candidates.len() >= 2 {
            let ratio = candidates[0].1 / (candidates[1].1 + 1e-6);
            ratio.min(0.9)
        } else {
            0.45
        };

        FeatureResult {
            score: candidates.first().map(|(_, s)| *s).unwrap_or(0.0),
            confidence,
            candidates,
        }
    }
}
