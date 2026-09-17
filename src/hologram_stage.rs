// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! hologram_stage.rs — Stage 3: holographic cross-validation and final consensus buffer.

use crate::hash_stage::FeatureResult;
use crate::hybrid_mesh::PAP_512_BYTES;

pub struct HologramStage;

impl HologramStage {
    pub fn new() -> Self {
        Self
    }

    pub fn query(
        &self,
        _pap_512: &[u8; PAP_512_BYTES],
        wedge_result: &FeatureResult,
    ) -> FeatureResult {
        // Hologram: final confidence calibration. If top score isn't clearly ahead,
        // suppress confidence so downstream consensus can request more stages next time.
        let mut candidates = wedge_result.candidates.clone();
        candidates.truncate(5);

        let confidence = if candidates.len() >= 2 && candidates[0].1 > 0.0 {
            let ratio = candidates[0].1 / (candidates[1].1 + 1e-6);
            if ratio > 1.5 {
                0.98
            } else if ratio > 1.2 {
                0.85
            } else {
                0.55
            }
        } else {
            0.4
        };

        FeatureResult {
            score: candidates.first().map(|(_, s)| *s).unwrap_or(0.0),
            confidence,
            candidates,
        }
    }
}
