// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! spectral_stage.rs — Stage 1: spectral dot product re-rank and anomaly detection.

use std::collections::HashMap;

use crate::hash_stage::FeatureResult;
use crate::hybrid_mesh::{pap_distance_512, PAP_128_BYTES, PAP_512_BYTES};
use crate::spectral_coords::SpectralCoords;

pub struct SpectralStage {
    /// Map from candidate id to its stored 512-bit PAP.
    /// Populated from the mesh so we score real hashes instead of a stub.
    pap_map: HashMap<u64, [u8; PAP_512_BYTES]>,
    /// M3.3: precomputed spectral coordinates per candidate.
    coords_map: HashMap<u64, SpectralCoords>,
}

impl SpectralStage {
    pub fn new() -> Self {
        Self {
            pap_map: HashMap::new(),
            coords_map: HashMap::new(),
        }
    }

    pub fn with_paps(paps: HashMap<u64, [u8; PAP_512_BYTES]>) -> Self {
        Self {
            pap_map: paps,
            coords_map: HashMap::new(),
        }
    }

    pub fn with_paps_and_coords(
        paps: HashMap<u64, [u8; PAP_512_BYTES]>,
        coords: HashMap<u64, SpectralCoords>,
    ) -> Self {
        Self {
            pap_map: paps,
            coords_map: coords,
        }
    }

    pub fn query(
        &self,
        _pap_128: &[u8; PAP_128_BYTES],
        pap_512: &[u8; PAP_512_BYTES],
        hash_result: &FeatureResult,
    ) -> FeatureResult {
        let mut candidates: Vec<(u64, f32)> = hash_result
            .candidates
            .iter()
            .map(|&(id, _)| {
                // M3.3: prefer precomputed spectral coords (3-float distance)
                if let Some(coords) = self.coords_map.get(&id) {
                    let query_coords = SpectralCoords::from_hash(pap_512);
                    let score = query_coords.score(coords);
                    return (id, score);
                }
                // Fall back to real 512-bit distance; stub only if missing entirely.
                let fallback = int_hash(id);
                let candidate_pap = self.pap_map.get(&id).unwrap_or(&fallback);
                let d512 = pap_distance_512(pap_512, candidate_pap);
                (id, d512)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(20);

        let confidence = if candidates.len() >= 2 {
            let ratio = candidates[0].1 / (candidates[1].1 + 1e-6);
            ratio.min(0.95)
        } else {
            0.5
        };

        FeatureResult {
            score: candidates.first().map(|(_, s)| *s).unwrap_or(0.0),
            confidence,
            candidates,
        }
    }

    /// Rebuild the PAP map from a mesh. Useful after mesh updates.
    pub fn rebuild_from_mesh(&mut self, mesh: &crate::hybrid_mesh::HybridMesh) {
        self.pap_map.clear();
        self.pap_map.reserve(mesh.fine.slots.len());
        self.coords_map.clear();
        self.coords_map.reserve(mesh.spectral_coords.len());
        for slot in &mesh.fine.slots {
            self.pap_map.insert(slot.id, slot.pap);
        }
        self.coords_map.extend(mesh.spectral_coords.iter().map(|(&k, &v)| (k, v)));
    }
}

/// Deterministic 512-bit hash from a u64 id (fallback only).
fn int_hash(id: u64) -> [u8; 64] {
    let mut h = [0u8; 64];
    let bytes = id.to_le_bytes();
    for i in 0..64 {
        h[i] = bytes[i % 8].wrapping_add(i as u8);
    }
    h
}
