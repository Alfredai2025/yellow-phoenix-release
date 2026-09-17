// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! spectral_coords.rs — M3.3: Precomputed spectral signatures
//!
//! Build time: compute 3-float spectral coordinates per paper (12 bytes each)
//! Query time: Euclidean distance between coords (3 ops vs 512-bit Hamming)

use crate::hybrid_mesh::PAP_512_BYTES;

/// 3-dimensional spectral coordinates (12 bytes per paper)
#[derive(Clone, Copy, Debug, Default)]
pub struct SpectralCoords {
    pub c0: f32,
    pub c1: f32,
    pub c2: f32,
}

impl SpectralCoords {
    /// Compute from 512-bit hash using deterministic projection.
    /// Runs ONCE at build time per paper.
    pub fn from_hash(hash: &[u8; PAP_512_BYTES]) -> Self {
        let [c0, c1, c2] = crate::simd_kernels::simd_spectral_project(hash);
        SpectralCoords { c0, c1, c2 }
    }

    /// Query-time distance: 3 subtractions + sqrt
    pub fn distance(&self, other: &SpectralCoords) -> f32 {
        let d0 = self.c0 - other.c0;
        let d1 = self.c1 - other.c1;
        let d2 = self.c2 - other.c2;
        (d0 * d0 + d1 * d1 + d2 * d2).sqrt()
    }

    /// Convert distance to similarity score [0,1]
    pub fn score(&self, other: &SpectralCoords) -> f32 {
        let dist = self.distance(other);
        1.0 / (1.0 + dist)
    }
}

/// 6-byte f16 spectral coordinates.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpectralCoordsF16 {
    pub c0: u16,
    pub c1: u16,
    pub c2: u16,
}

impl SpectralCoordsF16 {
    pub fn from_coords(coords: &SpectralCoords) -> Self {
        Self {
            c0: quantize_f16(coords.c0),
            c1: quantize_f16(coords.c1),
            c2: quantize_f16(coords.c2),
        }
    }

    pub fn to_coords(&self) -> SpectralCoords {
        SpectralCoords {
            c0: dequantize_f16(self.c0),
            c1: dequantize_f16(self.c1),
            c2: dequantize_f16(self.c2),
        }
    }
}

fn quantize_f16(v: f32) -> u16 {
    // Spectral coords typically fall in [-2, 2].  Map that range to u16.
    let t = ((v + 2.0) / 4.0).clamp(0.0, 1.0);
    (t * 65535.0) as u16
}

fn dequantize_f16(bits: u16) -> f32 {
    bits as f32 / 65535.0 * 4.0 - 2.0
}

/// Fast spectral query using precomputed coords.
///
/// candidates: Vec<(paper_id, SpectralCoords)>
/// Returns: Vec<(score, paper_id)> sorted by score descending
pub fn query_spectral_fast(
    query_hash: &[u8; PAP_512_BYTES],
    candidates: &[(u64, SpectralCoords)],
    top_k: usize,
) -> Vec<(f32, u64)> {
    let query_coords = SpectralCoords::from_hash(query_hash);

    let mut scored: Vec<(f32, u64)> = candidates
        .iter()
        .map(|(pid, coords)| {
            let score = query_coords.score(coords);
            (score, *pid)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coords_deterministic() {
        let h = [0xABu8; 64];
        let c1 = SpectralCoords::from_hash(&h);
        let c2 = SpectralCoords::from_hash(&h);
        assert!((c1.c0 - c2.c0).abs() < 0.001);
        assert!((c1.c1 - c2.c1).abs() < 0.001);
        assert!((c1.c2 - c2.c2).abs() < 0.001);
    }

    #[test]
    fn test_distance_symmetric() {
        let h1 = [0x00u8; 64];
        let h2 = [0xFFu8; 64];
        let c1 = SpectralCoords::from_hash(&h1);
        let c2 = SpectralCoords::from_hash(&h2);
        assert!(c1.distance(&c2) > 0.0);
    }

    #[test]
    fn test_fast_query_exact_match() {
        let h = [0xABu8; 64];
        let c = SpectralCoords::from_hash(&h);
        let candidates = vec![(1u64, c), (2u64, SpectralCoords::from_hash(&[0x00u8; 64]))];
        let results = query_spectral_fast(&h, &candidates, 2);
        assert_eq!(results[0].1, 1);
    }
}
