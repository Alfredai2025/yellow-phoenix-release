// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use alloc::vec::Vec;
use crate::types::multivector::BinaryMultivector;
use crate::types::ternary::TernaryMultivector;

/// Compute the PAP distance between two binary multivectors.
///
/// The distance is defined as `1.0 - similarity`, where similarity = overlap / sqrt(|a| * |b|).
/// Overlap is the number of bits set in both vectors (AND popcount), computed via the XOR popcount
/// identity: overlap = (popcnt(a) + popcnt(b) - popcnt(a ^ b)) / 2.
#[inline(always)]
pub fn pap_distance(a: &BinaryMultivector, b: &BinaryMultivector) -> f32 {
    let pop_a = (a.0[0].count_ones() + a.0[1].count_ones()) as u32;
    let pop_b = (b.0[0].count_ones() + b.0[1].count_ones()) as u32;

    // Both vectors are empty → identical
    if pop_a == 0 && pop_b == 0 {
        return 0.0_f32;
    }
    // One vector is empty, the other is not → orthogonal
    if pop_a == 0 || pop_b == 0 {
        return 1.0_f32;
    }

    let hamming = a.hamming_distance(b);
    let overlap = (pop_a + pop_b - hamming) / 2;
    let similarity = (overlap as f32) / ((pop_a as f32).sqrt() * (pop_b as f32).sqrt());
    1.0_f32 - similarity
}

/// Compute `pap_distance` for a batch of candidates.
#[inline(always)]
pub fn batch_pap_distance(query: &BinaryMultivector, candidates: &[BinaryMultivector]) -> Vec<f32> {
    candidates.iter().map(|c| pap_distance(query, c)).collect()
}

/// Compute PAP distance between two ternary multivectors using signed overlap.
#[inline(always)]
pub fn ternary_pap_distance(a: &TernaryMultivector, b: &TernaryMultivector) -> f32 {
    let mag_a = a.magnitude();
    let mag_b = b.magnitude();
    if mag_a == 0 && mag_b == 0 {
        return 0.0_f32;
    }
    if mag_a == 0 || mag_b == 0 {
        return 1.0_f32;
    }
    let overlap = (a.signed_overlap(b).abs()) as f32;
    let similarity = overlap / ((mag_a as f32).sqrt() * (mag_b as f32).sqrt());
    1.0_f32 - similarity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::multivector::BinaryMultivector;

    #[test]
    fn test_identical_distance_zero() {
        let a = BinaryMultivector([0b1100, 0b1010]);
        let b = BinaryMultivector([0b1100, 0b1010]);
        let d = pap_distance(&a, &b);
        assert!((d - 0.0).abs() < 1e-6, "expected 0.0, got {}", d);
    }

    #[test]
    fn test_orthogonal_distance_one() {
        let a = BinaryMultivector([0b1111, 0b0000]); // popcnt 4
        let b = BinaryMultivector([0b0000, 0b1111]); // popcnt 4, zero overlap
        let d = pap_distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-6, "expected 1.0, got {}", d);
    }

    #[test]
    fn test_known_overlap() {
        // a has bits 0,1,2 => popcnt 3
        let a = BinaryMultivector([0b111, 0]);
        // b has bits 1,2,3 => popcnt 3
        let b = BinaryMultivector([0b1110, 0]);
        let pop_a = 3u32;
        let pop_b = 3u32;
        let overlap = 2u32;
        let expected_sim = overlap as f32 / ((pop_a as f32).sqrt() * (pop_b as f32).sqrt());
        let expected_dist = 1.0 - expected_sim;
        let d = pap_distance(&a, &b);
        assert!((d - expected_dist).abs() < 1e-6, "expected {}, got {}", expected_dist, d);
    }

    #[test]
    fn test_batch() {
        let a = BinaryMultivector([0b111, 0]);
        let b = BinaryMultivector([0b111, 0]);
        let c = BinaryMultivector([0b000, 0b111]);
        let candidates = vec![a, b, c];
        let dists = batch_pap_distance(&a, &candidates);
        assert_eq!(dists.len(), 3);
        assert!((dists[0] - 0.0).abs() < 1e-6);
        assert!((dists[1] - 0.0).abs() < 1e-6);
        assert!((dists[2] - 1.0).abs() < 1e-6);
    }

    use crate::types::ternary::TernaryMultivector;

    #[test]
    fn test_ternary_pap_distance_zero() {
        let a = TernaryMultivector::new();
        let b = TernaryMultivector::new();
        let d = ternary_pap_distance(&a, &b);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_pap_distance_one() {
        let a = TernaryMultivector::new();
        let b = TernaryMultivector::from_binary(&BinaryMultivector([1, 0]));
        let d = ternary_pap_distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_pap_distance_identity() {
        let a = TernaryMultivector::from_binary(&BinaryMultivector([0b111, 0b101]));
        let d = ternary_pap_distance(&a, &a);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_pap_distance_orthogonal() {
        let a = TernaryMultivector::from_binary(&BinaryMultivector([0b111, 0]));
        let b = TernaryMultivector::from_binary(&BinaryMultivector([0b111000, 0]));
        let mag_a = a.magnitude();
        let mag_b = b.magnitude();
        let overlap = a.signed_overlap(&b).abs() as f32;
        let similarity = overlap / ((mag_a as f32).sqrt() * (mag_b as f32).sqrt());
        let expected = 1.0_f32 - similarity;
        let d = ternary_pap_distance(&a, &b);
        assert!((d - expected).abs() < 1e-6);
    }
}
