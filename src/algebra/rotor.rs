// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Algebraic rotor utilities for geometric binary multivectors.
//!
//! Provides geodesic distance, bridge scoring, and bridge discovery.

use crate::types::multivector::BinaryMultivector;
use crate::distance::pap_distance;

/// Hamming distance (XOR popcount) between two binary multivectors.
#[inline]
pub fn geodesic_distance(a: &BinaryMultivector, b: &BinaryMultivector) -> u32 {
    a.hamming_distance(b)
}

/// Bridge score designed to favour a candidate `c` that is close to both `a` and `b`
/// and equally distant from them.
///
/// Lower values are better.
/// `lambda` controls the penalty for imbalance; a typical helpful value is **‑1.0**.
#[inline]
pub fn bridge_score(
    a: &BinaryMultivector,
    b: &BinaryMultivector,
    c: &BinaryMultivector,
    lambda: f32,
) -> f32 {
    let d_ac = pap_distance(a, c);
    let d_cb = pap_distance(c, b);
    let min_val = d_ac.min(d_cb);
    let diff = (d_ac - d_cb).abs();
    min_val - lambda * diff
}

/// Find the best bridge candidate among `candidates`.
///
/// Returns `Some((index, score))` where `score` is the lowest `bridge_score` obtained
/// with a fixed `lambda = -1.0`. Returns `None` if the slice is empty.
#[inline]
pub fn find_bridge(
    a: &BinaryMultivector,
    b: &BinaryMultivector,
    candidates: &[BinaryMultivector],
) -> Option<(usize, f32)> {
    if candidates.is_empty() {
        return None;
    }
    let lambda = -1.0_f32;
    let mut best_idx = 0;
    let mut best_score = f32::MAX;
    for (i, c) in candidates.iter().enumerate() {
        let score = bridge_score(a, b, c, lambda);
        if score < best_score {
            best_score = score;
            best_idx = i;
        }
    }
    Some((best_idx, best_score))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::multivector::BinaryMultivector;
    use crate::distance::pap_distance;

    #[test]
    fn identical_a_b_best_closest() {
        let a = BinaryMultivector([0b1111, 0]);
        let b = a;
        let c0 = BinaryMultivector([0b1111, 0]); // distance 0
        let c1 = BinaryMultivector([0b0000, 0b1111]); // distance 1
        let candidates = vec![c0, c1];
        let result = find_bridge(&a, &b, &candidates);
        assert!(result.is_some());
        let (idx, score) = result.unwrap();
        assert_eq!(idx, 0);
        let expected = 0.0_f32;
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn three_candidates_middle_bridge() {
        // a has bits 0..3 set, b has bits 4..7 set, no overlap
        let a = BinaryMultivector([0b1111, 0]);
        let b = BinaryMultivector([0b1111_0000, 0]);
        // c0 = identical to a, c1 = identical to b, c2 = middle (bits 2..5)
        let c0 = BinaryMultivector([0b1111, 0]);
        let c1 = BinaryMultivector([0b1111_0000, 0]);
        let c2 = BinaryMultivector([0b0011_1100, 0]); // bits 2..5
        let candidates = vec![c0, c1, c2];
        let result = find_bridge(&a, &b, &candidates);
        assert!(result.is_some());
        let (idx, score) = result.unwrap();
        assert_eq!(idx, 2);
        // expected score for lambda = -1.0: min(0.5,0.5) + |0.5-0.5| = 0.5
        let expected = pap_distance(&a, &c2);
        assert!((score - expected).abs() < 1e-6);
    }
}
