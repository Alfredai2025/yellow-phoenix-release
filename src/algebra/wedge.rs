// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Geometric binary wedge product and novelty metrics for `BinaryMultivector`.

use crate::types::multivector::BinaryMultivector;

/// Compute the wedge product metric between two binary multivectors.
///
/// Formula:
/// `((hamming_distance(a,b) - overlap_count(a,b)) / max_popcount(a,b))`
/// clamped to `[-1.0, 1.0]`.
///
/// * Positive → novel (more different than overlapping).
/// * Negative → redundant (more overlapping than different).
/// * Returns `0.0` if both vectors are identical or if the maximum popcount is zero.
#[inline]
pub fn wedge_product(a: &BinaryMultivector, b: &BinaryMultivector) -> f32 {
    // Identical vectors always yield 0.
    if a == b {
        return 0.0;
    }

    let max_pop = core::cmp::max(total_ones(a), total_ones(b));
    if max_pop == 0 {
        return 0.0;
    }

    let ham = a.hamming_distance(b) as f32;
    let overlap = ((a.0[0] & b.0[0]).count_ones() + (a.0[1] & b.0[1]).count_ones()) as f32;
    let numerator = ham - overlap;
    let raw = numerator / (max_pop as f32);

    if raw > 1.0 {
        1.0
    } else if raw < -1.0 {
        -1.0
    } else {
        raw
    }
}

/// Total number of set bits in a `BinaryMultivector`.
#[inline]
fn total_ones(v: &BinaryMultivector) -> u32 {
    v.0[0].count_ones() + v.0[1].count_ones()
}

/// Novelty score: `max(0.0, wedge_product(base, candidate))`.
#[inline]
pub fn novelty_score(base: &BinaryMultivector, candidate: &BinaryMultivector) -> f32 {
    let w = wedge_product(base, candidate);
    if w > 0.0 {
        w
    } else {
        0.0
    }
}

/// Return the index and novelty score of the candidate with the highest
/// `novelty_score` relative to `base`, or `None` if the candidate list is empty.
#[inline]
pub fn most_novel(
    base: &BinaryMultivector,
    candidates: &[BinaryMultivector],
) -> Option<(usize, f32)> {
    let mut best_idx = None;
    let mut best_score: f32 = 0.0;

    for (i, cand) in candidates.iter().enumerate() {
        let score = novelty_score(base, cand);
        if score > best_score {
            best_score = score;
            best_idx = Some(i);
        }
    }

    best_idx.map(|i| (i, best_score))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::multivector::BinaryMultivector;

    #[test]
    fn identical_vectors_return_zero() {
        let a = BinaryMultivector([0b1111, 0b0]);
        assert_eq!(wedge_product(&a, &a), 0.0);
    }

    #[test]
    fn disjoint_equal_size_vectors_return_one() {
        let a = BinaryMultivector([0b1010, 0b0]); // popcount 2
        let b = BinaryMultivector([0b0101, 0b0]); // popcount 2
        // hamming distance = 4, overlap = 0, max pop = 2 → raw = 2.0 → clamped 1.0
        assert_eq!(wedge_product(&a, &b), 1.0);
    }

    #[test]
    fn subset_returns_negative() {
        // a is a subset of b (bits of a are also set in b)
        let a = BinaryMultivector([0b0011, 0b0]); // popcount 2
        let b = BinaryMultivector([0b0111, 0b0]); // popcount 3
        let w = wedge_product(&a, &b);
        assert!(w < 0.0);
    }

    #[test]
    fn most_novel_picks_correct_candidate() {
        let base = BinaryMultivector([0b1111, 0b0]);

        // c1 is disjoint → novelty 1.0
        let c1 = BinaryMultivector([0b0000, 0b1111]);
        // c2 is identical → novelty 0.0
        let c2 = base;
        // c3 is a subset → novelty 0.0 (negative wedge)
        let c3 = BinaryMultivector([0b0011, 0b0]);

        let result = most_novel(&base, &[c1, c2, c3]);
        assert_eq!(result, Some((0, 1.0)));
    }
}
