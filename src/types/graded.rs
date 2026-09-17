// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use crate::distance::pap_distance;
use crate::types::multivector::BinaryMultivector;

// Grade masks as defined in the geometric calculus
pub const GRADE0_MASK: [u64; 2] = [0x000000000000FFFF, 0];
pub const GRADE1_MASK: [u64; 2] = [0xFFFFFFFFFFFF0000, 0x00000000000FFFFF];
pub const GRADE2_MASK: [u64; 2] = [0, 0xFFFFFFFFFFF00000];

/// A binary multivector with explicit grade projection.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GradedMultivector(pub BinaryMultivector);

impl GradedMultivector {
    /// Create a new `GradedMultivector` wrapping the given underlying multivector.
    #[inline]
    pub fn new(mv: BinaryMultivector) -> Self {
        Self(mv)
    }

    /// Project the multivector onto the specified grade (0, 1, or 2).
    #[inline]
    pub fn grade_project(&self, grade: u8) -> BinaryMultivector {
        let raw = self.0;
        match grade {
            0 => BinaryMultivector([raw.0[0] & GRADE0_MASK[0], raw.0[1] & GRADE0_MASK[1]]),
            1 => BinaryMultivector([raw.0[0] & GRADE1_MASK[0], raw.0[1] & GRADE1_MASK[1]]),
            2 => BinaryMultivector([raw.0[0] & GRADE2_MASK[0], raw.0[1] & GRADE2_MASK[1]]),
            _ => panic!("grade (0/1/2) expected, got {}", grade),
        }
    }

    /// Return the population count (number of set bits) in the projected grade.
    #[inline]
    pub fn grade_popcount(&self, grade: u8) -> u32 {
        let proj = self.grade_project(grade);
        let limbs = &proj.0;
        (limbs[0].count_ones() + limbs[1].count_ones()) as u32
    }

    /// Weighted PAP distance between two graded multivectors.
    ///
    /// The distance is computed as
    ///   (d0 + d1 + 2 * d2) / 4.0
    /// where dk is the PAP distance restricted to grade k.
    #[inline]
    pub fn grade_weighted_distance(&self, other: &Self) -> f32 {
        let p0_a = self.grade_project(0);
        let p0_b = other.grade_project(0);
        let d0 = pap_distance(&p0_a, &p0_b);

        let p1_a = self.grade_project(1);
        let p1_b = other.grade_project(1);
        let d1 = pap_distance(&p1_a, &p1_b);

        let p2_a = self.grade_project(2);
        let p2_b = other.grade_project(2);
        let d2 = pap_distance(&p2_a, &p2_b);

        (d0 + d1 + 2.0 * d2) / 4.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to generate a full test multivector
    fn full() -> BinaryMultivector {
        // All 128 bits set
        BinaryMultivector([0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF])
    }

    #[test]
    fn grade0_project_isolates_low_16_bits() {
        let mv = GradedMultivector(full());
        let p = mv.grade_project(0);
        assert_eq!(p.0[0], 0x000000000000FFFF);
        assert_eq!(p.0[1], 0);
    }

    #[test]
    fn grade1_project_isolates_correct_bits() {
        let mv = GradedMultivector(full());
        let p = mv.grade_project(1);
        assert_eq!(p.0[0], 0xFFFFFFFFFFFF0000);
        assert_eq!(p.0[1], 0x00000000000FFFFF);
    }

    #[test]
    fn grade2_project_isolates_correct_bits() {
        let mv = GradedMultivector(full());
        let p = mv.grade_project(2);
        assert_eq!(p.0[0], 0);
        assert_eq!(p.0[1], 0xFFFFFFFFFFF00000);
    }

    #[test]
    #[should_panic]
    fn grade_project_panics_on_unknown_grade() {
        let mv = GradedMultivector(full());
        mv.grade_project(3);
    }

    #[test]
    fn grade_popcount_counts_each_grade() {
        let mv = GradedMultivector(full());
        assert_eq!(mv.grade_popcount(0), 16);
        // grade1: 48 bits in limb 0 + 20 bits in limb 1 = 68
        assert_eq!(mv.grade_popcount(1), 68);
        // grade2: high bits in limb 1 only, count set bits inside mask.
        let mask = GRADE2_MASK[1];
        let bits2 = mask.count_ones();
        assert_eq!(mv.grade_popcount(2), bits2);
    }

    #[test]
    fn identical_multivectors_yield_zero_distance() {
        let mv = GradedMultivector(full());
        let d = mv.grade_weighted_distance(&mv);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn grade2_only_difference_penalized_twice() {
        // Create two graded multivectors that differ only in the grade‑2 subspace.
        // Use a single bit inside the grade‑2 mask for a, zero for b.
        let limb_a = 1u64 << 32; // bit 96 overall, inside GRADE2_MASK
        let mv_a = GradedMultivector(BinaryMultivector([0, limb_a]));
        let mv_b = GradedMultivector(BinaryMultivector([0, 0]));

        // d0 and d1 are 0 (projections identical, zero population)
        // d2 = 1.0 because one population non‑zero, the other zero
        // expected = (0 + 0 + 2*1)/4 = 0.5
        let expected = 0.5;
        let d = mv_a.grade_weighted_distance(&mv_b);
        assert!((d - expected).abs() < 1e-6);
    }

    #[test]
    fn complete_coverage_bits() {
        // Check that summing grade popcounts equals total popcount of underlying multivector.
        let raw = BinaryMultivector([0xAAAA_5555_AAAA_5555, 0x5555_AAAA_5555_AAAA]);
        let mv = GradedMultivector(raw);
        let pop0 = mv.grade_popcount(0);
        let pop1 = mv.grade_popcount(1);
        let pop2 = mv.grade_popcount(2);
        let pop_total = (raw.0[0].count_ones() + raw.0[1].count_ones()) as u32;
        assert_eq!(pop0 + pop1 + pop2, pop_total);
    }
}
