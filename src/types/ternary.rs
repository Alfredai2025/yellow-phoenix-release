// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Ternary multivector (±1) representation for signed geometric operations.

use crate::types::multivector::BinaryMultivector;

/// Ternary multivector storing +1 bits and -1 bits separately.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TernaryMultivector {
    pub pos: [u64; 2],
    pub neg: [u64; 2],
}

impl TernaryMultivector {
    /// Create an empty ternary multivector (all bits 0).
    #[inline]
    pub fn new() -> Self {
        Self {
            pos: [0; 2],
            neg: [0; 2],
        }
    }

    /// Convert a binary multivector into a ternary one where every set bit becomes +1.
    #[inline]
    pub fn from_binary(b: &BinaryMultivector) -> Self {
        Self {
            pos: b.0,
            neg: [0; 2],
        }
    }

    /// Superpose two ternary vectors with clamping to {-1,0,+1}.
    ///
    /// Cancellation: +1 + -1 = 0.
    /// Reinforcement: +1 + +1 = +1, -1 + -1 = -1.
    /// Zero reinforcement: 0 + anything = anything.
    #[inline]
    pub fn superpose(&self, other: &Self) -> Self {
        let pos_a = self.pos;
        let neg_a = self.neg;
        let pos_b = other.pos;
        let neg_b = other.neg;

        let new_pos = [
            (pos_a[0] | pos_b[0]) & !(neg_a[0] | neg_b[0]),
            (pos_a[1] | pos_b[1]) & !(neg_a[1] | neg_b[1]),
        ];
        let new_neg = [
            (neg_a[0] | neg_b[0]) & !(pos_a[0] | pos_b[0]),
            (neg_a[1] | neg_b[1]) & !(pos_a[1] | pos_b[1]),
        ];

        TernaryMultivector {
            pos: new_pos,
            neg: new_neg,
        }
    }

    /// Signed overlap: sum over bits of +1 when both are +1, +1 when both are -1,
    /// -1 when one is +1 and the other -1, and 0 otherwise.
    #[inline]
    pub fn signed_overlap(&self, other: &Self) -> i32 {
        let mut sum: i32 = 0;
        for i in 0..2 {
            let pp = (self.pos[i] & other.pos[i]).count_ones() as i32;
            let nn = (self.neg[i] & other.neg[i]).count_ones() as i32;
            let pn = (self.pos[i] & other.neg[i]).count_ones() as i32;
            let np = (self.neg[i] & other.pos[i]).count_ones() as i32;
            sum += pp + nn - pn - np;
        }
        sum
    }

    /// Total number of +1 and -1 bits (magnitude).
    #[inline]
    pub fn magnitude(&self) -> u32 {
        let mut mag = 0u32;
        for i in 0..2 {
            mag += self.pos[i].count_ones() + self.neg[i].count_ones();
        }
        mag
    }

    /// Convert back to a binary multivector, discarding the sign.
    #[inline]
    pub fn to_binary(&self) -> BinaryMultivector {
        BinaryMultivector([self.pos[0] | self.neg[0], self.pos[1] | self.neg[1]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::multivector::BinaryMultivector;

    #[test]
    fn test_new_is_empty() {
        let t = TernaryMultivector::new();
        assert_eq!(t.pos, [0; 2]);
        assert_eq!(t.neg, [0; 2]);
        assert_eq!(t.magnitude(), 0);
    }

    #[test]
    fn test_from_binary_sets_positive_only() {
        let b = BinaryMultivector([0b101, 0b110]);
        let t = TernaryMultivector::from_binary(&b);
        assert_eq!(t.pos, b.0);
        assert_eq!(t.neg, [0; 2]);
        assert_eq!(t.magnitude(), b.0[0].count_ones() + b.0[1].count_ones());
    }

    #[test]
    fn test_superpose_identity() {
        let b = BinaryMultivector([0b111, 0b101]);
        let t = TernaryMultivector::from_binary(&b);
        let s = t.superpose(&t);
        // pos unchanged, neg zero
        assert_eq!(s.pos, t.pos);
        assert_eq!(s.neg, [0; 2]);
    }

    #[test]
    fn test_superpose_cancellation() {
        let b1 = BinaryMultivector([0b111, 0]);
        let b2 = BinaryMultivector([0b111, 0]);
        let pos = TernaryMultivector::from_binary(&b1);
        let mut neg = TernaryMultivector::new();
        neg.neg = b2.0; // all bits set to -1
        let s = pos.superpose(&neg);
        // all bits cancel
        assert_eq!(s.pos, [0; 2]);
        assert_eq!(s.neg, [0; 2]);
    }

    #[test]
    fn test_signed_overlap_identical() {
        let b = BinaryMultivector([0b111, 0b101]);
        let t = TernaryMultivector::from_binary(&b);
        let overlap = t.signed_overlap(&t);
        let expected = (t.pos[0].count_ones() + t.pos[1].count_ones()) as i32;
        assert_eq!(overlap, expected);
    }

    #[test]
    fn test_signed_overlap_opposite() {
        let b = BinaryMultivector([0b111, 0]);
        let pos = TernaryMultivector::from_binary(&b);
        let mut neg = TernaryMultivector::new();
        neg.neg = b.0;
        let overlap = pos.signed_overlap(&neg);
        let expected = -((b.0[0].count_ones() + b.0[1].count_ones()) as i32);
        assert_eq!(overlap, expected);
    }

    #[test]
    fn test_to_binary_discards_sign() {
        let b = BinaryMultivector([0b111, 0b101]);
        let mut t = TernaryMultivector::from_binary(&b);
        t.neg[0] = 0b001; // adds a bit via -1
        let bin = t.to_binary();
        assert_eq!(bin.0[0], t.pos[0] | t.neg[0]);
        assert_eq!(bin.0[1], t.pos[1] | t.neg[1]);
    }
}
