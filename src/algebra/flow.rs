// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Signed and unsigned geometric flow primitives for binary and ternary multivectors.

use core::cmp::min;

use crate::types::multivector::BinaryMultivector;
use crate::types::ternary::TernaryMultivector;

/// Returns the bitwise XOR of `a` and `b` — the bits that changed between them.
#[inline]
pub fn velocity_binary(a: &BinaryMultivector, b: &BinaryMultivector) -> BinaryMultivector {
    BinaryMultivector([a.0[0] ^ b.0[0], a.0[1] ^ b.0[1]])
}

/// Predicts the state after moving from `a` towards `b` by a fraction `t` of the differing bits.
///
/// The number of bits flipped is `min(round(t * 128), total_diff)`.
/// Bits are considered in order chunk‑0, chunk‑1, low‑to‑high within each chunk.
/// `t = 0` returns `a`, `t = 1` returns `b`.
pub fn predict_binary(a: &BinaryMultivector, b: &BinaryMultivector, t: f32) -> BinaryMultivector {
    let vel = velocity_binary(a, b);
    let total_diff = (vel.0[0].count_ones() + vel.0[1].count_ones()) as usize;
    let t_clamped = t.clamp(0.0, 1.0);
    let max_flip = (t_clamped * 128.0_f32).round() as usize;
    let n_flip = min(max_flip, total_diff);

    let mut result = *a;
    let mut remaining = n_flip;

    'outer: for chunk in 0..2 {
        let v = vel.0[chunk];
        for bit in 0..64 {
            if (v >> bit) & 1 == 1 {
                if remaining == 0 {
                    break 'outer;
                }
                result.0[chunk] ^= 1 << bit;
                remaining -= 1;
            }
        }
    }

    result
}

/// Signed difference between two ternary multivectors.
///
/// The resulting `pos` bits correspond to transitions that end in `+1`
/// (`-1 → +1` or `0 → +1`), and the `neg` bits correspond to transitions
/// that end in `-1` (`+1 → -1` or `0 → -1`).
#[inline]
pub fn velocity_ternary(a: &TernaryMultivector, b: &TernaryMultivector) -> TernaryMultivector {
    let pos = [
        b.pos[0] & !a.pos[0] & !b.neg[0],
        b.pos[1] & !a.pos[1] & !b.neg[1],
    ];
    let neg = [
        b.neg[0] & !a.neg[0] & !b.pos[0],
        b.neg[1] & !a.neg[1] & !b.pos[1],
    ];
    TernaryMultivector { pos, neg }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::multivector::BinaryMultivector;
    use crate::types::ternary::TernaryMultivector;

    #[test]
    fn test_predict_binary_t_zero_returns_a() {
        let a = BinaryMultivector([0b101, 0b110]);
        let b = BinaryMultivector([0b111, 0b001]);
        let p = predict_binary(&a, &b, 0.0);
        assert_eq!(p, a);
    }

    #[test]
    fn test_predict_binary_t_one_returns_b() {
        let a = BinaryMultivector([0b101, 0b110]);
        let b = BinaryMultivector([0b111, 0b001]);
        let p = predict_binary(&a, &b, 1.0);
        assert_eq!(p, b);
    }

    #[test]
    fn test_predict_binary_t_half_flips_half_differing() {
        let a = BinaryMultivector([0, 0]);
        let b = BinaryMultivector([u64::MAX, u64::MAX]);
        let p = predict_binary(&a, &b, 0.5);
        // t=0.5 => n_flip = 64, first 64 bits flipped
        assert_eq!(p.0[0], u64::MAX);
        assert_eq!(p.0[1], 0);
        let ones = p.0[0].count_ones() + p.0[1].count_ones();
        assert_eq!(ones, 64);
    }

    #[test]
    fn test_velocity_ternary_known_transitions() {
        let mut a = TernaryMultivector::new();
        a.pos[0] = 0b001; // bit0 +1
        a.neg[0] = 0b010; // bit1 -1

        let mut b = TernaryMultivector::new();
        b.pos[0] = 0b100; // bit2 +1
        b.neg[0] = 0b001; // bit0 -1

        let v = velocity_ternary(&a, &b);
        // pos: bit2 was 0 → +1
        assert_eq!(v.pos[0], 0b100);
        // neg: bit0 was +1 → -1
        assert_eq!(v.neg[0], 0b001);
        assert_eq!(v.pos[1], 0);
        assert_eq!(v.neg[1], 0);
    }

    #[test]
    fn test_velocity_ternary_no_transition_to_zero() {
        let mut a = TernaryMultivector::new();
        a.pos[0] = 0b001; // bit0 +1

        let b = TernaryMultivector::new(); // all zeros

        let v = velocity_ternary(&a, &b);
        // bit0 goes from +1 to 0 → not a transition to -1 or +1
        assert_eq!(v.pos[0], 0);
        assert_eq!(v.neg[0], 0);
    }
}
