// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Multi-Base Confidence Layer — Yellow Phoenix v3.6 Phase 1
//!
//! Provides two fast confidence signals:
//! - `vote_bucket`: congruence majority vote on a bucket ID (spec hook).
//! - `agreement`: real spatial agreement between two 64-byte PAPs across bases.

const BASES: [u64; 4] = [3, 4, 6, 8];

/// Compute confidence for a bucket ID via multi-base majority vote.
/// Returns confidence in [0.0, 1.0].
pub fn vote_bucket(bucket: u32) -> f32 {
    let b = bucket as u64;
    let mut residues = [0u64; 4];
    for (i, base) in BASES.iter().enumerate() {
        residues[i] = b % base;
    }

    let mut best_count = 0;
    for &r in &residues {
        let count = residues.iter().filter(|&&x| x == r).count();
        if count > best_count {
            best_count = count;
        }
    }

    best_count as f32 / BASES.len() as f32
}

/// Real multi-base spatial agreement between two 64-byte PAPs.
/// Counts how many of the configured bases place the two PAPs in
/// adjacent coordinates (circular distance <= 1).
pub fn agreement(pap_a: &[u8; 64], pap_b: &[u8; 64]) -> f32 {
    let mut matches = 0;
    for &base in &BASES {
        let coord_a = compute_coord(pap_a, base);
        let coord_b = compute_coord(pap_b, base);
        let dist = circular_dist(coord_a, coord_b, base);
        if dist <= 1 {
            matches += 1;
        }
    }
    matches as f32 / BASES.len() as f32
}

fn compute_coord(pap: &[u8; 64], base: u64) -> u64 {
    let mut sum: u64 = 0;
    for i in 0..8 {
        sum = sum.wrapping_mul(base).wrapping_add(pap[i] as u64);
    }
    sum % base
}

fn circular_dist(a: u64, b: u64, base: u64) -> u64 {
    let d = if a > b { a - b } else { b - a };
    if d > base / 2 {
        base - d
    } else {
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vote_bucket_zero() {
        assert!((vote_bucket(0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_vote_bucket_lcm() {
        assert!((vote_bucket(24) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_vote_bucket_one() {
        assert!((vote_bucket(1) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_vote_bucket_mixed() {
        let c = vote_bucket(5);
        assert!(c >= 0.25 && c <= 1.0);
    }

    #[test]
    fn test_agreement_identical() {
        let pap = [0u8; 64];
        assert!((agreement(&pap, &pap) - 1.0).abs() < 0.01);
    }
}
