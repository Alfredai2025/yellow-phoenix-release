// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use crate::types::multivector::BinaryMultivector;

/// Bitwise AND across all papers (shared bits).
/// Returns an empty multivector if the input slice is empty.
pub fn meet(papers: &[BinaryMultivector]) -> BinaryMultivector {
    if papers.is_empty() {
        return BinaryMultivector::new();
    }
    let mut a = papers[0].0[0];
    let mut b = papers[0].0[1];
    for p in &papers[1..] {
        a &= p.0[0];
        b &= p.0[1];
    }
    BinaryMultivector([a, b])
}

/// Bitwise OR across all papers (combined bits).
/// Returns an empty multivector if the input slice is empty.
pub fn join(papers: &[BinaryMultivector]) -> BinaryMultivector {
    if papers.is_empty() {
        return BinaryMultivector::new();
    }
    let mut a = papers[0].0[0];
    let mut b = papers[0].0[1];
    for p in &papers[1..] {
        a |= p.0[0];
        b |= p.0[1];
    }
    BinaryMultivector([a, b])
}

fn popcount(mv: &BinaryMultivector) -> u32 {
    (mv.0[0].count_ones() + mv.0[1].count_ones()) as u32
}

/// Ratio of shared bits to the smallest paper (meet_popcount / min_popcount).
/// Returns 1.0 when there is perfect agreement, 0.0 when there is no overlap.
/// If the smallest popcount is zero, 1.0 is returned.
pub fn consensus_strength(papers: &[BinaryMultivector]) -> f32 {
    if papers.is_empty() {
        return 1.0;
    }
    let meet_mv = meet(papers);
    let meet_pop = popcount(&meet_mv);

    let min_pop = papers.iter().map(|p| popcount(p)).min().unwrap_or(0);
    if min_pop == 0 {
        // No bits in the smallest paper → treat as perfect agreement.
        return 1.0;
    }
    meet_pop as f32 / min_pop as f32
}

/// Ratio of combined bits to the largest paper (join_popcount / max_popcount).
/// Returns 1.0 when no new information is added; >1.0 indicates breadth increase.
/// If the largest popcount is zero, 1.0 is returned.
pub fn coverage_breadth(papers: &[BinaryMultivector]) -> f32 {
    if papers.is_empty() {
        return 1.0;
    }
    let join_mv = join(papers);
    let join_pop = popcount(&join_mv);

    let max_pop = papers.iter().map(|p| popcount(p)).max().unwrap_or(1);
    if max_pop == 0 {
        return 1.0;
    }
    join_pop as f32 / max_pop as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::multivector::BinaryMultivector;

    fn popc(mv: &BinaryMultivector) -> u32 {
        mv.hamming_distance(&BinaryMultivector::new())
    }

    #[test]
    fn test_identical_papers() {
        let mv = BinaryMultivector([0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0]);
        let papers = [mv, mv, mv];
        let m = meet(&papers);
        assert_eq!(m, mv, "meet should equal input for identical papers");
        let j = join(&papers);
        assert_eq!(j, mv, "join should equal input for identical papers");

        let cs = consensus_strength(&papers);
        assert!((cs - 1.0).abs() < 1e-7, "consensus should be 1.0 for identical papers");

        let cb = coverage_breadth(&papers);
        assert!((cb - 1.0).abs() < 1e-7, "coverage should be 1.0 for identical papers");
    }

    #[test]
    fn test_disjoint_papers() {
        let p0 = BinaryMultivector([0x1, 0]); // pop 1
        let p1 = BinaryMultivector([0x2, 0]); // pop 1
        let p2 = BinaryMultivector([0x4, 0]); // pop 1
        let papers = [p0, p1, p2];

        let m = meet(&papers);
        assert_eq!(m, BinaryMultivector::new(), "meet should be empty for disjoint papers");

        let j = join(&papers);
        assert_eq!(popc(&j), 3, "join should contain three bits for three disjoint papers");

        let cs = consensus_strength(&papers);
        assert!((cs - 0.0).abs() < 1e-7, "consensus should be 0.0 for disjoint papers");

        let cb = coverage_breadth(&papers);
        // each paper has popcount 1, combined 3
        assert!((cb - 3.0).abs() < 1e-7, "coverage should be 3.0 for three disjoint papers of equal size");
    }

    #[test]
    fn test_overlapping_papers() {
        let a = BinaryMultivector([0x55, 0]); // pop 4
        let b = BinaryMultivector([0xCC, 0]); // pop 4
        let papers = [a, b];

        let m = meet(&papers);
        let expected_meet = BinaryMultivector([0x44, 0]); // pop 2
        assert_eq!(m, expected_meet, "meet should have 2 common bits");

        let j = join(&papers);
        let expected_join = BinaryMultivector([0xDD, 0]); // pop 6
        assert_eq!(j, expected_join, "join should have 6 combined bits");

        let cs = consensus_strength(&papers);
        assert!((cs - 0.5).abs() < 1e-7, "consensus should be 0.5 for the overlapping pair");

        let cb = coverage_breadth(&papers);
        assert!((cb - 1.5).abs() < 1e-7, "coverage should be 1.5 for the overlapping pair");
    }
}
