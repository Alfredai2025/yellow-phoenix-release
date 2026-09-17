// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Hodge dual and algebraic gap detection for binary multivectors.

use crate::types::multivector::BinaryMultivector;

/// Bitwise complement (Hodge dual) of a binary multivector.
#[inline]
pub fn hodge_dual(a: &BinaryMultivector) -> BinaryMultivector {
    BinaryMultivector([!a.0[0], !a.0[1]])
}

/// Gap score between a query and a candidate.
///
/// A high score indicates the candidate lies far from the query
/// while also differing from the query's complement (i.e., not noise).
/// The score is defined as the product of the two PAP distances:
///
///   d_PAP(query, candidate) * d_PAP(hodge_dual(query), candidate)
///
/// Multiplying ensures that candidates which match the query exactly
/// (distance 0) or its complement exactly (distance 0) receive score 0.
#[inline]
pub fn gap_score(query: &BinaryMultivector, candidate: &BinaryMultivector) -> f32 {
    use crate::distance::pap_distance;

    let d = pap_distance(query, candidate);
    let d_dual = pap_distance(&hodge_dual(query), candidate);
    d * d_dual
}

/// Find the candidate with the largest gap score.
///
/// Returns `None` if the slice is empty; otherwise returns the index
/// and the score of the best candidate.
pub fn find_gap(query: &BinaryMultivector, candidates: &[BinaryMultivector]) -> Option<(usize, f32)> {
    let mut best_score = f32::NEG_INFINITY;
    let mut best_idx = None;

    for (i, cand) in candidates.iter().enumerate() {
        let score = gap_score(query, cand);
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
    fn identical_query_candidate_scores_zero() {
        let q = BinaryMultivector::from_seed_text("same");
        let s = gap_score(&q, &q);
        assert!((s - 0.0).abs() < 1e-6, "identical should score 0, got {s}");
    }

    #[test]
    fn exact_complement_scores_zero() {
        let q = BinaryMultivector::from_seed_text("query");
        let comp = hodge_dual(&q);
        let s = gap_score(&q, &comp);
        assert!((s - 0.0).abs() < 1e-6, "complement should score 0, got {s}");
    }

    #[test]
    fn different_valid_region_scores_positive() {
        let q = BinaryMultivector::from_seed_text("hello");
        let cand = BinaryMultivector::from_seed_text("world");
        let s = gap_score(&q, &cand);
        assert!(s > 0.0, "expected positive gap score, got {s}");
        assert!(s <= 1.0, "gap score should be ≤ 1, got {s}");
    }

    #[test]
    fn find_gap_empty_returns_none() {
        let q = BinaryMultivector::from_seed_text("q");
        assert!(find_gap(&q, &[]).is_none());
    }

    #[test]
    fn find_gap_picks_correct_candidate() {
        let q = BinaryMultivector::from_seed_text("query");
        let identical = q;
        let complement = hodge_dual(&q);
        let different = BinaryMultivector::from_seed_text("other");
        let candidates = [identical, complement, different];
        let res = find_gap(&q, &candidates);
        assert!(res.is_some(), "expected some result");
        let (idx, score) = res.unwrap();

        let s0 = gap_score(&q, &candidates[0]);
        let s1 = gap_score(&q, &candidates[1]);
        let s2 = gap_score(&q, &candidates[2]);
        let expected_score = s0.max(s1).max(s2);
        assert!((score - expected_score).abs() < 1e-6);

        assert!(s2 > s0 && s2 > s1, "different region should have higher score");
        assert_eq!(idx, 2);
    }
}
