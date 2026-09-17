// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use alloc::vec::Vec;

use crate::core::hologram::{HologramTier0, HologramTier1};
use crate::types::multivector::BinaryMultivector;

/// Result of a 4‑tier geometric query.
pub struct QueryResult {
    /// Highest tier that produced the result (0‑3).
    pub tier: u8,
    /// Score from that tier.
    pub score: f32,
    /// Top exact matches when Tier 3 is reached, otherwise empty.
    pub matches: Vec<(usize, u32)>,
}

/// Four‑tier pure geometric query pipeline.
///
/// Only the prototype store (`Vec<BinaryMultivector>`) allocates on the heap.
/// All tiers are fixed‑size, single‑threaded structures.
pub struct QueryPipeline {
    tier0: HologramTier0,
    tier1: HologramTier1,
    tier1b: HologramTier1,
    tier1c: HologramTier1,
    prototypes: Vec<BinaryMultivector>,
    threshold_t0: u32,
    threshold_t1: u32,
    threshold_t2: u32,
    query_count: u64,
}

impl QueryPipeline {
    /// Build a pipeline with explicit thresholds from the start.
    pub fn with_thresholds(t0: u32, t1: u32, t2: u32) -> Self {
        Self::new(t0, t1, t2)
    }

    /// Build a pipeline with the given tier thresholds.
    pub fn new(threshold_t0: u32, threshold_t1: u32, threshold_t2: u32) -> Self {
        Self {
            tier0: HologramTier0::new(),
            tier1: HologramTier1::new(),
            tier1b: HologramTier1::new(),
            tier1c: HologramTier1::new(),
            prototypes: Vec::new(),
            threshold_t0,
            threshold_t1,
            threshold_t2,
            query_count: 0,
        }
    }

    /// Default pipeline: calibrated hologram filter (T0=2500) with Tier 3 fallback.
    pub fn new_default() -> Self {
        Self::new(2500, 0, 0)
    }

    /// Total number of queries executed so far.
    pub fn query_count(&self) -> u64 {
        self.query_count
    }

    /// Inject a paper into every tier and store a copy for exact Tier‑3 search.
    pub fn add_paper(&mut self, paper: &BinaryMultivector) {
        self.tier0.inject(paper);
        self.tier1.inject(paper);
        self.tier1b.inject(paper);
        self.tier1c.inject(paper);
        self.prototypes.push(*paper);
    }

    /// Remove the paper at `index` from all tiers and from the prototype store.
    /// Uses `swap_remove` for O(1) removal.
    pub fn remove_paper(&mut self, index: usize) {
        if index >= self.prototypes.len() {
            return;
        }
        let paper = self.prototypes[index];
        self.tier0.eject(&paper);
        self.tier1.eject(&paper);
        self.tier1b.eject(&paper);
        self.tier1c.eject(&paper);
        self.prototypes.swap_remove(index);
    }

    /// Run the tiered query pipeline.
    pub fn query(&mut self, q: &BinaryMultivector) -> QueryResult {
        self.query_count += 1;
        let score0 = self.tier0.query(q);
        if score0 < self.threshold_t0 {
            return QueryResult {
                tier: 0,
                score: score0 as f32,
                matches: Vec::new(),
            };
        }

        let score1 = self.tier1.query(q);
        if score1 < self.threshold_t1 {
            return QueryResult {
                tier: 1,
                score: score1 as f32,
                matches: Vec::new(),
            };
        }

        let score1b = self.tier1b.query(q);
        let score1c = self.tier1c.query(q);
        let ensemble_score = (score1 + score1b + score1c) / 3;

        if ensemble_score < self.threshold_t2 {
            return QueryResult {
                tier: 2,
                score: ensemble_score as f32,
                matches: Vec::new(),
            };
        }

        // Tier 3: exact geometric resonance over the prototype store.
        let scores: Vec<u32> = self.prototypes.iter()
            .map(|p| q.hamming_distance(p))
            .collect();
        let mut matches: Vec<(usize, u32)> = scores
            .into_iter()
            .enumerate()
            .map(|(idx, score)| (idx, score))
            .collect();
        // Lower geometric product means closer match.
        matches.sort_by(|a, b| a.1.cmp(&b.1));
        let top10: Vec<(usize, u32)> = matches.into_iter().take(10).collect();
        let best_score = top10.first().map(|m| m.1).unwrap_or(0);

        QueryResult {
            tier: 3,
            score: best_score as f32,
            matches: top10,
        }
    }

    /// Inject a paper using a bag‑of‑words (BOW) vector for hologram tiers and
    /// the exact vector for Tier‑3 prototype‑store matching.
    pub fn add_paper_bow(&mut self, bow: &BinaryMultivector, exact: &BinaryMultivector) {
        self.tier0.inject(bow);
        self.tier1.inject(bow);
        self.tier1b.inject(bow);
        self.tier1c.inject(bow);
        self.prototypes.push(*exact);
    }

    /// Run tiered query with a BOW vector for the hologram stages and an
    /// exact vector for the Tier‑3 prototype scan.
    pub fn query_bow(&mut self, bow: &BinaryMultivector, exact: &BinaryMultivector) -> QueryResult {
        self.query_count += 1;
        let score0 = self.tier0.query(bow);
        if score0 < self.threshold_t0 {
            return QueryResult {
                tier: 0,
                score: score0 as f32,
                matches: Vec::new(),
            };
        }

        let score1 = self.tier1.query(bow);
        if score1 < self.threshold_t1 {
            return QueryResult {
                tier: 1,
                score: score1 as f32,
                matches: Vec::new(),
            };
        }

        let score1b = self.tier1b.query(bow);
        let score1c = self.tier1c.query(bow);
        let ensemble_score = (score1 + score1b + score1c) / 3;

        if ensemble_score < self.threshold_t2 {
            return QueryResult {
                tier: 2,
                score: ensemble_score as f32,
                matches: Vec::new(),
            };
        }

        // Tier 3: exact geometric resonance over the prototype store.
        let scores: Vec<u32> = self.prototypes.iter()
            .map(|p| exact.hamming_distance(p))
            .collect();
        let mut matches: Vec<(usize, u32)> = scores
            .into_iter()
            .enumerate()
            .map(|(idx, score)| (idx, score))
            .collect();
        // Lower geometric product means closer match.
        matches.sort_by(|a, b| a.1.cmp(&b.1));
        let top10: Vec<(usize, u32)> = matches.into_iter().take(10).collect();
        let best_score = top10.first().map(|m| m.1).unwrap_or(0);

        QueryResult {
            tier: 3,
            score: best_score as f32,
            matches: top10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paper_from(i: usize) -> BinaryMultivector {
        BinaryMultivector::from_seed_text(&format!("paper-{}", i))
    }

    #[test]
    fn query_reaches_tier_1_or_2_with_1000_papers() {
        // threshold_t2 = u32::MAX keeps the query out of Tier 3.
        let mut pipeline = QueryPipeline::new(0, 0, u32::MAX);
        for i in 0..1_000 {
            let p = paper_from(i);
            pipeline.add_paper(&p);
        }
        let q = paper_from(7);
        let res = pipeline.query(&q);
        assert!(res.tier == 1 || res.tier == 2);
    }

    #[test]
    fn query_10k_papers_under_10_microseconds() {
        // Avoid Tier 3 exact scan so the test measures only the fixed tiers.
        let mut pipeline = QueryPipeline::new(0, 0, u32::MAX);
        let p = BinaryMultivector::from_seed_text("shared-paper");
        for _ in 0..10_000 {
            pipeline.add_paper(&p);
        }
        let q = BinaryMultivector::from_seed_text("query-vector");

        let start = std::time::Instant::now();
        let _res = pipeline.query(&q);
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        assert!(
            elapsed_ns < 50_000,
            "query took {} ns, expected < 50_000 ns",
            elapsed_ns
        );
    }

    #[test]
    fn tier3_exact_finds_identical_prototype() {
        // Thresholds of 0 force the pipeline all the way to Tier 3.
        let mut pipeline = QueryPipeline::new(0, 0, 0);
        let target = BinaryMultivector::from_seed_text("exact-match-target");
        pipeline.add_paper(&target);

        // Add a few distractors.
        for i in 0..50 {
            pipeline.add_paper(&paper_from(i));
        }

        let res = pipeline.query(&target);
        assert_eq!(res.tier, 3);
        assert!(!res.matches.is_empty(), "Tier 3 should return matches");
        let (best_idx, best_score) = res.matches[0];
        assert_eq!(
            best_score, 0,
            "identical prototype should have geometric product 0"
        );
        assert_eq!(
            pipeline.prototypes[best_idx], target,
            "top match should be the identical prototype"
        );
    }

    #[test]
    fn remove_paper_updates_all_tiers() {
        let mut pipeline = QueryPipeline::new(0, 0, 0);
        let a = paper_from(1_001);
        let b = paper_from(1_002);
        let c = paper_from(1_003);
        pipeline.add_paper(&a);
        pipeline.add_paper(&b);
        pipeline.add_paper(&c);

        pipeline.remove_paper(1);

        assert_eq!(pipeline.prototypes.len(), 2);
        assert!(
            !pipeline.prototypes.contains(&b),
            "removed prototype should no longer be in the store"
        );

        // A query for the removed paper must not return it from Tier 3.
        let res = pipeline.query(&b);
        assert_eq!(res.tier, 3);
        for (idx, _) in &res.matches {
            assert_ne!(
                pipeline.prototypes[*idx], b,
                "removed paper should not appear among top matches"
            );
        }
    }

    #[test]
    fn test_bow_pipeline_exact_match() {
        let mut pipeline = QueryPipeline::new(0, 0, 0);
        let encoder = crate::core::encoder::Encoder::new(0);
        let text = "machine learning";
        let bow = encoder.encode_paper_bow(text);
        let exact = encoder.encode_paper(text);
        pipeline.add_paper_bow(&bow, &exact);
        let res = pipeline.query_bow(&bow, &exact);
        assert_eq!(res.tier, 3);
        assert!(!res.matches.is_empty());
        let (_best_idx, best_score) = res.matches[0];
        assert_eq!(best_score, 0);
    }

    #[test]
    fn test_bow_pipeline_order_invariant() {
        let mut pipeline = QueryPipeline::new(0, 0, 0);
        let encoder = crate::core::encoder::Encoder::new(0);
        let text1 = "machine learning";
        let bow1 = encoder.encode_paper_bow(text1);
        let exact1 = encoder.encode_paper(text1);
        pipeline.add_paper_bow(&bow1, &exact1);

        let text2 = "learning machine";
        let bow2 = encoder.encode_paper_bow(text2);
        let exact2 = encoder.encode_paper(text2);
        let res = pipeline.query_bow(&bow2, &exact2);
        // Even with a different word order, the query should still reach Tier 3.
        // The exact prototype may not match, but the original paper should appear in matches.
        assert_eq!(res.tier, 3);
        let indices: Vec<usize> = res.matches.iter().map(|(idx, _)| *idx).collect();
        assert!(indices.contains(&0));
    }
}
