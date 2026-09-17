// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! intent_classifier.rs — M4.3 rule-based query intent classifier.
//!
//! Routes queries into one of three buckets based on how confident the hash
//! stage is and how crowded the target bucket is.

/// Query intent used to choose the execution path in the collaborative engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent {
    /// A single exact-match candidate: run hash only and return immediately.
    ExactLookup,
    /// A small, confident candidate set: hash + spectral re-rank is enough.
    SemanticSearch,
    /// Low confidence or crowded bucket: use the full geometric index.
    Exploration,
}

/// Tiny rule-based classifier. No ML, no allocations.
pub struct IntentClassifier;

impl IntentClassifier {
    pub fn new() -> Self {
        Self
    }

    /// Classify a query based on hash-stage confidence and coarse bucket size.
    ///
    /// Thresholds:
    /// - 0.98: skip all later stages when the top candidate dominates
    ///   (confidence) and the bucket contains exactly one record.
    /// - 0.85: skip wedge/hologram when the bucket is small (<5).
    /// - otherwise: run the full feature chain.
    pub fn classify(&self, hash_confidence: f32, bucket_size: usize) -> Intent {
        if hash_confidence >= 0.98 && bucket_size == 1 {
            Intent::ExactLookup
        } else if hash_confidence >= 0.85 && bucket_size < 5 {
            Intent::SemanticSearch
        } else {
            Intent::Exploration
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_lookup_when_confident_and_single_bucket() {
        let c = IntentClassifier::new();
        assert_eq!(c.classify(0.99, 1), Intent::ExactLookup);
    }

    #[test]
    fn semantic_search_when_confident_and_small_bucket() {
        let c = IntentClassifier::new();
        assert_eq!(c.classify(0.90, 3), Intent::SemanticSearch);
    }

    #[test]
    fn exploration_when_uncertain_or_crowded() {
        let c = IntentClassifier::new();
        assert_eq!(c.classify(0.50, 1), Intent::Exploration);
        assert_eq!(c.classify(0.90, 10), Intent::Exploration);
    }
}
