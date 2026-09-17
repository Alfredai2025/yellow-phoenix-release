// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — Exam Boundary Monitor (EXM)
//!
//! EXM guards the held-out exam set. Any content whose fingerprint matches the
//! exam index is blocked at ingestion time. The index is intended to be loaded
//! from 106,298 pre-computed exam fingerprints.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// Fingerprint type for the exam boundary.
pub type ExamFingerprint = u64;

/// Exam boundary monitor.
pub struct ExamBoundary {
    index: HashSet<ExamFingerprint>,
    blocks: AtomicU64,
    false_positives: AtomicU64,
}

impl ExamBoundary {
    /// Create an empty boundary monitor.
    pub fn new() -> Self {
        Self {
            index: HashSet::new(),
            blocks: AtomicU64::new(0),
            false_positives: AtomicU64::new(0),
        }
    }

    /// Create a monitor seeded with a collection of exam fingerprints.
    pub fn from_fingerprints(fingerprints: &[ExamFingerprint]) -> Self {
        let mut s = Self::new();
        s.load(fingerprints);
        s
    }

    /// Load (or reload) the exam fingerprint index.
    pub fn load(&mut self, fingerprints: &[ExamFingerprint]) {
        self.index.clear();
        self.index.extend(fingerprints.iter().copied());
    }

    /// Number of fingerprints currently in the index.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// True if the index contains no fingerprints.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Check whether a fingerprint is in the exam set.
    pub fn contains(&self, fp: ExamFingerprint) -> bool {
        self.index.contains(&fp)
    }

    /// Decide whether to allow ingestion of a raw fingerprint.
    /// Returns `true` if the content should be blocked.
    pub fn block_fingerprint(&self, fp: ExamFingerprint) -> bool {
        if self.index.contains(&fp) {
            self.blocks.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Hash raw content and decide whether to block ingestion.
    pub fn block_ingestion(&self, content: &[u8]) -> bool {
        let fp = hash_content(content);
        self.block_fingerprint(fp)
    }

    /// Record a false positive (allowed content that was later determined
    /// to be legitimate but initially flagged).
    pub fn record_false_positive(&self) {
        self.false_positives.fetch_add(1, Ordering::Relaxed);
    }

    /// Return `(blocks, false_positives)`.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.blocks.load(Ordering::Relaxed),
            self.false_positives.load(Ordering::Relaxed),
        )
    }
}

impl Default for ExamBoundary {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash raw content into a 64-bit exam fingerprint.
pub(crate) fn hash_content(content: &[u8]) -> u64 {
    use xxhash_rust::xxh3::xxh3_64;
    xxh3_64(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_exam() {
        let exam_content = b"this is a held-out exam paper";
        let exam_fp = hash_content(exam_content);
        let normal_content = b"this is ordinary training data";

        let exm = ExamBoundary::from_fingerprints(&[exam_fp]);
        assert_eq!(exm.len(), 1);
        assert!(exm.block_ingestion(exam_content));
        assert!(!exm.block_ingestion(normal_content));

        let (blocks, fps) = exm.stats();
        assert_eq!(blocks, 1);
        assert_eq!(fps, 0);
    }

    #[test]
    fn test_allow_normal() {
        let exm = ExamBoundary::new();
        assert!(exm.is_empty());
        assert!(!exm.block_ingestion(b"anything not in the index"));
        assert_eq!(exm.stats().0, 0);
    }

    #[test]
    fn test_load_many_fingerprints() {
        let fingerprints: Vec<u64> = (0..100_000).map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15)).collect();
        let exm = ExamBoundary::from_fingerprints(&fingerprints);
        assert_eq!(exm.len(), 100_000);
        assert!(exm.contains(fingerprints[42]));
        assert!(!exm.contains(0xDEADBEEF));
    }

    #[test]
    fn test_false_positive_tracking() {
        let exm = ExamBoundary::new();
        exm.record_false_positive();
        exm.record_false_positive();
        assert_eq!(exm.stats().1, 2);
    }
}
