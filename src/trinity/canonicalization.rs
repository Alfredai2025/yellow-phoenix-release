// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct QueryPattern {
    pub hash_prefix: [u8; 4],
    pub domain: String,
}

pub struct CanonicalizedPath {
    pub pattern: QueryPattern,
    pub hardcoded_bucket: u32,
    pub hardcoded_engine: String,
    pub is_canonicalized: bool,
    pub accuracy_history: Vec<bool>,
}

pub struct CanonicalizationEngine {
    pub paths: HashMap<QueryPattern, CanonicalizedPath>,
    pub threshold: f64,
    pub required_audits: usize,
}

impl CanonicalizationEngine {
    pub fn new() -> Self {
        CanonicalizationEngine {
            paths: HashMap::new(),
            threshold: 0.995,
            required_audits: 10_000,
        }
    }
    
    pub fn record(&mut self, pattern: &QueryPattern, correct: bool) {
        let path = self.paths.entry(pattern.clone()).or_insert_with(|| CanonicalizedPath {
            pattern: pattern.clone(),
            hardcoded_bucket: 0,
            hardcoded_engine: String::new(),
            is_canonicalized: false,
            accuracy_history: Vec::new(),
        });
        path.accuracy_history.push(correct);
        if path.accuracy_history.len() > self.required_audits {
            path.accuracy_history.remove(0);
        }
        
        if !path.is_canonicalized && path.accuracy_history.len() >= self.required_audits {
            let acc = path.accuracy_history.iter().filter(|&&x| x).count() as f64 / path.accuracy_history.len() as f64;
            if acc >= self.threshold {
                path.is_canonicalized = true;
            }
        }
    }
    
    pub fn is_canonicalized(&self, pattern: &QueryPattern) -> bool {
        self.paths.get(pattern).map(|p| p.is_canonicalized).unwrap_or(false)
    }
    
    pub fn get_path(&self, pattern: &QueryPattern) -> Option<(u32, String)> {
        self.paths.get(pattern).and_then(|p| {
            if p.is_canonicalized {
                Some((p.hardcoded_bucket, p.hardcoded_engine.clone()))
            } else {
                None
            }
        })
    }
    
    pub fn decanonize_if_stale(&mut self, pattern: &QueryPattern) {
        if let Some(path) = self.paths.get_mut(pattern) {
            if path.is_canonicalized {
                let recent_start = path.accuracy_history.len().saturating_sub(1000);
                let recent = &path.accuracy_history[recent_start..];
                let acc = recent.iter().filter(|&&x| x).count() as f64 / recent.len() as f64;
                if acc < 0.95 {
                    path.is_canonicalized = false;
                }
            }
        }
    }
    
    pub fn canonicalized_count(&self) -> usize {
        self.paths.values().filter(|p| p.is_canonicalized).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_canonicalization_canonize() {
        let mut b = CanonicalizationEngine::new();
        let p = QueryPattern { hash_prefix: [0xA3, 0xF2, 0, 0], domain: "CS".to_string() };
        for _ in 0..10_000 {
            b.record(&p, true);
        }
        assert!(b.is_canonicalized(&p));
        assert_eq!(b.canonicalized_count(), 1);
    }
    
    #[test]
    fn test_canonicalization_not_enough() {
        let mut b = CanonicalizationEngine::new();
        let p = QueryPattern { hash_prefix: [0xA3, 0xF2, 0, 0], domain: "CS".to_string() };
        for _ in 0..9_999 {
            b.record(&p, true);
        }
        assert!(!b.is_canonicalized(&p));
    }
    
    #[test]
    fn test_canonicalization_get_path() {
        let mut b = CanonicalizationEngine::new();
        let p = QueryPattern { hash_prefix: [0xA3, 0xF2, 0, 0], domain: "CS".to_string() };
        for _ in 0..10_000 {
            b.record(&p, true);
        }
        let path = b.get_path(&p);
        assert!(path.is_some());
    }
    
    #[test]
    fn test_canonicalization_demote_on_stale() {
        let mut b = CanonicalizationEngine::new();
        let p = QueryPattern { hash_prefix: [0xA3, 0xF2, 0, 0], domain: "CS".to_string() };
        for _ in 0..10_000 {
            b.record(&p, true);
        }
        assert!(b.is_canonicalized(&p));
        for _ in 0..1_000 {
            b.record(&p, false);
        }
        b.decanonize_if_stale(&p);
        assert!(!b.is_canonicalized(&p));
    }
}
