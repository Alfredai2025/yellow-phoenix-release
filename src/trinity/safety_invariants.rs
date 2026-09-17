// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::collections::HashSet;

pub struct SafetyInvariantSet {
    pub max_latency_ns: u64,
    pub max_prewarm_buckets: usize,
    pub valid_engines: HashSet<String>,
    pub valid_domains: HashSet<String>,
    pub max_memory_mb: f64,
}

pub enum InvariantViolation {
    LatencyExceeded,
    TooManyBuckets,
    InvalidEngine,
    InvalidDomain,
    MemoryExceeded,
}

pub struct ValidatedForecast {
    pub engine: String,
    pub bucket: u32,
    pub latency_ns: u64,
}

impl SafetyInvariantSet {
    pub fn new() -> Self {
        let mut engines = HashSet::new();
        engines.insert("CascadeL1".to_string());
        engines.insert("CascadeL2".to_string());
        engines.insert("CascadeL3".to_string());
        engines.insert("HybridMesh".to_string());
        engines.insert("TensorSpectral".to_string());
        engines.insert("HologramTier0".to_string());
        engines.insert("HologramTier1".to_string());
        engines.insert("FullFallback".to_string());
        
        let mut domains = HashSet::new();
        domains.insert("CS".to_string());
        domains.insert("Medical".to_string());
        domains.insert("Legal".to_string());
        domains.insert("General".to_string());
        
        SafetyInvariantSet {
            max_latency_ns: 20_000,
            max_prewarm_buckets: 3,
            valid_engines: engines,
            valid_domains: domains,
            max_memory_mb: 50.0,
        }
    }
    
    pub fn validate(&self, engine: &str, bucket_count: usize, latency: u64, domain: &str) -> Result<ValidatedForecast, InvariantViolation> {
        if latency > self.max_latency_ns {
            return Err(InvariantViolation::LatencyExceeded);
        }
        if bucket_count > self.max_prewarm_buckets {
            return Err(InvariantViolation::TooManyBuckets);
        }
        if !self.valid_engines.contains(engine) {
            return Err(InvariantViolation::InvalidEngine);
        }
        if !self.valid_domains.contains(domain) {
            return Err(InvariantViolation::InvalidDomain);
        }
        Ok(ValidatedForecast {
            engine: engine.to_string(),
            bucket: 0,
            latency_ns: latency,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_invariant_latency_pass() {
        let c = SafetyInvariantSet::new();
        assert!(c.validate("HybridMesh", 2, 10_000, "CS").is_ok());
    }
    
    #[test]
    fn test_invariant_latency_fail() {
        let c = SafetyInvariantSet::new();
        assert!(matches!(c.validate("HybridMesh", 2, 25_000, "CS"), Err(InvariantViolation::LatencyExceeded)));
    }
    
    #[test]
    fn test_invariant_bucket_limit() {
        let c = SafetyInvariantSet::new();
        assert!(matches!(c.validate("HybridMesh", 5, 10_000, "CS"), Err(InvariantViolation::TooManyBuckets)));
    }
    
    #[test]
    fn test_invariant_invalid_engine() {
        let c = SafetyInvariantSet::new();
        assert!(matches!(c.validate("FakeEngine", 2, 10_000, "CS"), Err(InvariantViolation::InvalidEngine)));
    }
    
    #[test]
    fn test_invariant_invalid_domain() {
        let c = SafetyInvariantSet::new();
        assert!(matches!(c.validate("HybridMesh", 2, 10_000, "FakeDomain"), Err(InvariantViolation::InvalidDomain)));
    }
}
