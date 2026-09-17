// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use sha2::{Sha256, Digest};
use super::predictor::{PredictorHint, Domain};
use super::provenance::ExecutionTrace;

pub struct FrozenLedger {
    pub entries: Vec<LedgerEntry>,
    pub head_hash: [u8; 32],
    pub capacity: usize,
}

pub struct LedgerEntry {
    pub query_id: u64,
    pub timestamp_ns: u64,
    pub query_hash_prefix: [u8; 8],
    pub predictor_hint: Option<PredictorHint>,
    pub core_bucket: u32,
    pub core_engine: String,
    pub execution_trace: ExecutionTrace,
    pub prev_hash: [u8; 32],
    pub entry_hash: [u8; 32],
}

pub struct HindsightRelabeler;

impl HindsightRelabeler {
    pub fn relabel(entry: &LedgerEntry) -> Option<SyntheticExample> {
        if let Some(pred) = &entry.predictor_hint {
            if pred.predicted_bucket != entry.core_bucket || pred.predicted_engine != entry.core_engine {
                return Some(SyntheticExample {
                    query_id: entry.query_id,
                    correct_bucket: entry.core_bucket,
                    correct_engine: entry.core_engine.clone(),
                    weight: 2.0,
                });
            }
        }
        None
    }
}

pub struct SyntheticExample {
    pub query_id: u64,
    pub correct_bucket: u32,
    pub correct_engine: String,
    pub weight: f64,
}

pub struct DisputeEngine {
    pub overconfident_threshold: f64,
    pub latency_multiplier: f64,
    pub entropy_threshold: f64,
}

impl DisputeEngine {
    pub fn should_audit(&self, hint: &Option<PredictorHint>, actual_bucket: u32, actual_engine: &str, actual_latency: u64) -> bool {
        if let Some(h) = hint {
            // Overconfident and wrong
            if h.confidence > self.overconfident_threshold {
                if h.predicted_bucket != actual_bucket || h.predicted_engine != actual_engine {
                    return true;
                }
            }
            // Latency spike
            if actual_latency > (h.predicted_latency_ns as f64 * self.latency_multiplier) as u64 {
                return true;
            }
        }
        false
    }
}

impl FrozenLedger {
    pub fn new(capacity: usize) -> Self {
        FrozenLedger { entries: Vec::new(), head_hash: [0u8; 32], capacity }
    }
    
    pub fn append(&mut self, entry: LedgerEntry) {
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.head_hash = entry.entry_hash;
        self.entries.push(entry);
    }
    
    pub fn verify_chain(&self) -> bool {
        for i in 1..self.entries.len() {
            if self.entries[i].prev_hash != self.entries[i-1].entry_hash {
                return false;
            }
        }
        true
    }
    
    pub fn audit_random_sample(&self, sample_size: usize) -> Vec<&LedgerEntry> {
        use std::collections::HashSet;
        let mut samples = HashSet::new();
        let mut rng = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        while samples.len() < sample_size.min(self.entries.len()) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng as usize) % self.entries.len();
            samples.insert(idx);
        }
        samples.iter().map(|&i| &self.entries[i]).collect()
    }
}

pub fn hash_entry(entry: &LedgerEntry) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(&entry.query_id.to_le_bytes());
    hasher.update(&entry.timestamp_ns.to_le_bytes());
    hasher.update(&entry.query_hash_prefix);
    if let Some(ref hint) = entry.predictor_hint {
        hasher.update(&hint.predicted_bucket.to_le_bytes());
        hasher.update(hint.predicted_engine.as_bytes());
    }
    hasher.update(&entry.core_bucket.to_le_bytes());
    hasher.update(entry.core_engine.as_bytes());
    hasher.update(&entry.prev_hash);
    hasher.finalize().into()
}

pub struct Auditor {
    pub ledger: FrozenLedger,
    pub dispute_engine: DisputeEngine,
    pub relabeler: HindsightRelabeler,
}

impl Auditor {
    pub fn new() -> Self {
        Auditor {
            ledger: FrozenLedger::new(100_000),
            dispute_engine: DisputeEngine {
                overconfident_threshold: 0.95,
                latency_multiplier: 3.0,
                entropy_threshold: 4.5,
            },
            relabeler: HindsightRelabeler,
        }
    }
    
    pub fn record(&mut self, query_id: u64, hint: Option<PredictorHint>, 
                  actual_bucket: u32, actual_engine: String, 
                  trace: ExecutionTrace, hash_prefix: [u8; 8]) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let prev_hash = self.ledger.head_hash;
        let mut entry = LedgerEntry {
            query_id,
            timestamp_ns: timestamp,
            query_hash_prefix: hash_prefix,
            predictor_hint: hint,
            core_bucket: actual_bucket,
            core_engine: actual_engine,
            execution_trace: trace,
            prev_hash,
            entry_hash: [0u8; 32],
        };
        entry.entry_hash = hash_entry(&entry);
        self.ledger.append(entry);
    }
    
    pub fn audit_last_n(&self, n: usize) -> Vec<SyntheticExample> {
        let start = self.ledger.entries.len().saturating_sub(n);
        self.ledger.entries[start..]
            .iter()
            .filter_map(|e| HindsightRelabeler::relabel(e))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ledger_append_and_chain() {
        let mut ledger = FrozenLedger::new(1000);
        let e1 = LedgerEntry {
            query_id: 1,
            timestamp_ns: 0,
            query_hash_prefix: [0u8; 8],
            predictor_hint: None,
            core_bucket: 12,
            core_engine: "HybridMesh".to_string(),
            execution_trace: ExecutionTrace {
                query_id: 1,
                predictor_hint: None,
                executor_action: "HybridMesh:12".to_string(),
                execution_mode: super::super::provenance::ExecutionMode::FollowedHint,
                proof_hash: [0u8; 32],
            },
            prev_hash: [0u8; 32],
            entry_hash: [0u8; 32],
        };
        let mut e1_hash = e1;
        e1_hash.entry_hash = hash_entry(&e1_hash);
        
        let e2 = LedgerEntry {
            query_id: 2,
            timestamp_ns: 0,
            query_hash_prefix: [0u8; 8],
            predictor_hint: None,
            core_bucket: 47,
            core_engine: "CascadeL2".to_string(),
            execution_trace: ExecutionTrace {
                query_id: 2,
                predictor_hint: None,
                executor_action: "CascadeL2:47".to_string(),
                execution_mode: super::super::provenance::ExecutionMode::IgnoredHint,
                proof_hash: [0u8; 32],
            },
            prev_hash: e1_hash.entry_hash,
            entry_hash: [0u8; 32],
        };
        ledger.append(e1_hash);
        ledger.append(e2);
        assert!(ledger.verify_chain());
    }
    
    #[test]
    fn test_dispute_overconfident_wrong() {
        let de = DisputeEngine {
            overconfident_threshold: 0.95,
            latency_multiplier: 3.0,
            entropy_threshold: 4.5,
        };
        let hint = Some(PredictorHint {
            query_id: 1,
            predicted_bucket: 12,
            predicted_engine: "HybridMesh".to_string(),
            predicted_latency_ns: 2000,
            confidence: 0.97,
            prewarm_buckets: vec![],
            domain: Domain::CS,
            source: "test".to_string(),
            tangent_id: None,
        });
        assert!(de.should_audit(&hint, 47, "CascadeL2", 1000));
    }
    
    #[test]
    fn test_dispute_latency_spike() {
        let de = DisputeEngine {
            overconfident_threshold: 0.95,
            latency_multiplier: 3.0,
            entropy_threshold: 4.5,
        };
        let hint = Some(PredictorHint {
            query_id: 1,
            predicted_bucket: 12,
            predicted_engine: "HybridMesh".to_string(),
            predicted_latency_ns: 1000,
            confidence: 0.90,
            prewarm_buckets: vec![],
            domain: Domain::CS,
            source: "test".to_string(),
            tangent_id: None,
        });
        assert!(de.should_audit(&hint, 12, "HybridMesh", 5000)); // 5x predicted
    }
    
    #[test]
    fn test_auditor_record_and_audit() {
        let mut auditor = Auditor::new();
        let trace = ExecutionTrace {
            query_id: 1,
            predictor_hint: Some("HybridMesh:12".to_string()),
            executor_action: "HybridMesh:12".to_string(),
            execution_mode: super::super::provenance::ExecutionMode::FollowedHint,
            proof_hash: [0u8; 32],
        };
        auditor.record(1, None, 12, "HybridMesh".to_string(), trace, [0u8; 8]);
        assert_eq!(auditor.ledger.entries.len(), 1);
    }
    
    #[test]
    fn test_ledger_capacity_eviction() {
        let mut ledger = FrozenLedger::new(2);
        for i in 0..5 {
            let entry = LedgerEntry {
                query_id: i,
                timestamp_ns: 0,
                query_hash_prefix: [0u8; 8],
                predictor_hint: None,
                core_bucket: i as u32,
                core_engine: "Test".to_string(),
                execution_trace: ExecutionTrace {
                    query_id: i,
                    predictor_hint: None,
                    executor_action: "Test".to_string(),
                    execution_mode: super::super::provenance::ExecutionMode::FollowedHint,
                    proof_hash: [0u8; 32],
                },
                prev_hash: [0u8; 32],
                entry_hash: [0u8; 32],
            };
            ledger.append(entry);
        }
        assert_eq!(ledger.entries.len(), 2);
    }
    
    #[test]
    fn test_relabel_mismatch() {
        let entry = LedgerEntry {
            query_id: 1,
            timestamp_ns: 0,
            query_hash_prefix: [0u8; 8],
            predictor_hint: Some(PredictorHint {
                query_id: 1,
                predicted_bucket: 12,
                predicted_engine: "HybridMesh".to_string(),
                predicted_latency_ns: 2000,
                confidence: 0.95,
                prewarm_buckets: vec![12],
                domain: Domain::CS,
                source: "test".to_string(),
                tangent_id: None,
            }),
            core_bucket: 47,
            core_engine: "CascadeL2".to_string(),
            execution_trace: ExecutionTrace {
                query_id: 1,
                predictor_hint: None,
                executor_action: "CascadeL2:47".to_string(),
                execution_mode: super::super::provenance::ExecutionMode::IgnoredHint,
                proof_hash: [0u8; 32],
            },
            prev_hash: [0u8; 32],
            entry_hash: [0u8; 32],
        };
        let example = HindsightRelabeler::relabel(&entry);
        assert!(example.is_some());
        assert_eq!(example.unwrap().correct_bucket, 47);
    }
}
