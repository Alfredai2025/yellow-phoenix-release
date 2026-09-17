// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use sha2::{Sha256, Digest};

#[derive(Clone, Copy)]
pub enum ExecutionMode {
    FollowedHint,
    ModifiedHint,
    IgnoredHint,
    ForcedConservative,
    CanonicalizedPath,
    InvariantBlocked,
}

pub struct ExecutionTrace {
    pub query_id: u64,
    pub predictor_hint: Option<String>,
    pub executor_action: String,
    pub execution_mode: ExecutionMode,
    pub proof_hash: [u8; 32],
}

impl ExecutionTrace {
    pub fn compute_proof(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.query_id.to_le_bytes());
        if let Some(ref hint) = self.predictor_hint {
            hasher.update(hint.as_bytes());
        }
        hasher.update(self.executor_action.as_bytes());
        hasher.update(&[self.execution_mode as u8]);
        hasher.finalize().into()
    }
}

pub struct ProvenanceChain {
    pub traces: Vec<ExecutionTrace>,
    pub chain_hash: [u8; 32],
}

impl ProvenanceChain {
    pub fn new() -> Self {
        ProvenanceChain { traces: Vec::new(), chain_hash: [0u8; 32] }
    }
    
    pub fn append(&mut self, trace: ExecutionTrace) {
        let mut hasher = Sha256::new();
        hasher.update(&self.chain_hash);
        hasher.update(&trace.compute_proof());
        self.chain_hash = hasher.finalize().into();
        self.traces.push(trace);
    }
    
    pub fn verify(&self) -> bool {
        let mut hash = [0u8; 32];
        for trace in &self.traces {
            let mut hasher = Sha256::new();
            hasher.update(&hash);
            hasher.update(&trace.compute_proof());
            hash = hasher.finalize().into();
        }
        hash == self.chain_hash
    }
    
    pub fn len(&self) -> usize {
        self.traces.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_provenance_proof() {
        let log = ExecutionTrace {
            query_id: 1,
            predictor_hint: Some("bucket 12".to_string()),
            executor_action: "bucket 12".to_string(),
            execution_mode: ExecutionMode::FollowedHint,
            proof_hash: [0u8; 32],
        };
        let hash = log.compute_proof();
        assert_ne!(hash, [0u8; 32]);
    }
    
    #[test]
    fn test_provenance_chain_append() {
        let mut chain = ProvenanceChain::new();
        let trace = ExecutionTrace {
            query_id: 1,
            predictor_hint: None,
            executor_action: "bucket 47".to_string(),
            execution_mode: ExecutionMode::IgnoredHint,
            proof_hash: [0u8; 32],
        };
        chain.append(trace);
        assert_eq!(chain.len(), 1);
        assert_ne!(chain.chain_hash, [0u8; 32]);
    }
    
    #[test]
    fn test_provenance_chain_verify() {
        let mut chain = ProvenanceChain::new();
        for i in 0..100 {
            chain.append(ExecutionTrace {
                query_id: i,
                predictor_hint: Some(format!("hint {}", i)),
                executor_action: format!("action {}", i),
                execution_mode: ExecutionMode::FollowedHint,
                proof_hash: [0u8; 32],
            });
        }
        assert!(chain.verify());
    }
    
    #[test]
    fn test_provenance_chain_tamper() {
        let mut chain = ProvenanceChain::new();
        chain.append(ExecutionTrace {
            query_id: 1,
            predictor_hint: None,
            executor_action: "original".to_string(),
            execution_mode: ExecutionMode::IgnoredHint,
            proof_hash: [0u8; 32],
        });
        chain.traces[0].executor_action = "tampered".to_string();
        assert!(!chain.verify());
    }
}
