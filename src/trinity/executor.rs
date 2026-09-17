// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use super::predictor::{Predictor, PredictorHint, PredictorState};
use super::safety_invariants::{SafetyInvariantSet, ValidatedForecast};
use super::provenance::{ExecutionTrace, ExecutionMode};

pub struct Executor {
    pub predictor: Predictor,
    pub safety_invariants: SafetyInvariantSet,
}

impl Executor {
    pub fn new() -> Self {
        Executor {
            predictor: Predictor::new(),
            safety_invariants: SafetyInvariantSet::new(),
        }
    }
    
    pub fn query_with_trinity(&mut self, query_id: u64, text_hash: &[u8]) -> (u32, String, ExecutionTrace) {
        // Step 1: Ask Predictor
        let hint = self.predictor.predict(query_id);
        
        // Step 2: Validate with Safety Invariants
        let validated = hint.clone().and_then(|h| {
            self.safety_invariants.validate(
                &h.predicted_engine,
                h.prewarm_buckets.len(),
                h.predicted_latency_ns,
                &format!("{:?}", h.domain)
            ).ok().map(|_| h)
        });
        
        // Step 3: Execute based on validation result
        match validated {
            Some(h) => {
                let trace = ExecutionTrace {
                    query_id,
                    predictor_hint: Some(format!("{}:{}", h.predicted_engine, h.predicted_bucket)),
                    executor_action: format!("{}:{}", h.predicted_engine, h.predicted_bucket),
                    execution_mode: ExecutionMode::FollowedHint,
                    proof_hash: [0u8; 32],
                };
                (h.predicted_bucket, h.predicted_engine, trace)
            }
            None => {
                // Fallback: no predictor or invalid hint
                let trace = ExecutionTrace {
                    query_id,
                    predictor_hint: hint.map(|h| format!("{}:{}", h.predicted_engine, h.predicted_bucket)),
                    executor_action: "FullFallback".to_string(),
                    execution_mode: ExecutionMode::IgnoredHint,
                    proof_hash: [0u8; 32],
                };
                (0, "FullFallback".to_string(), trace)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_executor_follows_hint() {
        let mut e = Executor::new();
        let (bucket, engine, trace) = e.query_with_trinity(1, &[0u8; 8]);
        assert_eq!(engine, "HybridMesh");
        assert!(matches!(trace.execution_mode, ExecutionMode::FollowedHint));
    }
    
    #[test]
    fn test_executor_locked_predictor_ignored() {
        let mut e = Executor::new();
        e.predictor.state = PredictorState::Locked;
        let (bucket, engine, trace) = e.query_with_trinity(1, &[0u8; 8]);
        assert_eq!(engine, "FullFallback");
        assert!(matches!(trace.execution_mode, ExecutionMode::IgnoredHint));
    }
    
    #[test]
    fn test_executor_invalid_hint_blocked() {
        let mut e = Executor::new();
        // Predictor hint with invalid engine would be caught by invariants
        // Since our stub predictor always returns HybridMesh (valid), we test via direct validation
        let result = e.safety_invariants.validate("FakeEngine", 2, 10_000, "CS");
        assert!(result.is_err());
    }
}
