// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use super::predictor::{Predictor, Domain};
use super::auditor::Auditor;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HealthState {
    Green,    // Trinity drives alone
    Yellow,   // Trinity runs parallel with legacy, picks best
    Red,      // Trinity steps back, legacy drives
    Recovery, // Shadow validation in progress
}

pub struct HealthMonitor {
    pub state: HealthState,
    pub audit_window: usize,
    pub green_threshold: f64,
    pub yellow_threshold: f64,
}

impl HealthMonitor {
    pub fn new() -> Self {
        HealthMonitor {
            state: HealthState::Green,
            audit_window: 100,
            green_threshold: 0.85,
            yellow_threshold: 0.60,
        }
    }

    pub fn assess(&mut self, predictor: &Predictor, auditor: &Auditor) -> HealthState {
        // Count bankrupt domains
        let domains = [Domain::CS, Domain::Medical, Domain::Legal, Domain::General];
        let bankrupt_count = domains.iter().filter(|d| predictor.is_bankrupt(d)).count();

        if bankrupt_count >= 3 {
            self.state = HealthState::Red;
            return self.state;
        }

        // Calculate recent accuracy from ledger
        let entries = &auditor.ledger.entries;
        let recent_count = entries.len().min(self.audit_window);
        if recent_count == 0 {
            self.state = HealthState::Green;
            return self.state;
        }

        let correct = entries.iter()
            .rev()
            .take(recent_count)
            .filter(|e| {
                if let Some(ref hint) = e.predictor_hint {
                    hint.predicted_bucket == e.core_bucket && hint.predicted_engine == e.core_engine
                } else {
                    false
                }
            })
            .count();

        let accuracy = correct as f64 / recent_count as f64;

        self.state = if accuracy >= self.green_threshold && bankrupt_count == 0 {
            HealthState::Green
        } else if accuracy >= self.yellow_threshold {
            HealthState::Yellow
        } else {
            HealthState::Red
        };

        self.state
    }

    pub fn can_control(&self) -> bool {
        matches!(self.state, HealthState::Green | HealthState::Yellow)
    }

    pub fn is_active(&self) -> bool {
        matches!(self.state, HealthState::Green)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::predictor::PredictorHint;
    use super::super::auditor::LedgerEntry;
    use super::super::provenance::ExecutionTrace;

    #[test]
    fn test_health_green() {
        let mut h = HealthMonitor::new();
        let p = Predictor::new();
        let a = Auditor::new();
        // All wallets full, no audits = default Green
        assert_eq!(h.assess(&p, &a), HealthState::Green);
    }

    #[test]
    fn test_health_red_bankrupt() {
        let mut h = HealthMonitor::new();
        let mut p = Predictor::new();
        for d in [Domain::CS, Domain::Medical, Domain::Legal] {
            p.update_wallet(&d, -100, "test", 1);
        }
        let a = Auditor::new();
        assert_eq!(h.assess(&p, &a), HealthState::Red);
    }

    #[test]
    fn test_health_yellow_low_accuracy() {
        let mut h = HealthMonitor::new();
        let p = Predictor::new();
        let mut a = Auditor::new();
        // Fill ledger with 70% correct entries -> Yellow (above 0.60, below 0.85)
        for i in 0..100 {
            let hint = Some(PredictorHint {
                query_id: i as u64,
                predicted_bucket: if i < 70 { 12 } else { 99 },
                predicted_engine: if i < 70 { "HybridMesh".to_string() } else { "Wrong".to_string() },
                predicted_latency_ns: 1000,
                confidence: 0.9,
                prewarm_buckets: vec![],
                domain: Domain::CS,
                source: "test".to_string(),
                tangent_id: None,
            });
            let entry = LedgerEntry {
                query_id: i as u64,
                timestamp_ns: 0,
                query_hash_prefix: [0u8; 8],
                predictor_hint: hint,
                core_bucket: 12,
                core_engine: "HybridMesh".to_string(),
                execution_trace: ExecutionTrace {
                    query_id: i as u64,
                    predictor_hint: None,
                    executor_action: "test".to_string(),
                    execution_mode: super::super::provenance::ExecutionMode::FollowedHint,
                    proof_hash: [0u8; 32],
                },
                prev_hash: [0u8; 32],
                entry_hash: [0u8; 32],
            };
            a.ledger.append(entry);
        }
        assert_eq!(h.assess(&p, &a), HealthState::Yellow);
    }
}
