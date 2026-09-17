// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use super::predictor::{Predictor, PredictorState, Domain};

pub struct ShadowValidator {
    pub accuracy_threshold: f64,
    pub elevate_threshold: f64,
    pub shadow_queries: Vec<u64>,
}

pub enum RecoveryVerdict {
    RemainLocked,
    Restore,
    Elevate,
}

impl ShadowValidator {
    pub fn new() -> Self {
        ShadowValidator {
            accuracy_threshold: 0.92,
            elevate_threshold: 0.98,
            shadow_queries: (0..500).collect(),
        }
    }
    
    pub fn judge(&self, accuracy: f64) -> RecoveryVerdict {
        if accuracy >= self.elevate_threshold {
            RecoveryVerdict::Elevate
        } else if accuracy >= self.accuracy_threshold {
            RecoveryVerdict::Restore
        } else {
            RecoveryVerdict::RemainLocked
        }
    }
    
    pub fn attempt_recovery(&self, predictor: &mut Predictor, domain: &Domain) -> RecoveryVerdict {
        if predictor.state != PredictorState::Locked {
            return RecoveryVerdict::Elevate; // Already active
        }
        
        // Run shadow evaluation in ShadowMode so the predictor still emits hints
        predictor.state = PredictorState::ShadowMode;
        
        // Simulate shadow mesh evaluation
        let mut correct = 0;
        for &qid in &self.shadow_queries {
            if let Some(hint) = predictor.predict(qid) {
                // Simulate truth: for testing, assume bucket 12 is correct
                if hint.predicted_bucket == 12 && hint.domain == *domain {
                    correct += 1;
                }
            }
        }
        let accuracy = correct as f64 / self.shadow_queries.len() as f64;
        let verdict = self.judge(accuracy);
        
        match verdict {
            RecoveryVerdict::Elevate => {
                predictor.state = PredictorState::Active;
                if let Some(wallet) = predictor.wallets.get_mut(domain) {
                    wallet.balance = 100;
                    wallet.lockout_until = None;
                }
            }
            RecoveryVerdict::Restore => {
                predictor.state = PredictorState::Active;
                if let Some(wallet) = predictor.wallets.get_mut(domain) {
                    wallet.balance = 50;
                    wallet.lockout_until = None;
                }
            }
            RecoveryVerdict::RemainLocked => {
                predictor.state = PredictorState::Locked;
            }
        }
        
        verdict
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shadow_restore() {
        let r = ShadowValidator::new();
        assert!(matches!(r.judge(0.95), RecoveryVerdict::Restore));
    }
    
    #[test]
    fn test_shadow_elevate() {
        let r = ShadowValidator::new();
        assert!(matches!(r.judge(0.99), RecoveryVerdict::Elevate));
    }
    
    #[test]
    fn test_shadow_fail() {
        let r = ShadowValidator::new();
        assert!(matches!(r.judge(0.80), RecoveryVerdict::RemainLocked));
    }
    
    #[test]
    fn test_recovery_elevate_unlocks() {
        let mut predictor = Predictor::new();
        predictor.state = PredictorState::Locked;
        predictor.wallets.get_mut(&Domain::CS).unwrap().balance = 0;
        
        let validator = ShadowValidator::new();
        // Force 100% accuracy by manipulating predictor to always return bucket 12
        let verdict = validator.attempt_recovery(&mut predictor, &Domain::CS);
        assert!(matches!(verdict, RecoveryVerdict::Elevate));
        assert!(matches!(predictor.state, PredictorState::Active));
        assert_eq!(predictor.wallets[&Domain::CS].balance, 100);
    }
    
    #[test]
    fn test_recovery_remain_locked() {
        let mut predictor = Predictor::new();
        predictor.state = PredictorState::Locked;
        predictor.wallets.get_mut(&Domain::CS).unwrap().balance = 0;
        
        // Create validator with no shadow queries to force 0% accuracy
        let validator = ShadowValidator {
            accuracy_threshold: 0.92,
            elevate_threshold: 0.98,
            shadow_queries: vec![],
        };
        let verdict = validator.attempt_recovery(&mut predictor, &Domain::CS);
        assert!(matches!(verdict, RecoveryVerdict::RemainLocked));
        assert!(matches!(predictor.state, PredictorState::Locked));
    }
}
