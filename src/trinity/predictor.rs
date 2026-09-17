// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::collections::HashMap;
use super::cgt_bridge::{TangentHypothesis, ConjectureStatus};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Domain {
    CS, Medical, Legal, General
}

#[derive(Clone)]
pub struct PredictorHint {
    pub query_id: u64,
    pub predicted_bucket: u32,
    pub predicted_engine: String,
    pub predicted_latency_ns: u64,
    pub confidence: f64,
    pub prewarm_buckets: Vec<u32>,
    pub domain: Domain,
    pub source: String,
    pub tangent_id: Option<u64>,
}

pub struct TrustWallet {
    pub domain: Domain,
    pub balance: i32,
    pub history: Vec<TokenTransaction>,
    pub lockout_until: Option<u64>,
}

pub struct TokenTransaction {
    pub query_id: u64,
    pub change: i32,
    pub reason: String,
    pub balance_after: i32,
}

#[derive(PartialEq)]
pub enum PredictorState {
    Active,
    Observer,
    ShadowMode,
    Locked,
}

pub struct Predictor {
    pub wallets: HashMap<Domain, TrustWallet>,
    pub state: PredictorState,
    pub tangents: Vec<TangentHypothesis>,
    /// Memorized query_id -> bucket mappings from explicit training/feedback.
    pub memory: HashMap<u64, u32>,
}

impl Predictor {
    pub fn new() -> Self {
        let mut wallets = HashMap::new();
        for domain in [Domain::CS, Domain::Medical, Domain::Legal, Domain::General] {
            wallets.insert(domain.clone(), TrustWallet {
                domain: domain.clone(),
                balance: 100,
                history: Vec::new(),
                lockout_until: None,
            });
        }
        Predictor { wallets, state: PredictorState::Active, tangents: Vec::new(), memory: HashMap::new() }
    }

    /// Memorize a query -> bucket mapping.
    pub fn record(&mut self, query_id: u64, bucket: u32) {
        self.memory.insert(query_id, bucket);
    }

    pub fn predict(&self, query_id: u64) -> Option<PredictorHint> {
        match self.state {
            PredictorState::Active | PredictorState::ShadowMode => {
                // Phase 1: CGT tangent routing
                for tangent in self.active_tangents() {
                    if tangent.status != ConjectureStatus::Validated
                        && tangent.status != ConjectureStatus::Testing
                    {
                        continue;
                    }
                    if tangent.validated_accuracy < 0.65 {
                        continue;
                    }

                    let (bucket, engine) = match &tangent.predicted_action {
                        super::cgt_bridge::PredictedAction::RouteToBucket { bucket } => (*bucket, "HybridMesh".to_string()),
                        super::cgt_bridge::PredictedAction::RouteToEngine { engine } => (12, engine.clone()),
                        _ => continue,
                    };

                    let conf = if tangent.status == ConjectureStatus::Validated {
                        tangent.validated_accuracy.min(0.99)
                    } else {
                        // Testing: discount confidence
                        tangent.validated_accuracy * 0.5
                    };
                    return Some(PredictorHint {
                        query_id,
                        predicted_bucket: bucket,
                        predicted_engine: engine,
                        predicted_latency_ns: 2_000,
                        confidence: conf,
                        prewarm_buckets: vec![bucket],
                        domain: Domain::CS,
                        source: "cgt_tangent".to_string(),
                        tangent_id: Some(tangent.cgt_id),
                    });
                }

                // Phase 2: Memorized fallback.
                let (bucket, conf) = if let Some(&b) = self.memory.get(&query_id) {
                    (b, 0.99)
                } else {
                    (12, 0.30)
                };
                Some(PredictorHint {
                    query_id,
                    predicted_bucket: bucket,
                    predicted_engine: "HybridMesh".to_string(),
                    predicted_latency_ns: 2_000,
                    confidence: conf,
                    prewarm_buckets: vec![bucket],
                    domain: Domain::CS,
                    source: "memory".to_string(),
                    tangent_id: None,
                })
            }
            _ => None,
        }
    }
    
    pub fn update_wallet(&mut self, domain: &Domain, change: i32, reason: &str, query_id: u64) {
        if let Some(wallet) = self.wallets.get_mut(domain) {
            wallet.balance += change;
            wallet.history.push(TokenTransaction {
                query_id,
                change,
                reason: reason.to_string(),
                balance_after: wallet.balance,
            });
            if wallet.balance <= 0 {
                wallet.lockout_until = Some(query_id + 1000);
                self.state = PredictorState::Locked;
            }
        }
    }
    
    pub fn is_bankrupt(&self, domain: &Domain) -> bool {
        self.wallets.get(domain).map(|w| w.balance <= 0).unwrap_or(true)
    }
    
    /// Return validated/testing tangents sorted by validated_accuracy descending.
    pub fn active_tangents(&self) -> Vec<&TangentHypothesis> {
        let mut active: Vec<&TangentHypothesis> = self.tangents
            .iter()
            .filter(|t| matches!(t.status, ConjectureStatus::Testing | ConjectureStatus::Validated))
            .collect();
        active.sort_by(|a, b| {
            b.validated_accuracy
                .partial_cmp(&a.validated_accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        active
    }

    /// Serialize active tangent state for Python CGT rerank.
    pub fn predictor_trace(&self, query_id: u64) -> serde_json::Value {
        use super::cgt_bridge::PredictedAction;
        let tangents: Vec<serde_json::Value> = self
            .active_tangents()
            .into_iter()
            .map(|t| {
                let (predicted_bucket, predicted_engine) = match &t.predicted_action {
                    PredictedAction::RouteToBucket { bucket } => (bucket.to_string(), "HybridMesh".to_string()),
                    PredictedAction::RouteToEngine { engine } => ("12".to_string(), engine.clone()),
                    PredictedAction::PreWarmBuckets { buckets } => (buckets.first().map(|b| b.to_string()).unwrap_or_else(|| "12".to_string()), "HybridMesh".to_string()),
                    PredictedAction::SkipEngine { engine } => ("12".to_string(), engine.clone()),
                    PredictedAction::ConservativeMode => ("12".to_string(), "HybridMesh".to_string()),
                };
                serde_json::json!({
                    "id": t.cgt_id,
                    "condition_type": format!("{:?}", t.condition),
                    "predicted_bucket": predicted_bucket,
                    "predicted_engine": predicted_engine,
                    "status": format!("{:?}", t.status),
                    "validated_accuracy": t.validated_accuracy,
                    "injected_confidence": t.injected_confidence,
                    "conservative_mode": matches!(t.predicted_action, PredictedAction::ConservativeMode),
                })
            })
            .collect();

        serde_json::json!({
            "query_id": query_id,
            "active_tangents": tangents,
            "memory_size": self.memory.len(),
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_predictor_predict_active() {
        let g = Predictor::new();
        assert!(g.predict(1).is_some());
    }
    
    #[test]
    fn test_predictor_predict_locked() {
        let mut g = Predictor::new();
        g.state = PredictorState::Locked;
        assert!(g.predict(1).is_none());
    }
    
    #[test]
    fn test_wallet_credit() {
        let mut g = Predictor::new();
        g.update_wallet(&Domain::CS, 1, "correct", 1);
        assert_eq!(g.wallets[&Domain::CS].balance, 101);
    }
    
    #[test]
    fn test_wallet_debit() {
        let mut g = Predictor::new();
        g.update_wallet(&Domain::CS, -10, "wrong", 1);
        assert_eq!(g.wallets[&Domain::CS].balance, 90);
    }
    
    #[test]
    fn test_domain_bankruptcy() {
        let mut g = Predictor::new();
        g.update_wallet(&Domain::Medical, -100, "catastrophic", 1);
        assert!(g.is_bankrupt(&Domain::Medical));
        assert!(!g.is_bankrupt(&Domain::CS));
    }
}
