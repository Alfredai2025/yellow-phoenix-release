// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use super::predictor::{Predictor, Domain};

#[derive(Clone, Debug)]
pub struct TangentHypothesis {
    pub name: String,
    pub cgt_id: u64,
    pub condition: TangentCondition,
    pub predicted_action: PredictedAction,
    pub injected_confidence: f64,
    pub validated_accuracy: f64,
    pub query_count: u64,
    pub hit_count: u64,
    pub status: ConjectureStatus,
}

#[derive(Clone, Debug)]
pub enum TangentCondition {
    EigenvalueDrift { index: usize, threshold: f64 },
    DomainPattern { domain: Domain },
    HashPrefix { prefix: [u8; 4] },
    BucketFillRate { bucket: u32, threshold: f64 },
    Custom { description: String },
}

#[derive(Clone, Debug)]
pub enum PredictedAction {
    RouteToEngine { engine: String },
    RouteToBucket { bucket: u32 },
    PreWarmBuckets { buckets: Vec<u32> },
    SkipEngine { engine: String },
    ConservativeMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ConjectureStatus {
    Proposed,
    Testing,
    Validated,
    Refuted,
    Deprecated,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CGTConjecture {
    pub cgt_id: u64,
    pub name: String,
    pub description: String,
    pub confidence: f64,
    pub condition_type: String,
    pub condition_params: HashMap<String, String>,
    pub predicted_action_type: String,
    pub predicted_action_params: HashMap<String, String>,
}

impl CGTConjecture {
    pub fn to_tangent(&self) -> Option<TangentHypothesis> {
        let condition = match self.condition_type.as_str() {
            "eigenvalue_drift" => {
                let idx = self.condition_params.get("index")?.parse().ok()?;
                let thresh = self.condition_params.get("threshold")?.parse().ok()?;
                TangentCondition::EigenvalueDrift { index: idx, threshold: thresh }
            }
            "domain_pattern" => {
                let dom = match self.condition_params.get("domain")?.as_str() {
                    "CS" => Domain::CS,
                    "Medical" => Domain::Medical,
                    "Legal" => Domain::Legal,
                    _ => Domain::General,
                };
                TangentCondition::DomainPattern { domain: dom }
            }
            "hash_prefix" => {
                let hex = self.condition_params.get("prefix")?;
                let mut prefix = [0u8; 4];
                let bytes = hex::decode(hex).ok()?;
                if bytes.len() >= 4 {
                    prefix.copy_from_slice(&bytes[..4]);
                }
                TangentCondition::HashPrefix { prefix }
            }
            "bucket_fill_rate" => {
                let bucket = self.condition_params.get("bucket")?.parse().ok()?;
                let thresh = self.condition_params.get("threshold")?.parse().ok()?;
                TangentCondition::BucketFillRate { bucket, threshold: thresh }
            }
            _ => TangentCondition::Custom { description: self.description.clone() },
        };
        
        let action = match self.predicted_action_type.as_str() {
            "route_to_engine" => {
                let engine = self.predicted_action_params.get("engine")?.clone();
                PredictedAction::RouteToEngine { engine }
            }
            "route_to_bucket" => {
                let bucket = self.predicted_action_params.get("bucket")?.parse().ok()?;
                PredictedAction::RouteToBucket { bucket }
            }
            "prewarm_buckets" => {
                let buckets_str = self.predicted_action_params.get("buckets")?;
                let buckets: Vec<u32> = buckets_str.split(',').filter_map(|s| s.parse().ok()).collect();
                PredictedAction::PreWarmBuckets { buckets }
            }
            "skip_engine" => {
                let engine = self.predicted_action_params.get("engine")?.clone();
                PredictedAction::SkipEngine { engine }
            }
            _ => PredictedAction::ConservativeMode,
        };
        
        Some(TangentHypothesis {
            name: self.name.clone(),
            cgt_id: self.cgt_id,
            condition,
            predicted_action: action,
            injected_confidence: self.confidence,
            validated_accuracy: 0.0,
            query_count: 0,
            hit_count: 0,
            status: ConjectureStatus::Proposed,
        })
    }
}

pub struct ConjectureInjector;

impl ConjectureInjector {
    pub fn inject(predictor: &mut Predictor, conjecture: CGTConjecture) -> Result<u64, String> {
        let tangent = conjecture.to_tangent()
            .ok_or("Failed to parse CGT conjecture into tangent")?;
        let id = tangent.cgt_id;
        predictor.tangents.push(tangent);
        Ok(id)
    }
    
    pub fn remove(predictor: &mut Predictor, cgt_id: u64) -> bool {
        let before = predictor.tangents.len();
        predictor.tangents.retain(|t| t.cgt_id != cgt_id);
        predictor.tangents.len() < before
    }
    
    pub fn list(predictor: &Predictor) -> Vec<(u64, String, ConjectureStatus, f64)> {
        predictor.tangents.iter().map(|t| {
            (t.cgt_id, t.name.clone(), t.status, t.validated_accuracy)
        }).collect()
    }
}

pub struct ConjectureValidator;

impl ConjectureValidator {
    pub fn validate(tangent: &mut TangentHypothesis, predicted: &PredictedAction, actual_bucket: u32, actual_engine: &str) {
        tangent.query_count += 1;
        
        let correct = match (&tangent.predicted_action, predicted) {
            (PredictedAction::RouteToBucket { bucket }, _) => *bucket == actual_bucket,
            (PredictedAction::RouteToEngine { engine }, _) => engine == actual_engine,
            _ => false,
        };
        
        if correct {
            tangent.hit_count += 1;
        }
        
        tangent.validated_accuracy = tangent.hit_count as f64 / tangent.query_count as f64;
        
        if tangent.query_count >= 1000 {
            if tangent.validated_accuracy >= 0.85 {
                tangent.status = ConjectureStatus::Validated;
            } else if tangent.validated_accuracy < 0.60 {
                tangent.status = ConjectureStatus::Refuted;
            } else {
                tangent.status = ConjectureStatus::Testing;
            }
        } else if tangent.query_count >= 100 {
            tangent.status = ConjectureStatus::Testing;
        }
    }
    
    pub fn report(predictor: &Predictor) -> Vec<CGTValidationReport> {
        predictor.tangents.iter().map(|t| CGTValidationReport {
            cgt_id: t.cgt_id,
            name: t.name.clone(),
            status: t.status,
            injected_confidence: t.injected_confidence,
            validated_accuracy: t.validated_accuracy,
            query_count: t.query_count,
            hit_count: t.hit_count,
        }).collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CGTValidationReport {
    pub cgt_id: u64,
    pub name: String,
    pub status: ConjectureStatus,
    pub injected_confidence: f64,
    pub validated_accuracy: f64,
    pub query_count: u64,
    pub hit_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cgt_to_tangent_eigenvalue() {
        let cgt = CGTConjecture {
            cgt_id: 47,
            name: "lambda2_overflow".to_string(),
            description: "When lambda2 > 0.15, overflow imminent".to_string(),
            confidence: 0.73,
            condition_type: "eigenvalue_drift".to_string(),
            condition_params: {
                let mut m = HashMap::new();
                m.insert("index".to_string(), "2".to_string());
                m.insert("threshold".to_string(), "0.15".to_string());
                m
            },
            predicted_action_type: "route_to_bucket".to_string(),
            predicted_action_params: {
                let mut m = HashMap::new();
                m.insert("bucket".to_string(), "47".to_string());
                m
            },
        };
        let tangent = cgt.to_tangent().unwrap();
        assert_eq!(tangent.cgt_id, 47);
        assert_eq!(tangent.injected_confidence, 0.73);
        assert!(matches!(tangent.status, ConjectureStatus::Proposed));
    }
    
    #[test]
    fn test_inject_and_list() {
        let mut predictor = Predictor::new();
        let cgt = CGTConjecture {
            cgt_id: 99,
            name: "test".to_string(),
            description: "test".to_string(),
            confidence: 0.5,
            condition_type: "domain_pattern".to_string(),
            condition_params: {
                let mut m = HashMap::new();
                m.insert("domain".to_string(), "CS".to_string());
                m
            },
            predicted_action_type: "conservative_mode".to_string(),
            predicted_action_params: HashMap::new(),
        };
        let id = ConjectureInjector::inject(&mut predictor, cgt).unwrap();
        assert_eq!(id, 99);
        let list = ConjectureInjector::list(&predictor);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].0, 99);
    }
    
    #[test]
    fn test_validate_conjecture() {
        let mut tangent = TangentHypothesis {
            name: "test".to_string(),
            cgt_id: 1,
            condition: TangentCondition::Custom { description: "test".to_string() },
            predicted_action: PredictedAction::RouteToBucket { bucket: 12 },
            injected_confidence: 0.8,
            validated_accuracy: 0.0,
            query_count: 0,
            hit_count: 0,
            status: ConjectureStatus::Proposed,
        };
        
        ConjectureValidator::validate(&mut tangent, &PredictedAction::RouteToBucket { bucket: 12 }, 12, "HybridMesh");
        assert_eq!(tangent.hit_count, 1);
        assert_eq!(tangent.query_count, 1);
        
        ConjectureValidator::validate(&mut tangent, &PredictedAction::RouteToBucket { bucket: 12 }, 99, "HybridMesh");
        assert_eq!(tangent.hit_count, 1);
        assert_eq!(tangent.query_count, 2);
        assert_eq!(tangent.validated_accuracy, 0.5);
    }
    
    #[test]
    fn test_status_promotion() {
        let mut tangent = TangentHypothesis {
            name: "test".to_string(),
            cgt_id: 1,
            condition: TangentCondition::Custom { description: "test".to_string() },
            predicted_action: PredictedAction::RouteToBucket { bucket: 12 },
            injected_confidence: 0.9,
            validated_accuracy: 0.0,
            query_count: 0,
            hit_count: 0,
            status: ConjectureStatus::Proposed,
        };
        
        for _ in 0..900 {
            ConjectureValidator::validate(&mut tangent, &PredictedAction::RouteToBucket { bucket: 12 }, 12, "HybridMesh");
        }
        for _ in 0..100 {
            ConjectureValidator::validate(&mut tangent, &PredictedAction::RouteToBucket { bucket: 12 }, 99, "HybridMesh");
        }
        assert!(matches!(tangent.status, ConjectureStatus::Validated));
    }
}
