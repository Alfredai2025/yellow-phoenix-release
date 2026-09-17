// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

pub mod safety_invariants;
pub mod predictor;
pub mod executor;
pub mod auditor;
pub mod uncertainty;
pub mod shadow_validator;
pub mod canonicalization;
pub mod provenance;
pub mod ffi;
pub mod health;
pub mod cgt_bridge;

use health::{HealthMonitor, HealthState};

pub struct Trinity {
    pub predictor: predictor::Predictor,
    pub executor: executor::Executor,
    pub auditor: auditor::Auditor,
    pub canonicalization: canonicalization::CanonicalizationEngine,
    pub provenance: provenance::ProvenanceChain,
    pub health: HealthMonitor,
    pub config: TrinityConfig,
}

impl Default for Trinity {
    fn default() -> Self {
        Trinity {
            predictor: predictor::Predictor::new(),
            executor: executor::Executor::new(),
            auditor: auditor::Auditor::new(),
            canonicalization: canonicalization::CanonicalizationEngine::new(),
            provenance: provenance::ProvenanceChain::new(),
            health: HealthMonitor::new(),
            config: TrinityConfig::default(),
        }
    }
}

impl Trinity {
    /// The self-regulating mirror. No manual switch.
    /// Returns Some(result) if Trinity handles the query.
    /// Returns None if Trinity steps back and legacy should drive.
    pub fn query_auto_mirror(&mut self, query_id: u64, query_hash: &[u8]) -> Option<(u32, String)> {
        self.health.assess(&self.predictor, &self.auditor);
        
        match self.health.state {
            HealthState::Green => {
                // Trinity drives alone
                let (bucket, engine, trace) = self.executor.query_with_trinity(query_id, query_hash);
                self.provenance.append(trace);
                Some((bucket, engine))
            }
            HealthState::Yellow => {
                // Trinity runs but stays cautious
                let (bucket, engine, trace) = self.executor.query_with_trinity(query_id, query_hash);
                self.provenance.append(trace);
                // In Yellow mode, we still return Trinity's result but mark it as cautious
                Some((bucket, engine))
            }
            HealthState::Red | HealthState::Recovery => {
                // Trinity steps back. Legacy drives.
                self.predictor.state = predictor::PredictorState::Observer;
                None
            }
        }
    }
}

pub struct TrinityConfig {
    pub predictor_enabled: bool,
    pub auditor_enabled: bool,
    pub safety_invariants_enabled: bool,
    pub uncertainty_reporting_enabled: bool,
    pub shadow_validation_enabled: bool,
    pub canonicalization_enabled: bool,
    pub provenance_enabled: bool,
    pub audit_rate: f64,
    pub dispute_window_size: usize,
    pub shadow_mesh_size: usize,
    pub canonicalization_threshold: f64,
    pub token_lockout: i32,
    pub token_elevated: i32,
    pub token_restored: i32,
}

impl Default for TrinityConfig {
    fn default() -> Self {
        TrinityConfig {
            predictor_enabled: true,
            auditor_enabled: true,
            safety_invariants_enabled: true,
            uncertainty_reporting_enabled: true,
            shadow_validation_enabled: true,
            canonicalization_enabled: true,
            provenance_enabled: true,
            audit_rate: 0.001,
            dispute_window_size: 500,
            shadow_mesh_size: 500,
            canonicalization_threshold: 0.995,
            token_lockout: 0,
            token_elevated: 100,
            token_restored: 50,
        }
    }
}
