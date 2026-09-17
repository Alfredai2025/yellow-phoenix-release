// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use super::predictor::{Predictor, Domain};

pub struct UncertaintyReport {
    pub query_id: u64,
    pub flags: Vec<UncertaintyFlag>,
}

pub enum UncertaintyFlag {
    DomainUncertain,
    BucketBoundary,
    EigenvalueStale,
    NovelTerritory,
    LowTokenBalance,
}

pub struct UncertaintyEngine;

impl UncertaintyEngine {
    pub fn reward_report(predictor: &mut Predictor, domain: &Domain, query_id: u64) {
        predictor.update_wallet(domain, 2, "honest_uncertainty_report", query_id);
    }
    
    pub fn punish_silence(predictor: &mut Predictor, domain: &Domain, query_id: u64) {
        predictor.update_wallet(domain, -10, "silent_and_wrong", query_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_uncertainty_reward() {
        let mut g = Predictor::new();
        UncertaintyEngine::reward_report(&mut g, &Domain::CS, 1);
        assert_eq!(g.wallets[&Domain::CS].balance, 102);
    }
    
    #[test]
    fn test_uncertainty_punish_silence() {
        let mut g = Predictor::new();
        UncertaintyEngine::punish_silence(&mut g, &Domain::CS, 1);
        assert_eq!(g.wallets[&Domain::CS].balance, 90);
    }
}
