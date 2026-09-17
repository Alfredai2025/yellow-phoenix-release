// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Geometric Autopoiesis kernel — wires algebraic phases into Yellow's loop.

use alloc::vec::Vec;

use crate::algebra::conformal::ConformalPoint;
use crate::algebra::dual::hodge_dual;
use crate::algebra::flow::predict_binary;
use crate::algebra::meet_join::{consensus_strength, coverage_breadth};
use crate::algebra::wedge::novelty_score;
use crate::algebra::rotor::find_bridge;
use crate::distance::pap_distance;
use crate::types::multivector::BinaryMultivector;
use crate::types::ternary::TernaryMultivector;
use crate::types::graded::{GRADE0_MASK, GRADE1_MASK, GRADE2_MASK};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Total number of bits set in a `BinaryMultivector`.
#[inline]
fn total_pop(mv: &BinaryMultivector) -> u32 {
    mv.0[0].count_ones() + mv.0[1].count_ones()
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// The geometric state held by the kernel.
#[derive(Debug)]
pub struct GeometricState {
    pub scalar: BinaryMultivector,
    pub vector: BinaryMultivector,
    pub bivector: BinaryMultivector,
    pub history: Vec<BinaryMultivector>,
}

impl GeometricState {
    /// Flatten the three grades into a single 128‑bit multivector using the
    /// grade masks defined in `crate::types::graded`.
    ///
    /// * scalar   → bits 0..15
    /// * vector   → bits 16..95
    /// * bivector → bits 96..127
    pub fn to_binary(&self) -> BinaryMultivector {
        let mut res = BinaryMultivector([0, 0]);

        // grade 0 (low 16 bits of limb 0)
        res.0[0] |= self.scalar.0[0] & GRADE0_MASK[0];

        // grade 1 (bits 16..63 in limb 0, bits 64..83 in limb 1)
        res.0[0] |= self.vector.0[0] & GRADE1_MASK[0];
        res.0[1] |= self.vector.0[1] & GRADE1_MASK[1];

        // grade 2 (bits 96..127 in limb 1)
        res.0[1] |= self.bivector.0[1] & GRADE2_MASK[1];

        res
    }

    /// Push `to_binary()` onto `history`, capping at 64 entries (drop oldest).
    pub fn snapshot(&mut self) {
        let bin = self.to_binary();
        self.history.push(bin);
        if self.history.len() > 64 {
            // drop oldest entry
            self.history.remove(0);
        }
    }
}

/// Result of the diagnostic phase.
pub struct Diagnosis {
    pub stuck: bool,
    pub novelty_score: f32,
    pub blind_spot: f32,
    pub exploration_radius: i16,
}

/// Result of the proposal phase.
pub struct Proposal {
    pub bridge_idx: usize,
    pub bridge_score: f32,
    pub superposition: TernaryMultivector,
    pub conformal_distance: f32,
}

/// Result of the evaluation phase.
pub struct Evaluation {
    pub consensus: f32,
    pub coverage: f32,
    pub novelty: f32,
    pub approved: bool,
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

pub struct GeometricAutopoiesis {
    pub state: GeometricState,
    pub novelty_threshold: f32,
    pub bridge_lambda: f32,
    pub safety_radius: f32,
}

impl GeometricAutopoiesis {
    /// Create a new kernel with default hyper‑parameters.
    pub fn new(state: GeometricState) -> Self {
        Self {
            state,
            novelty_threshold: 0.25,
            bridge_lambda: 0.5,
            safety_radius: 10.0,
        }
    }

    /// Observe healthy and alarmed bits, returning a ternary superposition.
    ///
    /// pos = health & !alarm, neg = alarm.
    pub fn observe(
        &self,
        health_bits: &BinaryMultivector,
        alarm_bits: &BinaryMultivector,
    ) -> TernaryMultivector {
        let mut pos = TernaryMultivector::new();
        pos.pos[0] = health_bits.0[0] & !alarm_bits.0[0];
        pos.pos[1] = health_bits.0[1] & !alarm_bits.0[1];

        let mut neg = TernaryMultivector::new();
        neg.neg[0] = alarm_bits.0[0];
        neg.neg[1] = alarm_bits.0[1];

        pos.superpose(&neg)
    }

    /// Diagnose the current state.
    ///
    /// * novelty_score compares the current flattened state with the last
    ///   history entry (or defaults to 1.0 if no history).
    /// * stuck = novelty_score < novelty_threshold.
    pub fn diagnose(&self) -> Diagnosis {
        let current = self.state.to_binary();
        let last = self.state.history.last();
        let (novelty, stuck) = if let Some(last) = last {
            let n = novelty_score(last, &current);
            let stuck = n < self.novelty_threshold;
            (n, stuck)
        } else {
            (1.0_f32, false)
        };
        let blind_spot = pap_distance(&hodge_dual(&current), &current);
        let exploration_radius = self.safety_radius as i16;

        Diagnosis {
            stuck,
            novelty_score: novelty,
            blind_spot,
            exploration_radius,
        }
    }

    /// Propose a bridge between the current state and a desired target.
    pub fn propose(
        &self,
        desired: &BinaryMultivector,
        candidates: &[BinaryMultivector],
    ) -> Option<Proposal> {
        let current = self.state.to_binary();
        let result = find_bridge(&current, desired, candidates);
        result.map(|(bridge_idx, bridge_score)| {
            let superpos = TernaryMultivector::from_binary(&current)
                .superpose(&TernaryMultivector::from_binary(desired));

            let cp_cur = ConformalPoint::from_lsh(total_pop(&current) as u16, 0, 0);
            let cp_des = ConformalPoint::from_lsh(total_pop(desired) as u16, 0, 0);
            let conformal_distance = cp_cur.conformal_distance(&cp_des);

            Proposal {
                bridge_idx,
                bridge_score,
                superposition: superpos,
                conformal_distance,
            }
        })
    }

    /// Evaluate a set of test results.
    pub fn evaluate(&self, test_results: &[BinaryMultivector]) -> Evaluation {
        let consensus = if test_results.is_empty() {
            0.0
        } else {
            consensus_strength(test_results)
        };
        let coverage = if test_results.is_empty() {
            0.0
        } else {
            coverage_breadth(test_results)
        };

        let current = self.state.to_binary();
        let novelty = if test_results.is_empty() {
            0.0
        } else {
            test_results
                .iter()
                .map(|r| novelty_score(&current, r))
                .fold(0.0_f32, f32::max)
        };

        let approved = consensus > 0.7 && coverage > 0.9;

        Evaluation {
            consensus,
            coverage,
            novelty,
            approved,
        }
    }

    /// Predict the state after moving from the current state towards `proposed`
    /// by a fraction `t` (0 → current, 1 → proposed).
    pub fn predict(&self, proposed: &BinaryMultivector, t: f32) -> BinaryMultivector {
        let current = self.state.to_binary();
        predict_binary(&current, proposed, t.clamp(0.0, 1.0))
    }

    /// Consolidate a list of health‑bit vectors, keeping only those that are
    /// sufficiently novel relative to already‑kept entries.
    ///
    /// An entry is kept if its `novelty_score` against every previously kept
    /// entry is ≥ 0.1.
    pub fn consolidate(&self, hbs: &[BinaryMultivector]) -> Vec<usize> {
        let mut kept = Vec::new();
        for (i, hb) in hbs.iter().enumerate() {
            let keep = kept.iter().all(|&j| novelty_score(hb, &hbs[j]) >= 0.1);
            if keep {
                kept.push(i);
            }
        }
        kept
    }

    /// Safety check: the conformal distance between `before` and `after` must
    /// be strictly less than `safety_radius`.
    pub fn safety_check(&self, before: &BinaryMultivector, after: &BinaryMultivector) -> bool {
        let cp_before = ConformalPoint::from_lsh(total_pop(before) as u16, 0, 0);
        let cp_after = ConformalPoint::from_lsh(total_pop(after) as u16, 0, 0);
        let dist = cp_before.conformal_distance(&cp_after);
        dist < self.safety_radius
    }

    /// Update meta‑parameters based on the outcome of the last iteration.
    pub fn update(&mut self, diagnosis: &Diagnosis, evaluation: &Evaluation, feedback: f32) {
        if diagnosis.stuck && feedback < 0.5 {
            self.novelty_threshold *= 0.9;
        } else if !diagnosis.stuck && feedback > 0.8 {
            self.novelty_threshold *= 1.05;
        }
        if evaluation.coverage > 1.2 && feedback > 0.8 {
            self.safety_radius *= 1.02;
        }
        if evaluation.coverage < 0.8 || feedback < 0.3 {
            self.safety_radius *= 0.95;
        }
    }

    /// Persist current hyperparameters and state to a JSON file.
    pub fn save_config(&self, path: &std::path::Path) {
        let json = format!(
            r#"{{"novelty_threshold":{},"bridge_lambda":{},"safety_radius":{},"state":"{:?}","timestamp":"{}"}}"#,
            self.novelty_threshold,
            self.bridge_lambda,
            self.safety_radius,
            self.state,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );
        let _ = std::fs::create_dir_all(path.parent().unwrap_or_else(|| std::path::Path::new(".")));
        let _ = std::fs::write(path, json);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn make_state() -> GeometricState {
        GeometricState {
            scalar: BinaryMultivector::new(),
            vector: BinaryMultivector::new(),
            bivector: BinaryMultivector::new(),
            history: Vec::new(),
        }
    }

    #[test]
    fn test_observe() {
        let kernel = GeometricAutopoiesis::new(make_state());
        let health = BinaryMultivector([0b101, 0]);
        let alarm = BinaryMultivector([0b010, 0]);
        let obs = kernel.observe(&health, &alarm);
        // pos = health & !alarm → bits 0 & 2
        assert_eq!(obs.pos[0], 0b101);
        assert_eq!(obs.neg[0], 0b010);
        assert_eq!(obs.pos[1], 0);
        assert_eq!(obs.neg[1], 0);
    }

    #[test]
    fn test_diagnose_first_run() {
        let kernel = GeometricAutopoiesis::new(make_state());
        let diag = kernel.diagnose();
        assert!((diag.novelty_score - 1.0).abs() < 1e-6);
        assert!(!diag.stuck);
    }

    #[test]
    fn test_diagnose_detects_stuck() {
        let mut state = make_state();
        state.snapshot(); // push current (all zeros) onto history
        let kernel = GeometricAutopoiesis::new(state);
        let diag = kernel.diagnose();
        assert!((diag.novelty_score - 0.0).abs() < 1e-6);
        assert!(diag.stuck);
    }

    #[test]
    fn test_propose_finds_bridge() {
        let kernel = GeometricAutopoiesis::new(
            GeometricState {
                scalar: BinaryMultivector([0b1111, 0]),
                vector: BinaryMultivector::new(),
                bivector: BinaryMultivector::new(),
                history: Vec::new(),
            },
        );
        let desired = BinaryMultivector([0b1111_0000, 0]);
        let candidates = [
            BinaryMultivector([0b1111, 0]),
            BinaryMultivector([0b0011_1100, 0]),
            desired,
        ];
        let prop = kernel.propose(&desired, &candidates);
        assert!(prop.is_some());
        let p = prop.unwrap();
        assert_eq!(p.bridge_idx, 1);
    }

    #[test]
    fn test_evaluate_approved() {
        let kernel = GeometricAutopoiesis::new(make_state());
        let mv = BinaryMultivector([0xFFFF, 0]);
        let results = [mv, mv, mv];
        let eval = kernel.evaluate(&results);
        assert!((eval.consensus - 1.0).abs() < 1e-6);
        assert!((eval.coverage - 1.0).abs() < 1e-6);
        assert!(eval.approved);
    }

    #[test]
    fn test_predict_interpolates() {
        let kernel = GeometricAutopoiesis::new(make_state());
        let current = kernel.state.to_binary();
        let proposed = BinaryMultivector([u64::MAX, 0]);
        // t=0
        let p0 = kernel.predict(&proposed, 0.0);
        assert_eq!(p0, current);
        // t=1
        let p1 = kernel.predict(&proposed, 1.0);
        assert_eq!(p1, proposed);
    }

    #[test]
    fn test_consolidate_keeps_novel() {
        let kernel = GeometricAutopoiesis::new(make_state());
        let a = BinaryMultivector([0b111, 0]);
        let b = BinaryMultivector([0b111000, 0]);
        let hbs = [a, a, b];
        let kept = kernel.consolidate(&hbs);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0], 0);
        assert_eq!(kept[1], 2);
    }

    #[test]
    fn test_safety_check_rejects_large_warp() {
        let mut kernel = GeometricAutopoiesis::new(make_state());
        kernel.safety_radius = 1.0;
        let before = BinaryMultivector::new();
        let after = BinaryMultivector([u64::MAX, u64::MAX]);
        let safe = kernel.safety_check(&before, &after);
        assert!(!safe);
    }

    #[test]
    fn test_meta_update_lowers_threshold_when_stuck() {
        let mut kernel = GeometricAutopoiesis::new(make_state());
        let diag = Diagnosis {
            stuck: true,
            novelty_score: 0.0,
            blind_spot: 0.0,
            exploration_radius: 0,
        };
        let eval = Evaluation {
            consensus: 1.0,
            coverage: 1.0,
            novelty: 0.0,
            approved: true,
        };
        let initial = kernel.novelty_threshold;
        kernel.update(&diag, &eval, 0.0);
        // threshold should decrease (stuck && feedback < 0.5)
        assert!(kernel.novelty_threshold < initial);
    }
}
