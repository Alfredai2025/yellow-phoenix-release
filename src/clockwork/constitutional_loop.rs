// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — Constitutional Loop
//!
//! Integration glue that drives the ClockworkEngine with live telemetry,
//! runs the Golden Hash + ITF + CGT, and applies override actions to real
//! gears. This is the organism-level feedback loop.

use crate::aut_bridge_health::AutBridgeHealth;
use crate::cgt::conjecture::ConjectureRouter;
use crate::clockwork::*;
use crate::exm::ExamBoundary;
use crate::golden_hash::{AdjudicationLevel, GoldenHash, TelemetrySnapshot};
use crate::itf::IntelligentTelemetryFilter;
use crate::scar::{Scar, ScarKind};
use crate::scar_equilibrium::ScarType;
use crate::sentinels::{MemorySentinel, TimeoutSentinel};

/// Action taken by the loop on a given tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopAction {
    None,
    TightenedPitch(GearId),
    EncodedScar(GearId),
    ThermalCritical,
    Quarantined(GearId),
}

/// Outcome of one constitutional tick.
#[derive(Debug, Clone)]
pub struct LoopOutcome {
    pub tick: u64,
    pub sovereign_cycle: u64,
    pub active_gear: GearId,
    pub alert_count: usize,
    pub alert_word: u32,
    pub override_level: AdjudicationLevel,
    pub target_gear: Option<GearId>,
    pub action_taken: LoopAction,
    pub thermal_override: bool,
}

/// Drives the full constitutional organism from telemetry.
pub struct ConstitutionalLoop {
    pub engine: ClockworkEngine,
    pub golden_hash: GoldenHash,
    pub itf: IntelligentTelemetryFilter,
    pub cgt: ConjectureRouter,
    pub exm: ExamBoundary,
    pub bridge_health: AutBridgeHealth,
    pub memory_sentinel: MemorySentinel,
    pub timeout_sentinel: TimeoutSentinel,
    last_override_level: AdjudicationLevel,
    last_override_target: Option<GearId>,
}

impl ConstitutionalLoop {
    /// Create a new organism loop with default thresholds.
    pub fn new() -> Self {
        Self {
            engine: ClockworkEngine::new(),
            golden_hash: GoldenHash::new(),
            itf: IntelligentTelemetryFilter::new(),
            cgt: ConjectureRouter::new(16),
            exm: ExamBoundary::new(),
            bridge_health: AutBridgeHealth::new(),
            memory_sentinel: MemorySentinel::new(90.0),
            timeout_sentinel: TimeoutSentinel::new(5000),
            last_override_level: AdjudicationLevel::None,
            last_override_target: None,
        }
    }

    /// Advance one tick with the given telemetry. Returns the loop outcome.
    pub fn tick(&mut self, telemetry: &TelemetrySnapshot) -> LoopOutcome {
        // If the thermal event has subsided, reset the interrupt (3°C hysteresis).
        if self.engine.thermal.is_critical() && telemetry.cpu_temp < 72.0 {
            self.engine.thermal.set_state(ThermalState::Normal, 0);
        }

        // 1. Constitutional engine tick (advances gears).
        let state = self.engine.tick();

        // 2. Apply dynamic scar-bias from each gear's equilibrium regulator.
        let effective = self.equilibrium_biased_telemetry(telemetry);

        // 3. ITF filter.
        let frames = self.itf.filter(state.tick, &effective);
        let alert_word = self.itf.compress(&frames);

        // 4. Golden Hash adjudication.
        let (level, target) = self.golden_hash.adjudicate(&effective);

        // 5. CGT observes and learns.
        self.cgt.observe(telemetry.clone());
        let thermal_pred = self.cgt.predict_thermal_critical() > 0.5;
        let accuracy_pred = self.cgt.predict_accuracy_drop() > 0.5;
        self.cgt.update_thermal(thermal_pred, telemetry.cpu_temp > 75.0);
        self.cgt.update_accuracy(accuracy_pred, telemetry.r1 < 0.85);

        // 6. Save snapshots before any mutation.
        self.save_snapshots(state.tick);

        // 7. Apply override action.
        let action_taken = self.apply_action(state.tick, level, target, telemetry);

        // 8. Update equilibrium regulators and watch for hypochondria.
        let transition = level != self.last_override_level || target != self.last_override_target;
        self.update_equilibrium(state.tick, level, target, transition);
        self.check_hypochondria();

        self.last_override_level = level;
        self.last_override_target = target;

        LoopOutcome {
            tick: state.tick,
            sovereign_cycle: state.sovereign_cycle,
            active_gear: state.active_gear,
            alert_count: frames.len(),
            alert_word,
            override_level: level,
            target_gear: target,
            action_taken,
            thermal_override: state.thermal_override,
        }
    }

    /// Check whether a byte slice is in the exam boundary.
    pub fn check_exam(&self, content: &[u8]) -> bool {
        self.exm.block_ingestion(content)
    }

    /// Load exam fingerprints into the EXM index.
    pub fn load_exam_fingerprints(&mut self, fingerprints: &[u64]) {
        self.exm.load(fingerprints);
    }

    /// Roll back the given gear to its newest snapshot at or before `tick`.
    pub fn rollback(&self, gear_id: GearId, tick: u64) -> bool {
        self.gear(gear_id).rollback(tick)
    }

    /// Return true if the gear currently holds a scar in its Reserve window.
    pub fn has_scar(&self, gear_id: GearId) -> bool {
        Scar::read_from_gear(self.gear(gear_id)).is_some()
    }

    /// Return the gear by id.
    pub fn gear(&self, gear_id: GearId) -> &Gear {
        &self.engine.gears[gear_id.index()]
    }

    /// Apply equilibrium scar biases to telemetry. Each gear's regulator
    /// modulates the sensor stream it is responsible for.
    fn equilibrium_biased_telemetry(&self, telemetry: &TelemetrySnapshot) -> TelemetrySnapshot {
        let mut t = *telemetry;

        // QRY (G0) regulates latency.
        let qry_eq = self.gear(GearId::G0).equilibrium.lock().unwrap();
        t.p50_ms = qry_eq.apply_worse(ScarType::LatencySpike, t.p50_ms);
        t.p99_ms = qry_eq.apply_worse(ScarType::LatencySpike, t.p99_ms);

        // ENC (G1) regulates accuracy/recall scores.
        let enc_eq = self.gear(GearId::G1).equilibrium.lock().unwrap();
        t.r1 = enc_eq.apply_score(ScarType::AccuracyCollapse, t.r1);
        t.r5 = enc_eq.apply_score(ScarType::AccuracyCollapse, t.r5);

        // THR (G8) regulates thermal.
        let thr_eq = self.gear(GearId::G8).equilibrium.lock().unwrap();
        t.cpu_temp = thr_eq.apply_worse(ScarType::ThermalBreach, t.cpu_temp);

        // GRW (G6) regulates memory.
        let grw_eq = self.gear(GearId::G6).equilibrium.lock().unwrap();
        t.memory_pct = grw_eq.apply_worse(ScarType::ResourceExhaustion, t.memory_pct);

        // AUT (G4) regulates autopoiesis score.
        let aut_eq = self.gear(GearId::G4).equilibrium.lock().unwrap();
        t.autopoiesis = aut_eq.apply_score(ScarType::AutopoiesisFailure, t.autopoiesis);

        t
    }

    /// Update each gear's equilibrium regulator. The targeted gear strengthens
    /// the scar type matching the override; all others decay.
    fn update_equilibrium(
        &mut self,
        _tick: u64,
        level: AdjudicationLevel,
        target: Option<GearId>,
        transition: bool,
    ) {
        let alert_type = match (level, target) {
            (AdjudicationLevel::None, _) => None,
            (AdjudicationLevel::Soft | AdjudicationLevel::Hard, Some(g)) => {
                Some(scar_type_for_gear(g))
            }
            (AdjudicationLevel::Soft | AdjudicationLevel::Hard, None) => {
                Some(ScarType::LatencySpike as u8)
            }
            (AdjudicationLevel::Severe, Some(GearId::G8)) | (AdjudicationLevel::Severe, _) => {
                Some(ScarType::ThermalBreach as u8)
            }
        };

        for gear_arc in &self.engine.gears {
            let st = if target == Some(gear_arc.id) && transition {
                alert_type
            } else {
                None
            };
            if let Ok(mut eq) = gear_arc.equilibrium.lock() {
                eq.tick(st);
            }
        }
    }

    /// Reset any gear that has become hypochondriac.
    fn check_hypochondria(&self) {
        for gear_arc in &self.engine.gears {
            let reset = {
                if let Ok(eq) = gear_arc.equilibrium.lock() {
                    eq.is_hypochondriac()
                } else {
                    false
                }
            };
            if reset {
                if let Ok(mut eq) = gear_arc.equilibrium.lock() {
                    eq.reset();
                }
            }
        }
    }

    fn save_snapshots(&self, tick: u64) {
        for gear_arc in &self.engine.gears {
            gear_arc.save_snapshot(tick);
        }
    }

    fn apply_action(
        &mut self,
        tick: u64,
        level: AdjudicationLevel,
        target: Option<GearId>,
        telemetry: &TelemetrySnapshot,
    ) -> LoopAction {
        match level {
            AdjudicationLevel::None => LoopAction::None,
            AdjudicationLevel::Soft | AdjudicationLevel::Hard => {
                let gear_id = target.unwrap_or(GearId::G0);
                self.engine.gears[gear_id.index()]
                    .pitch
                    .store(1, Ordering::Relaxed);

                // Encode a scar appropriate to the target.
                let scar_kind = match gear_id {
                    GearId::G8 => ScarKind::Thermal,
                    GearId::G7 => ScarKind::Recall,
                    GearId::G6 => ScarKind::Resource,
                    _ => ScarKind::Latency,
                };
                let scar = Scar::new(scar_kind, gear_id.index() as u8, level as u8, tick as u16);
                scar.write_to_gear(self.gear(gear_id));

                LoopAction::TightenedPitch(gear_id)
            }
            AdjudicationLevel::Severe => {
                if telemetry.cpu_temp > 75.0 || target == Some(GearId::G8) {
                    self.engine.thermal.set_state(ThermalState::Critical, tick);
                    LoopAction::ThermalCritical
                } else if let Some(g) = target {
                    LoopAction::Quarantined(g)
                } else {
                    LoopAction::None
                }
            }
        }
    }

}

/// Map a gear to the scar type it regulates.
fn scar_type_for_gear(gear: GearId) -> u8 {
    match gear {
        GearId::G8 => ScarType::ThermalBreach as u8,
        GearId::G1 | GearId::G7 => ScarType::AccuracyCollapse as u8,
        GearId::G0 => ScarType::LatencySpike as u8,
        GearId::G4 => ScarType::AutopoiesisFailure as u8,
        GearId::G6 => ScarType::ResourceExhaustion as u8,
        _ => ScarType::IntegrityViolation as u8,
    }
}

impl Default for ConstitutionalLoop {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_telemetry() -> TelemetrySnapshot {
        TelemetrySnapshot {
            p50_ms: 3.5,
            p99_ms: 5.0,
            r1: 0.92,
            r5: 0.98,
            cpu_temp: 45.0,
            memory_pct: 50.0,
            autopoiesis: 0.95,
            exam_violation: false,
            git_violations: 0,
            circuit_trips: 0,
        }
    }

    #[test]
    fn test_loop_normal_no_override() {
        let mut loop_ = ConstitutionalLoop::new();
        for _ in 0..10 {
            let outcome = loop_.tick(&normal_telemetry());
            assert_eq!(outcome.override_level, AdjudicationLevel::None);
            assert_eq!(outcome.action_taken, LoopAction::None);
        }
    }

    #[test]
    fn test_loop_thermal_severe() {
        let mut loop_ = ConstitutionalLoop::new();
        let mut t = normal_telemetry();
        t.cpu_temp = 80.0;
        let outcome = loop_.tick(&t);
        assert_eq!(outcome.override_level, AdjudicationLevel::Severe);
        assert!(matches!(outcome.action_taken, LoopAction::ThermalCritical));
        assert!(loop_.engine.thermal.is_critical());
    }

    #[test]
    fn test_loop_latency_override_tightens_qry() {
        let mut loop_ = ConstitutionalLoop::new();
        let mut t = normal_telemetry();
        t.p50_ms = 12.0;
        t.p99_ms = 18.0;
        let outcome = loop_.tick(&t);
        assert!(
            outcome.override_level == AdjudicationLevel::Soft
                || outcome.override_level == AdjudicationLevel::Hard
        );
        assert_eq!(outcome.target_gear, Some(GearId::G0));
        assert!(loop_.has_scar(GearId::G0));
    }

    #[test]
    fn test_loop_rollback_restores_hash() {
        let mut loop_ = ConstitutionalLoop::new();

        // Degrade: override writes a scar into the QRY gear's hash.
        let mut t = normal_telemetry();
        t.p50_ms = 20.0;
        t.p99_ms = 30.0;
        let outcome = loop_.tick(&t);
        let tick = outcome.tick;
        assert!(loop_.has_scar(GearId::G0), "Override should have encoded a scar");

        // Rollback to the snapshot taken just before the mutation.
        assert!(loop_.rollback(GearId::G0, tick));
        assert!(!loop_.has_scar(GearId::G0), "Rollback should restore the pre-scar hash");
    }
}
