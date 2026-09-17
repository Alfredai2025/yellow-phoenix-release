// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — Golden Hash Sovereign
//!
//! The Golden Hash is the constitutional adjudicator. It observes telemetry
//! from all gears, scores constitutional divergence, and issues override
//! decisions when meta-laws are threatened.
//!
//! Three override levels are emitted:
//!   0 = None     — normal operation
//!   1 = Soft     — prompt autopoiesis (gear self-corrects)
//!   2 = Hard     — sovereign pitch override
//!   3 = Severe   — quarantine / thermal interrupt
//!
//! Meta-laws are absolute: exam violations, git violations, and thermal
//! emergency always produce a Severe response regardless of other telemetry.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::clockwork::GearId;
use crate::genesis::GENESIS_SEED;

/// Adjudication levels produced by the Golden Hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
pub enum AdjudicationLevel {
    /// No override. The organism is within constitutional bounds.
    None = 0,
    /// Soft override: prompt autopoiesis on the targeted gear.
    Soft = 1,
    /// Hard override: sovereign pitch/domain override on the targeted gear.
    Hard = 2,
    /// Severe override: quarantine or thermal interrupt. Meta-law triggered.
    Severe = 3,
}

impl AdjudicationLevel {
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(AdjudicationLevel::None),
            1 => Some(AdjudicationLevel::Soft),
            2 => Some(AdjudicationLevel::Hard),
            3 => Some(AdjudicationLevel::Severe),
            _ => None,
        }
    }
}

/// A snapshot of organism telemetry used for adjudication.
#[derive(Debug, Clone, Copy, Default)]
pub struct TelemetrySnapshot {
    /// Median query latency in milliseconds.
    pub p50_ms: f32,
    /// 99th percentile query latency in milliseconds.
    pub p99_ms: f32,
    /// Recall at rank 1 (0.0..1.0).
    pub r1: f32,
    /// Recall at rank 5 (0.0..1.0).
    pub r5: f32,
    /// CPU temperature in °C.
    pub cpu_temp: f32,
    /// Memory utilization in percent (0.0..100.0).
    pub memory_pct: f32,
    /// Autopoiesis health score (0.0..1.0).
    pub autopoiesis: f32,
    /// True if an exam-boundary violation was detected.
    pub exam_violation: bool,
    /// Number of git/source-integrity violations observed.
    pub git_violations: u32,
    /// Number of circuit-breaker trips observed.
    pub circuit_trips: u32,
}

/// The Golden Hash sovereign.
pub struct GoldenHash {
    seed: u64,
    epoch: AtomicU64,
}

impl GoldenHash {
    /// Create a new sovereign from the genesis seed.
    pub fn new() -> Self {
        Self {
            seed: GENESIS_SEED,
            epoch: AtomicU64::new(0),
        }
    }

    /// Create a sovereign from an explicit seed (useful for tests).
    pub fn from_seed(seed: u64) -> Self {
        Self {
            seed,
            epoch: AtomicU64::new(0),
        }
    }

    /// Current RTC epoch in seconds, floored to a 60s boundary.
    fn rtc_epoch() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            / 60
    }

    /// Refresh the sovereign cycle. Returns the new epoch.
    pub fn tick(&self) -> u64 {
        let epoch = Self::rtc_epoch();
        self.epoch.store(epoch, Ordering::Relaxed);
        epoch
    }

    /// Read the current sovereign epoch without refreshing.
    pub fn cycle(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Compute a constitutional divergence score from telemetry.
    ///
    /// Score is in [0, ∞). Rough scale:
    ///   < 1.0  — healthy
    ///   1.0–2.0 — soft override warranted
    ///   > 2.0   — hard override warranted
    pub fn divergence(&self, t: &TelemetrySnapshot) -> f32 {
        // Latency divergence: P50 anchor at 8ms, P99 anchor at 12ms.
        let latency_div = (t.p50_ms / 8.0 + t.p99_ms / 12.0) * 0.5;

        // Recall divergence: drop below 0.85 R1 / 0.95 R5 increases divergence.
        let r1_div = (1.0 - t.r1.clamp(0.0, 1.0)) / 0.15;
        let r5_div = (1.0 - t.r5.clamp(0.0, 1.0)) / 0.05;
        let recall_div = (r1_div + r5_div) * 0.5;

        // Resource divergence.
        let thermal_div = t.cpu_temp / 75.0;
        let memory_div = t.memory_pct / 90.0;

        // Autopoiesis inverse: lower score means higher divergence.
        let autopoiesis_div = 1.0 - t.autopoiesis.clamp(0.0, 1.0);

        // Circuit trips: each trip contributes a hard divergence unit.
        let circuit_div = t.circuit_trips as f32;

        // Weighted combination emphasizing latency, with modest thermal safety.
        0.50 * latency_div
            + 0.15 * recall_div
            + 0.10 * thermal_div
            + 0.05 * memory_div
            + 0.05 * autopoiesis_div
            + 0.05 * circuit_div
    }

    /// Check absolute meta-laws. Returns (level, gear) if a meta-law fires.
    ///
    /// Meta-laws override all other telemetry and always produce Severe.
    fn meta_law_check(&self, t: &TelemetrySnapshot) -> Option<(AdjudicationLevel, GearId)> {
        if t.exam_violation {
            // EXM gear (G5) guards the exam boundary.
            return Some((AdjudicationLevel::Severe, GearId::G5));
        }
        if t.git_violations > 0 {
            // Sovereign (G9) quarantines the whole train on integrity failure.
            return Some((AdjudicationLevel::Severe, GearId::G9));
        }
        if t.cpu_temp > 75.0 {
            // Thermal gear (G8) seizes control.
            return Some((AdjudicationLevel::Severe, GearId::G8));
        }
        if t.memory_pct > 90.0 {
            // Growth gear (G6) is responsible for resource autopoiesis.
            return Some((AdjudicationLevel::Severe, GearId::G6));
        }
        None
    }

    /// Adjudicate telemetry and produce an override decision.
    ///
    /// Returns `(level, target_gear)`. The gear is `None` only when the level
    /// is `None`.
    pub fn adjudicate(&self, t: &TelemetrySnapshot) -> (AdjudicationLevel, Option<GearId>) {
        // Meta-laws are absolute.
        if let Some(decision) = self.meta_law_check(t) {
            return (decision.0, Some(decision.1));
        }

        let score = self.divergence(t);

        if score > 1.5 {
            // Hard override: choose the gear most responsible for divergence.
            let gear = self.hard_target(t);
            (AdjudicationLevel::Hard, Some(gear))
        } else if score > 0.5 {
            // Soft override: prompt autopoiesis on the primary stressed gear.
            let gear = self.soft_target(t);
            (AdjudicationLevel::Soft, Some(gear))
        } else {
            (AdjudicationLevel::None, None)
        }
    }

    /// Choose the target for a hard override using normalized metrics so
    /// incomparable units (ms, °C, %) are not compared directly.
    fn hard_target(&self, t: &TelemetrySnapshot) -> GearId {
        // Normalize each stress signal to [0, 1].
        let latency_norm = (t.p50_ms / 20.0).clamp(0.0, 1.0);
        let thermal_norm = (t.cpu_temp / 100.0).clamp(0.0, 1.0);
        let memory_norm = (t.memory_pct / 100.0).clamp(0.0, 1.0);

        if latency_norm >= thermal_norm && latency_norm >= memory_norm {
            GearId::G0 // QRY
        } else if thermal_norm >= memory_norm {
            GearId::G8 // THR
        } else {
            GearId::G6 // GRW
        }
    }

    /// Choose the target for a soft override using normalized metrics.
    fn soft_target(&self, t: &TelemetrySnapshot) -> GearId {
        let latency_norm = (t.p50_ms / 8.0).clamp(0.0, 1.0);
        let recall_norm = ((1.0 - t.r1) / 0.15).clamp(0.0, 1.0);
        let thermal_norm = ((t.cpu_temp - 60.0) / 15.0).clamp(0.0, 1.0);

        if latency_norm >= recall_norm && latency_norm >= thermal_norm {
            GearId::G0
        } else if recall_norm >= thermal_norm {
            GearId::G7 // CGT: conjecture/ranking quality
        } else if thermal_norm > 0.0 {
            GearId::G8
        } else {
            GearId::G0
        }
    }

    /// 8 pinnacle goals encoded as a deterministic constitutional window.
    /// Returns a 64-bit fingerprint of which goals are currently satisfied.
    pub fn pinnacle_goals(&self, t: &TelemetrySnapshot) -> u64 {
        let mut goals = 0u64;
        if t.p50_ms < 8.0 { goals |= 1 << 0; }      // Speed
        if t.p99_ms < 12.0 { goals |= 1 << 1; }     // Predictability
        if t.r1 > 0.85 { goals |= 1 << 2; }         // Accuracy
        if t.r5 > 0.95 { goals |= 1 << 3; }         // Coverage
        if t.cpu_temp < 65.0 { goals |= 1 << 4; }   // Thermal safety
        if t.memory_pct < 80.0 { goals |= 1 << 5; } // Memory safety
        if t.autopoiesis > 0.8 { goals |= 1 << 6; } // Self-regulation
        if t.circuit_trips == 0 { goals |= 1 << 7; } // Stability
        goals
    }
}

impl Default for GoldenHash {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_telemetry() -> TelemetrySnapshot {
        TelemetrySnapshot {
            p50_ms: 4.0,
            p99_ms: 6.0,
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
    fn test_normal_adjudication_none() {
        let gh = GoldenHash::new();
        let t = normal_telemetry();
        let (level, gear) = gh.adjudicate(&t);
        assert_eq!(level, AdjudicationLevel::None);
        assert_eq!(gear, None);
    }

    #[test]
    fn test_exam_violation_is_severe() {
        let gh = GoldenHash::new();
        let mut t = normal_telemetry();
        t.exam_violation = true;
        let (level, gear) = gh.adjudicate(&t);
        assert_eq!(level, AdjudicationLevel::Severe);
        assert_eq!(gear, Some(GearId::G5));
    }

    #[test]
    fn test_thermal_emergency_is_severe() {
        let gh = GoldenHash::new();
        let mut t = normal_telemetry();
        t.cpu_temp = 80.0;
        let (level, gear) = gh.adjudicate(&t);
        assert_eq!(level, AdjudicationLevel::Severe);
        assert_eq!(gear, Some(GearId::G8));
    }

    #[test]
    fn test_git_violation_is_severe() {
        let gh = GoldenHash::new();
        let mut t = normal_telemetry();
        t.git_violations = 1;
        let (level, gear) = gh.adjudicate(&t);
        assert_eq!(level, AdjudicationLevel::Severe);
        assert_eq!(gear, Some(GearId::G9));
    }

    #[test]
    fn test_latency_spike_triggers_soft() {
        let gh = GoldenHash::new();
        let mut t = normal_telemetry();
        t.p50_ms = 10.0;
        t.p99_ms = 15.0;
        let (level, gear) = gh.adjudicate(&t);
        assert_eq!(level, AdjudicationLevel::Soft);
        assert_eq!(gear, Some(GearId::G0));
    }

    #[test]
    fn test_severe_latency_triggers_hard() {
        let gh = GoldenHash::new();
        let mut t = normal_telemetry();
        t.p50_ms = 25.0;
        t.p99_ms = 40.0;
        t.r1 = 0.60;
        t.r5 = 0.75;
        let (level, _gear) = gh.adjudicate(&t);
        assert_eq!(level, AdjudicationLevel::Hard);
    }

    #[test]
    fn test_pinnacle_goals_normal_all_satisfied() {
        let gh = GoldenHash::new();
        let t = normal_telemetry();
        let goals = gh.pinnacle_goals(&t);
        assert_eq!(goals, 0b1111_1111, "All 8 pinnacle goals should be satisfied");
    }

    #[test]
    fn test_tick_advances_epoch() {
        let gh = GoldenHash::new();
        let e1 = gh.tick();
        let e2 = gh.cycle();
        assert_eq!(e1, e2);
        assert!(e1 > 0);
    }
}
