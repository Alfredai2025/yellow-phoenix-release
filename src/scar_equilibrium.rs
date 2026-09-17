// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Equilibrium Regulator — Intelligent scar damping
//!
//! Lives in the Intelligence domain of every gear (opcode 0x63).
//! Prevents hypochondria: scars are useful memory, not permanent anxiety.
//!
//! Rules:
//! 1. Cap total bias multiplier at 2.0x (never more than double sensitivity)
//! 2. Decay bias by 1% per tick when no alert fires
//! 3. If same scar type recurs, strengthen that specific scar
//! 4. If no recurrence for 1000 ticks, that scar type fades to 1.0x
//! 5. Emergency override: thermal scars never decay below 1.5x (safety)

use std::collections::HashMap;
use std::sync::Mutex;

/// Classification of scar sensitivities regulated per gear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ScarType {
    ThermalBreach = 0,
    AccuracyCollapse = 1,
    LatencySpike = 2,
    AutopoiesisFailure = 3,
    ResourceExhaustion = 4,
    IntegrityViolation = 5,
}

impl ScarType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(ScarType::ThermalBreach),
            1 => Some(ScarType::AccuracyCollapse),
            2 => Some(ScarType::LatencySpike),
            3 => Some(ScarType::AutopoiesisFailure),
            4 => Some(ScarType::ResourceExhaustion),
            5 => Some(ScarType::IntegrityViolation),
            _ => None,
        }
    }

    pub fn count() -> usize {
        6
    }
}

/// Scar bias state per gear. Lives in the gear's Intelligence domain window.
#[derive(Clone, Debug)]
pub struct ScarEquilibrium {
    /// Current bias multiplier per scar type (1.0 = neutral, 2.0 = max)
    pub bias: HashMap<u8, f32>,

    /// Ticks since last alert per scar type
    pub ticks_since_alert: HashMap<u8, u64>,

    /// How many times each scar type has fired (strengthens on repeat)
    pub recurrence_count: HashMap<u8, u64>,

    /// Global cap — each individual bias never exceeds this
    pub global_cap: f32,

    /// Decay rate per tick when quiet (0.99 = 1% decay)
    pub decay_rate: f32,

    /// Emergency floor for thermal scars (safety)
    pub thermal_floor: f32,
}

impl ScarEquilibrium {
    pub fn new() -> Self {
        let mut bias = HashMap::new();
        for i in 0..ScarType::count() as u8 {
            bias.insert(i, 1.0);
        }

        Self {
            bias,
            ticks_since_alert: HashMap::new(),
            recurrence_count: HashMap::new(),
            global_cap: 2.0,
            decay_rate: 0.999,
            thermal_floor: 1.5,
        }
    }

    /// Call this EVERY tick, even when no alert fires.
    pub fn tick(&mut self, scar_type: Option<u8>) {
        match scar_type {
            Some(st) => {
                // ALERT FIRED — strengthen this scar type
                let count = self.recurrence_count.entry(st).or_insert(0);
                *count += 1;

                // Strengthen: +0.15 per recurrence, capped at global_cap
                let current = self.bias.get(&st).copied().unwrap_or(1.0);
                let mut strengthened = (current + 0.15).min(self.global_cap);

                // If this is a repeat (count > 1), extra boost
                if *count > 1 {
                    strengthened = (strengthened + 0.1).min(self.global_cap);
                }
                self.bias.insert(st, strengthened);

                // Reset quiet timer
                self.ticks_since_alert.insert(st, 0);
            }
            None => {
                // NO ALERT — decay all scar types
                for (st, current_bias) in self.bias.iter_mut() {
                    let mut new_bias = *current_bias * self.decay_rate;

                    if *st == ScarType::ThermalBreach as u8 {
                        new_bias = new_bias.max(self.thermal_floor);
                    } else {
                        new_bias = new_bias.max(1.0);
                    }

                    *current_bias = new_bias;

                    let quiet = self.ticks_since_alert.entry(*st).or_insert(0);
                    *quiet += 1;

                    if *quiet > 1000 {
                        self.recurrence_count.insert(*st, 0);
                    }
                }
            }
        }

        // Enforce per-type cap
        for bias in self.bias.values_mut() {
            *bias = (*bias).min(self.global_cap);
        }
    }

    /// Apply bias to raw telemetry value for higher-is-worse metrics.
    pub fn apply_worse(&self, scar_type: ScarType, raw_value: f32) -> f32 {
        let bias = self.bias.get(&(scar_type as u8)).copied().unwrap_or(1.0);
        raw_value * bias
    }

    /// Apply bias to raw telemetry value for lower-is-worse (score) metrics.
    pub fn apply_score(&self, scar_type: ScarType, raw_score: f32) -> f32 {
        let bias = self.bias.get(&(scar_type as u8)).copied().unwrap_or(1.0);
        (raw_score / bias).clamp(0.0, 1.0)
    }

    /// Get current bias for a scar type.
    pub fn current_bias(&self, scar_type: ScarType) -> f32 {
        self.bias.get(&(scar_type as u8)).copied().unwrap_or(1.0)
    }

    /// Check if gear is becoming hypochondriac (too many high biases).
    pub fn is_hypochondriac(&self) -> bool {
        self.bias.values().filter(|&&b| b > 1.8).count() >= 3
    }

    /// Force reset all biases to neutral (emergency calm-down).
    pub fn reset(&mut self) {
        for bias in self.bias.values_mut() {
            *bias = 1.0;
        }
        self.recurrence_count.clear();
        self.ticks_since_alert.clear();
    }
}

impl Default for ScarEquilibrium {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience type for the Gear struct.
pub type GearEquilibrium = Mutex<ScarEquilibrium>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_start() {
        let eq = ScarEquilibrium::new();
        assert_eq!(eq.current_bias(ScarType::ThermalBreach), 1.0);
        assert_eq!(eq.current_bias(ScarType::IntegrityViolation), 1.0);
    }

    #[test]
    fn test_strengthen_on_alert() {
        let mut eq = ScarEquilibrium::new();
        eq.tick(Some(ScarType::ThermalBreach as u8));
        assert!(eq.current_bias(ScarType::ThermalBreach) > 1.0);
    }

    #[test]
    fn test_cap_at_2x() {
        let mut eq = ScarEquilibrium::new();
        for _ in 0..20 {
            eq.tick(Some(ScarType::ThermalBreach as u8));
        }
        assert_eq!(eq.current_bias(ScarType::ThermalBreach), 2.0);
    }

    #[test]
    fn test_decay_when_quiet() {
        let mut eq = ScarEquilibrium::new();
        eq.tick(Some(ScarType::LatencySpike as u8));
        let after_alert = eq.current_bias(ScarType::LatencySpike);

        for _ in 0..100 {
            eq.tick(None);
        }
        let after_quiet = eq.current_bias(ScarType::LatencySpike);

        assert!(after_quiet < after_alert, "Bias should decay when quiet");
        assert!(after_quiet >= 1.0, "Non-thermal bias should floor at neutral");
    }

    #[test]
    fn test_thermal_never_below_floor() {
        let mut eq = ScarEquilibrium::new();
        eq.tick(Some(ScarType::ThermalBreach as u8));
        for _ in 0..1000 {
            eq.tick(None);
        }
        assert_eq!(eq.current_bias(ScarType::ThermalBreach), 1.5);
    }

    #[test]
    fn test_non_thermal_decays_to_neutral() {
        let mut eq = ScarEquilibrium::new();
        eq.tick(Some(ScarType::AccuracyCollapse as u8));
        for _ in 0..500 {
            eq.tick(None);
        }
        assert_eq!(eq.current_bias(ScarType::AccuracyCollapse), 1.0);
    }

    #[test]
    fn test_repeat_boosts() {
        let mut eq = ScarEquilibrium::new();
        eq.tick(Some(ScarType::LatencySpike as u8));
        let first = eq.current_bias(ScarType::LatencySpike);

        for _ in 0..50 {
            eq.tick(None);
        }
        eq.tick(Some(ScarType::LatencySpike as u8));
        let second = eq.current_bias(ScarType::LatencySpike);

        assert!(second > first, "Repeat scar should boost more");
    }

    #[test]
    fn test_hypochondriac_detection() {
        let mut eq = ScarEquilibrium::new();
        for _ in 0..15 {
            eq.tick(Some(ScarType::ThermalBreach as u8));
            eq.tick(Some(ScarType::AccuracyCollapse as u8));
            eq.tick(Some(ScarType::LatencySpike as u8));
        }
        assert!(eq.is_hypochondriac());
    }

    #[test]
    fn test_reset_calm_down() {
        let mut eq = ScarEquilibrium::new();
        for _ in 0..10 {
            eq.tick(Some(ScarType::ThermalBreach as u8));
        }
        assert!(eq.current_bias(ScarType::ThermalBreach) > 1.5);

        eq.reset();
        assert_eq!(eq.current_bias(ScarType::ThermalBreach), 1.0);
        assert!(!eq.is_hypochondriac());
    }

    #[test]
    fn test_apply_bias() {
        let mut eq = ScarEquilibrium::new();
        eq.tick(Some(ScarType::LatencySpike as u8));
        let raw = 8.0f32;
        let biased = eq.apply_worse(ScarType::LatencySpike, raw);
        assert!(biased > raw);
        assert!(biased <= raw * 2.0);
    }

    #[test]
    fn test_recurrence_resets_after_1000_quiet_ticks() {
        let mut eq = ScarEquilibrium::new();
        eq.tick(Some(ScarType::AutopoiesisFailure as u8));
        eq.tick(Some(ScarType::AutopoiesisFailure as u8));
        assert_eq!(eq.recurrence_count.get(&(ScarType::AutopoiesisFailure as u8)).copied().unwrap_or(0), 2);

        for _ in 0..1001 {
            eq.tick(None);
        }

        assert_eq!(eq.recurrence_count.get(&(ScarType::AutopoiesisFailure as u8)).copied().unwrap_or(0), 0);
    }
}
