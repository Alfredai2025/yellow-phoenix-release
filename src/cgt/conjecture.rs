// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — Conjecture Generator & Tester (CGT)
//!
//! CGT observes recent telemetry, extrapolates trends, and predicts
//! constitutional failure before the Golden Hash needs to intervene.
//! It maintains a trust score based on how often its predictions come true.

use std::collections::VecDeque;

use crate::golden_hash::TelemetrySnapshot;

/// A conjecture router that predicts thermal and accuracy divergence.
pub struct ConjectureRouter {
    /// Recent telemetry samples, oldest first.
    history: VecDeque<TelemetrySnapshot>,
    max_history: usize,
    /// Recent thermal predictions and their outcomes (predicted, actual).
    thermal_outcomes: VecDeque<(bool, bool)>,
    /// Recent accuracy predictions and their outcomes.
    accuracy_outcomes: VecDeque<(bool, bool)>,
    /// Smoothed trust score in [0.0, 1.0].
    trust: f32,
}

impl ConjectureRouter {
    /// Create a router with a bounded history.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history: max_history.max(2),
            thermal_outcomes: VecDeque::with_capacity(64),
            accuracy_outcomes: VecDeque::with_capacity(64),
            trust: 0.5,
        }
    }

    /// Observe a new telemetry sample.
    pub fn observe(&mut self, t: TelemetrySnapshot) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(t);
    }

    /// Number of samples currently buffered.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Return true if enough history exists to make a prediction.
    pub fn ready(&self) -> bool {
        self.history.len() >= 2
    }

    /// Predict the probability that the CPU temperature will exceed 75°C
    /// on the next tick, based on linear extrapolation of recent samples.
    pub fn predict_thermal_critical(&self) -> f32 {
        if !self.ready() {
            return 0.0;
        }
        let (slope, last) = self.linear_fit(|t| t.cpu_temp);
        let next = last + slope;
        if next >= 75.0 {
            ((next - 60.0) / 15.0).min(1.0)
        } else {
            0.0
        }
    }

    /// Predict the probability that R1 recall will drop below 0.85
    /// on the next tick.
    pub fn predict_accuracy_drop(&self) -> f32 {
        if !self.ready() {
            return 0.0;
        }
        let (slope, last) = self.linear_fit(|t| t.r1);
        let next = last + slope;
        if next < 0.85 {
            ((0.85 - next) / 0.15).min(1.0)
        } else {
            0.0
        }
    }

    /// Register a thermal prediction outcome and update trust.
    /// `predicted` = true if the router predicted thermal critical.
    /// `actual`    = true if thermal actually became critical.
    pub fn update_thermal(&mut self, predicted: bool, actual: bool) {
        if self.thermal_outcomes.len() >= 64 {
            self.thermal_outcomes.pop_front();
        }
        self.thermal_outcomes.push_back((predicted, actual));
        self.update_trust(predicted, actual);
    }

    /// Register an accuracy prediction outcome and update trust.
    pub fn update_accuracy(&mut self, predicted: bool, actual: bool) {
        if self.accuracy_outcomes.len() >= 64 {
            self.accuracy_outcomes.pop_front();
        }
        self.accuracy_outcomes.push_back((predicted, actual));
        self.update_trust(predicted, actual);
    }

    /// Current trust score in [0.0, 1.0].
    pub fn trust(&self) -> f32 {
        self.trust
    }

    /// Compute linear slope and last value over the history using the given extractor.
    fn linear_fit<F: Fn(&TelemetrySnapshot) -> f32>(&self, f: F) -> (f32, f32) {
        let n = self.history.len() as f32;
        let last = f(self.history.back().unwrap());
        let first = f(self.history.front().unwrap());
        let slope = (last - first) / (n - 1.0).max(1.0);
        (slope, last)
    }

    /// Update trust with a single outcome. Correct predictions increase trust;
    /// incorrect ones decrease it.
    fn update_trust(&mut self, predicted: bool, actual: bool) {
        let alpha = 0.1;
        let delta = if predicted == actual { alpha } else { -alpha };
        self.trust = (self.trust + delta).clamp(0.0, 1.0);
    }
}

impl Default for ConjectureRouter {
    fn default() -> Self {
        Self::new(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(temp: f32, r1: f32) -> TelemetrySnapshot {
        TelemetrySnapshot {
            cpu_temp: temp,
            r1,
            ..Default::default()
        }
    }

    #[test]
    fn test_thermal_prediction_rising() {
        let mut router = ConjectureRouter::new(8);
        for t in [40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0] {
            router.observe(sample(t, 0.95));
        }
        let p = router.predict_thermal_critical();
        assert!(p > 0.5, "Expected high thermal critical probability, got {}", p);
    }

    #[test]
    fn test_thermal_prediction_stable() {
        let mut router = ConjectureRouter::new(8);
        for _ in 0..5 {
            router.observe(sample(45.0, 0.95));
        }
        let p = router.predict_thermal_critical();
        assert_eq!(p, 0.0, "Stable low temperature should not predict critical");
    }

    #[test]
    fn test_accuracy_prediction_falling() {
        let mut router = ConjectureRouter::new(8);
        for r in [0.95, 0.92, 0.88, 0.84, 0.80, 0.77, 0.74] {
            router.observe(sample(45.0, r));
        }
        let p = router.predict_accuracy_drop();
        assert!(p > 0.5, "Expected high accuracy-drop probability, got {}", p);
    }

    #[test]
    fn test_trust_learning() {
        let mut router = ConjectureRouter::new(4);
        for _ in 0..10 {
            router.update_thermal(true, true); // correct predictions
        }
        assert!(
            router.trust() > 0.6,
            "Trust should rise after correct predictions"
        );

        for _ in 0..20 {
            router.update_thermal(true, false); // wrong predictions
        }
        assert!(
            router.trust() < 0.5,
            "Trust should fall after wrong predictions"
        );
    }

    #[test]
    fn test_not_ready() {
        let router = ConjectureRouter::new(8);
        assert!(!router.ready());
        assert_eq!(router.predict_thermal_critical(), 0.0);
    }
}
