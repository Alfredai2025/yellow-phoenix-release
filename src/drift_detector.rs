// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! drift_detector.rs — M4.4: Production accuracy monitoring.
//!
//! Tracks a rolling window of recent query results and alerts when the rolling
//! R@1 drops below a configurable threshold.  The detector is intentionally
//! simple (a `VecDeque<bool>`) so it adds negligible overhead per query.

use std::collections::VecDeque;

/// Rolling accuracy + feature logging.
#[derive(Clone, Debug)]
pub struct DriftDetector {
    window: VecDeque<bool>,
    feature_sums: [f64; 8],
    feature_count: usize,
    window_size: usize,
    alert_threshold: f32,
    min_samples: usize,
}

impl DriftDetector {
    /// Create a new detector.
    ///
    /// * `window_size` — maximum number of recent results to retain.
    /// * `alert_threshold` — R@1 below this value triggers drift (e.g. 0.95).
    /// * `min_samples` — minimum window length before alerting.
    pub fn new(window_size: usize, alert_threshold: f32, min_samples: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            feature_sums: [0.0; 8],
            feature_count: 0,
            window_size,
            alert_threshold,
            min_samples,
        }
    }

    /// Record whether the most recent query was correct (true) or not (false).
    pub fn record(&mut self, correct: bool) {
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(correct);
    }

    /// Record a query's feature vector (always) and optional correctness.
    pub fn record_query(&mut self, features: [f32; 8], correct: Option<bool>) {
        if let Some(c) = correct {
            self.record(c);
        }
        for i in 0..8 {
            self.feature_sums[i] += features[i] as f64;
        }
        self.feature_count += 1;
    }

    /// Mean feature vector over all logged queries.
    pub fn feature_mean(&self) -> [f32; 8] {
        if self.feature_count == 0 {
            return [0.0; 8];
        }
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = (self.feature_sums[i] / self.feature_count as f64) as f32;
        }
        out
    }

    /// Number of queries logged.
    pub fn queries_logged(&self) -> usize {
        self.feature_count
    }

    /// Current rolling R@1 over the retained window.
    pub fn current_r1(&self) -> f32 {
        if self.window.is_empty() {
            return 0.0;
        }
        let correct = self.window.iter().filter(|&&x| x).count();
        correct as f32 / self.window.len() as f32
    }

    /// Configured alert threshold.
    pub fn threshold(&self) -> f32 {
        self.alert_threshold
    }

    /// True if enough samples have been collected and R@1 is below threshold.
    pub fn is_drift(&self) -> bool {
        if self.window.len() < self.min_samples {
            return false;
        }
        self.current_r1() < self.alert_threshold
    }

    /// Human-readable status string.
    pub fn status(&self) -> String {
        let r1 = self.current_r1();
        let n = self.window.len();
        if n < self.min_samples {
            format!("Warming up ({}/{})", n, self.min_samples)
        } else if self.is_drift() {
            format!(
                "ALERT: R@1 = {:.2}% (below {:.2}%)",
                r1 * 100.0,
                self.alert_threshold * 100.0
            )
        } else {
            format!("OK: R@1 = {:.2}%", r1 * 100.0)
        }
    }

    /// JSON status suitable for FFI consumers.
    pub fn status_json(&self) -> String {
        let means = self.feature_mean();
        format!(
            r#"{{"drift_detected":{},"r1":{:.4},"threshold":{:.4},"queries_logged":{},"window_size":{},"min_samples":{},"feature_mean":[{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}]}}"#,
            self.is_drift(),
            self.current_r1(),
            self.alert_threshold,
            self.queries_logged(),
            self.window_size,
            self.min_samples,
            means[0], means[1], means[2], means[3], means[4], means[5], means[6], means[7]
        )
    }
}

/// Callback trait for drift alerts.
pub trait DriftAlert {
    fn on_drift(&self, r1: f32, threshold: f32);
}

/// Simple console alert implementation.
pub struct ConsoleAlert;

impl DriftAlert for ConsoleAlert {
    fn on_drift(&self, r1: f32, threshold: f32) {
        eprintln!(
            "[DRIFT ALERT] R@1 = {:.2}% (threshold: {:.2}%)",
            r1 * 100.0,
            threshold * 100.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_detector() {
        let mut detector = DriftDetector::new(100, 0.95, 50);

        // Warm-up: 50 correct queries
        for _ in 0..50 {
            detector.record(true);
        }
        assert!(!detector.is_drift());
        assert!(detector.current_r1() > 0.99);

        // Inject 20 incorrect queries
        for _ in 0..20 {
            detector.record(false);
        }
        // R@1 should now be ~30/50 = 60%, below 95% threshold
        assert!(detector.is_drift());
    }

    #[test]
    fn test_window_rolloff() {
        let mut detector = DriftDetector::new(10, 0.90, 10);
        for _ in 0..10 {
            detector.record(true);
        }
        assert!(!detector.is_drift());

        // Old results roll off
        for _ in 0..10 {
            detector.record(false);
        }
        assert!(detector.is_drift());
    }
}
