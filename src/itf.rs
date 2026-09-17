// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — Intelligent Telemetry Filter (ITF)
//!
//! The ITF compares live telemetry against ten constitutional anchors. When
//! an anchor is breached, it emits an `AlertFrame`. Multiple breaches in a
//! single tick are compressed into a compact 32-bit alert word.

use crate::golden_hash::TelemetrySnapshot;

/// A single breached anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlertFrame {
    /// Bit position of the anchor (0..9).
    pub anchor: u8,
    /// Human-readable domain name.
    pub domain: &'static str,
    /// Severity: 1 = warning, 2 = critical.
    pub severity: u8,
}

impl AlertFrame {
    /// Encode this frame into a single bit flag.
    pub fn flag(&self) -> u32 {
        1u32 << self.anchor
    }
}

/// Constitutional anchor thresholds.
#[derive(Debug, Clone, Copy)]
pub struct TelemetryAnchors {
    pub p50_ms: f32,
    pub p99_ms: f32,
    pub r1_min: f32,
    pub r5_min: f32,
    pub cpu_temp: f32,
    pub memory_pct: f32,
    pub autopoiesis_min: f32,
}

impl Default for TelemetryAnchors {
    fn default() -> Self {
        Self {
            p50_ms: 8.0,
            p99_ms: 12.0,
            r1_min: 0.85,
            r5_min: 0.95,
            cpu_temp: 75.0,
            memory_pct: 90.0,
            autopoiesis_min: 0.5,
        }
    }
}

/// Intelligent Telemetry Filter.
pub struct IntelligentTelemetryFilter {
    pub anchors: TelemetryAnchors,
}

impl IntelligentTelemetryFilter {
    /// Create a filter with default constitutional anchors.
    pub fn new() -> Self {
        Self {
            anchors: TelemetryAnchors::default(),
        }
    }

    /// Create a filter with custom anchors.
    pub fn with_anchors(anchors: TelemetryAnchors) -> Self {
        Self { anchors }
    }

    /// Evaluate telemetry at `tick` and return all alert frames.
    pub fn filter(&self, _tick: u64, t: &TelemetrySnapshot) -> Vec<AlertFrame> {
        let mut frames = Vec::with_capacity(4);

        if t.p50_ms > self.anchors.p50_ms {
            frames.push(AlertFrame {
                anchor: 0,
                domain: "latency_p50",
                severity: 1,
            });
        }
        if t.p99_ms > self.anchors.p99_ms {
            frames.push(AlertFrame {
                anchor: 1,
                domain: "latency_p99",
                severity: 1,
            });
        }
        if t.r1 < self.anchors.r1_min {
            frames.push(AlertFrame {
                anchor: 2,
                domain: "recall_r1",
                severity: 2,
            });
        }
        if t.r5 < self.anchors.r5_min {
            frames.push(AlertFrame {
                anchor: 3,
                domain: "recall_r5",
                severity: 2,
            });
        }
        if t.cpu_temp > self.anchors.cpu_temp {
            frames.push(AlertFrame {
                anchor: 4,
                domain: "thermal",
                severity: 2,
            });
        }
        if t.memory_pct > self.anchors.memory_pct {
            frames.push(AlertFrame {
                anchor: 5,
                domain: "memory",
                severity: 2,
            });
        }
        if t.autopoiesis < self.anchors.autopoiesis_min {
            frames.push(AlertFrame {
                anchor: 6,
                domain: "autopoiesis",
                severity: 1,
            });
        }
        if t.exam_violation {
            frames.push(AlertFrame {
                anchor: 7,
                domain: "exam",
                severity: 2,
            });
        }
        if t.git_violations > 0 {
            frames.push(AlertFrame {
                anchor: 8,
                domain: "git",
                severity: 2,
            });
        }
        if t.circuit_trips > 0 {
            frames.push(AlertFrame {
                anchor: 9,
                domain: "circuit",
                severity: 1,
            });
        }

        frames
    }

    /// Compress a slice of alert frames into a 32-bit alert word.
    pub fn compress(&self, frames: &[AlertFrame]) -> u32 {
        frames.iter().fold(0u32, |acc, f| acc | f.flag())
    }

    /// Evaluate and compress in one call.
    pub fn compressed(&self, tick: u64, t: &TelemetrySnapshot) -> u32 {
        let frames = self.filter(tick, t);
        self.compress(&frames)
    }
}

impl Default for IntelligentTelemetryFilter {
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
    fn test_normal_returns_empty() {
        let itf = IntelligentTelemetryFilter::new();
        let t = normal_telemetry();
        let frames = itf.filter(0, &t);
        assert!(frames.is_empty());
        assert_eq!(itf.compressed(0, &t), 0);
    }

    #[test]
    fn test_thermal_alert() {
        let itf = IntelligentTelemetryFilter::new();
        let mut t = normal_telemetry();
        t.cpu_temp = 80.0;
        let frames = itf.filter(0, &t);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].domain, "thermal");
        assert_eq!(frames[0].severity, 2);
        assert!(itf.compressed(0, &t) & (1 << 4) != 0);
    }

    #[test]
    fn test_git_critical() {
        let itf = IntelligentTelemetryFilter::new();
        let mut t = normal_telemetry();
        t.git_violations = 3;
        let frames = itf.filter(0, &t);
        assert!(frames.iter().any(|f| f.domain == "git" && f.severity == 2));
        assert!(itf.compressed(0, &t) & (1 << 8) != 0);
    }

    #[test]
    fn test_multiple_breaches_compress() {
        let itf = IntelligentTelemetryFilter::new();
        let mut t = normal_telemetry();
        t.p50_ms = 10.0;
        t.cpu_temp = 80.0;
        t.git_violations = 1;
        let word = itf.compressed(0, &t);
        assert_eq!(word.count_ones(), 3);
        assert!(word & (1 << 0) != 0); // latency
        assert!(word & (1 << 4) != 0); // thermal
        assert!(word & (1 << 8) != 0); // git
    }
}
