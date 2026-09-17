// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — AUT Bridge Health Monitor
//!
//! Tracks the health of the Rust ↔ Python FFI bridge. Too many consecutive
//! errors without a successful call marks the bridge as unhealthy, which the
//! autonomic layer can use to throttle or reset the connection.

use std::sync::atomic::{AtomicU64, Ordering};

/// Health monitor for the AUT (autonomic) FFI bridge.
pub struct AutBridgeHealth {
    errors: AtomicU64,
    successes: AtomicU64,
    threshold: u64,
}

impl AutBridgeHealth {
    /// Create a monitor with the default error threshold of 5.
    pub fn new() -> Self {
        Self::with_threshold(5)
    }

    /// Create a monitor with a custom error threshold.
    pub fn with_threshold(threshold: u64) -> Self {
        Self {
            errors: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            threshold,
        }
    }

    /// Record a successful bridge call. Resets the error counter.
    pub fn record_success(&self) {
        self.successes.fetch_add(1, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
    }

    /// Record a bridge error.
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Return true if the bridge is currently healthy.
    pub fn is_healthy(&self) -> bool {
        self.errors.load(Ordering::Relaxed) < self.threshold
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.errors.store(0, Ordering::Relaxed);
        self.successes.store(0, Ordering::Relaxed);
    }

    /// Current number of consecutive errors.
    pub fn error_count(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Total number of successes recorded.
    pub fn success_count(&self) -> u64 {
        self.successes.load(Ordering::Relaxed)
    }

    /// Current health status as a string for telemetry.
    pub fn status(&self) -> &'static str {
        if self.is_healthy() {
            "healthy"
        } else {
            "unhealthy"
        }
    }
}

impl Default for AutBridgeHealth {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initially_healthy() {
        let health = AutBridgeHealth::new();
        assert!(health.is_healthy());
        assert_eq!(health.status(), "healthy");
    }

    #[test]
    fn test_unhealthy_after_errors() {
        let health = AutBridgeHealth::with_threshold(3);
        health.record_error();
        assert!(health.is_healthy());
        health.record_error();
        assert!(health.is_healthy());
        health.record_error();
        assert!(!health.is_healthy());
        assert_eq!(health.status(), "unhealthy");
    }

    #[test]
    fn test_success_resets_errors() {
        let health = AutBridgeHealth::with_threshold(3);
        health.record_error();
        health.record_error();
        health.record_success();
        assert!(health.is_healthy());
        assert_eq!(health.error_count(), 0);
        assert_eq!(health.success_count(), 1);
    }

    #[test]
    fn test_reset() {
        let health = AutBridgeHealth::with_threshold(2);
        health.record_error();
        health.record_error();
        assert!(!health.is_healthy());
        health.reset();
        assert!(health.is_healthy());
        assert_eq!(health.error_count(), 0);
        assert_eq!(health.success_count(), 0);
    }
}
