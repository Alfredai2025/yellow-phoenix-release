// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — Fast Sentinels
//!
//! Sub-constitutional safety checks that run without holding the gear mutex.
//! Currently covers query timeout and memory distress.

use std::time::{Duration, Instant};

/// A simple timeout sentinel for a single operation.
pub struct TimeoutSentinel {
    limit: Duration,
    start: Instant,
}

impl TimeoutSentinel {
    /// Create a sentinel with the given limit in milliseconds.
    pub fn new(limit_ms: u64) -> Self {
        Self {
            limit: Duration::from_millis(limit_ms),
            start: Instant::now(),
        }
    }

    /// Create a sentinel with an explicit duration.
    pub fn with_duration(limit: Duration) -> Self {
        Self {
            limit,
            start: Instant::now(),
        }
    }

    /// Elapsed time since the sentinel was created.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Elapsed milliseconds as a floating point value.
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1000.0
    }

    /// True if the operation has exceeded its time budget.
    pub fn is_expired(&self) -> bool {
        self.elapsed() >= self.limit
    }

    /// True if the operation is still within its time budget.
    pub fn is_ok(&self) -> bool {
        !self.is_expired()
    }
}

/// Memory sentinel. Can check live system memory or be tested with explicit values.
pub struct MemorySentinel {
    threshold_pct: f32,
}

impl MemorySentinel {
    /// Create a sentinel with a distress threshold (0.0..100.0).
    pub fn new(threshold_pct: f32) -> Self {
        Self {
            threshold_pct: threshold_pct.clamp(0.0, 100.0),
        }
    }

    /// Check distress using explicit used/total bytes (deterministic, testable).
    pub fn is_distressed_with(&self, used_bytes: u64, total_bytes: u64) -> bool {
        if total_bytes == 0 {
            return false;
        }
        let pct = (used_bytes as f64 / total_bytes as f64) * 100.0;
        pct as f32 >= self.threshold_pct
    }

    /// Check live system memory distress.
    #[cfg(feature = "std")]
    pub fn is_distressed(&self) -> bool {
        use sysinfo::{MemoryRefreshKind, RefreshKind, System};
        let mut sys = System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
        );
        sys.refresh_memory();
        let used = sys.used_memory();
        let total = sys.total_memory();
        self.is_distressed_with(used, total)
    }
}

impl Default for MemorySentinel {
    fn default() -> Self {
        Self::new(90.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_timeout_not_expired() {
        let sentinel = TimeoutSentinel::new(5000);
        assert!(!sentinel.is_expired());
        assert!(sentinel.is_ok());
    }

    #[test]
    fn test_timeout_expired() {
        let sentinel = TimeoutSentinel::new(1);
        thread::sleep(Duration::from_millis(5));
        assert!(sentinel.is_expired());
        assert!(!sentinel.is_ok());
    }

    #[test]
    fn test_memory_distressed() {
        let sentinel = MemorySentinel::new(80.0);
        assert!(sentinel.is_distressed_with(85, 100));
        assert!(!sentinel.is_distressed_with(70, 100));
    }

    #[test]
    fn test_memory_zero_total() {
        let sentinel = MemorySentinel::new(80.0);
        assert!(!sentinel.is_distressed_with(0, 0));
    }
}
