// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Flat Array Query Engine v0.4
//!
//! O(1) direct-index queries with optional stats, health monitoring,
//! and a circuit breaker for graceful degradation.

use crate::flat_array::{FlatArray, FlatError, FLAT_HASH_LEN};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Ok,
    Degraded,
    Critical,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Ok => write!(f, "OK"),
            HealthStatus::Degraded => write!(f, "DEGRADED"),
            HealthStatus::Critical => write!(f, "CRITICAL"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueryStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub total_time_ns: u64,
    pub batch_count: u64,
}

impl QueryStatsSnapshot {
    pub fn avg_time_ns(&self) -> u64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0
        } else {
            self.total_time_ns / total
        }
    }
}

#[derive(Debug)]
pub struct QueryStats {
    hits: AtomicU64,
    misses: AtomicU64,
    total_time_ns: AtomicU64,
    batch_count: AtomicU64,
}

impl QueryStats {
    pub fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
            batch_count: AtomicU64::new(0),
        }
    }

    pub fn snapshot(&self) -> QueryStatsSnapshot {
        QueryStatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            total_time_ns: self.total_time_ns.load(Ordering::Relaxed),
            batch_count: self.batch_count.load(Ordering::Relaxed),
        }
    }

    pub fn record(&self, hit: bool, elapsed_ns: u64) {
        if hit {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        self.total_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
    }
}

impl Default for QueryStats {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct HealthReport {
    pub status: HealthStatus,
    pub corruption_detected: bool,
    pub memory_pressure: bool,
    pub circuit_open: bool,
    pub avg_query_ns: u64,
}

#[derive(Debug)]
pub struct HealthMonitor {
    corruption_detected: AtomicUsize, // 0 false, 1 true
    memory_pressure: AtomicUsize,
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            corruption_detected: AtomicUsize::new(0),
            memory_pressure: AtomicUsize::new(0),
        }
    }

    pub fn set_corruption(&self, detected: bool) {
        self.corruption_detected.store(detected as usize, Ordering::Relaxed);
    }

    pub fn set_memory_pressure(&self, pressure: bool) {
        self.memory_pressure.store(pressure as usize, Ordering::Relaxed);
    }

    pub fn corruption_detected(&self) -> bool {
        self.corruption_detected.load(Ordering::Relaxed) != 0
    }

    pub fn memory_pressure(&self) -> bool {
        self.memory_pressure.load(Ordering::Relaxed) != 0
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug)]
pub struct CircuitBreaker {
    state: Mutex<CircuitState>,
    failures: AtomicU64,
    successes: AtomicU64,
    last_failure: Mutex<Option<Instant>>,
    threshold: u64,
    timeout: Duration,
}

impl CircuitBreaker {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(CircuitState::Closed),
            failures: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            last_failure: Mutex::new(None),
            threshold: 3,
            timeout: Duration::from_secs(30),
        }
    }

    /// Returns true if the circuit is open and calls should be rejected/fallback.
    pub fn is_open(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        if *state == CircuitState::Open {
            // See if enough time has passed to try half-open.
            let should_try = self
                .last_failure
                .lock()
                .unwrap()
                .map(|t| t.elapsed() >= self.timeout)
                .unwrap_or(true);
            if should_try {
                *state = CircuitState::HalfOpen;
                false
            } else {
                true
            }
        } else {
            false
        }
    }

    pub fn record_success(&self) {
        self.failures.store(0, Ordering::Relaxed);
        self.successes.fetch_add(1, Ordering::Relaxed);
        let mut state = self.state.lock().unwrap();
        if *state == CircuitState::HalfOpen {
            *state = CircuitState::Closed;
        }
    }

    pub fn record_failure(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
        *self.last_failure.lock().unwrap() = Some(Instant::now());
        let mut state = self.state.lock().unwrap();
        if self.failures.load(Ordering::Relaxed) >= self.threshold
            || *state == CircuitState::HalfOpen
        {
            *state = CircuitState::Open;
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct FlatQueryEngine {
    array: Arc<FlatArray>,
    stats: QueryStats,
    health: HealthMonitor,
    circuit: CircuitBreaker,
}

impl FlatQueryEngine {
    pub fn new(array: FlatArray) -> Result<Self, FlatError> {
        if !array.verify_checksum() {
            return Err(FlatError::ChecksumMismatch);
        }
        Ok(Self {
            array: Arc::new(array),
            stats: QueryStats::new(),
            health: HealthMonitor::new(),
            circuit: CircuitBreaker::new(),
        })
    }

    /// O(1) single query. Updates stats with relaxed atomics.
    #[inline]
    pub fn query(&self, cell_id: u64) -> Option<[u8; FLAT_HASH_LEN]> {
        if self.circuit.is_open() {
            return None; // fallback placeholder
        }

        let start = Instant::now();
        let result = self.array.query(cell_id);
        let elapsed = start.elapsed().as_nanos() as u64;

        if self.health.corruption_detected() {
            self.circuit.record_failure();
        }

        let hit = result.is_some();
        self.stats.record(hit, elapsed);
        result.copied()
    }

    pub fn query_batch(&self, cell_ids: &[u64]) -> Vec<Option<[u8; FLAT_HASH_LEN]>> {
        let start = Instant::now();
        let results: Vec<_> = cell_ids.iter().map(|&id| self.array.query(id).copied()).collect();
        let elapsed = start.elapsed().as_nanos() as u64;
        self.stats.batch_count.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_time_ns
            .fetch_add(elapsed, Ordering::Relaxed);
        let hits = results.iter().filter(|r| r.is_some()).count() as u64;
        let misses = results.len() as u64 - hits;
        self.stats.hits.fetch_add(hits, Ordering::Relaxed);
        self.stats.misses.fetch_add(misses, Ordering::Relaxed);
        results
    }

    pub fn stats(&self) -> QueryStatsSnapshot {
        self.stats.snapshot()
    }

    pub fn health(&self) -> HealthReport {
        let corruption = self.health.corruption_detected();
        let memory_pressure = self.health.memory_pressure();
        let circuit_open = self.circuit.is_open();
        let snapshot = self.stats.snapshot();

        let status = if corruption || circuit_open {
            HealthStatus::Critical
        } else if memory_pressure {
            HealthStatus::Degraded
        } else {
            HealthStatus::Ok
        };

        HealthReport {
            status,
            corruption_detected: corruption,
            memory_pressure,
            circuit_open,
            avg_query_ns: snapshot.avg_time_ns(),
        }
    }

    pub fn mark_corruption(&self) {
        self.health.set_corruption(true);
        self.circuit.record_failure();
    }

    pub fn mark_memory_pressure(&self, pressure: bool) {
        self.health.set_memory_pressure(pressure);
    }

    pub fn array(&self) -> &FlatArray {
        &self.array
    }

    pub fn memory_bytes(&self) -> usize {
        self.array.memory_bytes()
    }

    pub fn len(&self) -> usize {
        self.array.len()
    }
}
