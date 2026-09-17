// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — Snapshot Ring
//!
//! A fixed-capacity circular buffer of `(tick, state)` pairs. Each gear owns
//! a 1024-entry ring of its 512-bit constitutional hash, enabling rollback
//! to any recent tick.

use std::sync::Mutex;

/// Fixed-capacity circular snapshot buffer.
pub struct SnapshotRing<T: Clone> {
    capacity: usize,
    /// Ordered from oldest to newest. Pushed at the back; when over capacity,
    /// the oldest element at the front is dropped.
    buffer: Mutex<Vec<(u64, T)>>,
}

impl<T: Clone> SnapshotRing<T> {
    /// Create a new empty ring with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Mutex::new(Vec::with_capacity(capacity)),
        }
    }

    /// Save a snapshot at `tick`. Drops the oldest entry if at capacity.
    pub fn save(&self, tick: u64, state: T) {
        if let Ok(mut buf) = self.buffer.lock() {
            // Monotonic-tick assumption: remove any entries with the same or
            // later tick to keep the ring consistent.
            while buf.last().map(|(t, _)| *t >= tick).unwrap_or(false) {
                buf.pop();
            }
            buf.push((tick, state));
            while buf.len() > self.capacity {
                buf.remove(0);
            }
        }
    }

    /// Return the newest snapshot whose tick is <= `tick`, if any.
    pub fn restore(&self, tick: u64) -> Option<T> {
        let buf = self.buffer.lock().ok()?;
        buf.iter()
            .filter(|(t, _)| *t <= tick)
            .last()
            .map(|(_, state)| state.clone())
    }

    /// Return the most recent snapshot, regardless of tick.
    pub fn latest(&self) -> Option<(u64, T)> {
        let buf = self.buffer.lock().ok()?;
        buf.last().cloned()
    }

    /// Return the number of stored snapshots.
    pub fn len(&self) -> usize {
        self.buffer.lock().map(|b| b.len()).unwrap_or(0)
    }

    /// Return true if no snapshots are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the configured capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all snapshots.
    pub fn clear(&self) {
        if let Ok(mut buf) = self.buffer.lock() {
            buf.clear();
        }
    }

    /// Return the tick of the oldest stored snapshot, if any.
    pub fn oldest_tick(&self) -> Option<u64> {
        let buf = self.buffer.lock().ok()?;
        buf.first().map(|(t, _)| *t)
    }

    /// Return the tick of the newest stored snapshot, if any.
    pub fn newest_tick(&self) -> Option<u64> {
        let buf = self.buffer.lock().ok()?;
        buf.last().map(|(t, _)| *t)
    }
}

impl<T: Clone> Default for SnapshotRing<T> {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_restore_roundtrip() {
        let ring = SnapshotRing::new(4);
        ring.save(10, "alpha".to_string());
        ring.save(20, "beta".to_string());
        ring.save(30, "gamma".to_string());

        assert_eq!(ring.restore(10), Some("alpha".to_string()));
        assert_eq!(ring.restore(25), Some("beta".to_string()));
        assert_eq!(ring.restore(30), Some("gamma".to_string()));
        assert_eq!(ring.latest(), Some((30, "gamma".to_string())));
    }

    #[test]
    fn test_restore_too_far_back_returns_none() {
        let ring = SnapshotRing::new(4);
        ring.save(100, "snapshot".to_string());
        assert_eq!(ring.restore(99), None);
        assert_eq!(ring.oldest_tick(), Some(100));
    }

    #[test]
    fn test_overwrite_oldest() {
        let ring = SnapshotRing::new(3);
        ring.save(1, "a".to_string());
        ring.save(2, "b".to_string());
        ring.save(3, "c".to_string());
        ring.save(4, "d".to_string());

        assert_eq!(ring.len(), 3);
        assert_eq!(ring.restore(1), None); // overwritten
        assert_eq!(ring.restore(2), Some("b".to_string()));
        assert_eq!(ring.restore(4), Some("d".to_string()));
    }

    #[test]
    fn test_clear() {
        let ring = SnapshotRing::new(4);
        ring.save(1, "x".to_string());
        ring.clear();
        assert!(ring.is_empty());
        assert_eq!(ring.restore(1), None);
    }

    #[test]
    fn test_capacity_and_default() {
        let ring: SnapshotRing<i32> = SnapshotRing::default();
        assert_eq!(ring.capacity(), 1024);
        assert!(ring.is_empty());
    }
}
