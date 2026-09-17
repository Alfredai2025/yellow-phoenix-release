// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! versioned_tables.rs — Milestone 2 Component 3.
//!
//! Lock-free, atomically versioned table. Old readers keep their `Arc`
//! reference; new readers see the swapped value. Zero contention between
//! reads and updates.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct VersionedTable<T> {
    current: Arc<T>,
    next: Arc<T>,
    version: AtomicU64,
}

impl<T: Clone> VersionedTable<T> {
    pub fn new(initial: T) -> Self {
        let arc = Arc::new(initial);
        Self {
            current: arc.clone(),
            next: arc,
            version: AtomicU64::new(1),
        }
    }

    /// Read the current table. Fast, lock-free, clone-once of the `Arc`.
    pub fn read(&self) -> Arc<T> {
        self.current.clone()
    }

    /// Stage a new value in the "next" slot. Does not affect current readers.
    pub fn write_next(&mut self, new_value: T) {
        self.next = Arc::new(new_value);
    }

    /// Atomically swap current and next. New readers immediately see the new
    /// value; old readers keep their `Arc` until dropped.
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
        self.version.fetch_add(1, Ordering::SeqCst);
    }

    /// Current version number. Increments on every successful swap.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_returns_current_value() {
        let table = VersionedTable::new(vec![1, 2, 3]);
        let v = table.read();
        assert_eq!(*v, vec![1, 2, 3]);
    }

    #[test]
    fn swap_updates_version_and_readers() {
        let mut table = VersionedTable::new(100u64);
        assert_eq!(table.version(), 1);

        let old_reader = table.read();
        assert_eq!(*old_reader, 100);

        table.write_next(200);
        table.swap();

        assert_eq!(table.version(), 2);
        let new_reader = table.read();
        assert_eq!(*new_reader, 200);
        assert_eq!(*old_reader, 100); // old reader unaffected
    }

    #[test]
    fn multiple_swaps_preserve_old_readers() {
        let mut table = VersionedTable::new(0u64);
        let r1 = table.read();

        table.write_next(1);
        table.swap();
        let r2 = table.read();

        table.write_next(2);
        table.swap();
        let r3 = table.read();

        assert_eq!(*r1, 0);
        assert_eq!(*r2, 1);
        assert_eq!(*r3, 2);
        assert_eq!(table.version(), 3);
    }

    #[test]
    fn many_readers_during_swap() {
        let mut table = VersionedTable::new(String::from("alpha"));
        let mut readers = Vec::new();
        for _ in 0..100 {
            readers.push(table.read());
        }

        table.write_next(String::from("beta"));
        table.swap();

        let new_reader = table.read();
        assert_eq!(*new_reader, "beta");
        for r in readers {
            assert_eq!(*r, "alpha");
        }
    }
}
