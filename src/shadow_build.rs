// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! shadow_build.rs — Milestone 2 Component 4.
//!
//! Background table build with atomic swap. The new table is constructed
//! incrementally while the old table continues to serve queries. When ready,
//! a single atomic swap publishes the new table with zero downtime.

use crate::versioned_tables::VersionedTable;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct ShadowBuild<T: Clone> {
    state: BuildState<T>,
}

enum BuildState<T: Clone> {
    Idle,
    Building { value: T, progress: f32, start_time: u64 },
    Ready { value: T },
}

impl<T: Clone> ShadowBuild<T> {
    pub fn new() -> Self {
        Self {
            state: BuildState::Idle,
        }
    }

    /// Begin a background build from a seed value.
    pub fn start(&mut self, seed: T) {
        self.state = BuildState::Building {
            value: seed,
            progress: 0.0,
            start_time: now_ms(),
        };
    }

    /// Advance the build by one step. `step` receives the in-progress table
    /// and returns `true` when the build is complete.
    /// Returns the current progress in `[0, 1]`.
    pub fn tick<F>(&mut self, step: F) -> f32
    where
        F: FnOnce(&mut T) -> bool,
    {
        match &mut self.state {
            BuildState::Building { value, progress, .. } => {
                let done = step(value);
                *progress += 0.1;
                *progress = progress.min(0.99);
                if done {
                    let value = value.clone();
                    self.state = BuildState::Ready { value };
                    1.0
                } else {
                    *progress
                }
            }
            BuildState::Ready { .. } => 1.0,
            BuildState::Idle => 0.0,
        }
    }

    /// True if a new table is ready to be swapped in.
    pub fn is_ready(&self) -> bool {
        matches!(self.state, BuildState::Ready { .. })
    }

    /// True if a build is in progress.
    pub fn is_building(&self) -> bool {
        matches!(self.state, BuildState::Building { .. })
    }

    /// Atomically publish the ready table into the versioned table.
    /// Returns true on success, false if no table is ready.
    pub fn try_swap(&mut self, versioned: &mut VersionedTable<T>) -> bool {
        if let BuildState::Ready { value } = &self.state {
            versioned.write_next(value.clone());
            versioned.swap();
            self.state = BuildState::Idle;
            true
        } else {
            false
        }
    }

    /// Drop any in-progress or ready build without swapping.
    pub fn cancel(&mut self) {
        self.state = BuildState::Idle;
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_build_and_swap() {
        let mut versioned = VersionedTable::new(vec![1u8, 2, 3]);
        let mut shadow = ShadowBuild::new();

        shadow.start(vec![10u8]);
        assert!(shadow.is_building());

        // Incrementally append elements until done.
        loop {
            let done = shadow.tick(|v| {
                let next = v.last().copied().unwrap_or(0) + 1;
                v.push(next);
                v.len() >= 5
            });
            if done >= 1.0 {
                break;
            }
        }

        assert!(shadow.is_ready());
        assert!(shadow.try_swap(&mut versioned));

        assert_eq!(*versioned.read(), vec![10, 11, 12, 13, 14]);
        assert_eq!(versioned.version(), 2);
    }

    #[test]
    fn old_readers_preserved_after_swap() {
        let mut versioned = VersionedTable::new(String::from("old"));
        let old_reader = versioned.read();

        let mut shadow = ShadowBuild::new();
        shadow.start(String::from("new"));
        shadow.tick(|s| {
            s.push_str("-table");
            true
        });
        assert!(shadow.try_swap(&mut versioned));

        assert_eq!(*old_reader, "old");
        assert_eq!(*versioned.read(), "new-table");
    }

    #[test]
    fn try_swap_without_ready_fails() {
        let mut versioned = VersionedTable::new(0u64);
        let mut shadow: ShadowBuild<u64> = ShadowBuild::new();
        assert!(!shadow.try_swap(&mut versioned));
        assert_eq!(versioned.version(), 1);
    }

    #[test]
    fn cancel_aborts_build() {
        let mut shadow: ShadowBuild<u64> = ShadowBuild::new();
        shadow.start(0);
        assert!(shadow.is_building());
        shadow.cancel();
        assert!(!shadow.is_building());
        assert!(!shadow.is_ready());
    }
}
