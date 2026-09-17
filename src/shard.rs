// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use hashbrown::HashMap;
use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicUsize, Ordering};

pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// In-core payload stored for each paper in an ISM shard.
/// Fixed 32-byte PAP (matches `PAP_512_BYTES` used throughout the crate).
pub type IsmPap = [u8; 32];

/// Maximum papers a single shard may hold (v0.3 hard ceiling).
pub const ISM_MAX_SHARD_CAPACITY: usize = 50_000_000;

#[derive(Debug, Clone, PartialEq)]
pub enum ShardError {
    Full { capacity: usize },
    Duplicate { cell_id: u64 },
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShardError::Full { capacity } => write!(f, "shard capacity {} reached", capacity),
            ShardError::Duplicate { cell_id } => write!(f, "duplicate cell_id {}", cell_id),
        }
    }
}

impl std::error::Error for ShardError {}

/// A single ISM shard: an independent HashMap with a hard capacity limit.
pub struct Shard {
    pub id: u32,
    pub max_capacity: usize,
    current_count: AtomicUsize,
    pub data: FxHashMap<u64, IsmPap>,
}

impl Shard {
    /// Create a new empty shard.
    /// `max_capacity` is clamped to `ISM_MAX_SHARD_CAPACITY`.
    pub fn new(id: u32, max_capacity: usize) -> Self {
        let max_capacity = max_capacity.min(ISM_MAX_SHARD_CAPACITY);
        Self {
            id,
            max_capacity,
            current_count: AtomicUsize::new(0),
            data: FxHashMap::with_capacity_and_hasher(max_capacity, Default::default()),
        }
    }

    /// Insert a single paper. Returns `ShardError::Full` if the shard is at capacity.
    pub fn insert(&mut self, cell_id: u64, pap: IsmPap) -> Result<(), ShardError> {
        let count = self.current_count.load(Ordering::Relaxed);
        if count >= self.max_capacity {
            return Err(ShardError::Full {
                capacity: self.max_capacity,
            });
        }
        if self.data.contains_key(&cell_id) {
            return Err(ShardError::Duplicate { cell_id });
        }
        self.data.insert(cell_id, pap);
        self.current_count.store(count + 1, Ordering::Relaxed);
        Ok(())
    }

    /// Batch insert; useful for the parallel build path.
    pub fn insert_batch(&mut self, papers: &[(u64, IsmPap)]) -> Result<(), ShardError> {
        for (cell_id, pap) in papers {
            self.insert(*cell_id, *pap)?;
        }
        Ok(())
    }

    /// Look up a paper by its cell id.
    pub fn get(&self, cell_id: u64) -> Option<&IsmPap> {
        self.data.get(&cell_id)
    }

    /// Current number of papers stored in the shard.
    pub fn len(&self) -> usize {
        self.current_count.load(Ordering::Relaxed)
    }

    /// Whether the shard has reached its capacity limit.
    pub fn is_full(&self) -> bool {
        self.len() >= self.max_capacity
    }

    /// Whether the shard contains no papers.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_insert_and_get() {
        let mut shard = Shard::new(0, 100);
        let mut pap = [0u8; 32];
        pap[..3].copy_from_slice(&[1, 2, 3]);
        assert!(shard.insert(7, pap).is_ok());
        assert_eq!(shard.get(7), Some(&pap));
        assert_eq!(shard.len(), 1);
    }

    #[test]
    fn shard_capacity_limit() {
        let mut shard = Shard::new(0, 2);
        shard.insert(1, [1u8; 32]).unwrap();
        shard.insert(2, [2u8; 32]).unwrap();
        assert!(matches!(
            shard.insert(3, [3u8; 32]),
            Err(ShardError::Full { capacity: 2 })
        ));
    }

    #[test]
    fn shard_duplicate_rejected() {
        let mut shard = Shard::new(0, 100);
        shard.insert(5, [1u8; 32]).unwrap();
        assert!(matches!(
            shard.insert(5, [2u8; 32]),
            Err(ShardError::Duplicate { cell_id: 5 })
        ));
    }
}
