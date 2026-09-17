// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Snapshot Trait — Yellow Phoenix v3.6 Phase 3
//! Versioned snapshots of gravity, dynamic buckets, and energy overlay.

use crate::energy_overlay::EnergyOverlay;
use crate::hybrid_mesh::{DynamicBuckets, HybridMesh, PaperGravity};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Change in bucket population between two snapshots.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BucketChange {
    pub bucket: u32,
    pub old_count: usize,
    pub new_count: usize,
    pub delta: i64,
}

/// Serializable subset of the living mesh state.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct LivingState {
    pub gravity_map: HashMap<u64, PaperGravity>,
    pub dynamic_buckets: DynamicBuckets,
    pub energy_overlay: Option<EnergyOverlay>,
}

/// Trait for snapshotting and restoring mesh state.
pub trait MeshSnapshot {
    /// Serialize the living state to JSON bytes.
    fn snapshot(&self) -> Result<Vec<u8>, String>;

    /// Restore the living state from JSON bytes.
    fn restore(&mut self, data: &[u8]) -> Result<(), String>;

    /// Compute per-bucket population changes relative to a previous snapshot.
    fn delta(&self, old_data: &[u8]) -> Result<Vec<BucketChange>, String>;
}

impl MeshSnapshot for HybridMesh {
    fn snapshot(&self) -> Result<Vec<u8>, String> {
        let state = LivingState {
            gravity_map: self.gravity_map.clone(),
            dynamic_buckets: self.dynamic_buckets.clone(),
            energy_overlay: self.energy_overlay.clone(),
        };
        serde_json::to_vec(&state).map_err(|e| e.to_string())
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        let state: LivingState = serde_json::from_slice(data).map_err(|e| e.to_string())?;
        self.gravity_map = state.gravity_map;
        self.dynamic_buckets = state.dynamic_buckets;
        self.energy_overlay = state.energy_overlay;
        Ok(())
    }

    fn delta(&self, old_data: &[u8]) -> Result<Vec<BucketChange>, String> {
        let old: LivingState = serde_json::from_slice(old_data).map_err(|e| e.to_string())?;

        let mut counts_now: HashMap<u32, usize> = HashMap::new();
        for (&bucket, list) in &self.dynamic_buckets.buckets {
            *counts_now.entry(bucket).or_insert(0) += list.len();
        }

        let mut counts_old: HashMap<u32, usize> = HashMap::new();
        for (&bucket, list) in &old.dynamic_buckets.buckets {
            *counts_old.entry(bucket).or_insert(0) += list.len();
        }

        let all_buckets: HashSet<u32> =
            counts_now.keys().chain(counts_old.keys()).copied().collect();

        let mut changes = Vec::new();
        for bucket in all_buckets {
            let old_count = *counts_old.get(&bucket).unwrap_or(&0);
            let new_count = *counts_now.get(&bucket).unwrap_or(&0);
            if old_count != new_count {
                changes.push(BucketChange {
                    bucket,
                    old_count,
                    new_count,
                    delta: new_count as i64 - old_count as i64,
                });
            }
        }

        changes.sort_by_key(|c| c.bucket);
        Ok(changes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_mesh() -> HybridMesh {
        HybridMesh::new(10, 10)
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let mut mesh = empty_mesh();
        mesh.gravity_map.insert(7, PaperGravity::default());
        mesh.dynamic_buckets.insert(7, 3);

        let bytes = mesh.snapshot().unwrap();
        let mut restored = empty_mesh();
        restored.restore(&bytes).unwrap();

        assert_eq!(restored.gravity_map.len(), 1);
        assert_eq!(restored.dynamic_buckets.get(7), Some(3));
    }

    #[test]
    fn test_delta() {
        let mut mesh = empty_mesh();
        mesh.dynamic_buckets.insert(10, 1);
        mesh.dynamic_buckets.insert(11, 1);
        mesh.dynamic_buckets.insert(20, 2);

        let old = mesh.snapshot().unwrap();

        mesh.dynamic_buckets.insert(12, 1);

        let changes = mesh.delta(&old).unwrap();
        let by_bucket: HashMap<u32, i64> = changes.iter().map(|c| (c.bucket, c.delta)).collect();
        assert_eq!(by_bucket.get(&1), Some(&1));
        assert!(!by_bucket.contains_key(&2));
    }
}
