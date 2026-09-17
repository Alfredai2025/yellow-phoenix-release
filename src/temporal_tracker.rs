// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Temporal Tracker — Yellow Phoenix v3.6 Phase 4
//! Disk-persistent history ring + trending analysis.

use crate::hybrid_mesh::HybridMesh;
use crate::snapshot_trait::{BucketChange, MeshSnapshot};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrendEntry {
    pub bucket: u32,
    pub delta: i64,
    pub trend: String,
}

#[derive(Debug, Clone)]
pub struct TemporalTracker {
    pub ring: VecDeque<(u64, String)>, // (timestamp, snapshot path)
    pub capacity: usize,
}

impl TemporalTracker {
    pub fn new() -> Self {
        let mut tracker = Self {
            ring: VecDeque::with_capacity(24),
            capacity: 24,
        };
        tracker.load_manifest();
        tracker
    }

    pub fn record(&mut self, ts: u64, snap_path: String) {
        if self.ring.len() >= self.capacity {
            self.ring.pop_front();
        }
        self.ring.push_back((ts, snap_path));
        self.save_manifest();
    }

    pub fn state_at(&self, timestamp: u64) -> Option<String> {
        self.ring
            .iter()
            .filter(|(ts, _)| *ts <= timestamp)
            .last()
            .map(|(_, path)| path.clone())
    }

    pub fn trending(&self, mesh: &HybridMesh, since: u64) -> Result<Vec<TrendEntry>, String> {
        let path = self.state_at(since);
        if path.is_none() {
            return Ok(Vec::new());
        }
        let old_bytes = std::fs::read(path.unwrap()).map_err(|e| e.to_string())?;
        let changes = mesh.delta(&old_bytes)?;
        let mut entries = Vec::new();
        for c in changes {
            let trend = match c.delta {
                d if d > 5 => "growing",
                d if d < -5 => "shrinking",
                _ => "stable",
            }
            .to_string();
            entries.push(TrendEntry {
                bucket: c.bucket,
                delta: c.delta,
                trend,
            });
        }
        entries.sort_by(|a, b| b.delta.abs().cmp(&a.delta.abs()));
        Ok(entries)
    }

    pub fn history(&self) -> Vec<(u64, usize)> {
        self.ring
            .iter()
            .map(|(ts, path)| {
                let len = std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0);
                (*ts, len)
            })
            .collect()
    }

    fn save_manifest(&self) {
        let manifest: Vec<(u64, String)> = self.ring.iter().cloned().collect();
        let _ = std::fs::create_dir_all("data");
        let _ = std::fs::write(
            "data/temporal_manifest.json",
            serde_json::to_string(&manifest).unwrap_or_default(),
        );
    }

    fn load_manifest(&mut self) {
        if let Ok(data) = std::fs::read_to_string("data/temporal_manifest.json") {
            if let Ok(manifest) = serde_json::from_str::<Vec<(u64, String)>>(&data) {
                self.ring = manifest.into_iter().collect();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_mesh() -> HybridMesh {
        HybridMesh::new(10, 10)
    }

    #[test]
    fn test_record_and_history() {
        let mut tracker = TemporalTracker::new();
        tracker.record(1000, "/tmp/snap1.json".to_string());
        assert_eq!(tracker.history().len(), 1);
    }

    #[test]
    fn test_state_at() {
        let mut tracker = TemporalTracker::new();
        tracker.record(1000, "/tmp/snap1.json".to_string());
        tracker.record(2000, "/tmp/snap2.json".to_string());
        assert_eq!(tracker.state_at(1500), Some("/tmp/snap1.json".to_string()));
        assert_eq!(tracker.state_at(2500), Some("/tmp/snap2.json".to_string()));
        assert_eq!(tracker.state_at(500), None);
    }
}
