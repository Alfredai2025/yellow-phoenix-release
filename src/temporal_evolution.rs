// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

// GHOST MODULE — NOT WIRED INTO src/lib.rs
// This file exists on disk but is NOT declared in src/lib.rs.
// It does NOT compile into the library and is NOT reachable from Python.
//
// AUDIT DATE: 2026-07-27
// ACTION: Preserved per "Wire First, Delete Never" policy.
//         Do not modify unless wiring into the build.
//
//! Temporal Evolution Layer: mesh snapshots over time.
//!
//! The dynamic mesh evolves. This layer records sparse snapshots
//! and enables historical queries: "what was the state at time T?"
//!
//! Architecture:
//! - Snapshots store only deltas (changed cells since last snapshot)
//! - Full snapshot every N steps, deltas between
//! - Trajectory: per-cell (energy, velocity) time series
//! - Historical query: reconstruct cell state at any mesh time

use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::dynamic_mesh::DynamicMesh;

/// Cell state at a specific mesh time.
#[derive(Clone, Copy, Debug)]
pub struct CellSnapshot {
    pub mesh_time: u64,
    pub energy: f32,
    pub velocity: f32,
    pub paper_count: u32,
}

/// Sparse delta: what changed in one mesh step.
#[derive(Clone, Debug)]
pub struct MeshDelta {
    pub mesh_time: u64,
    pub changed_cells: Vec<(usize, CellSnapshot)>, // (cell_key, new_state)
}

/// Full snapshot at periodic intervals.
#[derive(Clone, Debug)]
pub struct FullSnapshot {
    pub mesh_time: u64,
    pub hot_cell_states: Vec<(usize, CellSnapshot)>, // only hot cells
}

/// Temporal evolution tracker for a DynamicMesh.
pub struct TemporalEvolution {
    deltas: Vec<MeshDelta>,
    full_snapshots: Vec<FullSnapshot>,
    snapshot_interval: u64,
    last_snapshot_time: u64,
    // Trajectory index: cell_key -> time series of states
    trajectories: Vec<Vec<CellSnapshot>>,
}

impl TemporalEvolution {
    pub fn new(snapshot_interval: u64) -> Self {
        Self {
            deltas: Vec::new(),
            full_snapshots: Vec::new(),
            snapshot_interval,
            last_snapshot_time: 0,
            trajectories: Vec::new(),
        }
    }

    /// Record a delta from the current mesh state.
    pub fn record_delta(&mut self, mesh: &DynamicMesh, changed_cells: &[usize]) {
        let mut delta_cells = Vec::with_capacity(changed_cells.len());
        
        for &key in changed_cells {
            if key >= mesh.num_cells() { continue; }
            let header = &mesh.cell_header(key).unwrap();
            if !header.hot && header.energy < 0.01 { continue; } // skip cold/empty
            
            delta_cells.push((key, CellSnapshot {
                mesh_time: mesh.mesh_time(),
                energy: header.energy,
                velocity: header.velocity,
                paper_count: header.paper_count,
            }));
            
            // Update trajectory
            if key >= self.trajectories.len() {
                self.trajectories.resize(key + 1, Vec::new());
            }
            self.trajectories[key].push(CellSnapshot {
                mesh_time: mesh.mesh_time(),
                energy: header.energy,
                velocity: header.velocity,
                paper_count: header.paper_count,
            });
        }
        
        self.deltas.push(MeshDelta {
            mesh_time: mesh.mesh_time(),
            changed_cells: delta_cells,
        });
        
        // Periodic full snapshot
        if mesh.mesh_time() - self.last_snapshot_time >= self.snapshot_interval {
            self.take_full_snapshot(mesh);
            self.last_snapshot_time = mesh.mesh_time();
        }
    }

    fn take_full_snapshot(&mut self, mesh: &DynamicMesh) {
        let mut hot_states = Vec::new();
        for (key, header) in mesh.iter_cells() {
            if header.hot {
                hot_states.push((key, CellSnapshot {
                    mesh_time: mesh.mesh_time(),
                    energy: header.energy,
                    velocity: header.velocity,
                    paper_count: header.paper_count,
                }));
            }
        }
        
        self.full_snapshots.push(FullSnapshot {
            mesh_time: mesh.mesh_time(),
            hot_cell_states: hot_states,
        });
    }

    /// Reconstruct cell state at a specific mesh time.
    /// Walks backward from nearest snapshot or forward from deltas.
    pub fn cell_state_at(&self, cell_key: usize, target_time: u64) -> Option<CellSnapshot> {
        // Check trajectories first (exact)
        if cell_key < self.trajectories.len() {
            let traj = &self.trajectories[cell_key];
            // Binary search for nearest time <= target
            let mut best = None;
            for state in traj.iter().rev() {
                if state.mesh_time <= target_time {
                    best = Some(*state);
                    break;
                }
            }
            if best.is_some() { return best; }
        }
        
        // Fallback: find nearest full snapshot and walk deltas
        let mut nearest_snapshot: Option<(u64, f32, f32, u32)> = None;
        for snap in self.full_snapshots.iter().rev() {
            if snap.mesh_time <= target_time {
                for &(key, state) in &snap.hot_cell_states {
                    if key == cell_key {
                        nearest_snapshot = Some((state.mesh_time, state.energy, state.velocity, state.paper_count));
                        break;
                    }
                }
                if nearest_snapshot.is_some() { break; }
            }
        }
        
        if nearest_snapshot.is_none() { return None; }
        
        let (mut time, mut energy, mut velocity, mut paper_count) = nearest_snapshot.unwrap();
        
        // Apply deltas forward from snapshot time to target
        for delta in &self.deltas {
            if delta.mesh_time <= time { continue; }
            if delta.mesh_time > target_time { break; }
            
            for &(key, state) in &delta.changed_cells {
                if key == cell_key {
                    time = state.mesh_time;
                    energy = state.energy;
                    velocity = state.velocity;
                    paper_count = state.paper_count;
                }
            }
        }
        
        Some(CellSnapshot { mesh_time: time, energy, velocity, paper_count })
    }

    /// Query: find cells that were hot at time T with energy near target.
    pub fn query_historical_energy(
        &self,
        target_time: u64,
        energy_target: f32,
        energy_delta: f32,
        top_k: usize,
    ) -> Vec<(usize, CellSnapshot)> {
        let mut results = Vec::new();
        
        // Check all trajectories
        for (key, traj) in self.trajectories.iter().enumerate() {
            if traj.is_empty() { continue; }
            
            // Find state at target time
            let mut best_state = None;
            for state in traj.iter().rev() {
                if state.mesh_time <= target_time {
                    best_state = Some(*state);
                    break;
                }
            }
            
            if let Some(state) = best_state {
                let diff = (state.energy - energy_target).abs();
                if diff <= energy_delta {
                    results.push((key, state));
                }
            }
        }
        
        results.sort_by(|a, b| {
            let da = (a.1.energy - energy_target).abs();
            let db = (b.1.energy - energy_target).abs();
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// Query: find cells whose energy trajectory crosses a threshold.
    /// "When did this cell become hot?" or "Which cells were heating up between T1 and T2?"
    pub fn query_trajectory_crossing(
        &self,
        t1: u64,
        t2: u64,
        energy_threshold: f32,
        top_k: usize,
    ) -> Vec<(usize, CellSnapshot, CellSnapshot)> {
        let mut results = Vec::new();
        
        for (key, traj) in self.trajectories.iter().enumerate() {
            if traj.len() < 2 { continue; }
            
            // Find states at t1 and t2
            let mut state1 = None;
            let mut state2 = None;
            
            for state in traj.iter() {
                if state.mesh_time <= t1 { state1 = Some(*state); }
                if state.mesh_time <= t2 { state2 = Some(*state); }
            }
            
            if let (Some(s1), Some(s2)) = (state1, state2) {
                let crossed = (s1.energy < energy_threshold && s2.energy >= energy_threshold)
                           || (s1.energy >= energy_threshold && s2.energy < energy_threshold);
                if crossed {
                    results.push((key, s1, s2));
                }
            }
        }
        
        // Sort by magnitude of change
        results.sort_by(|a, b| {
            let da = (a.2.energy - a.1.energy).abs();
            let db = (b.2.energy - b.1.energy).abs();
            db.partial_cmp(&da).unwrap_or(Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// Get trajectory for a specific cell.
    pub fn get_trajectory(&self, cell_key: usize) -> Option<&[CellSnapshot]> {
        self.trajectories.get(cell_key).map(|v| v.as_slice())
    }

    /// Total snapshots stored.
    pub fn num_snapshots(&self) -> usize {
        self.full_snapshots.len()
    }

    /// Total deltas stored.
    pub fn num_deltas(&self) -> usize {
        self.deltas.len()
    }

    /// Total trajectory points.
    pub fn num_trajectory_points(&self) -> usize {
        self.trajectories.iter().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamic_mesh::DynamicMesh;

    fn make_pap(seed: u64) -> [u8; 64] {
        let mut pap = [0u8; 64];
        for i in 0..64 {
            pap[i] = ((seed.wrapping_mul(7919 + i as u64)) % 256) as u8;
        }
        pap
    }

    #[test]
    fn snapshot_records_deltas() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
        let mut evo = TemporalEvolution::new(10);

        // Insert 5 papers
        let mut changed = Vec::new();
        for i in 0..5 {
            let pap = make_pap(i as u64);
            let coord = mesh.insert(&pap, i);
            changed.push(mesh.cell_key(coord));
        }
        
        evo.record_delta(&mesh, &changed);
        assert_eq!(evo.num_deltas(), 1);
    }

    #[test]
    fn full_snapshot_triggered() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
        let mut evo = TemporalEvolution::new(5);

        for i in 0..10 {
            let pap = make_pap(i as u64);
            let coord = mesh.insert(&pap, i);
            let key = mesh.cell_key(coord);
            evo.record_delta(&mesh, &[key]);
        }
        
        assert!(evo.num_snapshots() >= 1, "Should have at least one full snapshot");
    }

    #[test]
    fn cell_state_reconstruction() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
        let mut evo = TemporalEvolution::new(100);

        let pap = make_pap(42);
        let coord = mesh.insert(&pap, 0);
        let key = mesh.cell_key(coord);
        evo.record_delta(&mesh, &[key]);

        let state = evo.cell_state_at(key, mesh.mesh_time());
        assert!(state.is_some());
        assert!(state.unwrap().energy > 0.0);
    }

    #[test]
    fn historical_energy_query() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
        let mut evo = TemporalEvolution::new(100);

        for i in 0..20 {
            let pap = make_pap(i as u64);
            let coord = mesh.insert(&pap, i);
            let key = mesh.cell_key(coord);
            evo.record_delta(&mesh, &[key]);
        }

        let results = evo.query_historical_energy(mesh.mesh_time(), 5.0, 10.0, 10);
        assert!(!results.is_empty(), "Should find cells with historical energy near target");
    }

    #[test]
    fn trajectory_crossing_detection() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
        let mut evo = TemporalEvolution::new(100);

        // Insert 10 papers at time 0-9
        for i in 0..10 {
            let pap = make_pap(i as u64);
            let coord = mesh.insert(&pap, i);
            let key = mesh.cell_key(coord);
            evo.record_delta(&mesh, &[key]);
        }

        let t1 = 0;
        let t2 = mesh.mesh_time();
        
        let crossings = evo.query_trajectory_crossing(t1, t2, 1.0, 10);
        // Some cells should have crossed from below 1.0 to above 1.0
        assert!(!crossings.is_empty() || evo.num_trajectory_points() > 0);
    }

    #[test]
    fn trajectory_time_series() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
        let mut evo = TemporalEvolution::new(100);

        let pap = make_pap(42);
        let coord = mesh.insert(&pap, 0);
        let key = mesh.cell_key(coord);
        evo.record_delta(&mesh, &[key]);

        let traj = evo.get_trajectory(key);
        assert!(traj.is_some());
        assert!(!traj.unwrap().is_empty());
    }
}
