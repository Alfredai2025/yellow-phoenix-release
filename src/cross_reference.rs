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
//! Cross-reference layer: bridges static CrystalMesh and living DynamicMesh.
//!
//! Bidirectional links: crystal node ↔ dynamic cell.
//! Multi-map: one dynamic cell can link to multiple crystal nodes.

use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::crystal::CrystalMesh;
use crate::dynamic_mesh::DynamicMesh;

/// Bidirectional link between crystal node and dynamic cell.
pub struct CrossLink {
    pub crystal_id: u64,
    pub dynamic_key: usize,
    pub base_idx: usize,
}

/// Cross-reference index: maps crystal nodes ↔ dynamic cells.
pub struct CrossReference {
    // crystal_id -> (dynamic_key, base_idx)
    crystal_to_dynamic: Vec<Option<(usize, usize)>>,
    // dynamic_key -> [crystal_ids] (multi-map)
    dynamic_to_crystal: Vec<Vec<u64>>,
}

impl CrossReference {
    pub fn new() -> Self {
        Self {
            crystal_to_dynamic: Vec::new(),
            dynamic_to_crystal: Vec::new(),
        }
    }

    /// Register a link between a crystal node and its dynamic cell.
    pub fn link(&mut self, crystal_id: u64, dynamic_key: usize, base_idx: usize) {
        if (crystal_id as usize) >= self.crystal_to_dynamic.len() {
            self.crystal_to_dynamic.resize((crystal_id as usize) + 1, None);
        }
        self.crystal_to_dynamic[crystal_id as usize] = Some((dynamic_key, base_idx));
        
        if dynamic_key >= self.dynamic_to_crystal.len() {
            self.dynamic_to_crystal.resize(dynamic_key + 1, Vec::new());
        }
        if !self.dynamic_to_crystal[dynamic_key].contains(&crystal_id) {
            self.dynamic_to_crystal[dynamic_key].push(crystal_id);
        }
    }

    /// Get dynamic cell for a crystal node.
    pub fn crystal_to_dynamic(&self, crystal_id: u64) -> Option<(usize, usize)> {
        self.crystal_to_dynamic.get(crystal_id as usize).copied().flatten()
    }

    /// Get first crystal node for a dynamic cell.
    pub fn dynamic_to_crystal(&self, dynamic_key: usize) -> Option<u64> {
        self.dynamic_to_crystal.get(dynamic_key).and_then(|v| v.first().copied())
    }
    
    /// Get all crystal nodes for a dynamic cell.
    pub fn dynamic_to_crystals(&self, dynamic_key: usize) -> &[u64] {
        self.dynamic_to_crystal.get(dynamic_key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Query: dynamic energy finds region, crystal graph expands within it.
    pub fn query_cross(
        &self,
        crystal: &CrystalMesh,
        dynamic: &DynamicMesh,
        _query_pap: &[u8; 64],
        energy_target: f32,
        energy_delta: f32,
        graph_depth: usize,
        top_k: usize,
    ) -> Vec<(u64, f32)> {
        let hot_cells = dynamic.query_energy(energy_target, energy_delta, top_k * 3);
        
        let mut crystal_seeds = Vec::new();
        for (coord, energy, _paper) in &hot_cells {
            let key = dynamic.cell_key(*coord);
            if let Some(crystal_id) = self.dynamic_to_crystal(key) {
                crystal_seeds.push((crystal_id, *energy));
            }
        }

        let mut results = Vec::new();
        let mut seen = Vec::new();
        
        for (seed_id, seed_energy) in crystal_seeds {
            self.expand_crystal(seed_id, graph_depth, seed_energy, crystal, &mut results, &mut seen);
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(top_k);
        results
    }

    fn expand_crystal(
        &self,
        node_id: u64,
        depth: usize,
        base_energy: f32,
        crystal: &CrystalMesh,
        results: &mut Vec<(u64, f32)>,
        seen: &mut Vec<bool>,
    ) {
        let idx = node_id as usize;
        if idx >= seen.len() {
            seen.resize(idx + 1, false);
        }
        if seen[idx] {
            return;
        }
        seen[idx] = true;

        let attenuation = 0.8f32.powi(depth as i32);
        let energy = base_energy * attenuation;
        results.push((node_id, energy));

        if depth == 0 {
            return;
        }

        if let Some(neighbors) = crystal.get_neighbors(node_id) {
            for &nid in &neighbors {
                self.expand_crystal(nid as u64, depth - 1, base_energy, crystal, results, seen);
            }
        }
    }

    pub fn query_trajectory_cross(
        &self,
        crystal: &CrystalMesh,
        dynamic: &DynamicMesh,
        target_energy: f32,
        graph_depth: usize,
        top_k: usize,
    ) -> Vec<(u64, f32, f32)> {
        let trajectory = dynamic.query_trajectory(target_energy, top_k * 3);
        
        let mut results = Vec::new();
        let mut seen = Vec::new();
        
        for (coord, energy, velocity, _paper) in &trajectory {
            let key = dynamic.cell_key(*coord);
            if let Some(crystal_id) = self.dynamic_to_crystal(key) {
                self.expand_crystal_with_velocity(
                    crystal_id, graph_depth, *energy, *velocity, crystal, &mut results, &mut seen
                );
            }
        }

        results.sort_by(|a, b| {
            let score_a = a.1 * a.2;
            let score_b = b.1 * b.2;
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    fn expand_crystal_with_velocity(
        &self,
        node_id: u64,
        depth: usize,
        base_energy: f32,
        base_velocity: f32,
        crystal: &CrystalMesh,
        results: &mut Vec<(u64, f32, f32)>,
        seen: &mut Vec<bool>,
    ) {
        let idx = node_id as usize;
        if idx >= seen.len() {
            seen.resize(idx + 1, false);
        }
        if seen[idx] {
            return;
        }
        seen[idx] = true;

        let attenuation = 0.8f32.powi(depth as i32);
        results.push((node_id, base_energy * attenuation, base_velocity));

        if depth == 0 {
            return;
        }

        if let Some(neighbors) = crystal.get_neighbors(node_id) {
            for &nid in &neighbors {
                self.expand_crystal_with_velocity(nid as u64, depth - 1, base_energy, base_velocity, crystal, results, seen);
            }
        }
    }

    /// Get all crystal nodes linked to hot dynamic cells.
    pub fn hot_crystal_nodes(&self, dynamic: &DynamicMesh) -> Vec<u64> {
        let mut hot = Vec::new();
        for key in dynamic.hot_cell_keys() {
            for &crystal_id in self.dynamic_to_crystals(key) {
                hot.push(crystal_id);
            }
        }
        hot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crystal::CrystalMesh;
    use crate::dynamic_mesh::DynamicMesh;

    fn make_pap(seed: u64) -> [u8; 64] {
        let mut pap = [0u8; 64];
        for i in 0..64 {
            pap[i] = ((seed.wrapping_mul(7919 + i as u64)) % 256) as u8;
        }
        pap
    }

    #[test]
    fn bidirectional_linking() {
        let mut cross = CrossReference::new();
        cross.link(0, 5, 0);
        cross.link(1, 10, 0);
        cross.link(2, 5, 1); // same dynamic key, different base

        assert_eq!(cross.crystal_to_dynamic(0), Some((5, 0)));
        assert_eq!(cross.crystal_to_dynamic(1), Some((10, 0)));
        assert_eq!(cross.dynamic_to_crystal(5), Some(0)); // first link
        assert_eq!(cross.dynamic_to_crystals(5), &[0, 2]); // both links
        assert_eq!(cross.dynamic_to_crystal(10), Some(1));
    }

    #[test]
    fn cross_query_finds_hot_region() {
        let mut crystal = CrystalMesh::new();
        let mut dynamic = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
        let mut cross = CrossReference::new();

        for i in 0..20 {
            let p = make_pap(i as u64);
            let cid = i as u64;
            crystal.insert(cid, &p);
            let coord = dynamic.insert(&p, cid);
            let key = dynamic.cell_key(coord);
            cross.link(cid, key, 0);
        }

        let results = cross.query_cross(&crystal, &dynamic, &make_pap(0), 5.0, 10.0, 2, 10);
        assert!(!results.is_empty(), "Cross query should find hot crystal nodes");
    }

    #[test]
    fn trajectory_cross_query() {
        let mut crystal = CrystalMesh::new();
        let mut dynamic = DynamicMesh::new(5, 0.5, 0.99, 0.8, 0.1, 0.1, 0.01);
        let mut cross = CrossReference::new();

        for i in 0..30 {
            let p = make_pap(i as u64);
            let cid = i as u64;
            crystal.insert(cid, &p);
            let coord = dynamic.insert(&p, cid);
            let key = dynamic.cell_key(coord);
            cross.link(cid, key, 0);
        }

        let results = cross.query_trajectory_cross(&crystal, &dynamic, 10.0, 2, 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn hot_nodes_mapping() {
        let mut crystal = CrystalMesh::new();
        let mut dynamic = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
        let mut cross = CrossReference::new();

        for i in 0..10 {
            let p = make_pap(i as u64);
            let cid = i as u64;
            crystal.insert(cid, &p);
            let coord = dynamic.insert(&p, cid);
            let key = dynamic.cell_key(coord);
            cross.link(cid, key, 0);
        }

        let hot = cross.hot_crystal_nodes(&dynamic);
        assert!(!hot.is_empty(), "Should find hot crystal nodes linked to dynamic cells");
    }
}
