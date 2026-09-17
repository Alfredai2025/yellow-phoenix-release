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
//! Multi-base dynamic hierarchy: stacked DynamicMeshes.
//!
//! Query starts at fastest base (largest), falls back to slower bases if needed.

use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::dynamic_mesh::DynamicMesh;
use crate::multi_base_crystal::Coord3D;

pub struct DynamicLevel {
    pub base: u32,
    pub mesh: DynamicMesh,
    pub upweight: f32,
}

pub struct MultiBaseDynamic {
    levels: Vec<DynamicLevel>,
}

impl MultiBaseDynamic {
    pub fn new(
        bases: Vec<u32>,
        learning_rate: f32,
        decay_rate: f32,
        ema_alpha: f32,
        pulse_strength: f32,
        hot_threshold: f32,
        cold_threshold: f32,
        upweights: Vec<f32>,
    ) -> Self {
        let mut levels = Vec::with_capacity(bases.len());

        for (i, &base) in bases.iter().enumerate() {
            levels.push(DynamicLevel {
                base,
                mesh: DynamicMesh::new(base, learning_rate, decay_rate, ema_alpha, pulse_strength, hot_threshold, cold_threshold),
                upweight: upweights.get(i).copied().unwrap_or(0.1),
            });
        }

        Self { levels }
    }

    pub fn insert(&mut self, pap: &[u8; 64], entity_id: u64) -> Vec<Coord3D> {
        let mut coords = Vec::with_capacity(self.levels.len());
        for level in self.levels.iter_mut() {
            let coord = level.mesh.insert(pap, entity_id);
            coords.push(coord);
        }
        coords
    }

    pub fn query_hierarchical(
        &self,
        pap: &[u8; 64],
        top_k: usize,
        fast_radius: u16,
        _min_agreement: usize,
    ) -> Vec<(usize, f32, usize)> {
        let mut results = Vec::new();
        let mut seen = Vec::new();

        for level_idx in (0..self.levels.len()).rev() {
            let level = &self.levels[level_idx];
            let coord = crate::multi_base_crystal::MultiBaseCrystal::compute_coord(pap, level.base);
            let spatial = level.mesh.query_spatial(coord, fast_radius);

            for paper in spatial {
                let eid = paper.entity_id as usize;
                if eid >= seen.len() { seen.resize(eid + 1, false); }
                if !seen[eid] {
                    seen[eid] = true;
                    let score = paper.power * (1.0 + level_idx as f32 * 0.1);
                    results.push((eid, score, level_idx));
                }
            }

            if results.len() >= top_k {
                break;
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(top_k);
        results
    }

    pub fn level(&self, idx: usize) -> Option<&DynamicMesh> {
        self.levels.get(idx).map(|l| &l.mesh)
    }

    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pap(seed: u64) -> [u8; 64] {
        let mut pap = [0u8; 64];
        for i in 0..64 {
            pap[i] = ((seed.wrapping_mul(7919 + i as u64)) % 256) as u8;
        }
        pap
    }

    #[test]
    fn hierarchy_insert_all_levels() {
        let mut hierarchy = MultiBaseDynamic::new(
            vec![3u32, 5, 7],
            0.5, 0.99, 0.3, 0.1, 0.5, 0.01,
            vec![0.1, 0.1, 0.1],
        );

        let pap = make_pap(42);
        let coords = hierarchy.insert(&pap, 0);

        assert_eq!(coords.len(), 3);
        assert_eq!(hierarchy.num_levels(), 3);

        for i in 0..3 {
            let mesh = hierarchy.level(i).unwrap();
            assert_eq!(mesh.len(), 1);
        }
    }

    #[test]
    fn hierarchical_query_finds_paper() {
        let mut hierarchy = MultiBaseDynamic::new(
            vec![3u32, 5],
            0.5, 0.99, 0.3, 0.1, 0.1, 0.01,
            vec![0.1, 0.1],
        );

        let seed = make_pap(42);
        // Insert 10 similar papers to heat up the cell
        for i in 0..10 {
            let pap = make_pap(42);
            hierarchy.insert(&pap, i);
        }

        let results = hierarchy.query_hierarchical(&seed, 5, 1, 1);
        assert!(!results.is_empty(), "Hierarchical query should find papers in hot cells");
    }

    #[test]
    fn faster_levels_score_higher() {
        let mut hierarchy = MultiBaseDynamic::new(
            vec![3u32, 5, 7],
            0.5, 0.99, 0.3, 0.1, 0.1, 0.01,
            vec![0.1, 0.1, 0.1],
        );

        let seed = make_pap(42);
        for i in 0..20 {
            let pap = make_pap(42);
            hierarchy.insert(&pap, i);
        }

        let results = hierarchy.query_hierarchical(&seed, 5, 1, 1);
        assert!(!results.is_empty(), "Should find papers with hierarchical query");
        // Fastest level (index 2 = base 7) should have highest score multiplier
        let max_level = results.iter().map(|r| r.2).max().unwrap_or(0);
        assert!(max_level >= 1, "Should find papers in faster levels");
    }
}
