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
//! Hybrid intelligent router: exact → multi-base dynamic hierarchy → crystal graph → cascade.
//!
//! Architecture:
//! 1. Direct hash: O(1) exact lookup
//! 2. MultiBaseDynamic: hierarchical living field (fastest base first)
//! 3. CrystalMesh: static graph expansion from dynamic winners
//! 4. CascadeIndex: brute-force geometric fallback

use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::cascade_index::CascadeIndex;
use crate::crystal::CrystalMesh;
use crate::direct_hash::DirectIndex;
use crate::distance::pap_distance;
use crate::multi_base_dynamic::MultiBaseDynamic;
use crate::types::multivector::BinaryMultivector;

pub struct IntelligentRouter {
    direct: DirectIndex,
    multi_dynamic: MultiBaseDynamic,
    crystal: CrystalMesh,
    cascade: CascadeIndex,
}

impl IntelligentRouter {
    pub fn new(
        direct: DirectIndex,
        multi_dynamic: MultiBaseDynamic,
        crystal: CrystalMesh,
        cascade: CascadeIndex,
    ) -> Self {
        Self {
            direct,
            multi_dynamic,
            crystal,
            cascade,
        }
    }

    /// Store a paper across all layers.
    pub fn add(&mut self, pap: &[u8; 64], binary: BinaryMultivector, pq: &[f32], exact: &[f32]) {
        let idx = self.cascade.len() as u64;

        // Layer 1: Exact hash
        self.direct.insert(idx, pap);

        // Layer 2: Multi-base dynamic hierarchy
        self.multi_dynamic.insert(pap, idx);

        // Layer 3: Crystal mesh (static graph)
        self.crystal.insert(idx, pap);

        // Layer 4: Cascade
        self.cascade.add(binary, pq, exact);
    }

    /// Full hybrid query: exact → multi-base dynamic → crystal graph → cascade.
    pub fn query(
        &self,
        pap: &[u8; 64],
        binary: &BinaryMultivector,
        float: &[f32],
        top_k: usize,
        dynamic_radius: u16,
        graph_depth: usize,
    ) -> Vec<(usize, f32)> {
        // === STAGE 1: EXACT DIRECT HASH ===
        if let Some(id) = self.direct.query(pap) {
            let idx = id as usize;
            let d = pap_distance(binary, &self.cascade.lsh.get_prototype(idx));
            return vec![(idx, d)];
        }

        // === STAGE 2: MULTI-BASE DYNAMIC HIERARCHY ===
        let dynamic_results = self.multi_dynamic.query_hierarchical(pap, top_k * 3, dynamic_radius, 1);

        // === STAGE 3: CRYSTAL GRAPH EXPANSION ===
        let mut candidates = Vec::new();
        let mut seen = Vec::new();

        for (entity_id, _score, _level_idx) in &dynamic_results {
            self.expand_graph(*entity_id as u64, graph_depth, &mut candidates, &mut seen);
        }

        // If dynamic found nothing, try crystal direct query
        if candidates.is_empty() {
            if let Some(id) = self.crystal.query_crystal(pap) {
                self.expand_graph(id, graph_depth, &mut candidates, &mut seen);
            }
        }

        // Rank candidates with binary PAP
        if !candidates.is_empty() {
            let mut scored: Vec<(usize, f32)> = candidates
                .iter()
                .map(|&idx| {
                    let d = pap_distance(binary, &self.cascade.lsh.get_prototype(idx));
                    (idx, d)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            scored.truncate(top_k);
            return scored;
        }

        // === STAGE 4: CASCADE FALLBACK ===
        self.cascade.query(binary, float, top_k, 1, 100, 10)
    }

    /// Fast query: skip crystal graph, use multi-base dynamic only.
    pub fn query_fast(
        &self,
        pap: &[u8; 64],
        binary: &BinaryMultivector,
        float: &[f32],
        top_k: usize,
        dynamic_radius: u16,
    ) -> Vec<(usize, f32)> {
        if let Some(id) = self.direct.query(pap) {
            let idx = id as usize;
            let d = pap_distance(binary, &self.cascade.lsh.get_prototype(idx));
            return vec![(idx, d)];
        }

        let dynamic_results = self.multi_dynamic.query_hierarchical(pap, top_k * 2, dynamic_radius, 1);

        if !dynamic_results.is_empty() {
            let mut scored: Vec<(usize, f32)> = dynamic_results
                .iter()
                .take(top_k)
                .map(|(entity_id, _score, _level_idx)| {
                    let idx = *entity_id as usize;
                    let d = pap_distance(binary, &self.cascade.lsh.get_prototype(idx));
                    (idx, d)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            return scored;
        }

        self.cascade.query(binary, float, top_k, 1, 100, 10)
    }

    fn expand_graph(
        &self,
        entity_id: u64,
        depth: usize,
        candidates: &mut Vec<usize>,
        seen: &mut Vec<bool>,
    ) {
        let idx = entity_id as usize;
        if idx >= seen.len() {
            seen.resize(idx + 1, false);
        }
        if seen[idx] {
            return;
        }
        seen[idx] = true;
        candidates.push(idx);

        if depth == 0 {
            return;
        }

        if let Some(neighbors) = self.crystal.get_neighbors(entity_id) {
            for nid in neighbors {
                self.expand_graph(nid as u64, depth - 1, candidates, seen);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.cascade.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cascade.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_base_dynamic::MultiBaseDynamic;

    fn make_pap(seed: u64) -> [u8; 64] {
        let mut pap = [0u8; 64];
        for i in 0..64 {
            pap[i] = ((seed.wrapping_mul(7919 + i as u64)) % 256) as u8;
        }
        pap
    }

    fn near_pap(original: &[u8; 64], flips: usize, seed: u64) -> [u8; 64] {
        let mut pap = *original;
        for i in 0..flips {
            let pos = ((seed.wrapping_mul(104729 + i as u64)) % 64) as usize;
            pap[pos] = pap[pos].wrapping_add(1);
        }
        pap
    }

    fn pap_to_binary(pap: &[u8; 64]) -> BinaryMultivector {
        let mut chunks = [0u64; 2];
        for i in 0..8 {
            chunks[0] |= (pap[i] as u64) << (i * 8);
        }
        for i in 0..8 {
            chunks[1] |= (pap[i + 8] as u64) << (i * 8);
        }
        BinaryMultivector(chunks)
    }

    fn train_pq(cascade: &mut CascadeIndex) {
        let mut train = Vec::with_capacity(8000);
        for i in 0..1000 {
            for d in 0..8 {
                train.push(((i * 17 + d * 31) % 100) as f32 / 100.0);
            }
        }
        cascade.train_pq(&train, 1000);
    }

    #[test]
    fn hybrid_router_narrows_candidates() {
        let cascade = CascadeIndex::new(64, 1, 2, 4, 8);
        let direct = DirectIndex::new();
        let multi_dynamic = MultiBaseDynamic::new(
            vec![3u32, 5, 7],
            0.5, 0.99, 0.3, 0.1, 0.1, 0.01,
            vec![0.1, 0.1, 0.1],
        );
        let crystal = CrystalMesh::new();
        let mut router = IntelligentRouter::new(direct, multi_dynamic, crystal, cascade);
        train_pq(&mut router.cascade);

        let seed_pap = make_pap(42);

        // 50 cluster papers
        for i in 0..50 {
            let pap = near_pap(&seed_pap, 2, i as u64 * 17);
            let binary = pap_to_binary(&pap);
            let vec: Vec<f32> = (0..8).map(|d| ((i * 17 + d * 31) % 100) as f32 / 100.0).collect();
            router.add(&pap, binary, &vec, &vec);
        }

        // 50 noise papers
        for i in 0..50 {
            let pap = make_pap(i as u64 * 997);
            let binary = pap_to_binary(&pap);
            let vec: Vec<f32> = (0..8).map(|d| ((i * 997 + d * 31) % 100) as f32 / 100.0).collect();
            router.add(&pap, binary, &vec, &vec);
        }

        let q_binary = pap_to_binary(&seed_pap);
        let q_float = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let results = router.query(&seed_pap, &q_binary, &q_float, 5, 1, 2);

        assert!(!results.is_empty());
        assert!(results[0].1 < 0.3, "Top result should be cluster member, got distance {:.3}", results[0].1);
    }

    #[test]
    fn fast_query_uses_dynamic_only() {
        let cascade = CascadeIndex::new(64, 1, 2, 4, 8);
        let direct = DirectIndex::new();
        let multi_dynamic = MultiBaseDynamic::new(
            vec![3u32, 5, 7],
            0.5, 0.99, 0.3, 0.1, 0.1, 0.01,
            vec![0.1, 0.1, 0.1],
        );
        let crystal = CrystalMesh::new();
        let mut router = IntelligentRouter::new(direct, multi_dynamic, crystal, cascade);
        train_pq(&mut router.cascade);

        let seed = make_pap(77);

        for i in 0..30 {
            let pap = near_pap(&seed, 2, i as u64);
            let binary = pap_to_binary(&pap);
            let vec = vec![0.5f32; 8];
            router.add(&pap, binary, &vec, &vec);
        }

        let q_binary = pap_to_binary(&seed);
        let q_float = vec![0.5f32; 8];
        let results = router.query_fast(&seed, &q_binary, &q_float, 5, 1);

        assert!(!results.is_empty());
    }

    #[test]
    fn exact_hit_bypasses_all_layers() {
        let cascade = CascadeIndex::new(64, 1, 2, 4, 8);
        let direct = DirectIndex::new();
        let multi_dynamic = MultiBaseDynamic::new(
            vec![3u32, 5],
            0.5, 0.99, 0.3, 0.1, 0.1, 0.01,
            vec![0.1, 0.1],
        );
        let crystal = CrystalMesh::new();
        let mut router = IntelligentRouter::new(direct, multi_dynamic, crystal, cascade);
        train_pq(&mut router.cascade);

        let pap = make_pap(1);
        let binary = pap_to_binary(&pap);
        let vec = vec![1.0f32; 8];
        router.add(&pap, binary, &vec, &vec);

        let q_binary = pap_to_binary(&pap);
        let results = router.query(&pap, &q_binary, &vec, 1, 1, 1);

        assert_eq!(results.len(), 1);
        assert!(results[0].1 < 1e-6, "Exact hit should have zero distance");
    }

    #[test]
    fn multi_base_agreement_correlates_with_pap() {
        let bases = vec![3u32, 5, 7, 11];
        let seed = make_pap(123);

        let identical = seed;
        let near = near_pap(&seed, 4, 1);
        let medium = near_pap(&seed, 16, 2);
        let far = near_pap(&seed, 50, 3);

        let pairs = [
            ("identical", &seed, &identical),
            ("near", &seed, &near),
            ("medium", &seed, &medium),
            ("far", &seed, &far),
        ];

        for (name, a, b) in pairs {
            let mut agreement = 0;
            for &base in &bases {
                let ca = crate::multi_base_crystal::MultiBaseCrystal::compute_coord(a, base);
                let cb = crate::multi_base_crystal::MultiBaseCrystal::compute_coord(b, base);

                let dx = crate::multi_base_crystal::MultiBaseCrystal::circular_dist(ca.0, cb.0, base);
                let dy = crate::multi_base_crystal::MultiBaseCrystal::circular_dist(ca.1, cb.1, base);
                let dz = crate::multi_base_crystal::MultiBaseCrystal::circular_dist(ca.2, cb.2, base);

                if dx <= 1 && dy <= 1 && dz <= 1 {
                    agreement += 1;
                }
            }

            if name == "identical" {
                assert_eq!(agreement, 4, "Identical PAPs agree on all bases");
            } else if name == "near" {
                assert!(agreement >= 2, "Near PAPs agree on 2+ bases, got {}", agreement);
            } else if name == "far" {
                assert!(agreement <= 2, "Far PAPs agree on 0-2 bases, got {}", agreement);
            }
        }
    }

    #[test]
    fn statistical_correlation_200_vectors() {
        let bases = vec![3u32, 5, 7, 11];
        let seed = make_pap(42);

        let mut cluster_paps = Vec::with_capacity(100);
        let mut noise_paps = Vec::with_capacity(100);

        for i in 0..100 {
            cluster_paps.push(near_pap(&seed, 5, i as u64));
        }
        for i in 0..100 {
            noise_paps.push(make_pap(i as u64 * 104729));
        }

        let mut tp = 0usize;
        let mut fp = 0usize;

        for pap in &cluster_paps {
            let mut agree = 0;
            for &base in &bases {
                let coord = crate::multi_base_crystal::MultiBaseCrystal::compute_coord(pap, base);
                let seed_coord = crate::multi_base_crystal::MultiBaseCrystal::compute_coord(&seed, base);
                let dx = crate::multi_base_crystal::MultiBaseCrystal::circular_dist(coord.0, seed_coord.0, base);
                let dy = crate::multi_base_crystal::MultiBaseCrystal::circular_dist(coord.1, seed_coord.1, base);
                let dz = crate::multi_base_crystal::MultiBaseCrystal::circular_dist(coord.2, seed_coord.2, base);
                if dx <= 1 && dy <= 1 && dz <= 1 {
                    agree += 1;
                }
            }
            if agree >= 2 {
                tp += 1;
            }
        }

        for pap in &noise_paps {
            let mut agree = 0;
            for &base in &bases {
                let coord = crate::multi_base_crystal::MultiBaseCrystal::compute_coord(pap, base);
                let seed_coord = crate::multi_base_crystal::MultiBaseCrystal::compute_coord(&seed, base);
                let dx = crate::multi_base_crystal::MultiBaseCrystal::circular_dist(coord.0, seed_coord.0, base);
                let dy = crate::multi_base_crystal::MultiBaseCrystal::circular_dist(coord.1, seed_coord.1, base);
                let dz = crate::multi_base_crystal::MultiBaseCrystal::circular_dist(coord.2, seed_coord.2, base);
                if dx <= 1 && dy <= 1 && dz <= 1 {
                    agree += 1;
                }
            }
            if agree >= 2 {
                fp += 1;
            }
        }

        let precision = tp as f32 / (tp + fp).max(1) as f32;
        let recall = tp as f32 / cluster_paps.len() as f32;
        let f1 = 2.0 * precision * recall / (precision + recall).max(1e-6);

        assert!(f1 > 0.50, "Multi-base agreement must predict cluster membership (F1={:.2})", f1);
    }
}
