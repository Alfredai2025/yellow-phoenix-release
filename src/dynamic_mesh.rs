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
use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::multi_base_crystal::{Coord3D, MultiBaseCrystal};

/// Cell metadata — always in RAM.
pub struct CellHeader {
    pub energy: f32,
    pub velocity: f32,
    pub last_update: u64,
    pub paper_count: u32,
    pub last_paper_id: u64,
    pub hot: bool,
    pub column_energy: f32,
}

/// Paper record in the dynamic field.
#[derive(Clone, Copy, Debug)]
pub struct DynamicPaper {
    pub entity_id: u64,
    pub power: f32,
    pub mesh_time: u64,
    pub column_depth: u16,
}

/// Full cell state — hot cells only.
pub struct HotCell {
    pub papers: Vec<DynamicPaper>,
    pub pulse_energy: f32,
    pub thread_ids: Vec<u64>,
}

/// 4D dynamic mesh: (x,y,z,energy) with hot/cold, columns, pulses, threads.
/// 
/// Queries are non-mutating. Decay is applied lazily during insertion.
/// Effective energy is computed on-the-fly for queries.
pub struct DynamicMesh {
    base: u32,
    dim: usize,
    cells: Vec<CellHeader>,
    hot_cells: Vec<Option<HotCell>>,
    mesh_time: u64,
    learning_rate: f32,
    decay_rate: f32,
    ema_alpha: f32,
    pulse_strength: f32,
    hot_threshold: f32,
    cold_threshold: f32,
}

impl DynamicMesh {
    pub fn new(
        base: u32,
        learning_rate: f32,
        decay_rate: f32,
        ema_alpha: f32,
        pulse_strength: f32,
        hot_threshold: f32,
        cold_threshold: f32,
    ) -> Self {
        let dim = base as usize;
        let num_cells = dim * dim * dim;
        
        let mut cells = Vec::with_capacity(num_cells);
        for _ in 0..num_cells {
            cells.push(CellHeader {
                energy: 0.0,
                velocity: 0.0,
                last_update: 0,
                paper_count: 0,
                last_paper_id: 0,
                hot: false,
                column_energy: 0.0,
            });
        }
        
        let mut hot_cells = Vec::with_capacity(num_cells);
        for _ in 0..num_cells {
            hot_cells.push(None);
        }
        
        Self {
            base, dim, cells, hot_cells,
            mesh_time: 0,
            learning_rate, decay_rate, ema_alpha,
            pulse_strength, hot_threshold, cold_threshold,
        }
    }

    #[inline]
    pub fn cell_key(&self, coord: Coord3D) -> usize {
        let (x, y, z) = coord;
        (x as usize) * self.dim * self.dim + (y as usize) * self.dim + (z as usize)
    }

    /// USER HOOK: Replace with your power formula.
    pub fn compute_power(pap: &[u8; 64]) -> f32 {
        let mut ones = 0u32;
        for &byte in pap.iter() { ones += byte.count_ones(); }
        let p = ones as f32 / 512.0;
        let entropy = -(p * p.log2() + (1.0 - p) * (1.0 - p).log2());
        if entropy.is_nan() { 0.0 } else { entropy }
    }

    /// Effective energy at current mesh time (non-mutating).
    #[inline]
    pub fn effective_energy(&self, key: usize) -> f32 {
        let dt = self.mesh_time - self.cells[key].last_update;
        if dt > 0 {
            self.cells[key].energy * self.decay_rate.powi(dt as i32)
        } else {
            self.cells[key].energy
        }
    }

    /// Effective column energy at current mesh time (non-mutating).
    #[inline]
    fn effective_column_energy(&self, key: usize) -> f32 {
        let dt = self.mesh_time - self.cells[key].last_update;
        if dt > 0 {
            self.cells[key].column_energy * self.decay_rate.powi(dt as i32)
        } else {
            self.cells[key].column_energy
        }
    }

    /// Apply decay and update hot/cold status. Called during insertion only.
    fn apply_decay(&mut self, key: usize) {
        let dt = self.mesh_time - self.cells[key].last_update;
        if dt > 0 {
            self.cells[key].energy *= self.decay_rate.powi(dt as i32);
            self.cells[key].column_energy *= self.decay_rate.powi(dt as i32);
            if self.cells[key].energy < self.cold_threshold && self.cells[key].hot {
                self.cells[key].hot = false;
                self.hot_cells[key] = None;
            }
        }
        self.cells[key].last_update = self.mesh_time;
    }

    pub fn insert(&mut self, pap: &[u8; 64], entity_id: u64) -> Coord3D {
        let coord = MultiBaseCrystal::compute_coord(pap, self.base);
        let power = Self::compute_power(pap);
        let key = self.cell_key(coord);
        let z = coord.2 as usize;
        
        self.apply_decay(key);
        
        let header = &mut self.cells[key];
        header.energy += power * self.learning_rate;
        header.velocity = self.ema_alpha * power + (1.0 - self.ema_alpha) * header.velocity;
        header.paper_count += 1;
        header.last_paper_id = entity_id;
        
        if !header.hot && header.energy >= self.hot_threshold {
            header.hot = true;
            self.hot_cells[key] = Some(HotCell {
                papers: Vec::new(),
                pulse_energy: 0.0,
                thread_ids: Vec::new(),
            });
        }
        
        if header.hot {
            if let Some(hot) = &mut self.hot_cells[key] {
                hot.papers.push(DynamicPaper {
                    entity_id, power,
                    mesh_time: self.mesh_time,
                    column_depth: coord.2,
                });
            }
        }
        
        // Column update: same x,y, all z
        let col_base = key - z;
        for dz in 0..self.dim {
            let ck = col_base + dz;
            if ck < self.cells.len() {
                self.cells[ck].column_energy += power * self.learning_rate * 0.1;
            }
        }
        
        self.pulse(coord, power);
        self.update_threads(key, coord);
        self.mesh_time += 1;
        coord
    }

    fn pulse(&mut self, coord: Coord3D, power: f32) {
        let (x, y, z) = coord;
        let dim = self.dim as i32;
        let pulse = power * self.pulse_strength * self.learning_rate;
        
        let neighbors = [
            ((x as i32 + 1).rem_euclid(dim) as u16, y, z),
            ((x as i32 - 1).rem_euclid(dim) as u16, y, z),
            (x, (y as i32 + 1).rem_euclid(dim) as u16, z),
            (x, (y as i32 - 1).rem_euclid(dim) as u16, z),
            (x, y, (z as i32 + 1).rem_euclid(dim) as u16),
            (x, y, (z as i32 - 1).rem_euclid(dim) as u16),
        ];
        
        for ncoord in neighbors {
            let nkey = self.cell_key(ncoord);
            self.apply_decay(nkey);
            self.cells[nkey].energy += pulse;
            self.cells[nkey].velocity += pulse * 0.5;
            
            if !self.cells[nkey].hot && self.cells[nkey].energy >= self.hot_threshold {
                self.cells[nkey].hot = true;
                self.hot_cells[nkey] = Some(HotCell {
                    papers: Vec::new(),
                    pulse_energy: pulse,
                    thread_ids: Vec::new(),
                });
            }
            if let Some(hot) = &mut self.hot_cells[nkey] {
                hot.pulse_energy += pulse;
            }
        }
    }

    fn update_threads(&mut self, key: usize, coord: Coord3D) {
        if !self.cells[key].hot { return; }
        let (x, y, z) = coord;
        let dim = self.dim as i32;
        let energy = self.cells[key].energy;
        let radius = 5i32;
        
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                for dz in -radius..=radius {
                    if dx.abs() + dy.abs() + dz.abs() > radius { continue; }
                    if dx == 0 && dy == 0 && dz == 0 { continue; }
                    
                    let nx = ((x as i32 + dx).rem_euclid(dim)) as u16;
                    let ny = ((y as i32 + dy).rem_euclid(dim)) as u16;
                    let nz = ((z as i32 + dz).rem_euclid(dim)) as u16;
                    let nkey = self.cell_key((nx, ny, nz));
                    if nkey == key { continue; }
                    
                    self.apply_decay(nkey);
                    if self.cells[nkey].hot {
                        let diff = (self.cells[nkey].energy - energy).abs() / energy.max(1.0);
                        if diff < 0.2 {
                            if let Some(hot) = &mut self.hot_cells[key] {
                                if !hot.thread_ids.contains(&(nkey as u64)) {
                                    hot.thread_ids.push(nkey as u64);
                                }
                            }
                            if let Some(hot) = &mut self.hot_cells[nkey] {
                                if !hot.thread_ids.contains(&(key as u64)) {
                                    hot.thread_ids.push(key as u64);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // === QUERIES (all non-mutating) ===

    pub fn query_spatial(&self, coord: Coord3D, radius: u16) -> Vec<&DynamicPaper> {
        let mut results = Vec::new();
        let r = radius as i32;
        let dim = self.dim as i32;
        let (cx, cy, cz) = coord;
        
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if dx.abs() + dy.abs() + dz.abs() > r { continue; }
                    let nx = ((cx as i32 + dx).rem_euclid(dim)) as u16;
                    let ny = ((cy as i32 + dy).rem_euclid(dim)) as u16;
                    let nz = ((cz as i32 + dz).rem_euclid(dim)) as u16;
                    let key = self.cell_key((nx, ny, nz));
                    if let Some(hot) = &self.hot_cells[key] {
                        results.extend(hot.papers.iter());
                    }
                }
            }
        }
        results
    }

    pub fn query_energy(&self, target: f32, delta: f32, top_k: usize) -> Vec<(Coord3D, f32, &DynamicPaper)> {
        let mut scored = Vec::new();
        for (key, header) in self.cells.iter().enumerate() {
            if !header.hot { continue; }
            let eff = self.effective_energy(key);
            let diff = (eff - target).abs();
            if diff <= delta {
                if let Some(hot) = &self.hot_cells[key] {
                    let coord = self.key_to_coord(key);
                    for paper in &hot.papers {
                        scored.push((coord, eff, paper));
                    }
                }
            }
        }
        scored.sort_by(|a, b| {
            let da = (a.1 - target).abs();
            let db = (b.1 - target).abs();
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });
        scored.truncate(top_k);
        scored
    }

    pub fn query_trajectory(&self, target: f32, top_k: usize) -> Vec<(Coord3D, f32, f32, &DynamicPaper)> {
        let mut scored = Vec::new();
        for (key, header) in self.cells.iter().enumerate() {
            if !header.hot { continue; }
            let eff = self.effective_energy(key);
            let moving = if header.velocity > 0.0 { target > eff }
                        else if header.velocity < 0.0 { target < eff }
                        else { false };
            if moving {
                if let Some(hot) = &self.hot_cells[key] {
                    let coord = self.key_to_coord(key);
                    for paper in &hot.papers {
                        scored.push((coord, eff, header.velocity, paper));
                    }
                }
            }
        }
        scored.sort_by(|a, b| {
            let da = (a.1 - target).abs();
            let db = (b.1 - target).abs();
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });
        scored.truncate(top_k);
        scored
    }

    pub fn query_column(&self, coord: Coord3D, top_k: usize) -> Vec<(Coord3D, f32, &DynamicPaper)> {
        let mut scored = Vec::new();
        let (x, y, _) = coord;
        for z in 0..self.dim {
            let key = self.cell_key((x, y, z as u16));
            let col_e = self.effective_column_energy(key);
            if let Some(hot) = &self.hot_cells[key] {
                for paper in &hot.papers {
                    scored.push(((x, y, z as u16), col_e, paper));
                }
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    pub fn query_threads(&self, coord: Coord3D, depth: usize) -> Vec<(Coord3D, &DynamicPaper)> {
        let key = self.cell_key(coord);
        let mut results = Vec::new();
        let mut visited = Vec::new();
        let mut queue = Vec::new();
        queue.push((key, 0usize));
        
        while let Some((ck, d)) = queue.pop() {
            if ck >= visited.len() { visited.resize(ck + 1, false); }
            if visited[ck] { continue; }
            visited[ck] = true;
            
            if let Some(hot) = &self.hot_cells[ck] {
                let c = self.key_to_coord(ck);
                for paper in &hot.papers {
                    results.push((c, paper));
                }
                if d < depth {
                    for &tid in &hot.thread_ids {
                        let t = tid as usize;
                        if t >= visited.len() { visited.resize(t + 1, false); }
                        if !visited[t] { queue.push((t, d + 1)); }
                    }
                }
            }
        }
        results
    }

    pub fn query_temporal(&self, time: u64, delta: u64, top_k: usize) -> Vec<(u64, &DynamicPaper)> {
        let mut scored = Vec::new();
        let t_min = time.saturating_sub(delta);
        let t_max = time + delta;
        
        for (key, header) in self.cells.iter().enumerate() {
            if !header.hot { continue; }
            if let Some(hot) = &self.hot_cells[key] {
                for paper in &hot.papers {
                    if paper.mesh_time >= t_min && paper.mesh_time <= t_max {
                        scored.push((paper.mesh_time, paper));
                    }
                }
            }
        }
        scored.sort_by(|a, b| {
            let da = if a.0 > time { a.0 - time } else { time - a.0 };
            let db = if b.0 > time { b.0 - time } else { time - b.0 };
            da.cmp(&db)
        });
        scored.truncate(top_k);
        scored
    }

    #[inline]
    fn key_to_coord(&self, key: usize) -> Coord3D {
        let d = self.dim;
        let x = key / (d * d);
        let r = key % (d * d);
        let y = r / d;
        let z = r % d;
        (x as u16, y as u16, z as u16)
    }


    /// Return all cell keys that are currently hot.
    pub fn hot_cell_keys(&self) -> Vec<usize> {
        let mut keys = Vec::new();
        for (key, header) in self.cells.iter().enumerate() {
            if header.hot {
                keys.push(key);
            }
        }
        keys
    }

    pub fn mesh_time(&self) -> u64 { self.mesh_time }
    pub fn len(&self) -> usize { self.mesh_time as usize }
    pub fn base(&self) -> u32 { self.base }

    /// Number of cells in the mesh.
    pub fn num_cells(&self) -> usize { self.cells.len() }

    /// Get cell header by key.
    pub fn cell_header(&self, key: usize) -> Option<&CellHeader> {
        self.cells.get(key)
    }

    /// Iterate over all cell headers with their keys.
    pub fn iter_cells(&self) -> impl Iterator<Item = (usize, &CellHeader)> {
        self.cells.iter().enumerate()
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

    fn near_pap(original: &[u8; 64], flips: usize, seed: u64) -> [u8; 64] {
        let mut pap = *original;
        for i in 0..flips {
            let pos = ((seed.wrapping_mul(104729 + i as u64)) % 64) as usize;
            pap[pos] = pap[pos].wrapping_add(1);
        }
        pap
    }

    #[test]
    fn insertion_boosts_energy() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
        let pap = make_pap(42);
        let coord = mesh.insert(&pap, 0);
        let key = mesh.cell_key(coord);
        assert!(mesh.cells[key].energy > 0.0);
        assert!(mesh.cells[key].hot);
    }

    #[test]
    fn hot_cold_transition() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
        let pap = make_pap(42);
        let coord = mesh.insert(&pap, 0);
        let key = mesh.cell_key(coord);
        assert!(mesh.cells[key].hot);
        mesh.mesh_time += 100;
        mesh.apply_decay(key);
        if mesh.cells[key].energy < 0.01 {
            assert!(!mesh.cells[key].hot);
        }
    }

    #[test]
    fn pulse_propagates() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.2, 0.5, 0.01);
        let pap = make_pap(42);
        let coord = mesh.insert(&pap, 0);
        let (x, y, z) = coord;
        let dim = mesh.dim as i32;
        let mut neighbor_energy = 0.0;
        
        for dx in -1i32..=1i32 {
            for dy in -1i32..=1i32 {
                for dz in -1i32..=1i32 {
                    if dx == 0 && dy == 0 && dz == 0 { continue; }
                    let nx = ((x as i32 + dx).rem_euclid(dim)) as u16;
                    let ny = ((y as i32 + dy).rem_euclid(dim)) as u16;
                    let nz = ((z as i32 + dz).rem_euclid(dim)) as u16;
                    let nkey = mesh.cell_key((nx, ny, nz));
                    neighbor_energy += mesh.cells[nkey].energy;
                }
            }
        }
        assert!(neighbor_energy > 0.0);
    }

    #[test]
    fn column_query_finds_vertical() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
        let seed = make_pap(42);
        for i in 0..10 {
            let mut pap = near_pap(&seed, 1, i as u64);
            pap[42] = (i * 25) as u8;
            mesh.insert(&pap, i);
        }
        let seed_coord = MultiBaseCrystal::compute_coord(&seed, 5);
        let results = mesh.query_column(seed_coord, 20);
        assert!(!results.is_empty());
    }

    #[test]
    fn thread_connects_correlated() {
        let mut mesh = DynamicMesh::new(10, 0.5, 0.99, 0.3, 0.1, 0.5, 0.01);
        let seed1 = make_pap(1);
        let seed2 = make_pap(2);
        
        for i in 0..20 {
            let p = near_pap(&seed1, 1, i as u64);
            mesh.insert(&p, i);
        }
        for i in 20..40 {
            let p = near_pap(&seed2, 1, i as u64);
            mesh.insert(&p, i);
        }
        
        let mut thread_count = 0;
        for hot in &mesh.hot_cells {
            if let Some(h) = hot { thread_count += h.thread_ids.len(); }
        }
        assert!(thread_count > 0, "Threads should form between correlated hot cells");
    }

    #[test]
    fn trajectory_query() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.8, 0.1, 0.5, 0.01);
        let seed = make_pap(42);
        for i in 0..30 {
            let p = near_pap(&seed, 1, i as u64);
            mesh.insert(&p, i);
        }
        let results = mesh.query_trajectory(10.0, 20);
        assert!(!results.is_empty());
    }

    #[test]
    fn energy_query() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.5, 0.01);
        let seed = make_pap(42);
        for i in 0..50 {
            let p = near_pap(&seed, 2, i as u64);
            mesh.insert(&p, i);
        }
        let results = mesh.query_energy(5.0, 10.0, 20);
        assert!(!results.is_empty());
    }

    #[test]
    fn temporal_query() {
        let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
        for i in 0..100 {
            let p = make_pap(i as u64);
            mesh.insert(&p, i);
        }
        let results = mesh.query_temporal(50, 5, 10);
        assert!(!results.is_empty());
        for (time, _) in &results {
            assert!(*time >= 45 && *time <= 55);
        }
    }
}
