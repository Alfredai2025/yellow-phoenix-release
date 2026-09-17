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
//! Shared slot types for HybridMesh and DynamicMesh
//! Extracted to avoid circular dependency between hybrid_mesh and dynamic_mesh

use std::cmp::Ordering;

/// A single slot in an adaptive cell — compressed signature for fast querying
#[derive(Clone, Debug, PartialEq)]
pub struct Slot {
    pub paper_id: u64,
    pub distance: f32,
    /// Compressed binary signature (e.g., 128 bits = 16 bytes)
    pub signature: Vec<u8>,
}

impl Slot {
    pub fn new(paper_id: u64, distance: f32, signature: Vec<u8>) -> Self {
        Self { paper_id, distance, signature }
    }
}

impl Ord for Slot {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Slot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Eq for Slot {}

/// Metrics for an adaptive cell
#[derive(Clone, Debug, Default)]
pub struct CellMetrics {
    pub centroid: Vec<f32>,
    pub radius: f32,
    pub count: usize,
}

impl CellMetrics {
    pub fn new(dims: usize) -> Self {
        Self {
            centroid: vec![0.0; dims],
            radius: 0.0,
            count: 0,
        }
    }

    pub fn update(&mut self, signature: &[f32]) {
        let n = self.count as f32;
        let nf = n + 1.0;
        for (i, &v) in signature.iter().enumerate() {
            if i < self.centroid.len() {
                self.centroid[i] = (self.centroid[i] * n + v) / nf;
            }
        }
        self.count += 1;
        let dist = euclidean_distance(signature, &self.centroid);
        if dist > self.radius {
            self.radius = dist;
        }
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

/// An adaptive cell with slots that auto-resize based on query pressure
#[derive(Clone, Debug)]
pub struct AdaptiveCell {
    pub slots: Vec<Slot>,
    pub capacity: usize,
    pub metrics: CellMetrics,
    pub query_count: u64,
    pub insert_count: u64,
}

impl AdaptiveCell {
    pub fn new(capacity: usize, dims: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            capacity,
            metrics: CellMetrics::new(dims),
            query_count: 0,
            insert_count: 0,
        }
    }

    pub fn insert(&mut self, paper_id: u64, distance: f32, signature: Vec<u8>) {
        self.insert_count += 1;
        let slot = Slot::new(paper_id, distance, signature);
        if self.slots.len() < self.capacity {
            self.slots.push(slot);
        } else if let Some(worst_idx) = self.worst_slot_idx() {
            if self.slots[worst_idx].distance > distance {
                self.slots[worst_idx] = slot;
            }
        }
        self.auto_resize();
    }

    pub fn query(&self, signature: &[u8], k: usize) -> Vec<(u64, f32)> {
        let mut results: Vec<(u64, f32)> = self.slots
            .iter()
            .map(|slot| {
                let dist = hamming_distance(&slot.signature, signature);
                (slot.paper_id, dist as f32)
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results
    }

    pub fn merge(&mut self, other: &AdaptiveCell) {
        for slot in &other.slots {
            self.insert(slot.paper_id, slot.distance, slot.signature.clone());
        }
    }

    pub fn auto_resize(&mut self) {
        let pressure = self.query_count as f32 / self.insert_count.max(1) as f32;
        if pressure > 2.0 && self.capacity < 4096 {
            self.capacity = (self.capacity * 2).min(4096);
            self.slots.reserve(self.capacity - self.slots.len());
        } else if pressure < 0.5 && self.capacity > 64 {
            self.capacity = (self.capacity / 2).max(64);
            self.slots.truncate(self.capacity);
            self.slots.shrink_to_fit();
        }
    }

    fn worst_slot_idx(&self) -> Option<usize> {
        self.slots.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
    }

    pub fn len(&self) -> usize { self.slots.len() }
    pub fn is_empty(&self) -> bool { self.slots.is_empty() }

    pub fn memory_bytes(&self) -> usize {
        self.slots.capacity() * std::mem::size_of::<Slot>()
            + self.metrics.centroid.capacity() * std::mem::size_of::<f32>()
            + std::mem::size_of::<Self>()
    }
}

pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_creation() {
        let s = Slot::new(42, 0.5, vec![1, 2, 3]);
        assert_eq!(s.paper_id, 42);
        assert_eq!(s.distance, 0.5);
    }

    #[test]
    fn adaptive_cell_insert_and_query() {
        let mut cell = AdaptiveCell::new(10, 128);
        let sig = vec![0u8; 16];
        for i in 0..5 {
            let mut s = sig.clone();
            s[0] = i as u8;
            cell.insert(i as u64, i as f32, s);
        }
        assert_eq!(cell.len(), 5);
        let results = cell.query(&sig, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn adaptive_cell_capacity_limit() {
        let mut cell = AdaptiveCell::new(3, 128);
        let sig = vec![0u8; 16];
        for i in 0..10 {
            cell.insert(i as u64, i as f32, sig.clone());
        }
        assert_eq!(cell.len(), 3);
    }

    #[test]
    fn hamming_distance_works() {
        let a = vec![0b0000_0001, 0b0000_0010];
        let b = vec![0b0000_0001, 0b0000_0000];
        assert_eq!(hamming_distance(&a, &b), 1);
    }

    #[test]
    fn cell_memory_bytes_positive() {
        let cell = AdaptiveCell::new(100, 128);
        assert!(cell.memory_bytes() > 0);
    }
}
