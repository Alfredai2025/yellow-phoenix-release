// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Energy Overlay — Yellow Phoenix v3.6 Phase 2
//! Per-bucket heat map with spill and decay.

use alloc::vec::Vec;
use core::cmp::Ordering;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyOverlay {
    cells: Vec<(u32, f32)>,  // sparse: (bucket, temperature)
    pub spill_rate: f32,
    pub decay_rate: f32,
}

impl EnergyOverlay {
    pub fn new() -> Self {
        Self {
            cells: Vec::new(),
            spill_rate: 0.30,
            decay_rate: 0.05,
        }
    }

    fn index_of(&self, bucket: u32) -> Option<usize> {
        self.cells.binary_search_by_key(&bucket, |&(b, _)| b)
            .ok()
    }

    /// Add heat to a bucket. Spills to neighbors if provided, otherwise to
    /// the linear adjacent buckets.
    pub fn energize(&mut self, bucket: u32, amount: f32, graph_neighbors: Option<&[u32]>) {
        // Direct hit
        match self.index_of(bucket) {
            Some(i) => self.cells[i].1 += amount,
            None => {
                self.cells.push((bucket, amount));
                self.cells.sort_by_key(|&(b, _)| b);
            }
        }

        // Spill to neighbors
        let spill = amount * self.spill_rate;
        if let Some(nbrs) = graph_neighbors {
            for &n in nbrs {
                self.add_heat(n, spill);
            }
        } else {
            if bucket > 0 {
                self.add_heat(bucket - 1, spill);
            }
            self.add_heat(bucket + 1, spill);
        }
    }

    fn add_heat(&mut self, bucket: u32, amount: f32) {
        match self.index_of(bucket) {
            Some(i) => self.cells[i].1 += amount,
            None => {
                self.cells.push((bucket, amount));
                self.cells.sort_by_key(|&(b, _)| b);
            }
        }
    }

    /// Decay all temperatures. Removes cold cells.
    pub fn decay(&mut self) {
        let mut i = 0;
        while i < self.cells.len() {
            self.cells[i].1 *= 1.0 - self.decay_rate;
            if self.cells[i].1 < 0.001 {
                self.cells.swap_remove(i);
            } else {
                i += 1;
            }
        }
        self.cells.sort_by_key(|&(b, _)| b);
    }

    /// Get temperature of a bucket.
    pub fn get(&self, bucket: u32) -> f32 {
        self.index_of(bucket)
            .map(|i| self.cells[i].1)
            .unwrap_or(0.0)
    }

    /// Top-N hottest buckets.
    pub fn top_hot(&self, n: usize) -> Vec<(u32, f32)> {
        let mut items = self.cells.clone();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        items.into_iter().take(n).collect()
    }

    /// Total heat in the system.
    pub fn total_energy(&self) -> f32 {
        self.cells.iter().map(|(_, t)| t).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energize_and_get() {
        let mut e = EnergyOverlay::new();
        e.energize(10, 1.0, None);
        assert!((e.get(10) - 1.0).abs() < 0.01);
        assert!(e.get(9) > 0.0);
        assert!(e.get(11) > 0.0);
    }

    #[test]
    fn test_energize_graph_neighbors() {
        let mut e = EnergyOverlay::new();
        e.energize(10, 1.0, Some(&[20, 30]));
        assert!((e.get(10) - 1.0).abs() < 0.01);
        assert!(e.get(20) > 0.0);
        assert!(e.get(30) > 0.0);
        assert!(e.get(9) < 0.001);
    }

    #[test]
    fn test_decay() {
        let mut e = EnergyOverlay::new();
        e.energize(5, 1.0, None);
        e.decay();
        assert!(e.get(5) < 1.0 && e.get(5) > 0.0);
    }

    #[test]
    fn test_top_hot() {
        let mut e = EnergyOverlay::new();
        e.energize(1, 5.0, None);
        e.energize(2, 10.0, None);
        e.energize(3, 1.0, None);
        let top = e.top_hot(2);
        assert_eq!(top[0].0, 2);
        assert_eq!(top[1].0, 1);
    }
}
