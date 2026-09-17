// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

extern crate alloc;

use alloc::vec::Vec;
use crate::types::multivector::BinaryMultivector;
use crate::distance::pap_distance;

pub struct StandingHologram {
    pub wave: [u64; 8],
    pub count: usize,
    /// Indexed prototype store: (global_index, multivector).
    pub prototypes: Vec<(usize, BinaryMultivector)>,
}

impl StandingHologram {
    pub fn new() -> Self {
        StandingHologram {
            wave: [0u64; 8],
            count: 0,
            prototypes: Vec::new(),
        }
    }

    /// Build a hologram from an existing wave and indexed prototype store.
    pub fn from_parts(wave: [u64; 8], prototypes: Vec<(usize, BinaryMultivector)>) -> Self {
        let count = prototypes.len();
        Self { wave, count, prototypes }
    }

    pub fn inject(&mut self, vec: &BinaryMultivector) {
        for i in 0..8 {
            self.wave[i] ^= vec.0[i % 2];
        }
        self.prototypes.push((self.count, vec.clone()));
        self.count += 1;
    }

    pub fn batch_inject(&mut self, vecs: &[BinaryMultivector]) {
        for (i, v) in vecs.iter().enumerate() {
            self.inject(v);
            if (i + 1) % 100_000 == 0 {
                #[cfg(any(test, feature = "std"))]
                println!("batch_inject: processed {} vectors", i + 1);
            }
        }
    }

    pub fn resonance_query(&self, query: &BinaryMultivector, top_k: usize) -> Vec<(usize, f32)> {
        let mut distances: Vec<(f32, usize)> = self
            .prototypes
            .iter()
            .map(|(idx, proto)| {
                let d = pap_distance(proto, query);
                (d, *idx)
            })
            .collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        distances.truncate(top_k);
        distances.into_iter().map(|(d, idx)| (idx, d)).collect()
    }

    pub fn hologram_signature(&self) -> [u64; 8] {
        self.wave
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::multivector::BinaryMultivector;
    use std::time::Instant;

    #[test]
    fn inject_1m() {
        let mut hologram = StandingHologram::new();
        let mut vecs = Vec::with_capacity(1_000_000);
        for i in 0..1_000_000 {
            let v = BinaryMultivector([i as u64, (i as u64).wrapping_mul(0xDEADBEEF)]);
            vecs.push(v);
        }
        hologram.batch_inject(&vecs);
        assert_eq!(hologram.count, 1_000_000);
        assert_eq!(hologram.prototypes.len(), 1_000_000);
        let wave = hologram.hologram_signature();
        let mut non_zero = false;
        for &w in wave.iter() {
            if w != 0 {
                non_zero = true;
                break;
            }
        }
        assert!(non_zero);
    }

    #[test]
    fn resonance_finds_nearest() {
        let mut hologram = StandingHologram::new();
        let v0 = BinaryMultivector([1, 0]);
        let v1 = BinaryMultivector([0, 2]);
        let v2 = BinaryMultivector([3, 3]);
        hologram.inject(&v0);
        hologram.inject(&v1);
        hologram.inject(&v2);
        let results = hologram.resonance_query(&v0, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-6);
        let mut indices: Vec<usize> = results.iter().skip(1).map(|(idx, _)| *idx).collect();
        indices.sort_unstable();
        assert_eq!(indices, vec![1, 2]);
    }

    #[test]
    fn batch_speed() {
        let mut hologram = StandingHologram::new();
        let mut vecs = Vec::with_capacity(100_000);
        for i in 0..100_000 {
            let v = BinaryMultivector([i as u64, (i as u64).wrapping_mul(0xCAFEBABE)]);
            vecs.push(v);
        }
        let start = Instant::now();
        hologram.batch_inject(&vecs);
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs_f64() < 1.0);
    }
}
