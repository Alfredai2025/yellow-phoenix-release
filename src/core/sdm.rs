// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use crate::types::multivector::BinaryMultivector;
use alloc::vec::Vec;

/// A hard location in Sparse Distributed Memory.
/// Address is a 128-bit binary vector. Counters store write history.
/// Indices links back to stored papers.
#[derive(Debug, Clone)]
pub struct HardLocation {
    pub address: BinaryMultivector,
    pub counters: [i16; 128],
    pub indices: Vec<usize>,
}

/// One layer of the SDM (Sparse Distributed Memory).
/// Contains hard locations scattered through the address space.
pub struct SdmLayer {
    pub locations: Vec<HardLocation>,
    pub address_space: u32,
}

impl SdmLayer {
    /// Create a new SDM layer with `n` random hard locations.
    /// Seeded RNG for determinism in tests.
    pub fn new(n: usize, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut locations = Vec::with_capacity(n);

        for _ in 0..n {
            let a = rng.random::<u64>();
            let b = rng.random::<u64>();
            locations.push(HardLocation {
                address: BinaryMultivector([a, b]),
                counters: [0i16; 128],
                indices: Vec::new(),
            });
        }

        Self {
            locations,
            address_space: n as u32,
        }
    }

    /// Write a paper index into the layer.
    /// Finds all hard locations within `radius` Hamming distance of `paper`,
    /// increments counters for matching bits, and stores the index.
    pub fn write(&mut self, paper: &BinaryMultivector, index: usize, radius: u32) {
        for loc in self.locations.iter_mut() {
            let dist = paper.hamming_distance(&loc.address);
            if dist <= radius {
                // Increment counters where paper bits are 1, decrement where 0
                for i in 0..64 {
                    let bit = ((paper.0[0] >> i) & 1) != 0;
                    if bit {
                        loc.counters[i] = loc.counters[i].saturating_add(1);
                    } else {
                        loc.counters[i] = loc.counters[i].saturating_sub(1);
                    }
                }
                for i in 0..64 {
                    let bit = ((paper.0[1] >> i) & 1) != 0;
                    if bit {
                        loc.counters[64 + i] = loc.counters[64 + i].saturating_add(1);
                    } else {
                        loc.counters[64 + i] = loc.counters[64 + i].saturating_sub(1);
                    }
                }
                loc.indices.push(index);
            }
        }
    }

    /// Read from the layer given a query vector.
    /// Returns an "echo" vector (majority vote of activated locations) and confidence score.
    pub fn read(&self, query: &BinaryMultivector, radius: u32) -> (BinaryMultivector, f32) {
        let mut echo_counters = [0i32; 128];
        let mut activations = 0usize;

        for loc in self.locations.iter() {
            let dist = query.hamming_distance(&loc.address);
            if dist <= radius {
                activations += 1;
                for i in 0..128 {
                    echo_counters[i] += loc.counters[i] as i32;
                }
            }
        }

        if activations == 0 {
            return (BinaryMultivector([0, 0]), 0.0);
        }

        // Majority vote: positive counter → bit 1, negative → bit 0
        let mut chunks = [0u64; 2];
        for i in 0..64 {
            if echo_counters[i] > 0 {
                chunks[0] |= 1u64 << i;
            }
        }
        for i in 0..64 {
            if echo_counters[64 + i] > 0 {
                chunks[1] |= 1u64 << i;
            }
        }

        // Confidence = normalized dot product between query and echo
        let echo = BinaryMultivector(chunks);
        let confidence = Self::normalized_similarity(query, &echo);

        (echo, confidence)
    }

    /// Continuous read: weighted sum of counters across activated hard locations.
    /// Returns a 128‑element f32 vector and a confidence score.
    pub fn read_continuous(&self, query: &BinaryMultivector, radius: u32) -> (Vec<f32>, f32) {
        let mut sum = vec![0.0f32; 128];
        let mut sum_weights = 0.0f32;
        let mut activations = 0usize;

        for loc in self.locations.iter() {
            let dist = query.hamming_distance(&loc.address);
            if dist <= radius {
                let weight = 1.0 / (1.0 + (dist as f32 / 128.0));
                for i in 0..128 {
                    sum[i] += loc.counters[i] as f32 * weight;
                }
                sum_weights += weight;
                activations += 1;
            }
        }

        if sum.iter().all(|&x| x == 0.0) {
            return (vec![0.0f32; 128], 0.0);
        }

        if activations == 0 {
            return (vec![0.0f32; 128], 0.0);
        }

        let confidence = sum_weights / activations as f32;
        (sum, confidence)
    }

    /// Cosine-like similarity for binary vectors: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
    fn normalized_similarity(a: &BinaryMultivector, b: &BinaryMultivector) -> f32 {
        let same = (a.0[0] & b.0[0]).count_ones() + (a.0[1] & b.0[1]).count_ones();
        let diff = (a.0[0] ^ b.0[0]).count_ones() + (a.0[1] ^ b.0[1]).count_ones();
        let total = same + diff;
        if total == 0 {
            return 0.0;
        }
        (same as f32 - diff as f32) / total as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_layer_creation() {
        let layer = SdmLayer::new(100, 42);
        assert_eq!(layer.locations.len(), 100);
        assert_eq!(layer.address_space, 100);
    }

    #[test]
    fn write_and_read_identical() {
        let mut layer = SdmLayer::new(10_000, 123);
        let paper = BinaryMultivector([0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555]);
        
        layer.write(&paper, 0, 128); // activate all locations
        
        let (_echo, confidence) = layer.read(&paper, 128);
        assert!(confidence > 0.5, "Confidence should be high for identical query: got {}", confidence);
    }

    #[test]
    fn counters_increment_and_decrement() {
        let mut layer = SdmLayer::new(100, 7);
        let paper = BinaryMultivector([0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF]);
        
        layer.write(&paper, 0, 128); // radius 128 = all locations
        
        // All counters should be positive (all bits were 1)
        for loc in &layer.locations {
            for c in &loc.counters {
                assert!(*c > 0, "Counter should be positive after all-ones write");
            }
        }
    }

    #[test]
    fn with_index_extract_index_roundtrip() {
        let v = BinaryMultivector([0, 0]).with_index(999);
        assert_eq!(v.extract_index(), 999);
    }

    #[test]
    fn continuous_read_empty_field() {
        let layer = SdmLayer::new(100, 42);
        let q = BinaryMultivector([0, 0]);
        let (vec, conf) = layer.read_continuous(&q, 128);
        assert_eq!(vec.len(), 128);
        assert!(conf < 0.01);
    }

    #[test]
    fn continuous_read_single_write() {
        let mut layer = SdmLayer::new(100, 42);
        let paper = BinaryMultivector([0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF]);
        layer.write(&paper, 0, 128);
        let (vec, conf) = layer.read_continuous(&paper, 128);
        assert_eq!(vec.len(), 128);
        assert!(conf > 0.5);
        for v in &vec {
            assert!(*v > 0.0);
        }
    }

    #[test]
    fn continuous_read_interpolation() {
        let mut layer = SdmLayer::new(1000, 42);
        let a = BinaryMultivector([0xFFFF_FFFF_0000_0000, 0]);
        let b = BinaryMultivector([0, 0xFFFF_FFFF_0000_0000]);
        layer.write(&a, 0, 128);
        layer.write(&b, 1, 128);

        let q = BinaryMultivector([0xFFFF_0000_0000_0000, 0xFFFF_0000_0000_0000]);
        let (vec, conf) = layer.read_continuous(&q, 128);
        assert!(conf > 0.0);
        assert_eq!(vec.len(), 128);
        // First 64 dims should be mixed positive/negative (interference pattern)
    }
}
