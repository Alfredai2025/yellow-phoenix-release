// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! 384-bit Crystal Mesh — honest 48-byte PAP slots.
//! No padding. No waste. Every bit carries signal.

use std::cmp::Ordering;

pub const PAP_384_BYTES: usize = 48;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot384 {
    pub id: u64,
    pub pap: [u8; PAP_384_BYTES],
}

impl Slot384 {
    pub fn new(id: u64, pap: [u8; PAP_384_BYTES]) -> Self {
        Self { id, pap }
    }

    /// Hamming distance to another 384-bit PAP.
    pub fn hamming(&self, other: &[u8; PAP_384_BYTES]) -> u32 {
        self.pap.iter().zip(other.iter()).map(|(a, b)| (a ^ b).count_ones()).sum()
    }

    /// Check if this slot is empty (id zero and all-zero PAP).
    pub fn is_empty(&self) -> bool {
        self.id == 0 && self.pap.iter().all(|&b| b == 0)
    }
}

/// 384-bit crystal mesh with linear probing and exact-match query.
#[derive(Clone, Debug)]
pub struct CrystalMesh384 {
    pub slots: Vec<Slot384>,
    pub capacity: usize,
    pub len: usize,
}

impl CrystalMesh384 {
    pub fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, || Slot384::new(0, [0u8; PAP_384_BYTES]));
        Self { slots, capacity, len: 0 }
    }

    /// Insert a document. Returns true if inserted, false if full or duplicate.
    pub fn insert(&mut self, id: u64, pap: [u8; PAP_384_BYTES]) -> bool {
        if self.len >= self.capacity {
            return false;
        }
        let idx = (id as usize) % self.capacity;
        let mut probe = 0;
        loop {
            let i = (idx + probe) % self.capacity;
            if self.slots[i].is_empty() {
                self.slots[i] = Slot384::new(id, pap);
                self.len += 1;
                return true;
            }
            if self.slots[i].id == id {
                // Update existing
                self.slots[i].pap = pap;
                return true;
            }
            probe += 1;
            if probe >= self.capacity {
                return false;
            }
        }
    }

    /// Exact-match query by Hamming distance. Returns top-k matches.
    pub fn query_k(&self, pap: &[u8; PAP_384_BYTES], k: usize) -> Vec<(u32, u64)> {
        let mut results: Vec<(u32, u64)> = self.slots
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| (s.hamming(pap), s.id))
            .collect();
        results.sort_by(|a, b| a.0.cmp(&b.0));
        results.truncate(k);
        results
    }

    /// Single nearest neighbor.
    pub fn query_direct(&self, pap: &[u8; PAP_384_BYTES]) -> Option<u64> {
        self.slots
            .iter()
            .filter(|s| !s.is_empty())
            .min_by_key(|s| s.hamming(pap))
            .map(|s| s.id)
    }

    /// Brute-force all-pairs Hamming distance (for functor audits).
    pub fn all_pairs_hamming(&self) -> Vec<(u64, u64, u32)> {
        let active: Vec<&Slot384> = self.slots.iter().filter(|s| !s.is_empty()).collect();
        let mut pairs = Vec::with_capacity(active.len() * active.len() / 2);
        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                let d = active[i].hamming(&active[j].pap);
                pairs.push((active[i].id, active[j].id, d));
            }
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot384_hamming() {
        let a = Slot384::new(1, [0xFF; PAP_384_BYTES]);
        let b = Slot384::new(2, [0x00; PAP_384_BYTES]);
        assert_eq!(a.hamming(&b.pap), 384);
    }

    #[test]
    fn test_insert_and_query() {
        let mut mesh = CrystalMesh384::new(1024);
        let mut pap = [0u8; PAP_384_BYTES];
        pap[0] = 0xAB;
        assert!(mesh.insert(42, pap));
        let result = mesh.query_direct(&pap);
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_query_k_ordering() {
        let mut mesh = CrystalMesh384::new(1024);
        let pap1 = [0xFF; PAP_384_BYTES];
        let pap2 = [0x00; PAP_384_BYTES];
        let pap3 = [0x00; PAP_384_BYTES]; // duplicate distance
        mesh.insert(1, pap1);
        mesh.insert(2, pap2);
        mesh.insert(3, pap3);
        let results = mesh.query_k(&pap2, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // exact match to pap2
    }

    #[test]
    fn test_capacity_full() {
        let mut mesh = CrystalMesh384::new(4);
        for i in 0..4 {
            let mut pap = [0u8; PAP_384_BYTES];
            pap[0] = i as u8;
            assert!(mesh.insert(i as u64, pap));
        }
        let mut pap = [0u8; PAP_384_BYTES];
        pap[0] = 99;
        assert!(!mesh.insert(99, pap)); // full
    }
}
