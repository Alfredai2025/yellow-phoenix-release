// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Locality-Sensitive Hash index for fast approximate neighbor retrieval.

use std::collections::{HashMap, HashSet};
use xxhash_rust::xxh3::xxh3_128;

pub type EntityId = u64;

pub struct LSHIndex {
    tables: Vec<HashMap<u64, Vec<EntityId>>>,
    hash_seeds: [u64; 8],
    hash_bits: u32,
}

impl LSHIndex {
    pub fn new() -> Self {
        Self::with_tables(8, 16)
    }

    pub fn with_tables(n_tables: usize, hash_bits: u32) -> Self {
        assert!(n_tables <= 8, "LSHIndex supports at most 8 tables");
        assert!(hash_bits <= 64, "hash_bits must be <= 64");
        let mut tables = Vec::with_capacity(n_tables);
        for _ in 0..n_tables {
            tables.push(HashMap::new());
        }
        // Deterministic, distinct seeds per table.
        let mut hash_seeds = [0u64; 8];
        for (i, seed) in hash_seeds.iter_mut().enumerate() {
            *seed = i as u64 + 0x9E3779B97F4A7C15;
        }
        Self {
            tables,
            hash_seeds,
            hash_bits,
        }
    }

    /// Insert an entity into all LSH tables.
    pub fn insert(&mut self, id: EntityId, pap: &[u8; 64]) {
        for (table_id, seed) in self.hash_seeds.iter().enumerate() {
            let bucket = self.hash(pap, table_id, *seed);
            self.tables[table_id].entry(bucket).or_default().push(id);
        }
    }

    /// Collect candidate neighbors from all tables, deduplicated.
    pub fn candidates(&self, pap: &[u8; 64]) -> Vec<EntityId> {
        let mut set = HashSet::new();
        for (table_id, seed) in self.hash_seeds.iter().enumerate() {
            let bucket = self.hash(pap, table_id, *seed);
            if let Some(ids) = self.tables[table_id].get(&bucket) {
                for &id in ids {
                    set.insert(id);
                }
            }
        }
        set.into_iter().collect()
    }

    /// Compute the LSH bucket for a given table.
    fn hash(&self, pap: &[u8; 64], _table_id: usize, seed: u64) -> u64 {
        let mut buf = [0u8; 72];
        buf[0..64].copy_from_slice(pap);
        buf[64..72].copy_from_slice(&seed.to_le_bytes());
        let h = xxh3_128(&buf);
        let mask = if self.hash_bits == 64 {
            u64::MAX
        } else {
            (1u64 << self.hash_bits) - 1
        };
        (h as u64) & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pap_from_seed(seed: u64) -> [u8; 64] {
        let mut pap = [0u8; 64];
        let bytes = seed.to_le_bytes();
        for chunk in pap.chunks_exact_mut(8) {
            chunk.copy_from_slice(&bytes);
        }
        pap
    }

    #[test]
    fn test_lsh_insert_candidates() {
        let mut lsh = LSHIndex::new();
        for i in 0..100u64 {
            lsh.insert(i, &pap_from_seed(i));
        }
        let candidates = lsh.candidates(&pap_from_seed(42));
        assert!(!candidates.is_empty(), "expected some candidates");
    }

    #[test]
    fn test_lsh_candidates_are_similar() {
        let mut lsh = LSHIndex::new();
        // Identical vectors must hash to the same buckets.
        let base = pap_from_seed(1000);
        lsh.insert(1000, &base);
        lsh.insert(1001, &base);
        lsh.insert(1002, &base);

        let candidates = lsh.candidates(&base);
        assert!(candidates.contains(&1000));
        assert!(candidates.contains(&1001));
        assert!(candidates.contains(&1002));
    }
}
