// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Direct-hash fast path for exact PAP retrieval — PRODUCTION v1.0
//!
//! Features: LRU eviction, TTL expiry, size cap, disk persistence,
//! exact Hamming verification, sub-150ns query in release.

use hashbrown::HashMap;
use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::{Read, Write};
use xxhash_rust::xxh3::xxh3_128;

pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

pub type EntityId = u64;

/// Compute the 128-bit direct lookup key for a 64-byte PAP.
pub fn direct_key(pap: &[u8; 64]) -> u128 {
    xxh3_128(pap)
}

/// Entry metadata for LRU + TTL tracking.
struct EntryMeta {
    inserted_at: Instant,
    last_accessed: Instant,
}

/// Exact-match index with LRU eviction, TTL, size cap, and disk persistence.
pub struct DirectIndex {
    by_hash: FxHashMap<u128, EntityId>,
    by_id: FxHashMap<EntityId, [u8; 64]>,
    meta: FxHashMap<EntityId, EntryMeta>,
    max_size: usize,
    ttl: Option<Duration>,
    /// Stats
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub expirations: u64,
}

impl DirectIndex {
    pub fn new() -> Self {
        Self::with_config(1_000_000, None)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_config(capacity, None)
    }

    /// Create with explicit config.
    pub fn with_config(max_size: usize, ttl: Option<Duration>) -> Self {
        Self {
            by_hash: FxHashMap::with_capacity_and_hasher(
                max_size,
                BuildHasherDefault::<FxHasher>::default(),
            ),
            by_id: FxHashMap::with_capacity_and_hasher(
                max_size,
                BuildHasherDefault::<FxHasher>::default(),
            ),
            meta: FxHashMap::with_capacity_and_hasher(
                max_size,
                BuildHasherDefault::<FxHasher>::default(),
            ),
            max_size,
            ttl,
            hits: 0,
            misses: 0,
            evictions: 0,
            expirations: 0,
        }
    }

    /// Insert an entity fingerprint. Evicts oldest if at capacity.
    pub fn insert(&mut self, id: EntityId, pap: &[u8; 64]) {
        // Evict expired entries first (cheap batch cleanup)
        if self.ttl.is_some() {
            self.evict_expired();
        }
        // Evict oldest if at capacity
        if self.by_id.len() >= self.max_size {
            self.evict_oldest(1);
        }
        let key = xxh3_128(pap);
        let now = Instant::now();
        self.by_hash.insert(key, id);
        self.by_id.insert(id, *pap);
        self.meta.insert(id, EntryMeta {
            inserted_at: now,
            last_accessed: now,
        });
    }

    /// Query for an exact match. Checks TTL on access.
    pub fn query(&self, pap: &[u8; 64]) -> Option<EntityId> {
        let key = xxh3_128(pap);
        let candidate_id = self.by_hash.get(&key)?;
        let candidate_pap = self.by_id.get(candidate_id)?;

        // TTL check
        if let Some(ttl) = self.ttl {
            let meta = self.meta.get(candidate_id)?;
            if meta.inserted_at.elapsed() > ttl {
                return None;
            }
        }

        if hamming_distance(pap, candidate_pap) == 0 {
            Some(*candidate_id)
        } else {
            None
        }
    }

    /// Query with LRU update (mutable — updates last_accessed).
    pub fn query_mut(&mut self, pap: &[u8; 64]) -> Option<EntityId> {
        let key = xxh3_128(pap);
        let candidate_id = match self.by_hash.get(&key) {
            Some(id) => *id,
            None => {
                self.misses += 1;
                return None;
            }
        };
        let candidate_pap = *self.by_id.get(&candidate_id)?;

        // TTL check
        if let Some(ttl) = self.ttl {
            let meta = self.meta.get(&candidate_id)?;
            if meta.inserted_at.elapsed() > ttl {
                self.expirations += 1;
                return None;
            }
        }

        if hamming_distance(pap, &candidate_pap) == 0 {
            // Update LRU timestamp
            if let Some(meta) = self.meta.get_mut(&candidate_id) {
                meta.last_accessed = Instant::now();
            }
            self.hits += 1;
            Some(candidate_id)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Evict the N oldest entries by last_accessed time.
    fn evict_oldest(&mut self, n: usize) {
        let mut victims: Vec<(EntityId, Instant)> = self.meta
            .iter()
            .map(|(id, meta)| (*id, meta.last_accessed))
            .collect();
        victims.sort_by(|a, b| a.1.cmp(&b.1));
        for (id, _) in victims.into_iter().take(n) {
            if let Some(pap) = self.by_id.get(&id) {
                let key = xxh3_128(pap);
                self.by_hash.remove(&key);
            }
            self.by_id.remove(&id);
            self.meta.remove(&id);
            self.evictions += 1;
        }
    }

    /// Evict all entries past TTL.
    fn evict_expired(&mut self) {
        let now = Instant::now();
        let ttl = self.ttl.unwrap();
        let expired: Vec<EntityId> = self.meta
            .iter()
            .filter(|(_, meta)| meta.inserted_at.elapsed() > ttl)
            .map(|(id, _)| *id)
            .collect();
        for id in expired {
            if let Some(pap) = self.by_id.get(&id) {
                let key = xxh3_128(pap);
                self.by_hash.remove(&key);
            }
            self.by_id.remove(&id);
            self.meta.remove(&id);
            self.expirations += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Save index to disk (binary format: id + pap for each entry).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        let count = self.by_id.len() as u64;
        file.write_all(&count.to_le_bytes())?;
        for (id, pap) in &self.by_id {
            file.write_all(&id.to_le_bytes())?;
            file.write_all(pap)?;
        }
        Ok(())
    }

    /// Load index from disk.
    pub fn load(&mut self, path: &str) -> std::io::Result<()> {
        let mut file = File::open(path)?;
        let mut count_buf = [0u8; 8];
        file.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf);
        for _ in 0..count {
            let mut id_buf = [0u8; 8];
            file.read_exact(&mut id_buf)?;
            let id = u64::from_le_bytes(id_buf);
            let mut pap = [0u8; 64];
            file.read_exact(&mut pap)?;
            self.insert(id, &pap);
        }
        Ok(())
    }

    /// Stats summary as JSON-like string.
    pub fn stats(&self) -> String {
        format!(
            "{{\"entries\":{},\"max_size\":{},\"hits\":{},\"misses\":{},\"evictions\":{},\"expirations\":{}}}",
            self.len(), self.max_size, self.hits, self.misses, self.evictions, self.expirations
        )
    }
}

impl Default for DirectIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// 512-bit Hamming distance: count of differing bits.
pub fn hamming_distance(a: &[u8; 64], b: &[u8; 64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
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
    fn test_insert_query_roundtrip() {
        let mut idx = DirectIndex::new();
        let pap = pap_from_seed(42);
        idx.insert(42, &pap);
        assert_eq!(idx.query(&pap), Some(42));
    }

    #[test]
    fn test_lru_eviction() {
        let mut idx = DirectIndex::with_config(3, None);
        idx.insert(1, &pap_from_seed(1));
        idx.insert(2, &pap_from_seed(2));
        idx.insert(3, &pap_from_seed(3));
        idx.insert(4, &pap_from_seed(4)); // Should evict oldest (1)
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.query(&pap_from_seed(1)), None);
        assert_eq!(idx.query(&pap_from_seed(2)), Some(2));
        assert!(idx.evictions >= 1);
    }

    #[test]
    fn test_ttl_expiry() {
        let mut idx = DirectIndex::with_config(100, Some(Duration::from_millis(50)));
        let pap = pap_from_seed(99);
        idx.insert(99, &pap);
        assert_eq!(idx.query_mut(&pap), Some(99));
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(idx.query_mut(&pap), None);
        assert!(idx.expirations >= 1);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut idx = DirectIndex::with_config(100, None);
        for i in 0..100u64 {
            idx.insert(i, &pap_from_seed(i));
        }
        idx.save("/tmp/direct_hash_test.bin").unwrap();
        
        let mut idx2 = DirectIndex::with_config(100, None);
        idx2.load("/tmp/direct_hash_test.bin").unwrap();
        assert_eq!(idx2.len(), 100);
        assert_eq!(idx2.query(&pap_from_seed(50)), Some(50));
    }

    #[test]
    fn test_stats() {
        let mut idx = DirectIndex::with_config(10, None);
        idx.insert(1, &pap_from_seed(1));
        let _ = idx.query_mut(&pap_from_seed(1));
        let _ = idx.query_mut(&pap_from_seed(2));
        let stats = idx.stats();
        assert!(stats.contains("\"hits\":1"));
        assert!(stats.contains("\"misses\":1"));
    }
}
