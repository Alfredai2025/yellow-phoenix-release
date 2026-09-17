// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! result_cache.rs — LRU query-result cache with TTL.
//!
//! Caches top-5 results keyed by the first 8 bytes of the query hash.
//! Enterprise: memory cap, TTL enforcement, hit/miss metrics, invalidation API.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

const CACHE_SIZE: usize = 1024;
const CACHE_TTL: Duration = Duration::from_secs(60);

#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub results: Vec<(u64, f32, String)>,
    pub timestamp: Instant,
}

#[derive(Clone, Debug, Default)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub invalidations: u64,
}

pub struct ResultCache {
    map: HashMap<[u8; 8], CacheEntry>,
    order: VecDeque<[u8; 8]>,
    metrics: CacheMetrics,
    max_size: usize,
    ttl: Duration,
}

impl Default for ResultCache {
    fn default() -> Self {
        Self::new(CACHE_SIZE, CACHE_TTL)
    }
}

impl ResultCache {
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            map: HashMap::with_capacity(max_size),
            order: VecDeque::with_capacity(max_size),
            metrics: CacheMetrics::default(),
            max_size,
            ttl,
        }
    }

    /// Look up a cached result by query hash.
    pub fn lookup(&mut self, query_hash: &[u8; 64]) -> Option<Vec<(u64, f32, String)>> {
        let key = self.key(query_hash);
        let expired = self.map.get(&key).map_or(true, |entry| entry.timestamp.elapsed() >= self.ttl);
        if expired {
            self.metrics.misses += 1;
            return None;
        }
        let results = self.map.get(&key).unwrap().results.clone();
        self.touch(key);
        self.metrics.hits += 1;
        Some(results)
    }

    /// Insert a result into the cache.
    pub fn insert(&mut self, query_hash: &[u8; 64], results: Vec<(u64, f32, String)>) {
        let key = self.key(query_hash);

        // Evict stale entries first.
        self.evict_stale();

        // Enforce size cap.
        if self.map.len() >= self.max_size && !self.map.contains_key(&key) {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
                self.metrics.evictions += 1;
            }
        }

        self.map.insert(
            key,
            CacheEntry {
                results,
                timestamp: Instant::now(),
            },
        );
        self.touch(key);
    }

    /// Invalidate a specific cached query.
    pub fn invalidate(&mut self, query_hash: &[u8; 64]) {
        let key = self.key(query_hash);
        if self.map.remove(&key).is_some() {
            self.order.retain(|k| k != &key);
            self.metrics.invalidations += 1;
        }
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.metrics.invalidations += 1;
    }

    /// Current hit rate (0.0 - 1.0).
    pub fn hit_rate(&self) -> f32 {
        let total = self.metrics.hits + self.metrics.misses;
        if total == 0 {
            0.0
        } else {
            self.metrics.hits as f32 / total as f32
        }
    }

    pub fn metrics(&self) -> &CacheMetrics {
        &self.metrics
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    fn key(&self, query_hash: &[u8; 64]) -> [u8; 8] {
        let mut key = [0u8; 8];
        key.copy_from_slice(&query_hash[..8]);
        key
    }

    fn touch(&mut self, key: [u8; 8]) {
        self.order.retain(|k| k != &key);
        self.order.push_back(key);
    }

    fn evict_stale(&mut self) {
        let ttl = self.ttl;
        let mut to_remove = Vec::new();
        for key in &self.order {
            if let Some(entry) = self.map.get(key) {
                if entry.timestamp.elapsed() >= ttl {
                    to_remove.push(*key);
                }
            }
        }
        for key in to_remove {
            self.map.remove(&key);
            self.order.retain(|k| k != &key);
            self.metrics.evictions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(seed: u8) -> [u8; 64] {
        let mut h = [seed; 64];
        h
    }

    #[test]
    fn cache_hit() {
        let mut cache = ResultCache::default();
        let results = vec![(1u64, 0.9f32, "title".into())];
        cache.insert(&q(1), results.clone());
        assert_eq!(cache.lookup(&q(1)), Some(results));
        assert_eq!(cache.metrics().hits, 1);
    }

    #[test]
    fn cache_miss() {
        let mut cache = ResultCache::default();
        assert_eq!(cache.lookup(&q(1)), None);
        assert_eq!(cache.metrics().misses, 1);
    }

    #[test]
    fn ttl_eviction() {
        let mut cache = ResultCache::new(1024, Duration::from_millis(10));
        cache.insert(&q(1), vec![(1, 1.0, "a".into())]);
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(cache.lookup(&q(1)), None);
    }

    #[test]
    fn size_cap_eviction() {
        let mut cache = ResultCache::new(2, CACHE_TTL);
        cache.insert(&q(1), vec![(1, 1.0, "a".into())]);
        cache.insert(&q(2), vec![(2, 1.0, "b".into())]);
        cache.insert(&q(3), vec![(3, 1.0, "c".into())]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.lookup(&q(1)), None); // oldest evicted
        assert!(cache.lookup(&q(3)).is_some());
    }

    #[test]
    fn hit_rate_calculation() {
        let mut cache = ResultCache::default();
        cache.insert(&q(1), vec![(1, 1.0, "a".into())]);
        cache.lookup(&q(1));
        cache.lookup(&q(2));
        assert!((cache.hit_rate() - 0.5).abs() < 1e-6);
    }
}
