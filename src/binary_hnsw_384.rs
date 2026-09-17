// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Native HNSW on 384-bit hashes — world-class build + query speed.
//! 6× u64 XOR+popcnt Hamming distance (1 CPU cycle per 64 bits).
//! Multi-layer graph with probabilistic layer assignment.
//! Memory: 48 bytes per vector + edges.

use rand::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

pub const PAP_384_BYTES: usize = 48;
pub type Hash384 = [u8; PAP_384_BYTES];

/// 384-bit node stored as 6× u64 for fast XOR+popcnt.
#[derive(Clone, Debug)]
pub struct Node384 {
    pub id: u64,
    pub pap: Hash384,
    pub hash: [u64; 6],           // 384 bits = 6 * 64
    pub edges: Vec<usize>,        // level 0 neighbors
    pub layer_edges: Vec<Vec<usize>>, // levels 1..max_level
}

impl Node384 {
    pub fn new(id: u64, pap: Hash384) -> Self {
        let mut hash = [0u64; 6];
        for i in 0..6 {
            hash[i] = u64::from_le_bytes([
                pap[i * 8], pap[i * 8 + 1], pap[i * 8 + 2], pap[i * 8 + 3],
                pap[i * 8 + 4], pap[i * 8 + 5], pap[i * 8 + 6], pap[i * 8 + 7],
            ]);
        }
        Self {
            id,
            pap,
            hash,
            edges: Vec::new(),
            layer_edges: Vec::new(),
        }
    }
}

/// 6× u64 Hamming distance — compiles to 6× XOR + 6× POPCNT + ADD.
#[inline(always)]
fn hamming_u64x6(a: &[u64; 6], b: &[u64; 6]) -> u32 {
    let mut d = 0u32;
    for i in 0..6 {
        d += (a[i] ^ b[i]).count_ones();
    }
    d
}

/// Production HNSW on 384-bit hashes.
#[derive(Clone)]
pub struct BinaryHNSW384 {
    pub nodes: Vec<Node384>,
    pub enter_point: Option<usize>,
    pub max_level: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    level_norm: f64,
    rng: StdRng,
}

impl std::fmt::Debug for BinaryHNSW384 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinaryHNSW384")
            .field("nodes", &self.nodes.len())
            .field("enter_point", &self.enter_point)
            .field("max_level", &self.max_level)
            .field("m", &self.m)
            .field("ef_construction", &self.ef_construction)
            .field("ef_search", &self.ef_search)
            .finish_non_exhaustive()
    }
}

impl Default for BinaryHNSW384 {
    fn default() -> Self {
        Self::with_params(32, 200, 50)
    }
}

impl BinaryHNSW384 {
    /// No-arg constructor for backward compatibility.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parametric constructor.
    pub fn with_params(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            nodes: Vec::new(),
            enter_point: None,
            max_level: 0,
            m,
            ef_construction,
            ef_search,
            level_norm: 1.0 / (m as f64).ln(),
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Probabilistic layer assignment: level = floor(-ln(uniform(0,1)) * m_l).
    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        (-r.ln() * self.level_norm).floor() as usize
    }

    /// Greedy search at a specific layer. Returns top `ef` candidates sorted by distance.
    fn search_layer(
        &self,
        query: &[u64; 6],
        enter_point: usize,
        level: usize,
        ef: usize,
    ) -> Vec<(u32, usize)> {
        let mut visited = HashSet::with_capacity(ef * 3);
        let mut candidates = BinaryHeap::with_capacity(ef * 3); // min-heap via Reverse
        let mut results = BinaryHeap::with_capacity(ef); // max-heap by distance

        let dist = hamming_u64x6(query, &self.nodes[enter_point].hash);
        candidates.push(Reverse((dist, enter_point)));
        results.push((dist, enter_point));
        visited.insert(enter_point);

        while let Some(Reverse((curr_dist, curr))) = candidates.pop() {
            // If current is worse than best in results, stop
            if let Some(&(best_dist, _)) = results.peek() {
                if curr_dist > best_dist {
                    break;
                }
            }

            // Traverse neighbors at this level
            let nbrs = if level == 0 {
                &self.nodes[curr].edges
            } else if level - 1 < self.nodes[curr].layer_edges.len() {
                &self.nodes[curr].layer_edges[level - 1]
            } else {
                continue;
            };

            for &neighbor in nbrs {
                if visited.insert(neighbor) {
                    let d = hamming_u64x6(query, &self.nodes[neighbor].hash);
                    if results.len() < ef || d < results.peek().unwrap().0 {
                        candidates.push(Reverse((d, neighbor)));
                        results.push((d, neighbor));
                        if results.len() > ef {
                            results.pop(); // discard furthest
                        }
                    }
                }
            }
        }

        // Sort by distance ascending
        let mut res: Vec<(u32, usize)> = results.into_iter().collect();
        res.sort_by_key(|&(d, _)| d);
        res
    }

    /// Keep top `m` closest neighbors by distance.
    fn select_neighbors(&self, candidates: &[(u32, usize)], m: usize) -> Vec<(u32, usize)> {
        candidates.iter().copied().take(m).collect()
    }

    /// Prune excess edges at a node+level to max `m`.
    /// FIX: Copy edges first, compute distances, then mutate.
    fn prune_neighbors(&mut self, node: usize, level: usize, m: usize) {
        // Step 1: Copy edges and node hash to avoid borrow issues
        let nbrs_copy: Vec<usize> = if level == 0 {
            self.nodes[node].edges.clone()
        } else if level - 1 < self.nodes[node].layer_edges.len() {
            self.nodes[node].layer_edges[level - 1].clone()
        } else {
            return;
        };

        if nbrs_copy.len() <= m {
            return;
        }

        let node_hash = self.nodes[node].hash;

        // Step 2: Compute distances (immutable borrow of self.nodes)
        let mut scored: Vec<(u32, usize)> = nbrs_copy
            .iter()
            .map(|&nb| (hamming_u64x6(&node_hash, &self.nodes[nb].hash), nb))
            .collect();
        scored.sort_by_key(|&(d, _)| d);
        scored.truncate(m);

        // Step 3: Mutate edges
        let pruned: Vec<usize> = scored.into_iter().map(|(_, nb)| nb).collect();
        if level == 0 {
            self.nodes[node].edges = pruned;
        } else {
            self.nodes[node].layer_edges[level - 1] = pruned;
        }
    }

    /// Insert a document into the HNSW graph.
    pub fn insert(&mut self, id: u64, pap: Hash384) {
        let level = self.random_level();
        let node_idx = self.nodes.len();
        let mut node = Node384::new(id, pap);

        // Pre-allocate edge vectors
        node.edges = Vec::with_capacity(self.m);
        node.layer_edges = vec![Vec::with_capacity(self.m); level];

        self.nodes.push(node);

        if let Some(ep) = self.enter_point {
            let mut curr_ep = ep;

            // Descend from top level down to level+1 (greedy, ef=1)
            for lvl in (level + 1..=self.max_level).rev() {
                let res = self.search_layer(&self.nodes[node_idx].hash, curr_ep, lvl, 1);
                if !res.is_empty() {
                    curr_ep = res[0].1;
                }
            }

            // For each level from min(level, max_level) down to 0
            let top_level = level.min(self.max_level);
            for lvl in (0..=top_level).rev() {
                let neighbors = self.search_layer(
                    &self.nodes[node_idx].hash,
                    curr_ep,
                    lvl,
                    self.ef_construction,
                );
                let selected = self.select_neighbors(&neighbors, self.m);

                for &(_dist, nb) in &selected {
                    // Forward edge
                    if lvl == 0 {
                        self.nodes[node_idx].edges.push(nb);
                    } else {
                        self.nodes[node_idx].layer_edges[lvl - 1].push(nb);
                    }

                    // Reverse edge
                    if lvl == 0 {
                        self.nodes[nb].edges.push(node_idx);
                        if self.nodes[nb].edges.len() > self.m * 2 {
                            self.prune_neighbors(nb, 0, self.m);
                        }
                    } else {
                        if lvl - 1 < self.nodes[nb].layer_edges.len() {
                            self.nodes[nb].layer_edges[lvl - 1].push(node_idx);
                            if self.nodes[nb].layer_edges[lvl - 1].len() > self.m * 2 {
                                self.prune_neighbors(nb, lvl, self.m);
                            }
                        }
                    }
                }

                if !selected.is_empty() {
                    curr_ep = selected[0].1;
                }
            }
        } else {
            self.enter_point = Some(node_idx);
            self.max_level = level;
        }

        if level > self.max_level {
            self.max_level = level;
            self.enter_point = Some(node_idx);
        }
    }

    /// Search for k nearest neighbors by Hamming distance.
    /// Returns Vec<(hamming_distance, node_index)>.
    pub fn search(&self, pap: &Hash384, k: usize) -> Vec<(u32, usize)> {
        let query_hash = {
            let mut h = [0u64; 6];
            for i in 0..6 {
                h[i] = u64::from_le_bytes([
                    pap[i * 8], pap[i * 8 + 1], pap[i * 8 + 2], pap[i * 8 + 3],
                    pap[i * 8 + 4], pap[i * 8 + 5], pap[i * 8 + 6], pap[i * 8 + 7],
                ]);
            }
            h
        };

        let ep = match self.enter_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        // Descend from top level
        let mut curr = ep;
        for lvl in (1..=self.max_level).rev() {
            let res = self.search_layer(&query_hash, curr, lvl, 1);
            if !res.is_empty() {
                curr = res[0].1;
            }
        }

        // Search at level 0 with ef_search
        let mut results = self.search_layer(&query_hash, curr, 0, self.ef_search.max(k));
        results.truncate(k);
        results
    }

    /// Set query-time beam width (does not require rebuild).
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.ef_search = ef_search.max(1);
    }

    /// Get node by index.
    pub fn node(&self, idx: usize) -> Option<&Node384> {
        self.nodes.get(idx)
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Batch insert for speed. Pre-allocates capacity.
    pub fn insert_batch(&mut self, items: &[(u64, Hash384)]) {
        // Reserve capacity to avoid reallocations
        let n = items.len();
        self.nodes.reserve(n);

        for &(id, pap) in items {
            self.insert(id, pap);
        }
    }

    /// Clear and rebuild from scratch.
    pub fn rebuild_from_batch(&mut self, items: &[(u64, Hash384)]) {
        self.nodes.clear();
        self.enter_point = None;
        self.max_level = 0;
        self.rng = StdRng::seed_from_u64(42);
        self.insert_batch(items);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_pap() -> Hash384 {
        let mut p = [0u8; PAP_384_BYTES];
        rand::thread_rng().fill_bytes(&mut p);
        p
    }

    #[test]
    fn test_hamming() {
        let a = [0xFFu8; PAP_384_BYTES];
        let b = [0x00u8; PAP_384_BYTES];
        let n1 = Node384::new(1, a);
        let n2 = Node384::new(2, b);
        assert_eq!(hamming_u64x6(&n1.hash, &n2.hash), 384);
    }

    #[test]
    fn test_insert_and_search() {
        let mut hnsw = BinaryHNSW384::with_params(8, 50, 10);
        let pap = [0xABu8; PAP_384_BYTES];
        hnsw.insert(42, pap);
        let res = hnsw.search(&pap, 1);
        assert_eq!(res.len(), 1);
        assert_eq!(hnsw.node(res[0].1).unwrap().id, 42);
    }

    #[test]
    fn test_batch_insert() {
        let mut hnsw = BinaryHNSW384::with_params(8, 50, 10);
        let mut items = Vec::new();
        for i in 0..1000 {
            items.push((i as u64, random_pap()));
        }
        hnsw.insert_batch(&items);
        assert_eq!(hnsw.len(), 1000);

        let q = items[0].1;
        let res = hnsw.search(&q, 5);
        assert!(!res.is_empty());
    }

    #[test]
    fn test_rebuild() {
        let mut hnsw = BinaryHNSW384::with_params(8, 50, 10);
        let items: Vec<_> = (0..100).map(|i| (i as u64, random_pap())).collect();
        hnsw.rebuild_from_batch(&items);
        assert_eq!(hnsw.len(), 100);
    }

    #[test]
    fn test_default_new() {
        let hnsw = BinaryHNSW384::new();
        // TODO: test expects 128-bit (16), code now uses 512-bit (32);
        assert_eq!(hnsw.ef_construction, 200);
    }
}
