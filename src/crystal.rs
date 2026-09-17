// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Phase 4: Static Crystal Mesh
use std::collections::BinaryHeap;
pub const PAP_BYTES: usize = 64;
#[derive(Clone, Debug)]
pub struct Slot { pub id: u64, pub pap: [u8; PAP_BYTES], pub edges: Vec<usize> }
#[derive(Clone, Debug)]
struct Candidate { dist: f32, idx: usize, hops: usize }
impl PartialEq for Candidate { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Eq for Candidate {}
impl PartialOrd for Candidate { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { o.dist.partial_cmp(&self.dist) } }
impl Ord for Candidate { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.partial_cmp(o).unwrap_or(std::cmp::Ordering::Equal) } }
pub struct CrystalMesh { pub slots: Vec<Slot>, buckets: Vec<Vec<usize>>, bucket_bits: usize }
impl CrystalMesh {
    pub fn new() -> Self { Self::with_capacity(0) }
    pub fn with_capacity(cap: usize) -> Self { Self::with_bucket_bits(cap, 16) }
    pub fn with_bucket_bits(cap: usize, bits: usize) -> Self { Self { slots: Vec::with_capacity(cap), buckets: Vec::new(), bucket_bits: bits } }
    pub fn insert(&mut self, id: u64, pap: &[u8; PAP_BYTES]) { self.slots.push(Slot { id, pap: *pap, edges: Vec::new() }); }
    pub fn len(&self) -> usize { self.slots.len() }
    pub fn is_empty(&self) -> bool { self.slots.is_empty() }
                pub fn build_edges(&mut self, max_edges: usize) {
        let n = self.slots.len();
        if n == 0 { return; }

        // Build bucket index
        let n_buckets = 1 << self.bucket_bits;
        self.buckets = vec![Vec::new(); n_buckets];
        let hashes: Vec<usize> = self.slots.iter().map(|s| self.bucket_hash(&s.pap)).collect();
        for (i, h) in hashes.iter().enumerate() { self.buckets[*h].push(i); }

        // Phase 1: Compute all pairwise distances and find top-k for each node
        // For N > 50K, use bucket sampling to avoid O(N^2)
        let use_exact = n <= 50_000;
        let mut all_neighbors: Vec<Vec<(f32, usize)>> = vec![Vec::new(); n];

        for i in 0..n {
            let mut candidates: Vec<(f32, usize)> = Vec::new();

            if use_exact {
                // O(N^2) — exact all-pairs for small N
                for j in 0..n {
                    if i == j { continue; }
                    let d = pap_distance(&self.slots[i].pap, &self.slots[j].pap);
                    candidates.push((d, j));
                }
            } else {
                // O(N * bucket_size) — sample from same + nearby buckets
                let b = hashes[i];
                for &j in &self.buckets[b] {
                    if i != j { candidates.push((pap_distance(&self.slots[i].pap, &self.slots[j].pap), j)); }
                }
                // Expand rings if sparse
                let mut ring = 1;
                while candidates.len() < max_edges * 4 && ring <= n_buckets / 2 {
                    let b1 = (b + ring) & (n_buckets - 1);
                    let b2 = (b + n_buckets - ring) & (n_buckets - 1);
                    for &j in &self.buckets[b1] { if i != j { candidates.push((pap_distance(&self.slots[i].pap, &self.slots[j].pap), j)); } }
                    for &j in &self.buckets[b2] { if i != j { candidates.push((pap_distance(&self.slots[i].pap, &self.slots[j].pap), j)); } }
                    ring += 1;
                }
            }

            // Sort descending by PAP, keep top-k
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(max_edges);
            all_neighbors[i] = candidates;
        }

        // Phase 2: Make edges bidirectional (mutual k-NN)
        // Collect all edges first, then add reverse edges
        let mut reverse_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            for (_, j) in &all_neighbors[i] {
                reverse_edges[*j].push(i);
            }
        }

        // Add reverse edges to each node's neighbor list
        for j in 0..n {
            for i in &reverse_edges[j] {
                // Check if i is already in j's neighbors
                let already_connected = all_neighbors[j].iter().any(|(_, idx)| *idx == *i);
                if !already_connected {
                    let d = pap_distance(&self.slots[j].pap, &self.slots[*i].pap);
                    all_neighbors[j].push((d, *i));
                }
            }
            // Keep top-k after adding reverse edges
            all_neighbors[j].sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            all_neighbors[j].truncate(max_edges);
        }

        // Phase 3: Write edges to slots
        for i in 0..n {
            self.slots[i].edges = all_neighbors[i].iter().map(|(_, j)| *j).collect();
        }
    }
        pub fn query_graph(&self, pap: &[u8; PAP_BYTES], top_k: usize, max_hops: usize) -> Vec<(f32, &Slot)> {
        if self.slots.is_empty() { return Vec::new(); }

        let mut visited = vec![false; self.slots.len()];
        let mut results: Vec<(f32, usize)> = Vec::new();

        // Find the closest seed node in the query's bucket
        let sb = self.bucket_hash(pap);
        let mut best_seed = 0usize;
        let mut best_seed_dist = 0.0f32;

        for &i in &self.buckets[sb] {
            let d = pap_distance(pap, &self.slots[i].pap);
            if d > best_seed_dist {
                best_seed_dist = d;
                best_seed = i;
            }
            visited[i] = true;
            results.push((d, i));
        }

        // Fallback: if bucket empty, scan first 100 slots
        if best_seed_dist == 0.0 && !self.slots.is_empty() {
            for i in 0..self.slots.len().min(100) {
                if !visited[i] {
                    let d = pap_distance(pap, &self.slots[i].pap);
                    if d > best_seed_dist {
                        best_seed_dist = d;
                        best_seed = i;
                    }
                    visited[i] = true;
                    results.push((d, i));
                }
            }
        }

        // Greedy walk: always expand the best unexpanded node
        let mut to_expand: Vec<(f32, usize, usize)> = Vec::new(); // (dist, idx, hops)
        to_expand.push((best_seed_dist, best_seed, 0));

        while let Some((dist, idx, hops)) = to_expand.pop() {
            if hops >= max_hops { continue; }

            // Expand this node: add all its neighbors
            for &e in &self.slots[idx].edges {
                if !visited[e] {
                    visited[e] = true;
                    let d = pap_distance(pap, &self.slots[e].pap);
                    results.push((d, e));
                    to_expand.push((d, e, hops + 1));
                }
            }

            // Keep to_expand sorted by distance (best last, so we pop best first)
            to_expand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Sort all results by distance
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Deduplicate by slot index, keep top_k
        let mut seen = std::collections::HashSet::new();
        let mut unique: Vec<(f32, &Slot)> = Vec::new();
        for (score, idx) in results {
            if seen.insert(idx) {
                unique.push((score, &self.slots[idx]));
                if unique.len() >= top_k { break; }
            }
        }

        unique
    }
    pub fn query_k(&self, pap: &[u8; PAP_BYTES], top_k: usize) -> Vec<(f32, &Slot)> {
        let mut r: Vec<(f32, &Slot)> = self.slots.iter().map(|s| (pap_distance(pap, &s.pap), s)).collect();
        r.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        r.truncate(top_k);
        r
    }
    pub fn query_direct(&self, pap: &[u8; PAP_BYTES]) -> Option<u64> {
        self.query_k(pap, 1).first().map(|r| r.1.id)
    }
    pub fn query(&self, pap: &[u8; PAP_BYTES], top_k: usize) -> Vec<(f32, &Slot)> { self.query_k(pap, top_k) }
    pub fn query_crystal(&self, pap: &[u8; PAP_BYTES]) -> Option<u64> { self.query_direct(pap) }
    pub fn get_neighbors(&self, entity_id: u64) -> Option<Vec<u64>> {
        let idx = self.slots.iter().position(|s| s.id == entity_id)?;
        if !self.slots[idx].edges.is_empty() {
            return Some(self.slots[idx].edges.iter().map(|&i| self.slots[i].id).collect());
        }
        let mut candidates: Vec<(f32, u64)> = self.slots
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, s)| (pap_distance(&self.slots[idx].pap, &s.pap), s.id))
            .collect();
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        Some(candidates.into_iter().take(3).map(|(_, id)| id).collect())
    }
    pub fn get_cluster(&self, entity_id: u64) -> Option<Vec<u64>> {
        let slot = self.slots.iter().find(|s| s.id == entity_id)?;
        if !self.buckets.is_empty() {
            let b = self.bucket_hash(&slot.pap);
            return Some(self.buckets[b].iter().map(|&i| self.slots[i].id).collect());
        }
        Some(self.slots.iter().map(|s| s.id).collect())
    }
    fn bucket_hash(&self, pap: &[u8; PAP_BYTES]) -> usize {
        let b = self.bucket_bits;
        let n = (b + 7) / 8;
        let mut h = 0usize;
        for i in 0..n.min(PAP_BYTES) { h = (h << 8) | (pap[i] as usize); }
        h & ((1 << b) - 1)
    }
}
pub fn pap_distance(a: &[u8; PAP_BYTES], b: &[u8; PAP_BYTES]) -> f32 {
    let mut x = 0u32; let mut ac = 0u32; let mut bc = 0u32;
    for i in 0..PAP_BYTES { x += (a[i] & b[i]).count_ones(); ac += a[i].count_ones(); bc += b[i].count_ones(); }
    if ac == 0 || bc == 0 { return 0.0; }
    (x as f32) / ((ac as f32) * (bc as f32)).sqrt()
}
pub fn pap_from_seed(seed: u64) -> [u8; PAP_BYTES] {
    let mut p = [0u8; PAP_BYTES]; let mut s = seed.wrapping_mul(0x9e3779b97f4a7c15);
    for i in 0..PAP_BYTES { s = s.wrapping_mul(0x2545f4914f6cdd1d).wrapping_add(1); p[i] = (s >> 56) as u8; }
    p
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_pap_distance_self() { let p = pap_from_seed(42); assert!((pap_distance(&p, &p) - 1.0).abs() < 1e-6); }
    #[test] fn test_crystal_build_1m() {
        let t = std::time::Instant::now();
        let mut m = CrystalMesh::with_capacity(1_000_000);
        for i in 0..1_000_000u64 { m.insert(i, &pap_from_seed(i)); }
        println!("Built 1M in {:?}", t.elapsed());
        assert!(t.elapsed().as_secs_f64() < 300.0);
    }
    #[test] fn test_crystal_query_1m() {
        let mut m = CrystalMesh::with_capacity(1_000_000);
        for i in 0..1_000_000u64 { m.insert(i, &pap_from_seed(i)); }
        let q = pap_from_seed(500_000);
        let r = m.query(&q, 10);
        assert_eq!(r[0].1.id, 500_000); assert!(r[0].0 > 0.99);
    }
    #[test] fn test_crystal_edges() {
        let mut m = CrystalMesh::with_bucket_bits(10_000, 8);
        for i in 0..10_000u64 { m.insert(i, &pap_from_seed(i)); }
        m.build_edges(8);
        let ec: usize = m.slots.iter().map(|s| s.edges.len()).sum();
        println!("Edges: {}", ec); assert!(ec > 0);
        for s in &m.slots { for &e in &s.edges { assert!(e < m.slots.len()); } }
    }
    #[test]
    fn test_crystal_graph_query() {
        let mut m = CrystalMesh::with_bucket_bits(10_000, 10);
        for i in 0..10_000u64 { m.insert(i, &pap_from_seed(i)); }
        m.build_edges(8);
        let q = pap_from_seed(5_000);
        let g = m.query_graph(&q, 10, 5);
        let d = m.query_k(&q, 10);
        assert!(!g.is_empty(), "Graph query returned empty");
        assert_eq!(g[0].1.id, 5_000, "Graph missed exact match");
        let ds: std::collections::HashSet<u64> = d.iter().take(10).map(|r| r.1.id).collect();
        let gs: std::collections::HashSet<u64> = g.iter().take(20).map(|r| r.1.id).collect();
        let o = ds.intersection(&gs).count();
        println!("Graph recall: {}/10", o);
        assert!(o >= 1, "Graph recall too low: {}/10", o);
    }
    #[test] fn test_crystal_build_1m_with_edges() {
        let t = std::time::Instant::now();
        let mut m = CrystalMesh::with_capacity(1_000_000);
        for i in 0..1_000_000u64 { m.insert(i, &pap_from_seed(i)); }
        m.build_edges(8);
        println!("Built 1M+edges in {:?}", t.elapsed());
        assert!(t.elapsed().as_secs_f64() < 600.0);
    }
    #[test] fn test_backward_compat() {
        let mut m = CrystalMesh::new();
        m.insert(1, &pap_from_seed(1));
        m.insert(2, &pap_from_seed(2));
        m.build_edges(4);
        assert_eq!(m.len(), 2);
        assert!(!m.is_empty());
        assert!(m.query_crystal(&pap_from_seed(1)).is_some());
        assert!(m.get_neighbors(1).is_some());
        assert!(m.get_cluster(1).is_some());
        assert_eq!(m.query_direct(&pap_from_seed(1)), Some(1));
    }
}
