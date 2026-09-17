// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! hybrid_mesh.rs — Dual-mode Crystal Mesh: 128-bit (fast) + 512-bit (accurate)
//!
//! Three modes:
//!   Fast128  — 16-byte PAPs only. Edge/IoT/mobile. Fastest.
//!   Full512  — 64-byte PAPs only. Desktop/server. Most accurate.
//!   Hybrid   — 128-bit coarse filter → 512-bit re-rank top-K. Production.
//!
//! Auto-detect: if only 128-bit data exists, forces Fast128.
//!              if only 512-bit data exists, forces Full512.
//!              if both exist, uses Hybrid (or whatever you set).
//!
//! Rebuild contract: `insert()` appends a slot only. Bucket and edge
//! structures are (re)built by `build_edges()` — after ANY batch of
//! inserts, call `build_edges()` before graph/bucket queries, or those
//! queries silently miss the new records (a `dirty` flag makes
//! `query_graph` fall back to the correct linear scan instead).
//!
//! Graph-scope note: `query_graph` is a within-bucket approximation —
//! seeds and edges cover only the query's own 16-bit-prefix bucket.
//! Neighbors in other buckets are reachable only via `query_k` /
//! `query_hybrid` (linear coarse scan). Cross-bucket recall of
//! `query_auto` is bounded accordingly; this is by design, stated here
//! because it is not obvious from the API.

use crate::spectral_coords::SpectralCoords;
use serde::{Serialize, Deserialize};
use std::collections::{BinaryHeap, HashMap};
#[cfg(feature = "living_mesh")]
use crate::energy_overlay::EnergyOverlay;
#[cfg(feature = "living_mesh")]
use crate::temporal_tracker::TemporalTracker;

pub const PAP_128_BYTES: usize = 16;
pub const PAP_512_BYTES: usize = 64;

/// Gravity tracking for dynamic mesh movement
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PaperGravity {
    pub paper_id: u64,
    pub query_count: u64,
    pub last_query_ts: u64,
    pub co_occurrence: HashMap<u64, u32>, // paper_id -> times seen together
}

/// Logical buckets for autopoiesis-driven re-bucketing.
/// Independent of the PAP-hash crystal buckets; this is the “living” partition.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DynamicBuckets {
    pub paper_bucket: HashMap<u64, u32>,
    pub buckets: HashMap<u32, Vec<u64>>,
}

impl DynamicBuckets {
    pub fn get(&self, paper_id: u64) -> Option<u32> {
        self.paper_bucket.get(&paper_id).copied()
    }

    pub fn remove(&mut self, paper_id: u64) {
        if let Some(old) = self.paper_bucket.remove(&paper_id) {
            if let Some(list) = self.buckets.get_mut(&old) {
                list.retain(|&id| id != paper_id);
            }
        }
    }

    pub fn insert(&mut self, paper_id: u64, bucket: u32) {
        self.remove(paper_id);
        self.paper_bucket.insert(paper_id, bucket);
        self.buckets.entry(bucket).or_default().push(paper_id);
    }

    pub fn move_paper(&mut self, paper_id: u64, new_bucket: u32) -> bool {
        if self.get(paper_id) == Some(new_bucket) {
            return false;
        }
        self.insert(paper_id, new_bucket);
        true
    }

    pub fn count(&self) -> usize {
        self.paper_bucket.len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PAPMode {
    Fast128,
    Full512,
    Hybrid,
}

#[derive(Clone, Debug)]
pub struct Slot128 { pub id: u64, pub pap: [u8; PAP_128_BYTES], pub edges: Vec<usize> }

#[derive(Clone, Debug)]
pub struct Slot512 { pub id: u64, pub pap: [u8; PAP_512_BYTES], pub edges: Vec<usize> }


#[derive(Clone, Debug)]
struct Cand128 { dist: f32, idx: usize, hops: usize }
#[derive(Clone, Debug)]
struct Cand512 { dist: f32, idx: usize, hops: usize }

impl PartialEq for Cand128 { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Eq for Cand128 {}
impl PartialOrd for Cand128 { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { o.dist.partial_cmp(&self.dist) } }
impl Ord for Cand128 { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.partial_cmp(o).unwrap_or(std::cmp::Ordering::Equal) } }

impl PartialEq for Cand512 { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Eq for Cand512 {}
impl PartialOrd for Cand512 { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { o.dist.partial_cmp(&self.dist) } }
impl Ord for Cand512 { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.partial_cmp(o).unwrap_or(std::cmp::Ordering::Equal) } }

#[derive(Clone)]
pub struct CrystalMesh128 {
    pub slots: Vec<Slot128>, buckets: Vec<Vec<usize>>, bucket_bits: usize,
    /// Set by insert(), cleared by build_edges(). While dirty, graph
    /// queries fall back to the linear scan so new records are never
    /// silently missed (see module docs: rebuild contract).
    dirty: bool,
}

#[derive(Clone)]
pub struct CrystalMesh512 {
    pub slots: Vec<Slot512>, buckets: Vec<Vec<usize>>, bucket_bits: usize,
    dirty: bool,
    /// id -> slot index, maintained incrementally by insert() so hybrid
    /// re-rank never does a linear slots scan per candidate.
    by_id: HashMap<u64, usize>,
}

#[derive(Clone)]
pub struct HybridMesh {
    pub mode: PAPMode,
    pub coarse: CrystalMesh128,
    pub fine: CrystalMesh512,
    pub hybrid_top_k_coarse: usize,
    /// M3.3: precomputed spectral coordinates per paper id (12 bytes each)
    pub spectral_coords: HashMap<u64, SpectralCoords>,
    /// M3.2: exact 8-byte prefix map for O(1) identity lookups on the lazy
    /// path. Values are Vecs because two records can share a 64-bit prefix
    /// (last-write-wins here used to silently return the wrong identity).
    prefix_map: HashMap<u64, Vec<u64>>,
    /// Gravity tracking for dynamic mesh movement
    pub gravity_map: HashMap<u64, PaperGravity>,
    /// Logical dynamic buckets for autopoiesis-driven re-bucketing.
    pub dynamic_buckets: DynamicBuckets,
    /// Spectral eigenvectors from tensor_spectral.rs for drift checking.
    pub spectral_eigenvectors: Option<Vec<f32>>,
    /// Energy Overlay — per-bucket heat map (v3.6 Phase 2).
    #[cfg(feature = "living_mesh")]
    pub energy_overlay: Option<EnergyOverlay>,
    /// Temporal Tracker — mesh history ring (v3.6 Phase 4).
    #[cfg(feature = "living_mesh")]
    pub temporal_tracker: Option<TemporalTracker>,
}

impl CrystalMesh128 {
    pub fn new() -> Self { Self::with_capacity(0) }
    pub fn with_capacity(cap: usize) -> Self { Self::with_bucket_bits(cap, 16) }
    pub fn with_bucket_bits(cap: usize, bits: usize) -> Self { Self { slots: Vec::with_capacity(cap), buckets: Vec::new(), bucket_bits: bits, dirty: false } }
    /// Appends a slot. Graph/bucket queries see this record only after
    /// `build_edges()` is called (module docs: rebuild contract).
    pub fn insert(&mut self, id: u64, pap: &[u8; PAP_128_BYTES]) { self.slots.push(Slot128 { id, pap: *pap, edges: Vec::new() }); self.dirty = true; }
    pub fn len(&self) -> usize { self.slots.len() }
    pub fn is_empty(&self) -> bool { self.slots.is_empty() }

    pub fn build_edges(&mut self, max_edges: usize) {
        let n = 1 << self.bucket_bits;
        self.buckets = vec![Vec::new(); n];
        self.dirty = false;
        self.dirty = false;
        let hashes: Vec<usize> = self.slots.iter().map(|s| self.bucket_hash(&s.pap)).collect();
        for (i, h) in hashes.into_iter().enumerate() { self.buckets[h].push(i); }
        for i in 0..self.slots.len() {
            let b = self.bucket_hash(&self.slots[i].pap);
            let mut c: Vec<(f32, usize)> = self.buckets[b].iter().filter(|&&o| o != i).map(|&o| (pap_distance_128(&self.slots[i].pap, &self.slots[o].pap), o)).collect();
            c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            self.slots[i].edges = c.into_iter().take(max_edges).map(|(_, i)| i).collect();
        }
    }

    pub fn query_graph(&self, pap: &[u8; PAP_128_BYTES], top_k: usize, max_hops: usize) -> Vec<(f32, &Slot128)> {
        if self.slots.is_empty() { return Vec::new(); }
        if self.buckets.is_empty() || self.dirty { return self.query_k(pap, top_k); }
        let mut v = vec![false; self.slots.len()];
        let mut h = BinaryHeap::new();
        let mut r = Vec::new();
        let sb = self.bucket_hash(pap);
        if sb >= self.buckets.len() { return self.query_k(pap, top_k); }
        for &i in &self.buckets[sb] { if !v[i] { v[i] = true; h.push(Cand128 { dist: pap_distance_128(pap, &self.slots[i].pap), idx: i, hops: 0 }); } }
        while let Some(c) = h.pop() {
            r.push((c.dist, &self.slots[c.idx]));
            if c.hops < max_hops { for &e in &self.slots[c.idx].edges { if !v[e] { v[e] = true; h.push(Cand128 { dist: pap_distance_128(pap, &self.slots[e].pap), idx: e, hops: c.hops + 1 }); } } }
        }
        r.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        r.truncate(top_k); r
    }

    pub fn query_k(&self, pap: &[u8; PAP_128_BYTES], top_k: usize) -> Vec<(f32, &Slot128)> {
        let mut r: Vec<(f32, &Slot128)> = self.slots.iter().map(|s| (pap_distance_128(pap, &s.pap), s)).collect();
        r.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        r.truncate(top_k); r
    }

    pub fn query_direct(&self, pap: &[u8; PAP_128_BYTES]) -> Option<u64> { self.query_k(pap, 1).first().map(|r| r.1.id) }
    pub fn query(&self, pap: &[u8; PAP_128_BYTES], top_k: usize) -> Vec<(f32, &Slot128)> { self.query_k(pap, top_k) }

    /// Trinity fast path: exact-match PAPs inside a single predicted bucket.
    pub fn query_bucket_exact(&self, bucket_id: usize, pap: &[u8; PAP_128_BYTES], top_k: usize) -> Vec<(f32, u64)> {
        if bucket_id >= self.buckets.len() { return Vec::new(); }
        let mut results = Vec::new();
        for &idx in &self.buckets[bucket_id] {
            if self.slots[idx].pap == *pap {
                results.push((1.0, self.slots[idx].id));
                if results.len() >= top_k { break; }
            }
        }
        results
    }

    pub fn bucket_hash(&self, pap: &[u8; PAP_128_BYTES]) -> usize {
        let b = self.bucket_bits;
        let n = (b + 7) / 8;
        let mut h = 0usize;
        for i in 0..n.min(PAP_128_BYTES) { h = (h << 8) | (pap[i] as usize); }
        h & ((1 << b) - 1)
    }
}

impl CrystalMesh512 {
    pub fn new() -> Self { Self::with_capacity(0) }
    pub fn with_capacity(cap: usize) -> Self { Self::with_bucket_bits(cap, 16) }
    pub fn with_bucket_bits(cap: usize, bits: usize) -> Self { Self { slots: Vec::with_capacity(cap), buckets: Vec::new(), bucket_bits: bits, dirty: false, by_id: HashMap::with_capacity(cap) } }
    /// Appends a slot and indexes it by id. Graph/bucket queries see this
    /// record only after `build_edges()` (module docs: rebuild contract).
    pub fn insert(&mut self, id: u64, pap: &[u8; PAP_512_BYTES]) {
        self.by_id.insert(id, self.slots.len());
        self.slots.push(Slot512 { id, pap: *pap, edges: Vec::new() });
        self.dirty = true;
    }
    pub fn len(&self) -> usize { self.slots.len() }
    pub fn is_empty(&self) -> bool { self.slots.is_empty() }

    pub fn build_edges(&mut self, max_edges: usize) {
        let n = 1 << self.bucket_bits;
        self.buckets = vec![Vec::new(); n];
        let hashes: Vec<usize> = self.slots.iter().map(|s| self.bucket_hash(&s.pap)).collect();
        for (i, h) in hashes.into_iter().enumerate() { self.buckets[h].push(i); }
        for i in 0..self.slots.len() {
            let b = self.bucket_hash(&self.slots[i].pap);
            let mut c: Vec<(f32, usize)> = self.buckets[b].iter().filter(|&&o| o != i).map(|&o| (pap_distance_512(&self.slots[i].pap, &self.slots[o].pap), o)).collect();
            c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            self.slots[i].edges = c.into_iter().take(max_edges).map(|(_, i)| i).collect();
        }
    }

    pub fn query_graph(&self, pap: &[u8; PAP_512_BYTES], top_k: usize, max_hops: usize) -> Vec<(f32, &Slot512)> {
        if self.slots.is_empty() { return Vec::new(); }
        if self.buckets.is_empty() || self.dirty { return self.query_k(pap, top_k); }
        let mut v = vec![false; self.slots.len()];
        let mut h = BinaryHeap::new();
        let mut r = Vec::new();
        let sb = self.bucket_hash(pap);
        if sb >= self.buckets.len() { return self.query_k(pap, top_k); }
        for &i in &self.buckets[sb] { if !v[i] { v[i] = true; h.push(Cand512 { dist: pap_distance_512(pap, &self.slots[i].pap), idx: i, hops: 0 }); } }
        while let Some(c) = h.pop() {
            r.push((c.dist, &self.slots[c.idx]));
            if c.hops < max_hops { for &e in &self.slots[c.idx].edges { if !v[e] { v[e] = true; h.push(Cand512 { dist: pap_distance_512(pap, &self.slots[e].pap), idx: e, hops: c.hops + 1 }); } } }
        }
        r.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        r.truncate(top_k); r
    }

    pub fn query_k(&self, pap: &[u8; PAP_512_BYTES], top_k: usize) -> Vec<(f32, &Slot512)> {
        let mut r: Vec<(f32, &Slot512)> = self.slots.iter().map(|s| (pap_distance_512(pap, &s.pap), s)).collect();
        r.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        r.truncate(top_k); r
    }

    pub fn query_direct(&self, pap: &[u8; PAP_512_BYTES]) -> Option<u64> { self.query_k(pap, 1).first().map(|r| r.1.id) }
    pub fn query(&self, pap: &[u8; PAP_512_BYTES], top_k: usize) -> Vec<(f32, &Slot512)> { self.query_k(pap, top_k) }

    /// Trinity fast path: exact-match PAPs inside a single predicted bucket.
    pub fn query_bucket_exact(&self, bucket_id: usize, pap: &[u8; PAP_512_BYTES], top_k: usize) -> Vec<(f32, u64)> {
        if bucket_id >= self.buckets.len() { return Vec::new(); }
        let mut results = Vec::new();
        for &idx in &self.buckets[bucket_id] {
            if self.slots[idx].pap == *pap {
                results.push((1.0, self.slots[idx].id));
                if results.len() >= top_k { break; }
            }
        }
        results
    }

    pub fn bucket_hash(&self, pap: &[u8; PAP_512_BYTES]) -> usize {
        let b = self.bucket_bits;
        let n = (b + 7) / 8;
        let mut h = 0usize;
        for i in 0..n.min(PAP_512_BYTES) { h = (h << 8) | (pap[i] as usize); }
        h & ((1 << b) - 1)
    }
}

impl HybridMesh {
    pub fn new(coarse_cap: usize, fine_cap: usize) -> Self {
        Self {
            mode: PAPMode::Hybrid,
            coarse: CrystalMesh128::with_capacity(coarse_cap),
            fine: CrystalMesh512::with_capacity(fine_cap),
            hybrid_top_k_coarse: 100,
            spectral_coords: HashMap::with_capacity(coarse_cap.max(fine_cap)),
            prefix_map: HashMap::with_capacity(coarse_cap),
            gravity_map: HashMap::with_capacity(coarse_cap),
            dynamic_buckets: DynamicBuckets::default(),
            spectral_eigenvectors: None,
            #[cfg(feature = "living_mesh")]
            energy_overlay: Some(EnergyOverlay::new()),
            #[cfg(feature = "living_mesh")]
            temporal_tracker: Some(TemporalTracker::new()),
        }
    }
    pub fn set_mode(&mut self, mode: PAPMode) { self.mode = mode; }

    pub fn insert_dual(&mut self, id: u64, pap_128: &[u8; PAP_128_BYTES], pap_512: &[u8; PAP_512_BYTES]) {
        self.coarse.insert(id, pap_128);
        self.fine.insert(id, pap_512);
        self.spectral_coords.insert(id, SpectralCoords::from_hash(pap_512));
        let prefix = u64::from_le_bytes(pap_128[..8].try_into().unwrap());
        self.prefix_map.entry(prefix).or_default().push(id);
    }
    pub fn insert_128(&mut self, id: u64, pap: &[u8; PAP_128_BYTES]) {
        self.coarse.insert(id, pap);
        let prefix = u64::from_le_bytes(pap[..8].try_into().unwrap());
        self.prefix_map.entry(prefix).or_default().push(id);
    }
    pub fn insert_512(&mut self, id: u64, pap: &[u8; PAP_512_BYTES]) { self.fine.insert(id, pap); }

    /// O(1) exact match via 8-byte prefix hash. Returns Some(id) if found.
    /// On the (rare) 64-bit-prefix collision, returns the first matching
    /// record's id; callers needing full disambiguation must re-check the
    /// candidate's full PAP against their query context.
    pub fn query_prefix_exact(&self, pap_128: &[u8; PAP_128_BYTES]) -> Option<u64> {
        let prefix = u64::from_le_bytes(pap_128[..8].try_into().unwrap());
        self.prefix_map.get(&prefix).and_then(|v| v.first().copied())
    }

    /// Rebuild the exact prefix map from the coarse slots (used after loading a snapshot).
    pub fn rebuild_prefix_map(&mut self) {
        self.prefix_map.clear();
        self.prefix_map.reserve(self.coarse.slots.len());
        for slot in &self.coarse.slots {
            let prefix = u64::from_le_bytes(slot.pap[..8].try_into().unwrap());
            self.prefix_map.entry(prefix).or_default().push(slot.id);
        }
    }

    /// Store spectral eigenvectors for later FFI export.
    pub fn set_spectral_eigenvectors(&mut self, eigenvectors: Vec<f32>) {
        self.spectral_eigenvectors = Some(eigenvectors);
    }

    /// Move a paper to a new logical bucket. Returns true if moved.
    pub fn reinsert_paper(&mut self, paper_id: u64, new_bucket: u32) -> bool {
        if !self.dynamic_buckets.move_paper(paper_id, new_bucket) {
            return false;
        }
        if let Some(g) = self.gravity_map.get_mut(&paper_id) {
            g.query_count = 0; // fresh start in new bucket
        }
        true
    }

    /// Current logical bucket for a paper, if any.
    pub fn logical_bucket(&self, paper_id: u64) -> Option<u32> {
        self.dynamic_buckets.get(paper_id)
    }

    /// Return the paper id most often co-occurring with this one, if any.
    pub fn top_cooccurring_paper(&self, paper_id: u64) -> Option<u64> {
        self.gravity_map
            .get(&paper_id)
            .and_then(|g| {
                g.co_occurrence
                    .iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(id, _)| *id)
            })
    }

    /// Fast path: look at the coarse bucket only. If the bucket is small,
    /// score candidates by 128-bit similarity and return immediately.
    /// Returns `None` when the fast path cannot be used (empty/large bucket).
    pub fn query_fast_path(&self, pap_128: &[u8; PAP_128_BYTES], top_k: usize) -> Option<Vec<(f32, u64)>> {
        if self.coarse.is_empty() {
            return None;
        }
        const FAST_PATH_MAX_BUCKET: usize = 64;
        let bucket = self.coarse.bucket_hash(pap_128);
        let indices = self.coarse.buckets.get(bucket)?;
        if indices.len() > FAST_PATH_MAX_BUCKET {
            return None;
        }
        let mut results: Vec<(f32, u64)> = indices
            .iter()
            .map(|&idx| {
                let slot = &self.coarse.slots[idx];
                let score = pap_distance_128(pap_128, &slot.pap);
                (score, slot.id)
            })
            .collect();
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Some(results)
    }

    /// AUTO-DETECT QUERY
    ///   - Only 128-bit data → forces Fast128 (fastest)
    ///   - Only 512-bit data → forces Full512 (most accurate)
    ///   - Both exist → uses set mode (default Hybrid)
    pub fn query_auto(&self, pap_128: &[u8; PAP_128_BYTES], pap_512: &[u8; PAP_512_BYTES], top_k: usize) -> Vec<(f32, u64)> {
        let coarse_empty = self.coarse.is_empty();
        let fine_empty = self.fine.is_empty();

        if !coarse_empty && fine_empty {
            self.coarse.query_graph(pap_128, top_k, 32).into_iter().map(|(d, s)| (d, s.id)).collect()
        } else if coarse_empty && !fine_empty {
            self.fine.query_graph(pap_512, top_k, 32).into_iter().map(|(d, s)| (d, s.id)).collect()
        } else if !coarse_empty && !fine_empty {
            // Both indexes have data — query both and merge top-k
            let mut r128: Vec<(f32, u64)> = self.coarse.query_graph(pap_128, top_k, 32).into_iter().map(|(d, s)| (d, s.id)).collect();
            let mut r512: Vec<(f32, u64)> = self.fine.query_graph(pap_512, top_k, 32).into_iter().map(|(d, s)| (d, s.id)).collect();
            
            r128.extend(r512);
            r128.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            
            let mut seen = std::collections::HashSet::new();
            let mut merged = Vec::new();
            for (score, id) in r128 {
                if seen.insert(id) {
                    merged.push((score, id));
                    if merged.len() >= top_k { break; }
                }
            }
            merged
        } else {
            Vec::new()
        }
    }

    /// TRINITY-DRIVEN fast path: ask the Predictor for a bucket, exact-match there first.
    #[cfg(feature = "trinity")]
    pub fn query_auto_trinity(&self, query_id: u64, pap_128: &[u8; PAP_128_BYTES], pap_512: &[u8; PAP_512_BYTES], top_k: usize) -> Vec<(f32, u64)> {
        let predicted_bucket = crate::trinity::ffi::yp_trinity_predict_bucket(query_id);

        if predicted_bucket >= 0 {
            let bucket_id = predicted_bucket as usize;

            // 1. Try coarse (128-bit) exact match in the hinted bucket
            let coarse_hits = self.coarse.query_bucket_exact(bucket_id, pap_128, top_k);
            if !coarse_hits.is_empty() {
                crate::trinity::ffi::yp_trinity_record_result(query_id, predicted_bucket, predicted_bucket);
                return coarse_hits;
            }

            // 2. Try fine (512-bit) exact match in the hinted bucket
            let fine_hits = self.fine.query_bucket_exact(bucket_id, pap_512, top_k);
            if !fine_hits.is_empty() {
                crate::trinity::ffi::yp_trinity_record_result(query_id, predicted_bucket, predicted_bucket);
                return fine_hits;
            }

            // MISS — penalize predictor and fall back
            crate::trinity::ffi::yp_trinity_record_result(query_id, predicted_bucket, -1);
        }

        self.query_auto(pap_128, pap_512, top_k)
    }

    /// MANUAL query — respects your mode setting exactly
    pub fn query(&self, pap_128: &[u8; PAP_128_BYTES], pap_512: &[u8; PAP_512_BYTES], top_k: usize) -> Vec<(f32, u64)> {
        match self.mode {
            PAPMode::Fast128 => self.coarse.query_graph(pap_128, top_k, 32).into_iter().map(|(d, s)| (d, s.id)).collect(),
            PAPMode::Full512 => self.fine.query_graph(pap_512, top_k, 32).into_iter().map(|(d, s)| (d, s.id)).collect(),
            PAPMode::Hybrid => self.query_hybrid(pap_128, pap_512, top_k),
        }
    }

    fn query_hybrid(&self, pap_128: &[u8; PAP_128_BYTES], pap_512: &[u8; PAP_512_BYTES], top_k: usize) -> Vec<(f32, u64)> {
        let coarse_results = self.coarse.query_k(pap_128, self.hybrid_top_k_coarse);
        let mut fine_results: Vec<(f32, u64)> = coarse_results.iter()
            .filter_map(|(_, slot)| {
                self.fine.by_id.get(&slot.id)
                    .map(|&i| (pap_distance_512(pap_512, &self.fine.slots[i].pap), slot.id))
            }).collect();
        fine_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        fine_results.truncate(top_k);
        fine_results
    }

    pub fn build_edges(&mut self, max_edges: usize) {
        if !self.coarse.is_empty() { self.coarse.build_edges(max_edges); }
        if !self.fine.is_empty() { self.fine.build_edges(max_edges); }
    }

    /// Number of records in the coarse bucket for a given 128-bit hash.
    /// Used by the M4.3 intent classifier.
    pub fn bucket_size(&self, pap_128: &[u8; PAP_128_BYTES]) -> usize {
        if self.coarse.is_empty() {
            return 0;
        }
        let bucket = self.coarse.bucket_hash(pap_128);
        self.coarse.buckets.get(bucket).map(|b| b.len()).unwrap_or(0)
    }

    /// Slot indices inside the coarse bucket for a given 128-bit hash.
    /// Used by M3.5 batch fusion to score candidates per-query after a single
    /// bucket enumeration.
    pub fn bucket_slot_indices(&self, pap_128: &[u8; PAP_128_BYTES]) -> Option<Vec<usize>> {
        if self.coarse.is_empty() {
            return None;
        }
        let bucket = self.coarse.bucket_hash(pap_128);
        self.coarse.buckets.get(bucket).cloned()
    }
}

pub fn pap_distance_128(a: &[u8; PAP_128_BYTES], b: &[u8; PAP_128_BYTES]) -> f32 {
    let (x, ac, bc) = crate::simd_kernels::simd_pap_distance_128(a, b);
    if ac == 0 || bc == 0 { return 0.0; }
    (x as f32) / ((ac as f32) * (bc as f32)).sqrt()
}

pub fn pap_distance_512(a: &[u8; PAP_512_BYTES], b: &[u8; PAP_512_BYTES]) -> f32 {
    let (x, ac, bc) = crate::simd_kernels::simd_pap_distance_512(a, b);
    if ac == 0 || bc == 0 { return 0.0; }
    (x as f32) / ((ac as f32) * (bc as f32)).sqrt()
}

pub fn pap_128_from_seed(seed: u64) -> [u8; PAP_128_BYTES] {
    let mut p = [0u8; PAP_128_BYTES]; let mut s = seed.wrapping_mul(0x9e3779b97f4a7c15);
    for i in 0..PAP_128_BYTES { s = s.wrapping_mul(0x2545f4914f6cdd1d).wrapping_add(1); p[i] = (s >> 56) as u8; }
    p
}

pub fn pap_512_from_seed(seed: u64) -> [u8; PAP_512_BYTES] {
    let mut p = [0u8; PAP_512_BYTES]; let mut s = seed.wrapping_mul(0x9e3779b97f4a7c15);
    for i in 0..PAP_512_BYTES { s = s.wrapping_mul(0x2545f4914f6cdd1d).wrapping_add(1); p[i] = (s >> 56) as u8; }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_pap_128_distance_self() { let p = pap_128_from_seed(42); assert!((pap_distance_128(&p, &p) - 1.0).abs() < 1e-6); }
    #[test] fn test_pap_512_distance_self() { let p = pap_512_from_seed(42); assert!((pap_distance_512(&p, &p) - 1.0).abs() < 1e-6); }

    #[test] fn test_fast128_only() {
        let mut mesh = HybridMesh::new(1000, 0);
        for i in 0..1000u64 { mesh.insert_128(i, &pap_128_from_seed(i)); }
        mesh.build_edges(8);
        let q128 = pap_128_from_seed(500); let q512 = pap_512_from_seed(500);
        let r = mesh.query_auto(&q128, &q512, 10);
        assert_eq!(r[0].1, 500);
        mesh.set_mode(PAPMode::Fast128);
        let r = mesh.query(&q128, &q512, 10);
        assert_eq!(r[0].1, 500);
    }

    #[test] fn test_full512_only() {
        let mut mesh = HybridMesh::new(0, 1000);
        for i in 0..1000u64 { mesh.insert_512(i, &pap_512_from_seed(i)); }
        mesh.build_edges(8);
        let q128 = pap_128_from_seed(500); let q512 = pap_512_from_seed(500);
        let r = mesh.query_auto(&q128, &q512, 10);
        assert_eq!(r[0].1, 500);
        mesh.set_mode(PAPMode::Full512);
        let r = mesh.query(&q128, &q512, 10);
        assert_eq!(r[0].1, 500);
    }

    #[test] fn test_dual_index_hybrid() {
        let mut mesh = HybridMesh::new(10000, 10000);
        for i in 0..10000u64 { mesh.insert_dual(i, &pap_128_from_seed(i), &pap_512_from_seed(i)); }
        mesh.build_edges(8);
        let q128 = pap_128_from_seed(5000); let q512 = pap_512_from_seed(5000);
        let r = mesh.query_auto(&q128, &q512, 10);
        assert!(!r.is_empty()); assert_eq!(r[0].1, 5000);
        mesh.set_mode(PAPMode::Hybrid);
        let r = mesh.query(&q128, &q512, 10);
        assert_eq!(r[0].1, 5000);
    }

    #[test] fn test_mixed_data_auto_switch() {
        let mut mesh = HybridMesh::new(1000, 1000);
        for i in 0..500u64 { mesh.insert_128(i, &pap_128_from_seed(i)); }
        for i in 500..1000u64 { mesh.insert_512(i, &pap_512_from_seed(i)); }
        mesh.build_edges(8);
        let q128 = pap_128_from_seed(250); let q512 = pap_512_from_seed(750);
        let r = mesh.query_auto(&q128, &q512, 10);
        assert!(!r.is_empty());
    }

    #[test] fn test_speed_comparison() {
        let mut mesh = HybridMesh::new(100000, 100000);
        for i in 0..100000u64 { mesh.insert_dual(i, &pap_128_from_seed(i), &pap_512_from_seed(i)); }
        mesh.build_edges(8);
        let q128 = pap_128_from_seed(50000); let q512 = pap_512_from_seed(50000);
        let t0 = std::time::Instant::now(); mesh.set_mode(PAPMode::Fast128); let _ = mesh.query(&q128, &q512, 10); let fast = t0.elapsed();
        let t0 = std::time::Instant::now(); mesh.set_mode(PAPMode::Full512); let _ = mesh.query(&q128, &q512, 10); let full = t0.elapsed();
        let t0 = std::time::Instant::now(); mesh.set_mode(PAPMode::Hybrid); let _ = mesh.query(&q128, &q512, 10); let hybrid = t0.elapsed();
        println!("Fast128: {:?}, Full512: {:?}, Hybrid: {:?}", fast, full, hybrid);
    }

}

// === GRAVITY DECAY ===
impl HybridMesh {
    /// Decay gravity for all papers. Call periodically (e.g., nightly).
    pub fn decay_gravity(&mut self, decay_factor: f32) {
        let mut cleared = 0;
        for (_, gravity) in self.gravity_map.iter_mut() {
            gravity.query_count = (gravity.query_count as f32 * decay_factor) as u64;
            if gravity.query_count == 0 {
                gravity.co_occurrence.clear();
                cleared += 1;
            }
        }
        println!("[mesh] Gravity decayed by {:.0}%, {} entries cleared", (1.0 - decay_factor) * 100.0, cleared);
    }
}
