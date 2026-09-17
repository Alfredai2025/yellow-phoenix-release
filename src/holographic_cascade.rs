// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Multi-Mirror Holographic Cascade with Intelligent Insertion (Semantic Gravity).
//!
//! Each paper is injected into K mirror vaults (deterministic bit permutations)
//! PLUS top-3 vaults by centroid similarity (semantic gravity).
//!
//! Query uses adaptive arrow: probe mirrors sorted by centroid similarity,
//! stop when resonance exceeds threshold.
//!
//! Math:
//!   Bind:     bound[i] = content[i] * key[i]     (element-wise)
//!   Superpose: wave_field += bound                (vector add)
//!   Unbind:   raw[i] = wave_field[i] * query[i]   (element-wise)
//!   Cleanup:  score = dot(raw, content)           (dot product)

use alloc::vec::Vec;
use hashbrown::HashMap;

// ---------------------------------------------------------------------------
// VSA Primitives
// ---------------------------------------------------------------------------

#[inline]
pub fn hash_to_bipolar(hash: &[u8; 64]) -> Vec<f32> {
    let mut v = Vec::with_capacity(512);
    for byte in hash.iter() {
        for bit in 0..8 {
            v.push(if (byte >> (7 - bit)) & 1 == 1 { 1.0f32 } else { -1.0f32 });
        }
    }
    v
}

pub fn bind(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

pub fn superpose(vault: &mut [f32], bound: &[f32]) {
    for (i, &val) in bound.iter().enumerate() { vault[i] += val; }
}

#[inline]
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 { for x in v.iter_mut() { *x /= norm; } }
}

pub fn unbind(vault: &[f32], query_key: &[f32]) -> Vec<f32> {
    bind(vault, query_key)
}

pub fn cleanup(raw: &[f32], content: &[f32]) -> f32 {
    raw.iter().zip(content.iter()).map(|(a, b)| a * b).sum()
}

fn permute_bipolar(v: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; v.len()];
    let n = v.len();
    for i in 0..n {
        let j = (i.wrapping_mul(2654435761usize)) % n;
        out[j] = v[i];
    }
    out
}

// ---------------------------------------------------------------------------
// HologramTier1 — Wave-field superposition
// ---------------------------------------------------------------------------

pub struct HologramTier1 {
    pub wave_field: Vec<f32>,
    pub paper_ids: Vec<u64>,
    pub contents: Vec<Vec<f32>>,
    pub hashes: Vec<[u8; 64]>,
    dim: usize,
    phase: Vec<f32>,
}

impl HologramTier1 {
    pub fn new(dim: usize) -> Self {
        let mut phase = vec![0.0f32; dim]; phase[0] = 1.0;
        Self { wave_field: vec![0.0f32; dim], paper_ids: Vec::new(), contents: Vec::new(), hashes: Vec::new(), dim, phase }
    }

    pub fn set_phase(&mut self, eigenvector: &[f32]) {
        let n = eigenvector.len().min(self.dim);
        self.phase[..n].copy_from_slice(&eigenvector[..n]);
        normalize(&mut self.phase);
    }

    pub fn inject(&mut self, paper_id: u64, hash: &[u8; 64]) {
        let key = hash_to_bipolar(hash);
        let content = permute_bipolar(&key);
        let bound = bind(&content, &key);
        superpose(&mut self.wave_field, &bound);
        self.paper_ids.push(paper_id);
        self.contents.push(content);
        self.hashes.push(*hash);
    }

    pub fn query(&self, query_hash: &[u8; 64], top_k: usize) -> Vec<(u64, f32)> {
        let q = hash_to_bipolar(query_hash);
        let q_rotated = bind(&q, &self.phase);
        let raw = unbind(&self.wave_field, &q_rotated);
        let mut scores: Vec<(u64, f32)> = self.contents.iter().enumerate()
            .map(|(idx, content)| {
                let score = cleanup(&raw, content);
                (self.paper_ids[idx], score)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    pub fn len(&self) -> usize { self.paper_ids.len() }

    /// Compute vault centroid: average of all bipolar hashes
    pub fn centroid(&self) -> Vec<f32> {
        let mut avg = vec![0.0f32; self.dim];
        for hash in &self.hashes {
            let bip = hash_to_bipolar(hash);
            for (i, &val) in bip.iter().enumerate() { avg[i] += val; }
        }
        let n = self.hashes.len() as f32;
        if n > 0.0 { for x in avg.iter_mut() { *x /= n; } }
        normalize(&mut avg);
        avg
    }
}

// ---------------------------------------------------------------------------
// HolographicVault
// ---------------------------------------------------------------------------

pub struct HolographicVault {
    pub tier1: HologramTier1,
}

impl HolographicVault {
    pub fn new(dim: usize) -> Self {
        Self { tier1: HologramTier1::new(dim) }
    }
    pub fn inject(&mut self, paper_id: u64, hash: &[u8; 64]) {
        self.tier1.inject(paper_id, hash);
    }
    pub fn query(&self, query_hash: &[u8; 64], top_k: usize) -> Vec<(u64, f32)> {
        self.tier1.query(query_hash, top_k)
    }
    pub fn len(&self) -> usize { self.tier1.len() }
    pub fn centroid(&self) -> Vec<f32> { self.tier1.centroid() }
    pub fn set_phase(&mut self, ev: &[f32]) { self.tier1.set_phase(ev); }
}

// ---------------------------------------------------------------------------
// HolographicCascade — Multi-mirror + Semantic Gravity + Arrow Query
// ---------------------------------------------------------------------------

pub struct HolographicCascade {
    pub l1_bits: u8,
    pub l2_bits: u8,
    pub l3_bits: u8,
    pub vaults: HashMap<u64, HolographicVault>,
    pub dim: usize,
    pub n_mirrors: usize,
    pub n_gravity: usize,
    pub arrow_threshold: f32,
}

impl HolographicCascade {
    pub fn new(l1: u8, l2: u8, l3: u8, dim: usize) -> Self {
        Self {
            l1_bits: l1, l2_bits: l2, l3_bits: l3,
            vaults: HashMap::new(), dim,
            n_mirrors: 4, n_gravity: 3, arrow_threshold: 350.0,
        }
    }

    pub fn vault_id(&self, hash: &[u8; 64]) -> u64 {
        let total = (self.l1_bits + self.l2_bits + self.l3_bits) as usize;
        let mut id: u64 = 0;
        let mut bits = 0;
        for byte in hash {
            for bit in 0..8 {
                if bits >= total { break; }
                id = (id << 1) | (((byte >> (7 - bit)) & 1) as u64);
                bits += 1;
            }
            if bits >= total { break; }
        }
        id
    }

    /// Mirror hash: deterministic bit permutation
    pub fn mirror_hash(base: &[u8; 64], mirror_id: usize) -> [u8; 64] {
        let mut out = [0u8; 64];
        let seed = mirror_id.wrapping_mul(2654435761usize);
        for byte_idx in 0..64 {
            for bit_idx in 0..8 {
                let src_pos = ((byte_idx * 8 + bit_idx + seed) % 512) as usize;
                let src_byte = src_pos / 8;
                let src_bit = src_pos % 8;
                if (base[src_byte] >> (7 - src_bit)) & 1 == 1 {
                    out[byte_idx] |= 1 << (7 - bit_idx);
                }
            }
        }
        out
    }

    /// Phase 1: Insert into K mirror vaults
    pub fn insert_mirror(&mut self, paper_id: u64, hash: &[u8; 64]) {
        for m in 0..self.n_mirrors {
            let mirror = Self::mirror_hash(hash, m);
            let id = self.vault_id(&mirror);
            self.vaults.entry(id)
                .or_insert_with(|| HolographicVault::new(self.dim))
                .inject(paper_id, hash);
        }
    }

    /// Phase 2: Intelligent insertion (semantic gravity)
    /// After all papers are in mirror vaults, compute centroids,
    /// then re-insert each paper into top-N vaults by centroid similarity.
    pub fn apply_semantic_gravity(&mut self, hashes: &[[u8; 64]]) {
        // Compute all vault centroids
        let mut centroids: HashMap<u64, Vec<f32>> = HashMap::new();
        for (vid, vault) in &self.vaults {
            centroids.insert(*vid, vault.centroid());
        }
        // For each paper, find top-N vaults by centroid similarity
        for (paper_id, hash) in hashes.iter().enumerate() {
            let bip = hash_to_bipolar(hash);
            let mut vault_scores: Vec<(u64, f32)> = centroids.iter()
                .map(|(vid, cent)| {
                    let score: f32 = bip.iter().zip(cent.iter()).map(|(a, b)| a * b).sum();
                    (*vid, score)
                })
                .collect();
            vault_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
            for (vid, _) in vault_scores.iter().take(self.n_gravity) {
                self.vaults.entry(*vid)
                    .or_insert_with(|| HolographicVault::new(self.dim))
                    .inject(paper_id as u64, hash);
            }
        }
    }

    /// Auto-tune phases from vault centroids
    pub fn auto_tune_all_phases(&mut self) {
        let ids: Vec<u64> = self.vaults.keys().copied().collect();
        for vid in ids {
            if let Some(v) = self.vaults.get(&vid) {
                let cent = v.centroid();
                if let Some(v_mut) = self.vaults.get_mut(&vid) {
                    v_mut.set_phase(&cent);
                }
            }
        }
    }

    /// Adaptive arrow query: probe mirrors sorted by centroid similarity
    pub fn query_arrow(&self, query_hash: &[u8; 64], top_k: usize) -> Vec<(u64, f32)> {
        let q_bip = hash_to_bipolar(query_hash);
        // Compute centroid similarity for all mirror vaults
        let mut mirror_scores: Vec<(usize, f32)> = Vec::new();
        for m in 0..self.n_mirrors {
            let mirror = Self::mirror_hash(query_hash, m);
            let vid = self.vault_id(&mirror);
            if let Some(v) = self.vaults.get(&vid) {
                let cent = v.centroid();
                let sim: f32 = q_bip.iter().zip(cent.iter()).map(|(a, b)| a * b).sum();
                mirror_scores.push((m, sim));
            }
        }
        mirror_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        // Probe in order of expected relevance
        let mut all_results = Vec::new();
        for (m, _) in mirror_scores {
            let mirror = Self::mirror_hash(query_hash, m);
            let vid = self.vault_id(&mirror);
            if let Some(v) = self.vaults.get(&vid) {
                let mut r = v.query(query_hash, top_k);
                if !r.is_empty() && r[0].1 > self.arrow_threshold {
                    all_results.append(&mut r);
                    break; // Arrow hit target
                }
                all_results.append(&mut r);
            }
        }
        // Deduplicate
        let mut best: HashMap<u64, f32> = HashMap::new();
        for (pid, sc) in all_results { best.entry(pid).and_modify(|e| *e = e.max(sc)).or_insert(sc); }
        let mut out: Vec<_> = best.into_iter().collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        out.truncate(top_k);
        out
    }

    pub fn len(&self) -> usize {
        self.vaults.values().map(|v| v.len()).sum()
    }

    pub fn vault_count(&self) -> usize {
        self.vaults.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fake_hash(seed: u8) -> [u8; 64] {
        let mut h = [0u8; 64];
        for i in 0..64 { h[i] = seed.wrapping_add(i as u8); }
        h
    }
    #[test]
    fn test_mirror_hash() {
        let h = fake_hash(42);
        let m0 = HolographicCascade::mirror_hash(&h, 0);
        let m1 = HolographicCascade::mirror_hash(&h, 1);
        assert_ne!(m0, m1);
    }
    #[test]
    fn test_multi_mirror_insert() {
        let mut c = HolographicCascade::new(8, 4, 4, 512);
        let hashes: Vec<[u8; 64]> = (0..100).map(|i| fake_hash(i as u8)).collect();
        for (i, h) in hashes.iter().enumerate() {
            c.insert_mirror(i as u64, h);
        }
        assert!(c.len() >= 100); // at least 100 (could be more with duplicates)
    }
    #[test]
    fn test_arrow_query() {
        let mut c = HolographicCascade::new(8, 4, 4, 512);
        let hashes: Vec<[u8; 64]> = (0..200).map(|i| fake_hash(i as u8)).collect();
        for (i, h) in hashes.iter().enumerate() {
            c.insert_mirror(i as u64, h);
        }
        c.auto_tune_all_phases();
        let r = c.query_arrow(&fake_hash(50), 5);
        assert!(!r.is_empty());
    }
}