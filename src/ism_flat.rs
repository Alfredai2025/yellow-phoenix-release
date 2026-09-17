// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! ISM flat index: mmap-backed, zero-copy, brute-force top-k search over
//! 512-bit binary hashes. Replaces BinaryHNSW for small (≈100K) corpora where
//! the graph construction/load overhead is not worth the approximate speedup.

use crate::binary_hnsw::hamming_distance;
use memmap2::Mmap;
use std::collections::BinaryHeap;
use std::ffi::{c_char, c_int, CStr};
use std::fs::File;
use std::sync::RwLock;

const ISM_MAGIC: &[u8] = b"YISM";
const ISM_VERSION: u8 = 1;
const HASH_BYTES: usize = 64;
const HEADER_LEN: usize = 4 + 1 + 8 + 1; // magic + version + count + hash_len

/// Hamming distance on the first 16 bytes (128-bit prefix).
#[inline(always)]
fn prefix_hamming(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones() as u32).sum()
}

pub struct IsmFlatIndex {
    mmap: Mmap,
    count: usize,
    data_offset: usize,
    record_size: usize,
}

impl IsmFlatIndex {
    /// Memory-map an ISM flat file and validate its header.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_LEN {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file too short for header",
            ));
        }
        if &mmap[0..4] != ISM_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad magic",
            ));
        }
        if mmap[4] != ISM_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported version",
            ));
        }

        let count = u64::from_le_bytes(mmap[5..13].try_into().unwrap()) as usize;
        let hash_len = mmap[13] as usize;
        if hash_len != HASH_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported hash length",
            ));
        }

        let record_size = 8 + hash_len;
        let data_offset = HEADER_LEN;
        let expected = data_offset + count * record_size;
        if mmap.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file size does not match record count",
            ));
        }

        Ok(Self {
            mmap,
            count,
            data_offset,
            record_size,
        })
    }

    #[inline(always)]
    fn id_at(&self, idx: usize) -> u64 {
        let off = self.data_offset + idx * self.record_size;
        u64::from_le_bytes(self.mmap[off..off + 8].try_into().unwrap())
    }

    #[inline(always)]
    fn hash_at(&self, idx: usize) -> &[u8] {
        let off = self.data_offset + idx * self.record_size + 8;
        &self.mmap[off..off + HASH_BYTES]
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Brute-force scan returning top-k results as (id, score) with score in [0,1].
    pub fn search(&self, query: &[u8], top_k: usize) -> Vec<(u64, f32)> {
        if query.len() != HASH_BYTES || self.count == 0 || top_k == 0 {
            return Vec::new();
        }

        let query_arr: &[u8; HASH_BYTES] = match query.try_into() {
            Ok(a) => a,
            Err(_) => return Vec::new(),
        };

        let k = top_k.min(self.count);
        // Min-heap (largest distance at top) of the best k seen so far.
        let mut heap: BinaryHeap<(u32, u64)> = BinaryHeap::with_capacity(k);

        for i in 0..self.count {
            let h = self.hash_at(i);
            let hash_arr: &[u8; HASH_BYTES] = h.try_into().unwrap();
            let dist = hamming_distance(query_arr, hash_arr);

            if heap.len() < k {
                heap.push((dist, i as u64));
            } else if dist < heap.peek().unwrap().0 {
                heap.pop();
                heap.push((dist, i as u64));
            }
        }

        let max_bits = (HASH_BYTES * 8) as f32;
        let mut results: Vec<(u64, f32)> = heap
            .into_iter()
            .map(|(dist, idx)| (self.id_at(idx as usize), 1.0 - (dist as f32 / max_bits)))
            .collect();
        // Highest score first.
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    /// Prefix-filtered exact search: scan only the first 128 bits for all hashes,
    /// keep the best `prefix_candidates`, then compute full 512-bit Hamming on those.
    pub fn search_prefix(&self, query: &[u8], prefix_candidates: usize, top_k: usize) -> Vec<(u64, f32)> {
        if query.len() != HASH_BYTES || self.count == 0 || top_k == 0 || prefix_candidates == 0 {
            return Vec::new();
        }
        let query_arr: &[u8; HASH_BYTES] = match query.try_into() {
            Ok(a) => a,
            Err(_) => return Vec::new(),
        };

        let mut qprefix = [0u8; 16];
        qprefix.copy_from_slice(&query_arr[0..16]);

        // 1) 128-bit prefix scan
        let mut prefix_scored: Vec<(u32, usize)> = Vec::with_capacity(self.count);
        for i in 0..self.count {
            let h = self.hash_at(i);
            let hprefix: &[u8; 16] = h[0..16].try_into().unwrap();
            prefix_scored.push((prefix_hamming(&qprefix, hprefix), i));
        }

        let candidates = prefix_candidates.min(self.count).max(1);
        prefix_scored.select_nth_unstable_by(candidates - 1, |a, b| a.0.cmp(&b.0));
        prefix_scored.truncate(candidates);

        // 2) Full 512-bit Hamming on candidate subset
        let k = top_k.min(self.count).min(candidates);
        let mut heap: BinaryHeap<(u32, u64)> = BinaryHeap::with_capacity(k);
        for (_, idx) in prefix_scored {
            let h = self.hash_at(idx);
            let hash_arr: &[u8; HASH_BYTES] = h.try_into().unwrap();
            let dist = hamming_distance(query_arr, hash_arr);
            let id = self.id_at(idx);
            if heap.len() < k {
                heap.push((dist, id));
            } else if dist < heap.peek().unwrap().0 {
                heap.pop();
                heap.push((dist, id));
            }
        }

        let max_bits = (HASH_BYTES * 8) as f32;
        let mut results: Vec<(u64, f32)> = heap
            .into_iter()
            .map(|(dist, id)| (id, 1.0 - (dist as f32 / max_bits)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }
}

static ISM_FLAT: RwLock<Option<IsmFlatIndex>> = RwLock::new(None);

/// Load (or reload) the global ISM flat index from disk.
#[no_mangle]
pub extern "C" fn yp_ism_load(path: *const c_char) -> c_int {
    if path.is_null() {
        return -1;
    }
    let path = match unsafe { CStr::from_ptr(path).to_str() } {
        Ok(s) => s,
        Err(_) => return -2,
    };

    match IsmFlatIndex::load(path) {
        Ok(idx) => {
            *ISM_FLAT.write().unwrap() = Some(idx);
            0
        }
        Err(_) => -3,
    }
}

/// Return the number of indexed records, or 0 if not loaded.
#[no_mangle]
pub extern "C" fn yp_ism_count() -> usize {
    ISM_FLAT.read().unwrap().as_ref().map(|i| i.count()).unwrap_or(0)
}

/// Unload the global ISM flat index (drops the mmap). Safe to call when
/// nothing is loaded. Dataset switches call this so the previous corpus's
/// hash map does not stay resident (clean pages are free in jetsam
/// accounting, but the map still consumes VA space and file-backed cache).
#[no_mangle]
pub extern "C" fn yp_ism_unload() -> c_int {
    *ISM_FLAT.write().unwrap() = None;
    0
}

/// Search the loaded ISM index. Writes up to `out_cap` (id, score) pairs.
/// Returns the number of results written.
#[no_mangle]
pub extern "C" fn yp_ism_search(
    query_hash: *const u8,
    hash_len: usize,
    top_k: usize,
    out_ids: *mut u64,
    out_scores: *mut f32,
    out_cap: usize,
) -> usize {
    if query_hash.is_null() || out_ids.is_null() || out_scores.is_null() || out_cap == 0 {
        return 0;
    }

    let guard = ISM_FLAT.read().unwrap();
    let idx = match guard.as_ref() {
        Some(i) => i,
        None => return 0,
    };

    let query = unsafe { std::slice::from_raw_parts(query_hash, hash_len) };
    let results = idx.search(query, top_k.min(out_cap));
    let n = results.len();

    let ids = unsafe { std::slice::from_raw_parts_mut(out_ids, out_cap) };
    let scores = unsafe { std::slice::from_raw_parts_mut(out_scores, out_cap) };
    for (i, (id, score)) in results.iter().enumerate() {
        ids[i] = *id;
        scores[i] = *score;
    }
    n
}

/// Prefix-filtered ISM search. Writes up to `out_cap` (id, score) pairs.
/// Returns the number of results written.
#[no_mangle]
pub extern "C" fn yp_ism_search_prefix(
    query_hash: *const u8,
    hash_len: usize,
    prefix_candidates: usize,
    top_k: usize,
    out_ids: *mut u64,
    out_scores: *mut f32,
    out_cap: usize,
) -> usize {
    if query_hash.is_null() || out_ids.is_null() || out_scores.is_null() || out_cap == 0 {
        return 0;
    }

    let guard = ISM_FLAT.read().unwrap();
    let idx = match guard.as_ref() {
        Some(i) => i,
        None => return 0,
    };

    let query = unsafe { std::slice::from_raw_parts(query_hash, hash_len) };
    let results = idx.search_prefix(query, prefix_candidates, top_k.min(out_cap));
    let n = results.len();

    let ids = unsafe { std::slice::from_raw_parts_mut(out_ids, out_cap) };
    let scores = unsafe { std::slice::from_raw_parts_mut(out_scores, out_cap) };
    for (i, (id, score)) in results.iter().enumerate() {
        ids[i] = *id;
        scores[i] = *score;
    }
    n
}
