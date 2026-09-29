// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! AUDITED SHOOTOUT BENCHMARK — 2026-08-22
//!
//! Three honest test modes. All recall metrics verified mathematically correct.
//!
//! RECALL DEFINITION (standard):
//!   R@k = 1  if  ground_truth_id  ∈  hnsw_top_k_results
//!         0  otherwise
//!
//! This means we check whether the TRUE nearest neighbor (per exact brute-force)
//! appears somewhere in HNSW's returned top-k list. NOT whether HNSW's guess
//! appears in ISM's top-k.
//!
//! TIMING:
//!   Each timer wraps ONLY the search() call. No allocation, no I/O, no FFI
//!   marshalling inside the timed section.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::fs::File;
use std::io::Read;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;
use std::time::Instant;

#[cfg(target_os = "macos")]
extern "C" {
    /// Drop allocator caches after releasing large buffers (macOS-only API).
    fn malloc_zone_pressure_relief(zone: *mut c_void, relief: usize) -> usize;
}

/// Release allocator caches after dropping large buffers.
/// macOS: uses the zone-pressure API. Other platforms: no-op.
#[cfg(target_os = "macos")]
fn allocator_pressure_relief() {
    allocator_pressure_relief();
}

#[cfg(not(target_os = "macos"))]
fn allocator_pressure_relief() {}

use crate::binary_hnsw::{BinaryHNSW, Hash512, hamming_distance};

const HASH_BYTES: usize = 64;
const K: usize = 10;

// ────────────────────────────────
// Flat ISM index (exact brute-force)
// ────────────────────────────────
struct FlatIndex {
    ids: Vec<u64>,
    hashes: Vec<u8>,
    /// True when ids[i] == i for all i (true for the real 5M set where
    /// meta.id == row index). Lets us look up hashes by id without a map.
    identity_ids: bool,
}

impl FlatIndex {
    /// Load a flat ISM hash table. Two layouts are accepted:
    ///   • YISM v1 (current): 4B magic + 1B version + 8B count LE + 1B hash_len,
    ///     then count × (8B id LE + 64B hash) — the on-device format for the
    ///     106K and 1M corpora.
    ///   • legacy raw: [u64 count LE][count×64B hashes][count×8B IDs].
    fn load(path: &str) -> std::io::Result<Self> {
        use std::fs::File;
        use std::io::{Error, ErrorKind, Read};

        let mut file = File::open(path)?;
        let mut first4 = [0u8; 4];
        file.read_exact(&mut first4)?;
        if &first4 == b"YISM" {
            return Self::load_yism(file);
        }
        // Legacy raw: the first 8 bytes are the count LE; 4 were consumed above.
        let mut rest4 = [0u8; 4];
        file.read_exact(&mut rest4)?;
        let mut count_buf = [0u8; 8];
        count_buf[..4].copy_from_slice(&first4);
        count_buf[4..].copy_from_slice(&rest4);
        let count = u64::from_le_bytes(count_buf) as usize;
        Self::load_raw_after_count(file, count)
    }

    /// Parse a YISM v1 file (magic already consumed). Layout (matches
    /// `ism_flat.rs` HEADER_LEN = 4+1+8+1 = 14):
    ///   4B magic + 1B version + 8B count LE + 1B hash_len (=64),
    ///   then count × (8B id LE + 64B hash). Verified against the on-device
    ///   106K/1M corpora: file size == 14 + count×72 exactly, first id == 0.
    fn load_yism(mut file: File) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind, Read};

        let mut ver = [0u8; 1];
        file.read_exact(&mut ver)?;
        if ver[0] != 1 {
            return Err(Error::new(ErrorKind::InvalidData,
                format!("unsupported YISM version {}", ver[0])));
        }
        let mut count_buf = [0u8; 8];
        file.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf) as usize;
        let mut hash_len = [0u8; 1];
        file.read_exact(&mut hash_len)?;
        if hash_len[0] as usize != HASH_BYTES {
            return Err(Error::new(ErrorKind::InvalidData,
                format!("unsupported YISM hash length {}", hash_len[0])));
        }
        const YISM_HEADER_LEN: u64 = 14;
        let file_len = file.metadata()?.len();
        let expected = YISM_HEADER_LEN
            .checked_add((count as u64).checked_mul((8 + HASH_BYTES) as u64).ok_or_else(||
                Error::new(ErrorKind::InvalidData, "YISM count overflows"))?)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "YISM size overflows"))?;
        if file_len != expected {
            return Err(Error::new(ErrorKind::InvalidData,
                format!("YISM size mismatch: header count {count} needs {expected}B, file is {file_len}B")));
        }
        let mut ids: Vec<u64> = Vec::new();
        ids.try_reserve_exact(count)
            .map_err(|_| Error::new(ErrorKind::OutOfMemory, "not enough memory for ISM ids"))?;
        ids.resize(count, 0);
        let mut hashes: Vec<u8> = Vec::new();
        hashes.try_reserve_exact(count.saturating_mul(HASH_BYTES))
            .map_err(|_| Error::new(ErrorKind::OutOfMemory, "not enough memory for ISM hashes"))?;
        hashes.resize(count.saturating_mul(HASH_BYTES), 0);
        let mut identity_ids = true;
        let mut rec = [0u8; 8 + HASH_BYTES];
        for i in 0..count {
            file.read_exact(&mut rec)?;
            let id = u64::from_le_bytes(rec[0..8].try_into().unwrap());
            if id != i as u64 { identity_ids = false; }
            ids[i] = id;
            hashes[i * HASH_BYTES..(i + 1) * HASH_BYTES].copy_from_slice(&rec[8..]);
        }
        Ok(Self { ids, hashes, identity_ids })
    }

    /// Parse the legacy raw layout after its 8-byte count has been consumed.
    /// Layout: [count×64B hashes][count×8B IDs].
    fn load_raw_after_count(mut file: File, count: usize) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind, Read};

        // Validate count against the actual file size BEFORE allocating.
        // A corrupt/garbage count header used to reach `vec![0u8; count*64]`
        // directly: count*72 overflows usize (capacity overflow) or requests
        // absurdly much RAM, and Rust's default alloc-error handler ABORTS
        // the process — that was the Bench/Shootout tab crash family
        // (YPPhone SIGABRT in yp_shootout_load_ism, 2026-09-08 11:41 ×3).
        let file_len = file.metadata()?.len();
        // The 8-byte count was consumed before this fn was called.
        let expected = 8u64
            .checked_add((count as u64).checked_mul(HASH_BYTES as u64).ok_or_else(|| {
                Error::new(ErrorKind::InvalidData, "ISM count header overflows")
            })?)
            .and_then(|v| v.checked_add((count as u64).checked_mul(8)?))
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "ISM size overflows"))?;
        if file_len < expected {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("ISM file truncated: header count {count} needs {expected}B, file is {file_len}B — re-push the file"),
            ));
        }

        // Fallible allocations: on genuine memory pressure return an error
        // (surfaced in the UI) instead of aborting the whole app.
        let mut hashes: Vec<u8> = Vec::new();
        hashes.try_reserve_exact(count.saturating_mul(HASH_BYTES))
            .map_err(|_| Error::new(ErrorKind::OutOfMemory, "not enough memory for ISM hashes"))?;
        hashes.resize(count.saturating_mul(HASH_BYTES), 0);
        file.read_exact(&mut hashes)?;

        let mut ids: Vec<u64> = Vec::new();
        ids.try_reserve_exact(count)
            .map_err(|_| Error::new(ErrorKind::OutOfMemory, "not enough memory for ISM ids"))?;
        ids.resize(count, 0);
        let mut id_buf: Vec<u8> = Vec::new();
        id_buf.try_reserve_exact(count.saturating_mul(8))
            .map_err(|_| Error::new(ErrorKind::OutOfMemory, "not enough memory for ISM id buffer"))?;
        id_buf.resize(count.saturating_mul(8), 0);
        file.read_exact(&mut id_buf)?;
        let mut identity_ids = true;
        for i in 0..count {
            let id = u64::from_le_bytes(id_buf[i * 8..i * 8 + 8].try_into().unwrap());
            if id != i as u64 { identity_ids = false; }
            ids[i] = id;
        }
        Ok(Self { ids, hashes, identity_ids })
    }

    fn count(&self) -> usize { self.ids.len() }
    fn hash_at(&self, i: usize) -> &[u8] { &self.hashes[i * HASH_BYTES..(i + 1) * HASH_BYTES] }
    fn id_at(&self, i: usize) -> u64 { self.ids[i] }

    /// Hash for a user-facing id. Only supported when ids are identity-mapped.
    fn hash_by_id(&self, id: u64) -> Option<&[u8; HASH_BYTES]> {
        if !self.identity_ids { return None; }
        let i = id as usize;
        if i >= self.count() { return None; }
        Some(self.hash_at(i).try_into().unwrap())
    }

    /// Exact brute-force Hamming top-k. Returns (id, distance) sorted ascending.
    fn search(&self, query: &[u8; HASH_BYTES], k: usize) -> Vec<(u64, u32)> {
        let mut scored: Vec<(u32, u64)> = Vec::with_capacity(self.count());
        for i in 0..self.count() {
            let h: &[u8; HASH_BYTES] = self.hash_at(i).try_into().unwrap();
            let dist = hamming_distance(query, h);
            scored.push((dist, self.id_at(i)));
        }
        let k = k.min(scored.len());
        if k == 0 { return Vec::new(); }
        scored.select_nth_unstable_by(k - 1, |a, b| a.0.cmp(&b.0));
        scored.truncate(k);
        scored.sort_by(|a, b| a.0.cmp(&b.0));
        scored.into_iter().map(|(dist, id)| (id, dist)).collect()
    }

    /// Exact search excluding one index (for exclude-self mode).
    fn search_exclude(&self, query: &[u8; HASH_BYTES], k: usize, exclude_idx: usize) -> Vec<(u64, u32)> {
        let mut scored: Vec<(u32, u64)> = Vec::with_capacity(self.count().saturating_sub(1));
        for i in 0..self.count() {
            if i == exclude_idx { continue; }
            let h: &[u8; HASH_BYTES] = self.hash_at(i).try_into().unwrap();
            let dist = hamming_distance(query, h);
            scored.push((dist, self.id_at(i)));
        }
        if scored.is_empty() { return Vec::new(); }
        let k = k.min(scored.len());
        scored.select_nth_unstable_by(k - 1, |a, b| a.0.cmp(&b.0));
        scored.truncate(k);
        scored.sort_by(|a, b| a.0.cmp(&b.0));
        scored.into_iter().map(|(dist, id)| (id, dist)).collect()
    }

    fn memory_bytes(&self) -> u64 {
        (self.ids.len() * 8 + self.hashes.len()) as u64
    }
}

// ────────────────────────────────
// Globals
// ────────────────────────────────
static FLAT: RwLock<Option<FlatIndex>> = RwLock::new(None);
static HNSW: RwLock<Option<BinaryHNSW>> = RwLock::new(None);
static HNSW_EF_SEARCH: AtomicU32 = AtomicU32::new(128);

/// Saved query state for sequential ISM-then-HNSW benchmarking.
/// This keeps peak memory at max(ISM, HNSW) instead of ISM + HNSW.
struct Query {
    hash: [u8; HASH_BYTES],
    exclude_idx: Option<usize>,
    /// Source document the query hash was derived from. For perturb mode this
    /// is the un-perturbed document; for exclude_self it equals exclude_idx.
    /// Used as the re-rank query embedding row. Meaningless for random mode.
    base_idx: usize,
    gt_id: u64,
}

static QUERIES: RwLock<Option<Vec<Vec<Query>>>> = RwLock::new(None);

#[no_mangle]
pub extern "C" fn yp_shootout_get_hash(id: u64, out_hash: *mut u8) -> c_int {
    if out_hash.is_null() { return -1; }
    let out = unsafe { std::slice::from_raw_parts_mut(out_hash, HASH_BYTES) };
    if let Some(hnsw) = HNSW.read().unwrap().as_ref() {
        if let Some(h) = hnsw.hash_by_id(id) {
            out.copy_from_slice(h);
            return 0;
        }
    }
    if let Some(flat) = FLAT.read().unwrap().as_ref() {
        if let Some(h) = flat.hash_by_id(id) {
            out.copy_from_slice(h);
            return 0;
        }
    }
    -1
}

#[no_mangle]
pub extern "C" fn yp_shootout_unload_ism() -> c_int {
    {
        let mut guard = FLAT.write().unwrap();
        if let Some(ref flat) = *guard {
            // Hint the kernel to drop the ISM hash pages immediately so the
            // HNSW load does not peak at ISM + HNSW resident memory.
            let ptr = flat.hashes.as_ptr() as *mut c_void;
            let len = flat.hashes.len();
            unsafe { libc::madvise(ptr, len, libc::MADV_DONTNEED); }
        }
        *guard = None;
    }
    // Purge any malloc caches that may be holding the released pages.
    allocator_pressure_relief();
    0
}

#[no_mangle]
pub extern "C" fn yp_shootout_unload_hnsw() -> c_int {
    {
        let mut guard = HNSW.write().unwrap();
        if let Some(ref hnsw) = *guard {
            // neighbor_arena is the dominant allocation; hint the kernel to reclaim.
            hnsw.advise_dontneed();
        }
        *guard = None;
    }
    allocator_pressure_relief();
    0
}

/// Set the HNSW ef_search parameter used by the shootout HNSW phase.
/// Default is 128. Smaller values run faster; larger values improve recall.
#[no_mangle]
pub extern "C" fn yp_shootout_set_ef_search(ef: u32) {
    HNSW_EF_SEARCH.store(ef.max(1), Ordering::Relaxed);
}

#[no_mangle]
pub extern "C" fn yp_shootout_load_ism(path: *const c_char) -> c_int {
    let path = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };
    match FlatIndex::load(path) {
        Ok(idx) => { *FLAT.write().unwrap() = Some(idx); 0 }
        Err(e) => { crate::flat_ffi::set_shootout_error(format!("yp_shootout_load_ism {path}: {e}")); -1 }
    }
}

#[no_mangle]
pub extern "C" fn yp_shootout_load_hnsw(path: *const c_char) -> c_int {
    let path = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };
    match BinaryHNSW::load(path) {
        Ok(hnsw) => { *HNSW.write().unwrap() = Some(hnsw); 0 }
        Err(e) => { crate::flat_ffi::set_shootout_error(format!("yp_shootout_load_hnsw {path}: {e}")); -1 }
    }
}

/// Probe whether a file is in BinaryHNSW (YPH5v4/v5) format.
/// Returns 1 if yes, 0 if no (likely NativeEngine or older format), -1 on error.
#[no_mangle]
pub extern "C" fn yp_shootout_probe_hnsw_format(path: *const c_char) -> c_int {
    let path = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return -1,
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return -1;
    }
    if &magic == b"YPH5" {
        let mut version = [0u8; 1];
        if file.read_exact(&mut version).is_ok() && (version[0] == 4 || version[0] == 5) {
            return 1;
        }
    }
    0
}

// ────────────────────────────────
// Metrics structs (C-compatible)
// ────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub struct YpModeMetrics {
    pub ism_p50_us: u64,
    pub ism_p95_us: u64,
    pub ism_p99_us: u64,
    pub hnsw_p50_us: u64,
    pub hnsw_p95_us: u64,
    pub hnsw_p99_us: u64,
    pub ism_r1: f64,
    pub hnsw_r1: f64,
    pub ism_r5: f64,
    pub hnsw_r5: f64,
    pub ism_r10: f64,
    pub hnsw_r10: f64,
}

#[repr(C)]
pub struct YpShootoutMetrics {
    pub perturb_50: YpModeMetrics,
    pub exclude_self: YpModeMetrics,
    pub random_query: YpModeMetrics,
    pub ism_mem_kb: u64,
    pub hnsw_mem_kb: u64,
    pub count: u64,
}

/// Two-stage pipeline metrics: HNSW candidate generation + float cosine re-rank.
/// Re-rank is only defined for modes with a source document (perturb, exclude_self).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct YpRerankMetrics {
    pub perturb_r1: f64,
    pub perturb_r5: f64,
    pub perturb_r10: f64,
    pub exclude_r1: f64,
    pub exclude_r5: f64,
    pub exclude_r10: f64,
    pub p50_us: u64,
    pub p95_us: u64,
    /// Median per-query latency of the untimed warm-up pass (cold mmap pages).
    pub cold_p50_us: u64,
    /// How many exclude-self queries were scored against the float-space ground
    /// truth (0 = gt file missing/mismatched — exclude-* fields are not meaningful).
    pub float_gt_queries: u32,
    /// 1 if the phase ran, 0 otherwise (embeddings missing is reported by the caller).
    pub ran: i32,
}

// ────────────────────────────────
// LCG (deterministic, reproducible)
// ────────────────────────────────
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn range(&mut self, max: u64) -> u64 { self.next() % max }
    fn byte(&mut self) -> u8 { (self.next() % 256) as u8 }
}

// ────────────────────────────────
// Core benchmark runner (AUDITED)
// ────────────────────────────────

fn percentile(sorted: &[u64], p: usize) -> u64 {
    if sorted.is_empty() { return 0; }
    let idx = (sorted.len() * p) / 100;
    sorted[idx.min(sorted.len() - 1)]
}

/// Returns true if `target_id` appears in `ids` within the first `k` entries.
fn in_top_k(ids: &[u64], target_id: u64, k: usize) -> bool {
    ids.iter().take(k).any(|id| *id == target_id)
}

fn run_mode(flat: &FlatIndex, hnsw: &BinaryHNSW, n: usize, mode: u8) -> YpModeMetrics {
    let count = flat.count();
    let n = n.min(count);
    if n == 0 {
        return YpModeMetrics {
            ism_p50_us: 0, ism_p95_us: 0, ism_p99_us: 0, hnsw_p50_us: 0, hnsw_p95_us: 0, hnsw_p99_us: 0,
            ism_r1: 0.0, hnsw_r1: 0.0, ism_r5: 0.0, hnsw_r5: 0.0,
            ism_r10: 0.0, hnsw_r10: 0.0,
        };
    }

    let mut rng = Lcg::new(0x123456789ABCDEFu64);

    let mut ism_us: Vec<u64> = Vec::with_capacity(n);
    let mut hnsw_us: Vec<u64> = Vec::with_capacity(n);

    let mut ism_r1: usize = 0;
    let mut hnsw_r1: usize = 0;
    let mut hnsw_r5: usize = 0;
    let mut hnsw_r10: usize = 0;

    for _ in 0..n {
        let mut q = [0u8; HASH_BYTES];
        let mut exclude_idx: Option<usize> = None;

        match mode {
            // ── Mode 0: 50-bit perturbation ──
            0 => {
                let idx = rng.range(count as u64) as usize;
                q.copy_from_slice(flat.hash_at(idx));
                for _ in 0..50 {
                    let bit = rng.range(HASH_BYTES as u64 * 8) as usize;
                    q[bit / 8] ^= 1 << (bit % 8);
                }
            }
            // ── Mode 1: Exclude self ──
            1 => {
                let idx = rng.range(count as u64) as usize;
                q.copy_from_slice(flat.hash_at(idx));
                exclude_idx = Some(idx);
            }
            // ── Mode 2: Random query ──
            _ => {
                for byte in q.iter_mut() {
                    *byte = rng.byte();
                }
            }
        }

        // ── ISM exact search (timed) ──
        let t0 = Instant::now();
        let ism_top = match exclude_idx {
            Some(ex) => flat.search_exclude(&q, K, ex),
            None => flat.search(&q, K),
        };
        ism_us.push(t0.elapsed().as_micros() as u64);

        let gt_id = match ism_top.first() {
            Some((id, _)) => *id,
            None => continue,
        };
        ism_r1 += 1; // exact search is always correct

        // ── HNSW approximate search (timed) ──
        let t1 = Instant::now();
        let hnsw_top = hnsw.search(&q, K);
        hnsw_us.push(t1.elapsed().as_micros() as u64);

        // Convert internal (dist, idx, tag) tuples to user-facing IDs.
        // Drop any result whose hash is identical to the query (exclude-self
        // mode), otherwise the query vector trivially outranks its true NN.
        let hnsw_ids: Vec<u64> = hnsw_top.iter()
            .filter(|(_, idx, _)| hnsw.node(*idx).map(|n| n.hash != q).unwrap_or(false))
            .map(|(_, idx, _)| hnsw.node(*idx).map(|n| n.id).unwrap_or(0))
            .collect();

        // ── Recall computation (AUDITED) ──
        if hnsw_ids.first().copied() == Some(gt_id) {
            hnsw_r1 += 1;
        }
        if in_top_k(&hnsw_ids, gt_id, 5) {
            hnsw_r5 += 1;
        }
        if in_top_k(&hnsw_ids, gt_id, 10) {
            hnsw_r10 += 1;
        }
    }

    ism_us.sort_unstable();
    hnsw_us.sort_unstable();

    let n_f = n as f64;

    YpModeMetrics {
        ism_p50_us: percentile(&ism_us, 50),
        ism_p95_us: percentile(&ism_us, 95),
        ism_p99_us: percentile(&ism_us, 99),
        hnsw_p50_us: percentile(&hnsw_us, 50),
        hnsw_p95_us: percentile(&hnsw_us, 95),
        hnsw_p99_us: percentile(&hnsw_us, 99),
        // ISM is exact: R@k = 100% for all k
        ism_r1: 1.0,
        hnsw_r1: hnsw_r1 as f64 / n_f,
        ism_r5: 1.0,
        hnsw_r5: hnsw_r5 as f64 / n_f,
        ism_r10: 1.0,
        hnsw_r10: hnsw_r10 as f64 / n_f,
    }
}

/// Purge OS page cache for the ISM hash buffer (cold-start measurement).
#[no_mangle]
pub extern "C" fn yp_purge_mmap_cache() -> c_int {
    let flat_guard = FLAT.read().unwrap();
    let flat = match flat_guard.as_ref() {
        Some(f) => f,
        None => return -1,
    };
    let ptr = flat.hashes.as_ptr() as *mut c_void;
    let len = flat.hashes.len();
    unsafe {
        libc::madvise(ptr, len, libc::MADV_DONTNEED);
    }
    0
}

/// Generate a deterministic query for the given mode.
/// Returns (query hash, exclude_idx, base_idx).
fn generate_query(mode: u8, flat: &FlatIndex, rng: &mut Lcg) -> ([u8; HASH_BYTES], Option<usize>, usize) {
    let mut q = [0u8; HASH_BYTES];
    let count = flat.count();
    let mut exclude_idx: Option<usize> = None;
    let mut base_idx: usize = 0;
    match mode {
        0 => {
            let idx = rng.range(count as u64) as usize;
            q.copy_from_slice(flat.hash_at(idx));
            base_idx = idx;
            for _ in 0..50 {
                let bit = rng.range(HASH_BYTES as u64 * 8) as usize;
                q[bit / 8] ^= 1 << (bit % 8);
            }
        }
        1 => {
            let idx = rng.range(count as u64) as usize;
            q.copy_from_slice(flat.hash_at(idx));
            exclude_idx = Some(idx);
            base_idx = idx;
        }
        _ => {
            for byte in q.iter_mut() {
                *byte = rng.byte();
            }
        }
    }
    (q, exclude_idx, base_idx)
}

/// Run the ISM-only phase of the sequential shootout. Stores queries for the
/// later HNSW phase and returns ISM latency/recall metrics.
#[no_mangle]
pub extern "C" fn yp_shootout_run_ism_phase(n: usize, metrics: *mut YpShootoutMetrics) -> c_int {
    let flat_guard = FLAT.read().unwrap();
    let flat = match flat_guard.as_ref() {
        Some(f) => f,
        None => return -1,
    };

    let count = flat.count();
    let n = n.min(count);
    if n == 0 || metrics.is_null() {
        return -3;
    }

    let mut queries_per_mode: Vec<Vec<Query>> = Vec::with_capacity(3);
    let mut mode_metrics: [Option<YpModeMetrics>; 3] = [None, None, None];

    for mode in 0..3 {
        let mut rng = Lcg::new(0x123456789ABCDEFu64);
        let mut queries = Vec::with_capacity(n);
        let mut ism_us: Vec<u64> = Vec::with_capacity(n);

        for _ in 0..n {
            let (q, exclude_idx, base_idx) = generate_query(mode as u8, flat, &mut rng);
            let t0 = Instant::now();
            let ism_top = match exclude_idx {
                Some(ex) => flat.search_exclude(&q, K, ex),
                None => flat.search(&q, K),
            };
            ism_us.push(t0.elapsed().as_micros() as u64);

            let gt_id = match ism_top.first() {
                Some((id, _)) => *id,
                None => continue,
            };
            queries.push(Query { hash: q, exclude_idx, base_idx, gt_id });
        }

        queries_per_mode.push(queries);
        ism_us.sort_unstable();

        mode_metrics[mode] = Some(YpModeMetrics {
            ism_p50_us: percentile(&ism_us, 50),
            ism_p95_us: percentile(&ism_us, 95),
            ism_p99_us: percentile(&ism_us, 99),
            hnsw_p50_us: 0,
            hnsw_p95_us: 0,
            hnsw_p99_us: 0,
            ism_r1: 1.0,
            hnsw_r1: 0.0,
            ism_r5: 1.0,
            hnsw_r5: 0.0,
            ism_r10: 1.0,
            hnsw_r10: 0.0,
        });
    }

    *QUERIES.write().unwrap() = Some(queries_per_mode);

    unsafe {
        (*metrics).perturb_50 = mode_metrics[0].unwrap();
        (*metrics).exclude_self = mode_metrics[1].unwrap();
        (*metrics).random_query = mode_metrics[2].unwrap();
        (*metrics).ism_mem_kb = flat.memory_bytes() / 1024;
        (*metrics).hnsw_mem_kb = 0;
        (*metrics).count = count as u64;
    }
    0
}

/// Run the HNSW-only phase of the sequential shootout. Uses queries saved by
/// the ISM phase and returns HNSW latency/recall metrics.
#[no_mangle]
pub extern "C" fn yp_shootout_run_hnsw_phase(n: usize, metrics: *mut YpShootoutMetrics) -> c_int {
    let mut hnsw_guard = HNSW.write().unwrap();
    let hnsw = match hnsw_guard.as_mut() {
        Some(h) => h,
        None => return -2,
    };
    let ef = HNSW_EF_SEARCH.load(Ordering::Relaxed).max(1) as usize;
    hnsw.set_ef_search(ef);

    let queries_guard = QUERIES.read().unwrap();
    let queries_per_mode = match queries_guard.as_ref() {
        Some(q) => q,
        None => return -4,
    };

    if queries_per_mode.len() != 3 {
        return -5;
    }

    let count = hnsw.len();
    let n = n.min(count);
    if n == 0 || metrics.is_null() {
        return -3;
    }

    let mut mode_metrics: [Option<YpModeMetrics>; 3] = [None, None, None];

    for mode in 0..3 {
        let queries = &queries_per_mode[mode];
        let actual_n = queries.len().min(n);
        if actual_n == 0 {
            return -6;
        }

        let mut hnsw_us: Vec<u64> = Vec::with_capacity(actual_n);
        let mut hnsw_r1: usize = 0;
        let mut hnsw_r5: usize = 0;
        let mut hnsw_r10: usize = 0;

        for q in queries.iter().take(actual_n) {
            let t1 = Instant::now();
            let hnsw_top = hnsw.search(&q.hash, K);
            hnsw_us.push(t1.elapsed().as_micros() as u64);

            // Exclude the query vector itself (it is an indexed vector in
            // exclude-self mode and would otherwise trivially rank first).
            let hnsw_ids: Vec<u64> = hnsw_top.iter()
                .filter(|(_, idx, _)| hnsw.node(*idx).map(|n| n.hash != q.hash).unwrap_or(false))
                .map(|(_, idx, _)| hnsw.node(*idx).map(|n| n.id).unwrap_or(0))
                .collect();

            if hnsw_ids.first().copied() == Some(q.gt_id) {
                hnsw_r1 += 1;
            }
            if in_top_k(&hnsw_ids, q.gt_id, 5) {
                hnsw_r5 += 1;
            }
            if in_top_k(&hnsw_ids, q.gt_id, 10) {
                hnsw_r10 += 1;
            }
        }

        hnsw_us.sort_unstable();
        let n_f = actual_n as f64;

        mode_metrics[mode] = Some(YpModeMetrics {
            ism_p50_us: 0,
            ism_p95_us: 0,
            ism_p99_us: 0,
            hnsw_p50_us: percentile(&hnsw_us, 50),
            hnsw_p95_us: percentile(&hnsw_us, 95),
            hnsw_p99_us: percentile(&hnsw_us, 99),
            ism_r1: 0.0,
            hnsw_r1: hnsw_r1 as f64 / n_f,
            ism_r5: 0.0,
            hnsw_r5: hnsw_r5 as f64 / n_f,
            ism_r10: 0.0,
            hnsw_r10: hnsw_r10 as f64 / n_f,
        });
    }

    unsafe {
        (*metrics).perturb_50 = mode_metrics[0].unwrap();
        (*metrics).exclude_self = mode_metrics[1].unwrap();
        (*metrics).random_query = mode_metrics[2].unwrap();
        (*metrics).ism_mem_kb = 0;
        (*metrics).hnsw_mem_kb = hnsw.memory_bytes() / 1024;
        (*metrics).count = count as u64;
    }
    0
}

/// HNSW-only phase with EXACT BRUTE-FORCE ground truth — for corpora that
/// have no ISM file (3M, 5.1M, 10M). Same three honest modes as the
/// sequential shootout; the ground-truth id is computed by a full POPCNT
/// scan over the graph's node hashes (excluding the source node in
/// exclude-self mode), never by the index under test. Brute-force time is
/// NOT included in any latency field — each timer wraps only `search()`.
///
/// Recall definition is identical to the audited `run_mode`:
///   R@k = 1 if ground_truth_id ∈ hnsw_top_k_results
///
/// The `ism_*` fields of the returned metrics are zero — there is no ISM
/// reference here; the ground truth is exact brute force, which is stronger.
#[no_mangle]
pub extern "C" fn yp_shootout_run_hnsw_gt_phase(n: usize, metrics: *mut YpShootoutMetrics) -> c_int {
    let hnsw_guard = HNSW.read().unwrap();
    let hnsw = match hnsw_guard.as_ref() {
        Some(h) => h,
        None => return -2,
    };
    let ef = HNSW_EF_SEARCH.load(Ordering::Relaxed).max(1) as usize;
    hnsw.set_ef_search(ef); // the global is authoritative — files store build-time ef (observed 2026-09-10: without this the gt phase silently searched at the file's ef, making every sweep pass identical)

    let count = hnsw.len();
    let n = n.min(count);
    if n == 0 || metrics.is_null() {
        return -3;
    }

    /// Exact nearest neighbor by full POPCNT scan over node hashes.
    /// `exclude_idx` skips the source node (exclude-self mode).
    fn brute_force_gt(hnsw: &BinaryHNSW, query: &Hash512, exclude_idx: Option<u32>) -> Option<u64> {
        let mut best_dist = u32::MAX;
        let mut best_id = 0u64;
        let mut found = false;
        for i in 0..hnsw.len() {
            if exclude_idx == Some(i as u32) {
                continue;
            }
            let node = match hnsw.node(i as u32) {
                Some(nd) => nd,
                None => continue,
            };
            let d = hamming_distance(query, &node.hash);
            if d < best_dist {
                best_dist = d;
                best_id = node.id;
                found = true;
            }
        }
        if found { Some(best_id) } else { None }
    }

    let mut mode_metrics: [Option<YpModeMetrics>; 3] = [None, None, None];

    for mode in 0..3u8 {
        // Same deterministic LCG scheme as run_mode — one stream per mode.
        let mut rng = Lcg::new(0x2545_F491_4F6C_DD1D ^ (mode as u64));
        let mut hnsw_us: Vec<u64> = Vec::with_capacity(n);
        let mut hnsw_r1: usize = 0;
        let mut hnsw_r5: usize = 0;
        let mut hnsw_r10: usize = 0;
        let mut scored = 0usize;

        for _ in 0..n {
            // ── Query generation (identical semantics to generate_query) ──
            let mut q = [0u8; HASH_BYTES];
            let mut exclude_idx: Option<u32> = None;
            match mode {
                0 => {
                    let idx = rng.range(count as u64) as u32;
                    q.copy_from_slice(&hnsw.node(idx).map(|nd| nd.hash).unwrap_or([0u8; HASH_BYTES]));
                    for _ in 0..50 {
                        let bit = rng.range(HASH_BYTES as u64 * 8) as usize;
                        q[bit / 8] ^= 1 << (bit % 8);
                    }
                }
                1 => {
                    let idx = rng.range(count as u64) as u32;
                    q.copy_from_slice(&hnsw.node(idx).map(|nd| nd.hash).unwrap_or([0u8; HASH_BYTES]));
                    exclude_idx = Some(idx);
                }
                _ => {
                    for byte in q.iter_mut() {
                        *byte = rng.byte();
                    }
                }
            }
            let q_ref: &[u8; HASH_BYTES] = &q;

            // ── Exact brute-force ground truth (UNTIMED) ──
            let gt_id = match brute_force_gt(hnsw, q_ref, exclude_idx) {
                Some(id) => id,
                None => continue,
            };

            // ── HNSW approximate search (timed) ──
            let t1 = Instant::now();
            let hnsw_top = hnsw.search(q_ref, K);
            hnsw_us.push(t1.elapsed().as_micros() as u64);

            // Drop results whose hash is identical to the query (exclude-self),
            // matching the audited run_mode filtering.
            let hnsw_ids: Vec<u64> = hnsw_top.iter()
                .filter(|(_, idx, _)| hnsw.node(*idx).map(|nd| nd.hash != q).unwrap_or(false))
                .map(|(_, idx, _)| hnsw.node(*idx).map(|nd| nd.id).unwrap_or(0))
                .collect();

            scored += 1;
            if hnsw_ids.first().copied() == Some(gt_id) {
                hnsw_r1 += 1;
            }
            if in_top_k(&hnsw_ids, gt_id, 5) {
                hnsw_r5 += 1;
            }
            if in_top_k(&hnsw_ids, gt_id, 10) {
                hnsw_r10 += 1;
            }
        }

        if scored == 0 {
            return -6;
        }
        hnsw_us.sort_unstable();
        let n_f = scored as f64;

        mode_metrics[mode as usize] = Some(YpModeMetrics {
            ism_p50_us: 0,
            ism_p95_us: 0,
            ism_p99_us: 0,
            hnsw_p50_us: percentile(&hnsw_us, 50),
            hnsw_p95_us: percentile(&hnsw_us, 95),
            hnsw_p99_us: percentile(&hnsw_us, 99),
            ism_r1: 0.0,
            hnsw_r1: hnsw_r1 as f64 / n_f,
            ism_r5: 0.0,
            hnsw_r5: hnsw_r5 as f64 / n_f,
            ism_r10: 0.0,
            hnsw_r10: hnsw_r10 as f64 / n_f,
        });
    }

    unsafe {
        (*metrics).perturb_50 = mode_metrics[0].unwrap();
        (*metrics).exclude_self = mode_metrics[1].unwrap();
        (*metrics).random_query = mode_metrics[2].unwrap();
        (*metrics).ism_mem_kb = 0;
        (*metrics).hnsw_mem_kb = hnsw.memory_bytes() / 1024;
        (*metrics).count = count as u64;
    }
    0
}

// ────────────────────────────────
// Float-space recall of the production two-stage path (5.1M corpus):
//   ITQ hash → HNSW top-cand_k → float cosine re-rank → top-10
// Ground truth: exact cosine brute force over ALL float embeddings,
// computed in ONE batched pass (each file page is read once for the whole
// query batch, so the 3.9GB store costs ~4GB of traffic total, not 4GB×n).
// ────────────────────────────────

#[repr(C)]
pub struct YpFloatRecallMetrics {
    /// exclude_self: query = a real doc's own embedding (self filtered from
    /// results, mirroring a text query that never exactly matches a doc);
    /// GT = nearest OTHER doc by exact cosine.
    pub exclude_r1: f64, pub exclude_r5: f64, pub exclude_r10: f64,
    /// Hash-only baseline: GT in HNSW's first 10 (hamming order, no re-rank).
    pub exclude_hash_r10: f64,
    /// Ceiling: GT anywhere in the HNSW candidate set (what re-rank could fix).
    pub exclude_cand: f64,
    /// noise_0.05: query = normalized(doc embedding + gaussian σ=0.05) — a
    /// proxy for a real text encoding landing near (not on) a doc.
    pub noise_r1: f64, pub noise_r5: f64, pub noise_r10: f64,
    pub noise_hash_r10: f64,
    pub noise_cand: f64,
    pub hnsw_p50_us: u64, pub hnsw_p95_us: u64,
    pub count: u64,
    pub ran: i32,
}

const FLOAT_DIM: usize = 384;

/// Exact IEEE-754 binary16 → binary32 via the exponent-shift trick: the f16
/// mantissa+exponent bits land in an f32 subnormal, and one multiply by 2^112
/// normalizes exactly (handles f16 subnormals too). Sign applied separately.
/// Integer-only + one multiply — LLVM vectorizes this to NEON.
#[inline(always)]
fn f16_to_f32_exact(h: u16) -> f32 {
    let mag = f32::from_bits(((h & 0x7fff) as u32) << 13) * f32::from_bits(0x7780_0000); // ×2^112
    if h & 0x8000 != 0 { -mag } else { mag }
}

/// Decode one 384-dim f16 row (little-endian, 768 bytes) into a f32 buffer.
#[inline(always)]
fn decode_row(bytes: &[u8], out: &mut [f32; FLOAT_DIM]) {
    for d in 0..FLOAT_DIM {
        let h = u16::from_le_bytes([bytes[d * 2], bytes[d * 2 + 1]]);
        out[d] = f16_to_f32_exact(h);
    }
}

const F16_HDR: u64 = 32;
const ROW_BYTES: usize = FLOAT_DIM * 2;

/// Read one embedding row by index via pread — NO mmap, so the 3.9GB store
/// costs zero virtual address space (the graph+HNSW VA ceiling on iOS is the
/// reason this phase cannot mmap the store alongside the graph).
fn read_f16_row(file: &File, row: usize, raw: &mut [u8; ROW_BYTES],
                out: &mut [f32; FLOAT_DIM]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(raw, F16_HDR + (row * ROW_BYTES) as u64)?;
    decode_row(raw, out);
    Ok(())
}

/// Run the float-space recall phase. Requires the HNSW graph already loaded
/// via `yp_shootout_load_hnsw`. `cand_k` must mirror production (50 for
/// topK=10 semantic search). Returns 0 on success; metrics.ran == 1.
#[no_mangle]
pub extern "C" fn yp_shootout_run_float_recall_phase(
    f16_path: *const c_char,
    itq_path: *const c_char,
    hnsw_path: *const c_char,
    n: usize,
    cand_k: usize,
    ef: u32,
    out: *mut YpFloatRecallMetrics,
) -> c_int {
    use std::ffi::CString;
    if out.is_null() || n == 0 { return -1; }
    unsafe { (*out).ran = 0; }

    let f16_path = unsafe { CStr::from_ptr(f16_path).to_str().unwrap_or("") };
    let itq_path = unsafe { CStr::from_ptr(itq_path).to_str().unwrap_or("") };
    let hnsw_path = unsafe { CStr::from_ptr(hnsw_path).to_str().unwrap_or("") };

    // 1) ITQ model (same file the app bundles; no-op if already initialized).
    let itq_c = match CString::new(itq_path) {
        Ok(c) => c,
        Err(_) => return -8,
    };
    // Idempotent: returns 0 even if a model is already loaded.
    let itq_rc = crate::ffi_itq::yp_itq_init(itq_c.as_ptr());
    if itq_rc != 0 {
        crate::flat_ffi::set_shootout_error(format!("yp_itq_init({itq_path}) rc={itq_rc}"));
        return -2;
    }

    // 2) Open the f16 embedding store and validate its header. Deliberately
    //    NOT mmap'd: the 3.9GB map plus the ~3.1GB graph blows the iOS
    //    per-process VA ceiling either way (observed ENOMEM in both orders),
    //    which also means production cannot hold both — see isAvailable.
    let file = match File::open(f16_path) {
        Ok(f) => f,
        Err(e) => { crate::flat_ffi::set_shootout_error(format!("float store open {f16_path}: {e}")); return -3; }
    };
    let mut hdr = [0u8; F16_HDR as usize];
    use std::os::unix::fs::FileExt;
    if file.read_exact_at(&mut hdr, 0).is_err() || &hdr[0..4] != b"YPF1" {
        crate::flat_ffi::set_shootout_error(format!("{f16_path}: bad YPF1 header"));
        return -4;
    }
    let dim = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let store_count = u64::from_le_bytes(hdr[12..20].try_into().unwrap()) as usize;
    if dim != FLOAT_DIM || store_count == 0 {
        crate::flat_ffi::set_shootout_error(format!("{f16_path}: dim={dim} count={store_count} (expected dim 384)"));
        return -4;
    }

    // 3) HNSW graph — loaded locally, AFTER the float store is mapped.
    let hnsw = match BinaryHNSW::load(hnsw_path) {
        Ok(h) => h,
        Err(e) => {
            crate::flat_ffi::set_shootout_error(format!("float-recall HNSW load {hnsw_path}: {e}"));
            return -5;
        }
    };
    hnsw.set_ef_search(ef.max(1) as usize);
    let count = store_count.min(hnsw.len());
    if count < store_count {
        crate::flat_ffi::set_shootout_error(format!(
            "HNSW has {} docs, float store has {store_count} — counts must match", hnsw.len()));
        return -5;
    }

    // 4) Identity sanity: HNSW node id == float row index. Sampled ids must
    //    hash to ITQ(row). If this fails every recall number is garbage.
    let mut raw_row = [0u8; ROW_BYTES];
    for &id in &[0u64, (count / 2) as u64, (count - 1) as u64] {
        let node_hash = match hnsw.hash_by_id(id) {
            Some(h) => *h,
            None => {
                crate::flat_ffi::set_shootout_error(format!("identity check: HNSW id {id} missing"));
                return -7;
            }
        };
        let mut row = [0f32; FLOAT_DIM];
        if read_f16_row(&file, id as usize, &mut raw_row, &mut row).is_err() {
            crate::flat_ffi::set_shootout_error(format!("identity check: float row {id} unreadable"));
            return -7;
        }
        let mut q_hash = [0u8; HASH_BYTES];
        let rc = crate::ffi_itq::yp_itq_encode(row.as_ptr(), FLOAT_DIM, q_hash.as_mut_ptr());
        let hd = hamming_distance(&q_hash, &node_hash);
        if rc != 0 || hd > 8 {
            // Deterministic model fingerprint: ITQ of the all-zeros vector.
            // Lets us distinguish "wrong model loaded" from "wrong row read".
            let zeros = [0f32; FLOAT_DIM];
            let mut z_hash = [0u8; HASH_BYTES];
            let _ = crate::ffi_itq::yp_itq_encode(zeros.as_ptr(), FLOAT_DIM, z_hash.as_mut_ptr());
            crate::flat_ffi::set_shootout_error(format!(
                "identity check failed: HNSW id {id} hamming(ITQ(float row), node hash)={hd} \
                 (rc={rc}; itq={:?} node={:?}; itq_zero={:?}; itq_path={itq_path}; f16_path={f16_path})",
                &q_hash[..8], &node_hash[..8], &z_hash[..8]));
            return -7;
        }
    }

    // 5) Query batch: mode 0 = exclude_self (q = doc embedding, self excluded
    //    from GT and filtered from results), mode 1 = noise σ=0.05, renormalized.
    let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15);
    let mut queries: Vec<[f32; FLOAT_DIM]> = Vec::with_capacity(2 * n);
    let mut exclude: Vec<Option<u64>> = Vec::with_capacity(2 * n);
    {
        let mut row = [0f32; FLOAT_DIM];
        for mode in 0..2u8 {
            for _ in 0..n {
                let base = rng.range(count as u64) as usize;
                if read_f16_row(&file, base, &mut raw_row, &mut row).is_err() {
                    crate::flat_ffi::set_shootout_error(format!("query gen: float row {base} unreadable"));
                    return -6;
                }
                let mut q = row;
                if mode == 1 {
                    // Box-Muller gaussian σ=0.05 per dim (pairs).
                    let mut d = 0;
                    while d < FLOAT_DIM {
                        let u1 = ((rng.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
                        let u2 = ((rng.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
                        let r = (-2.0 * u1.ln()).sqrt() * 0.05;
                        let th = 2.0 * std::f64::consts::PI * u2;
                        q[d] += (r * th.cos()) as f32;
                        if d + 1 < FLOAT_DIM { q[d + 1] += (r * th.sin()) as f32; }
                        d += 2;
                    }
                }
                // Unit-normalize the query (production queries are normalized;
                // the re-ranker ranks by raw dot, identical to cosine then).
                let norm: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 0.0 { for v in q.iter_mut() { *v /= norm; } }
                queries.push(q);
                exclude.push(if mode == 0 { Some(base as u64) } else { None });
            }
        }
    }
    let nq = queries.len();

    // 6) ONE batched brute-force pass over the store (streaming buffered
    //    read — same single disk pass as mmap, zero VA): per row, decode
    //    once, then dot against every query. Best cosine per query wins.
    let t_gt = Instant::now();
    let mut best_cos = vec![f32::NEG_INFINITY; nq];
    let mut best_id = vec![u64::MAX; nq];
    let mut row = [0f32; FLOAT_DIM];
    let mut reader = std::io::BufReader::with_capacity(
        1024 * 1024,
        match file.try_clone() {
            Ok(f) => f,
            Err(e) => { crate::flat_ffi::set_shootout_error(format!("float store clone: {e}")); return -6; }
        },
    );
    use std::io::{Read as _, Seek, SeekFrom};
    if reader.seek(SeekFrom::Start(F16_HDR)).is_err() {
        crate::flat_ffi::set_shootout_error("float store seek failed".into());
        return -6;
    }
    for r in 0..count {
        if reader.read_exact(&mut raw_row).is_err() {
            crate::flat_ffi::set_shootout_error(format!("GT pass: row {r} truncated"));
            return -6;
        }
        decode_row(&raw_row, &mut row);
        let mut sumsq = 0f32;
        for &v in row.iter() { sumsq += v * v; }
        if sumsq <= 0.0 { continue; }
        let inv = 1.0 / sumsq.sqrt();
        // queries are unit norm → cosine = dot × inv
        for m in 0..nq {
            if exclude[m] == Some(r as u64) { continue; }
            let q = &queries[m];
            let mut dot = 0f32;
            for d in 0..FLOAT_DIM { dot += q[d] * row[d]; }
            let cos = dot * inv;
            if cos > best_cos[m] { best_cos[m] = cos; best_id[m] = r as u64; }
        }
    }
    let gt_ms = t_gt.elapsed().as_millis();
    eprintln!("[float-recall] batched GT pass: {gt_ms}ms for {nq} queries over {count} rows");

    // 7) Production path per query: ITQ hash → HNSW top-cand_k → float re-rank.
    let mut hnsw_us: Vec<u64> = Vec::with_capacity(nq);
    // per-mode accumulators: [r1, r5, r10, hash_r10, cand]
    let mut acc = [[0usize; 5]; 2];
    let mut cand_row = [0f32; FLOAT_DIM];
    for (m, q) in queries.iter().enumerate() {
        let mut q_hash = [0u8; HASH_BYTES];
        let rc = crate::ffi_itq::yp_itq_encode(q.as_ptr(), FLOAT_DIM, q_hash.as_mut_ptr());
        if rc != 0 { continue; }
        let t = Instant::now();
        let top = hnsw.search(&q_hash, cand_k.max(10));
        hnsw_us.push(t.elapsed().as_micros() as u64);

        let self_id = exclude[m];
        let cand_ids: Vec<u64> = top.iter()
            .filter_map(|(_, idx, _)| hnsw.node(*idx).map(|nd| nd.id))
            .filter(|id| Some(*id) != self_id)
            .collect();
        let gt = best_id[m];

        // Ceiling: GT in the candidate set at all.
        if cand_ids.contains(&gt) { acc[m / n][4] += 1; }

        // Hash-only baseline: GT in the first 10 (hamming order).
        if cand_ids.iter().take(10).any(|&id| id == gt) { acc[m / n][3] += 1; }

        // Production re-rank: exact float dot, sort desc, top-10.
        let mut scored: Vec<(f32, u64)> = cand_ids.iter().map(|&id| {
            if read_f16_row(&file, id as usize, &mut raw_row, &mut cand_row).is_err() {
                return (f32::NEG_INFINITY, id);
            }
            let mut dot = 0f32;
            for d in 0..FLOAT_DIM { dot += q[d] * cand_row[d]; }
            (dot, id)
        }).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let reranked: Vec<u64> = scored.into_iter().take(10).map(|(_, id)| id).collect();
        for (k, &id) in reranked.iter().enumerate() {
            if id == gt {
                if k == 0 { acc[m / n][0] += 1; }
                if k < 5 { acc[m / n][1] += 1; }
                acc[m / n][2] += 1;
                break;
            }
        }
    }

    hnsw_us.sort_unstable();
    let nf = n as f64;
    unsafe {
        (*out).exclude_r1 = acc[0][0] as f64 / nf;
        (*out).exclude_r5 = acc[0][1] as f64 / nf;
        (*out).exclude_r10 = acc[0][2] as f64 / nf;
        (*out).exclude_hash_r10 = acc[0][3] as f64 / nf;
        (*out).exclude_cand = acc[0][4] as f64 / nf;
        (*out).noise_r1 = acc[1][0] as f64 / nf;
        (*out).noise_r5 = acc[1][1] as f64 / nf;
        (*out).noise_r10 = acc[1][2] as f64 / nf;
        (*out).noise_hash_r10 = acc[1][3] as f64 / nf;
        (*out).noise_cand = acc[1][4] as f64 / nf;
        (*out).hnsw_p50_us = percentile(&hnsw_us, 50);
        (*out).hnsw_p95_us = percentile(&hnsw_us, 95);
        (*out).count = count as u64;
        (*out).ran = 1;
    }
    0
}

/// Run all three benchmark modes. Returns 0 on success.
#[no_mangle]
pub extern "C" fn yp_shootout_run(n: usize, metrics: *mut YpShootoutMetrics) -> c_int {
    let flat_guard = FLAT.read().unwrap();
    let flat = match flat_guard.as_ref() {
        Some(f) => f,
        None => return -1,
    };
    let hnsw_guard = HNSW.read().unwrap();
    let hnsw = match hnsw_guard.as_ref() {
        Some(h) => h,
        None => return -2,
    };

    let count = flat.count();
    let n = n.min(count);
    if n == 0 || metrics.is_null() {
        return -3;
    }

    let m1 = run_mode(flat, hnsw, n, 0);
    let m2 = run_mode(flat, hnsw, n, 1);
    let m3 = run_mode(flat, hnsw, n, 2);

    unsafe {
        (*metrics).perturb_50 = m1;
        (*metrics).exclude_self = m2;
        (*metrics).random_query = m3;
        (*metrics).ism_mem_kb = flat.memory_bytes() / 1024;
        (*metrics).hnsw_mem_kb = hnsw.memory_bytes() / 1024;
        (*metrics).count = count as u64;
    }
    0
}

// ────────────────────────────────
// Two-stage pipeline: HNSW candidates → float32 cosine re-rank
// ────────────────────────────────

/// Re-run the saved shootout queries (modes 0 and 1) through the two-stage
/// pipeline: HNSW top-`cand_k` candidates on hashes, then cosine re-rank of
/// those candidates against the pre-binarized embedding of the query's source
/// document (`base_idx`).
///
/// Ground truth per mode:
///   Mode 0 (perturb):   `gt_id` from the exact ISM phase — provably the source
///                       doc itself (50-bit perturbation << 146-bit nearest other).
///                       The source doc is NOT filtered from candidates here;
///                       excluding it made R@1 structurally 0%.
///   Mode 1 (exclude-self): the true float-space nearest OTHER document,
///                       precomputed offline in `gt_path` (query generation is
///                       a deterministic LCG, so base indices are reproducible).
///                       Hash-space gt is meaningless for re-rank scoring: the
///                       hamming-NN and float-NN agree only ~20% of the time.
///                       Entries are matched by base_idx at runtime; a mismatch
///                       skips the query, so a stale gt file can never produce
///                       wrong numbers (float_gt_queries reports how many ran).
///
/// Latency: two passes. The first pass warms the mmap page cache of the
/// embeddings file (untimed; its p50 is reported as `cold_p50_us`), the second
/// pass produces the hot `p50_us`/`p95_us`.
///
/// Requires: HNSW loaded (`yp_shootout_load_hnsw`) and queries saved by a
/// prior `yp_shootout_run_ism_phase` call. `npy_path` is a float32
/// (N × 512) `.npy` file with row i == doc id i (L2-normalized rows).
#[no_mangle]
pub extern "C" fn yp_shootout_rerank_phase(
    npy_path: *const c_char,
    gt_path: *const c_char,
    cand_k: usize,
    final_k: usize,
    out: *mut YpRerankMetrics,
) -> c_int {
    if out.is_null() || npy_path.is_null() {
        return -3;
    }
    let path = unsafe { CStr::from_ptr(npy_path).to_str().unwrap_or("") };
    let engine = match unsafe { crate::rerank_engine::RerankEngine::new(path) } {
        Ok(e) => e,
        Err(_) => return -7,
    };

    // Float-space ground truth for exclude-self: base_idx -> true cosine NN (other).
    let float_gt: std::collections::HashMap<u64, u64> = if !gt_path.is_null() {
        let gt = unsafe { CStr::from_ptr(gt_path).to_str().unwrap_or("") };
        load_float_gt(gt).unwrap_or_default()
    } else {
        std::collections::HashMap::new()
    };

    let mut hnsw_guard = HNSW.write().unwrap();
    let hnsw = match hnsw_guard.as_mut() {
        Some(h) => h,
        None => return -2,
    };
    let ef = HNSW_EF_SEARCH.load(Ordering::Relaxed).max(1) as usize;
    hnsw.set_ef_search(ef);

    let queries_guard = QUERIES.read().unwrap();
    let queries_per_mode = match queries_guard.as_ref() {
        Some(q) => q,
        None => return -4,
    };
    if queries_per_mode.len() != 3 {
        return -5;
    }

    let cand_k = cand_k.max(final_k).max(1);
    let final_k = final_k.max(1);

    let mut cold_us: Vec<u64> = Vec::new();
    let mut hot_us: Vec<u64> = Vec::new();
    let mut r1 = [0usize; 2];
    let mut r5 = [0usize; 2];
    let mut r10 = [0usize; 2];
    let mut cnt = [0usize; 2];
    let mut gt_cnt = 0usize;

    for timed in [false, true] {
        for mode in 0..2usize {
            for q in queries_per_mode[mode].iter() {
                let query_emb = match engine.embedding(q.base_idx as u64) {
                    Some(e) => e,
                    None => continue,
                };
                // Ground truth used for scoring this query.
                let gt_id = if mode == 0 {
                    q.gt_id
                } else {
                    match float_gt.get(&(q.base_idx as u64)) {
                        Some(g) => *g,
                        // No float gt for this query — skip rather than score
                        // against the hash-space gt (meaningless for re-rank).
                        None => continue,
                    }
                };

                let t0 = Instant::now();
                let hnsw_top = hnsw.search(&q.hash, cand_k);

                // Drop candidates whose hash is identical to the query hash.
                // Additionally, ONLY in exclude-self mode, drop the source doc
                // itself (it is the excluded document). In perturb mode the
                // source doc IS the ground truth and must stay.
                let cand_ids: Vec<u64> = hnsw_top.iter()
                    .filter(|(_, idx, _)| hnsw.node(*idx).map(|n| n.hash != q.hash).unwrap_or(false))
                    .map(|(_, idx, _)| hnsw.node(*idx).map(|n| n.id).unwrap_or(0))
                    .filter(|id| mode == 0 || *id != q.base_idx as u64)
                    .collect();

                let top = engine.rerank(&cand_ids, query_emb, final_k);
                let lat = t0.elapsed().as_micros() as u64;

                if timed {
                    hot_us.push(lat);
                    cnt[mode] += 1;
                    if mode == 1 {
                        gt_cnt += 1;
                    }
                    if top.first().map(|(id, _)| *id).unwrap_or(u64::MAX) == gt_id {
                        r1[mode] += 1;
                    }
                    let ids: Vec<u64> = top.iter().map(|(id, _)| *id).collect();
                    if in_top_k(&ids, gt_id, 5) {
                        r5[mode] += 1;
                    }
                    if in_top_k(&ids, gt_id, 10) {
                        r10[mode] += 1;
                    }
                } else {
                    // Warm-up pass: same work, untimed. Touch the gt row so its
                    // pages are resident for the timed pass.
                    cold_us.push(lat);
                    if mode == 1 {
                        let _ = engine.embedding(gt_id);
                    }
                }
            }
        }
    }

    let safe_div = |num: usize, den: usize| if den > 0 { num as f64 / den as f64 } else { 0.0 };

    unsafe {
        (*out).perturb_r1 = safe_div(r1[0], cnt[0]);
        (*out).perturb_r5 = safe_div(r5[0], cnt[0]);
        (*out).perturb_r10 = safe_div(r10[0], cnt[0]);
        (*out).exclude_r1 = safe_div(r1[1], cnt[1]);
        (*out).exclude_r5 = safe_div(r5[1], cnt[1]);
        (*out).exclude_r10 = safe_div(r10[1], cnt[1]);
        (*out).p50_us = percentile(&hot_us, 50);
        (*out).p95_us = percentile(&hot_us, 95);
        (*out).cold_p50_us = if cold_us.is_empty() { 0 } else { percentile(&cold_us, 50) };
        (*out).float_gt_queries = gt_cnt as u32;
        (*out).ran = 1;
    }
    0
}

/// Parse the precomputed float-space ground-truth file:
/// magic "YPFGT001" (8B) | count u32 LE | count × (base_idx u64 LE, float_gt u64 LE).
fn load_float_gt(path: &str) -> Option<std::collections::HashMap<u64, u64>> {
    let buf = std::fs::read(path).ok()?;
    if buf.len() < 12 || &buf[..8] != b"YPFGT001" {
        return None;
    }
    let count = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let mut map = std::collections::HashMap::with_capacity(count);
    let mut off = 12;
    for _ in 0..count {
        if off + 16 > buf.len() {
            break;
        }
        let base = u64::from_le_bytes(buf[off..off + 8].try_into().ok()?);
        let gt = u64::from_le_bytes(buf[off + 8..off + 16].try_into().ok()?);
        map.insert(base, gt);
        off += 16;
    }
    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

/// Raw exact search over the shootout ISM static (the index loaded by
/// `yp_shootout_load_ism`). Unlike `yp_ism_search` (which reads the
/// NativeEngine's separate `ISM_FLAT` static), this queries the same index the
/// test runner just loaded. Fills `out_ids`/`out_dists` with the top-k
/// (id, hamming distance) pairs, ascending by distance. Returns count written.
#[no_mangle]
pub extern "C" fn yp_shootout_raw_search_ism(
    hash: *const u8,
    hash_len: usize,
    k: usize,
    out_ids: *mut u64,
    out_dists: *mut u32,
) -> usize {
    if hash.is_null() || hash_len != HASH_BYTES || k == 0 || out_ids.is_null() || out_dists.is_null() {
        return 0;
    }
    let q: &[u8; HASH_BYTES] = unsafe { &*(hash as *const [u8; HASH_BYTES]) };
    let guard = FLAT.read().unwrap();
    let flat = match guard.as_ref() {
        Some(f) => f,
        None => return 0,
    };
    let top = flat.search(q, k);
    let n = top.len().min(k);
    for (i, (id, dist)) in top.iter().take(n).enumerate() {
        unsafe {
            *out_ids.add(i) = *id;
            *out_dists.add(i) = *dist;
        }
    }
    n
}

/// Raw search over the shootout HNSW static (the graph loaded by
/// `yp_shootout_load_hnsw`). Unlike `yp_hnsw_search` (which reads the
/// NativeEngine's separate `HNSW_CHUNK` static), this queries the same graph
/// the test runner just loaded. Fills `out_ids`/`out_dists` with the top-k
/// (id, hamming distance) pairs. Returns count written.
#[no_mangle]
pub extern "C" fn yp_shootout_raw_search_hnsw(
    hash: *const u8,
    hash_len: usize,
    k: usize,
    out_ids: *mut u64,
    out_dists: *mut u32,
) -> usize {
    if hash.is_null() || hash_len != HASH_BYTES || k == 0 || out_ids.is_null() || out_dists.is_null() {
        return 0;
    }
    let q: &[u8; HASH_BYTES] = unsafe { &*(hash as *const [u8; HASH_BYTES]) };
    let guard = HNSW.read().unwrap();
    let hnsw = match guard.as_ref() {
        Some(h) => h,
        None => return 0,
    };
    let top = hnsw.search(q, k);
    let mut n = 0usize;
    for (_, idx, _) in top.iter().take(k) {
        if let Some(node) = hnsw.node(*idx) {
            unsafe {
                *out_ids.add(n) = node.id;
                *out_dists.add(n) = hamming_distance(q, &node.hash);
            }
            n += 1;
        }
    }
    n
}
